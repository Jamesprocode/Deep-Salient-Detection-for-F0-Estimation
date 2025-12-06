#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate.py

Comprehensive evaluation script for F0 estimation methods:
- Autocorrelation baseline
- 6 trained deep learning models (SimpleNet & ResNet with different losses)

Evaluates on vocadito dataset using mir_eval melody metrics.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import csv

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from scipy.io import wavfile
import scipy.signal as sig
import pandas as pd

# Import mir_eval for melody evaluation
try:
    import mir_eval
except ImportError:
    print("Error: mir_eval not found. Install with: pip install mir_eval")
    sys.exit(1)


# ======================= HCQT Configuration =======================
SR = 44100
FMIN_NOTE = "C1"
FMIN = librosa.note_to_hz(FMIN_NOTE)
BINS_PER_OCTAVE = 60
N_OCTAVES = 6
N_BINS = BINS_PER_OCTAVE * N_OCTAVES
HARMONICS = [0.5, 1, 2, 3, 4, 5]
HOP_LENGTH = 512  # ~11.6 ms at 44100 Hz


# ======================= Model Definitions =======================

class SimpleHcqtCNN(nn.Module):
    """Simple CNN for HCQT -> salience prediction."""
    def __init__(self, in_channels=6, hidden_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(hidden_channels, 1, kernel_size=1)

        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.bn3 = nn.BatchNorm2d(hidden_channels)

    def forward(self, x):
        """x: (B, C=H, F, T) -> logits (B, F, T)"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv_out(x)
        x = x.squeeze(1)
        return x


class ResNetSalience(nn.Module):
    """ResNet-18 backbone + 1x1 Conv for salience."""
    def __init__(self, in_channels=6, pretrained=True):
        super().__init__()
        backbone = resnet18(pretrained=pretrained)

        if in_channels != 3:
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

            with torch.no_grad():
                if in_channels > 3:
                    w = old_conv.weight.data
                    w_mean = w.mean(dim=1, keepdim=True)
                    new_weight = w_mean.repeat(1, in_channels, 1, 1)
                    new_conv.weight.data[:, :, :, :] = new_weight[:, :, :, :]
                elif in_channels == 1:
                    w = old_conv.weight.data
                    w_mean = w.mean(dim=1, keepdim=True)
                    new_conv.weight.data = w_mean
                else:
                    w = old_conv.weight.data
                    new_conv.weight.data[:, :in_channels, :, :] = w[:, :in_channels, :, :]
            backbone.conv1 = new_conv

        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.salience_head = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        """x: (B, C, F, T) -> logits (B, F, T)"""
        B, C, F_bins, T = x.shape
        feat = self.backbone(x)
        logit_small = self.salience_head(feat)
        logit_up = F.interpolate(
            logit_small,
            size=(F_bins, T),
            mode="bilinear",
            align_corners=False,
        )
        return logit_up.squeeze(1)


# ======================= Autocorrelation Baseline =======================

def block_audio(audio_input, sr=None, frame_size=2048, hop_ratio=0.5, pad=True):
    """Block audio into frames."""
    if isinstance(audio_input, str):
        sr, audio_input = wavfile.read(audio_input)
    elif sr is None:
        raise ValueError("Must provide sampling rate")

    # Convert to float [-1, 1]
    if audio_input.dtype == np.float32 or audio_input.dtype == np.float64:
        pass
    else:
        if audio_input.dtype == np.uint8:
            nbits = 8
        elif audio_input.dtype == np.int16:
            nbits = 16
        elif audio_input.dtype == np.int32:
            nbits = 32
        else:
            raise ValueError(f"Unsupported audio dtype: {audio_input.dtype}")
        audio_input = audio_input / float(2**(nbits - 1))

    # Convert to mono
    if len(audio_input.shape) > 1:
        audio_input = np.mean(audio_input, axis=1)

    hop_size = int(hop_ratio * frame_size)

    if pad:
        num_blocks = max(1, int(np.ceil((len(audio_input) - frame_size) / hop_size)) + 1)
    else:
        num_blocks = max(0, (len(audio_input) - frame_size) // hop_size + 1)

    audio_blocks = np.zeros([num_blocks, frame_size])
    times = (np.arange(0, num_blocks) * hop_size) / sr

    for n in range(num_blocks):
        i_start = n * hop_size
        i_stop = i_start + frame_size

        if i_stop <= len(audio_input):
            audio_blocks[n] = audio_input[i_start:i_stop]
        else:
            remaining_samples = len(audio_input) - i_start
            if remaining_samples > 0:
                audio_blocks[n, :remaining_samples] = audio_input[i_start:]

    return audio_blocks, times


def estimate_f0(audio_frame, sr, minfreq=20, maxfreq=None, threshold=0.25):
    """Autocorrelation-based F0 estimation."""
    if maxfreq is None:
        maxfreq = sr / 8
    if maxfreq == 0 or minfreq == 0:
        raise ValueError('Frequency cannot be 0')

    f0 = np.nan

    # Normalize
    max_val = np.max(np.abs(audio_frame))
    if max_val > 0:
        audio_frame = audio_frame / max_val

    # Window
    audio_frame = audio_frame * sig.windows.blackmanharris(len(audio_frame), sym=False)

    # Detrend
    sig.detrend(audio_frame, type='constant', overwrite_data=True)

    # Convert frequencies to periods
    Tmax = 1 / minfreq
    Tmin = 1 / maxfreq
    Nmax = int(np.ceil(Tmax * sr))
    Nmin = int(np.floor(Tmin * sr))

    # Autocorrelation
    corr = np.correlate(audio_frame, audio_frame, mode='full')
    corr = corr[len(audio_frame)-1:]
    if corr[0] > 0:
        corr = corr / corr[0]

    corr = corr[Nmin:Nmax+1]

    # Find strongest peak
    peak_indices, props = sig.find_peaks(corr, height=threshold, distance=Nmin)
    if len(peak_indices) != 0:
        strongest_peak = peak_indices[np.argmax(props["peak_heights"])]
        k = Nmin + strongest_peak
        f0 = sr / k

    return f0


# ======================= HCQT Computation =======================

def compute_hcqt(y, sr):
    """Compute HCQT for audio."""
    hcqt_list = []
    lengths = []

    for h in HARMONICS:
        cqt = librosa.cqt(
            y,
            sr=sr,
            hop_length=HOP_LENGTH,
            fmin=FMIN * h,
            n_bins=N_BINS,
            bins_per_octave=BINS_PER_OCTAVE,
        )
        mag = np.abs(cqt).astype(np.float32)
        hcqt_list.append(mag)
        lengths.append(mag.shape[1])

    T_min = min(lengths)
    hcqt_list_trimmed = [h[:, :T_min] for h in hcqt_list]
    hcqt = np.stack(hcqt_list_trimmed, axis=0)  # (H, F, T)

    return hcqt


def hz_to_cqt_bin(f_hz):
    """Map frequency in Hz to CQT bin index."""
    if f_hz <= 0 or np.isnan(f_hz):
        return None
    k = BINS_PER_OCTAVE * np.log2(f_hz / FMIN)
    bin_idx = int(np.round(k))
    if bin_idx < 0 or bin_idx >= N_BINS:
        return None
    return bin_idx


def cqt_bin_to_hz(bin_idx):
    """Map CQT bin index to frequency in Hz."""
    if bin_idx < 0 or bin_idx >= N_BINS:
        return 0.0
    f_hz = FMIN * (2 ** (bin_idx / BINS_PER_OCTAVE))
    return f_hz


# ======================= F0 Extraction from Salience =======================

def salience_to_f0_contour(salience, threshold=0.5):
    """
    Extract F0 contour from salience map.

    Parameters
    ----------
    salience : np.ndarray
        Salience map of shape (F, T)
    threshold : float
        Activation threshold (0-1 for sigmoid output)

    Returns
    -------
    f0_contour : np.ndarray
        F0 values in Hz for each frame (shape T,)
        0.0 indicates unvoiced frames
    """
    F, T = salience.shape
    f0_contour = np.zeros(T)

    for t in range(T):
        sal_frame = salience[:, t]

        # Find peak in salience
        max_idx = np.argmax(sal_frame)
        max_val = sal_frame[max_idx]

        if max_val > threshold:
            # Convert bin to Hz
            f0_contour[t] = cqt_bin_to_hz(max_idx)
        else:
            f0_contour[t] = 0.0

    return f0_contour


# ======================= Model Loading =======================

def load_model(model_path: Path, model_type: str, device):
    """Load a trained model."""
    in_channels = len(HARMONICS)

    if model_type == "simple":
        model = SimpleHcqtCNN(in_channels=in_channels)
    elif model_type == "resnet":
        model = ResNetSalience(in_channels=in_channels, pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


# ======================= Evaluation Functions =======================

def evaluate_autocorrelation(audio_path: Path, gt_times, gt_freqs,
                            frame_size=2048, hop_ratio=0.5,
                            minfreq=50, maxfreq=800, threshold=0.25):
    """Evaluate autocorrelation baseline."""
    # Load audio
    sr, audio = wavfile.read(audio_path)

    # Block audio
    frames, times = block_audio(audio, sr=sr, frame_size=frame_size,
                                hop_ratio=hop_ratio, pad=True)

    # Estimate F0 for each frame
    f0s = []
    for frame in frames:
        f0 = estimate_f0(frame, sr, minfreq=minfreq, maxfreq=maxfreq, threshold=threshold)
        # Convert NaN to 0.0 for mir_eval
        f0s.append(0.0 if np.isnan(f0) else f0)

    f0s = np.array(f0s)

    # Evaluate with mir_eval
    scores = mir_eval.melody.evaluate(gt_times, gt_freqs, times, f0s)

    return scores


def evaluate_deep_model(model, audio_path: Path, gt_times, gt_freqs, device):
    """Evaluate a deep learning model."""
    # Load and preprocess audio
    y, sr = librosa.load(audio_path, sr=SR, mono=True)

    # Compute HCQT
    hcqt = compute_hcqt(y, sr)  # (H, F, T)

    # Log-compress
    hcqt = np.log1p(hcqt).astype(np.float32)

    # Convert to torch tensor
    hcqt_tensor = torch.from_numpy(hcqt).unsqueeze(0).to(device)  # (1, H, F, T)

    # Forward pass
    with torch.no_grad():
        logits = model(hcqt_tensor)  # (1, F, T)
        salience = torch.sigmoid(logits).cpu().numpy()[0]  # (F, T)

    # Extract F0 contour
    f0_contour = salience_to_f0_contour(salience, threshold=0.5)

    # Create time array (based on hop_length)
    est_times = librosa.frames_to_time(
        np.arange(len(f0_contour)),
        sr=SR,
        hop_length=HOP_LENGTH
    )

    # Evaluate with mir_eval
    scores = mir_eval.melody.evaluate(gt_times, gt_freqs, est_times, f0_contour)

    return scores


def load_ground_truth(annotation_path: Path):
    """Load ground truth F0 annotations."""
    df = pd.read_csv(annotation_path, header=None, names=['time', 'frequency'])
    times = df['time'].values
    freqs = df['frequency'].values
    return times, freqs


# ======================= Main Evaluation =======================

def main():
    # Configuration
    DATA_DIR = Path("Data 2/Evaluation/vocadito")
    AUDIO_DIR = DATA_DIR / "Audio"
    ANNOT_DIR = DATA_DIR / "Annotations" / "F0"
    MODEL_DIR = Path("model")

    # Model configurations
    MODELS = {
        "AutoCorrelation": {
            "type": "autocorrelation",
            "path": None,
        },
        "SimpleNet_BCE": {
            "type": "simple",
            "path": MODEL_DIR / "simpleNet_with_BCE(epoch30_lr1e-3_batchSize8))" / "best_model.pth",
        },
        "SimpleNet_BCE+MSE": {
            "type": "simple",
            "path": MODEL_DIR / "simpleNet_with_bce_mseloss" / "best_model_hybrid.pth",
        },
        "ResNet_BCE_Frozen": {
            "type": "resnet",
            "path": MODEL_DIR / "Resnet_with_Bce_freezen(epoch30_lr1e-4_batchSize8))" / "best_model_resnet_bce.pth",
        },
        "ResNet_BCE_Unfrozen": {
            "type": "resnet",
            "path": MODEL_DIR / "Resnet_with_BCE_no_freezen(epoch30_lr1e-4_batchSize8))" / "best_model_resnet_bce.pth",
        },
        "ResNet_BCE+MSE_Frozen": {
            "type": "resnet",
            "path": MODEL_DIR / "Resnet_with_BCE+MSE_Freezen(epoch30_lr1e-4_bs8	)" / "best_model_resnet_hybrid.pth",
        },
        "ResNet_BCE+MSE_Unfrozen": {
            "type": "resnet",
            "path": MODEL_DIR / "Resnet_with_BCE+mse_no_freezen(epoch30_lr1e-4_batchSize8))" / "best_model_resnet_hybrid.pth",
        },
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Load all deep learning models
    loaded_models = {}
    for model_name, config in MODELS.items():
        if config["type"] == "autocorrelation":
            continue

        print(f"Loading {model_name}...")
        try:
            model = load_model(config["path"], config["type"], device)
            loaded_models[model_name] = model
            print(f"   Loaded successfully")
        except Exception as e:
            print(f"   Error loading model: {e}")

    print()

    # Get list of test files
    audio_files = sorted(AUDIO_DIR.glob("vocadito_*.wav"))

    if len(audio_files) == 0:
        print("No audio files found!")
        return

    print(f"Found {len(audio_files)} audio files for evaluation\n")

    # Results storage
    all_results = {model_name: [] for model_name in MODELS.keys()}

    # Evaluate each file
    for audio_path in audio_files:
        track_id = audio_path.stem  # e.g., "vocadito_1"
        annot_path = ANNOT_DIR / f"{track_id}_f0.csv"

        if not annot_path.exists():
            print(f"� Skipping {track_id}: annotation not found")
            continue

        print(f"Evaluating: {track_id}")

        # Load ground truth
        gt_times, gt_freqs = load_ground_truth(annot_path)

        # Evaluate each method
        for model_name, config in MODELS.items():
            try:
                if config["type"] == "autocorrelation":
                    scores = evaluate_autocorrelation(
                        audio_path, gt_times, gt_freqs,
                        frame_size=2048, hop_ratio=0.5,
                        minfreq=50, maxfreq=800, threshold=0.25
                    )
                else:
                    model = loaded_models[model_name]
                    scores = evaluate_deep_model(model, audio_path, gt_times, gt_freqs, device)

                all_results[model_name].append(scores)
                print(f"  {model_name}: VR={scores['Voicing Recall']:.3f}, VFA={scores['Voicing False Alarm']:.3f}, RPA={scores['Raw Pitch Accuracy']:.3f}, RCA={scores['Raw Chroma Accuracy']:.3f}, OA={scores['Overall Accuracy']:.3f}")

            except Exception as e:
                print(f"   {model_name}: Error - {e}")

        print()

    # Aggregate results
    print("\n" + "="*80)
    print("FINAL RESULTS (Average across all test files)")
    print("="*80)

    metric_names = ['Voicing Recall', 'Voicing False Alarm', 'Raw Pitch Accuracy',
                   'Raw Chroma Accuracy', 'Overall Accuracy']

    summary_results = {}
    for model_name in MODELS.keys():
        if len(all_results[model_name]) == 0:
            continue

        # Average each metric
        avg_scores = {}
        for metric in metric_names:
            values = [scores[metric] for scores in all_results[model_name]]
            avg_scores[metric] = np.mean(values)

        summary_results[model_name] = avg_scores

        print(f"\n{model_name}:")
        for metric, value in avg_scores.items():
            print(f"  {metric:25s}: {value:.4f}")

    # Save results to CSV
    output_path = Path("evaluation_results.csv")
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # Header
        header = ['Model'] + metric_names
        writer.writerow(header)

        # Data
        for model_name, avg_scores in summary_results.items():
            row = [model_name] + [avg_scores[metric] for metric in metric_names]
            writer.writerow(row)

    print(f"\n Results saved to {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()
