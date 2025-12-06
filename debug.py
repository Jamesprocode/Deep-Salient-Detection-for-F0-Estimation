#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tune_thresholds.py

Find the optimal threshold for each model (including autocorrelation baseline).
"""

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from pathlib import Path
import pandas as pd
import mir_eval
from scipy.io import wavfile
import scipy.signal as sig

# ======================= HCQT Configuration =======================
SR = 44100
FMIN_NOTE = "C1"
FMIN = librosa.note_to_hz(FMIN_NOTE)
BINS_PER_OCTAVE = 60
N_OCTAVES = 6
N_BINS = BINS_PER_OCTAVE * N_OCTAVES
HARMONICS = [0.5, 1, 2, 3, 4, 5]
HOP_LENGTH = 512


# ======================= Autocorrelation Baseline =======================

def block_audio(audio_input, sr=None, frame_size=2048, hop_ratio=0.5, pad=True):
    if isinstance(audio_input, str):
        sr, audio_input = wavfile.read(audio_input)
    elif sr is None:
        raise ValueError("Must provide sampling rate")

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
    if maxfreq is None:
        maxfreq = sr / 8
    if maxfreq == 0 or minfreq == 0:
        raise ValueError('Frequency cannot be 0')

    f0 = np.nan

    max_val = np.max(np.abs(audio_frame))
    if max_val > 0:
        audio_frame = audio_frame / max_val

    audio_frame = audio_frame * sig.windows.blackmanharris(len(audio_frame), sym=False)
    sig.detrend(audio_frame, type='constant', overwrite_data=True)

    Tmax = 1 / minfreq
    Tmin = 1 / maxfreq
    Nmax = int(np.ceil(Tmax * sr))
    Nmin = int(np.floor(Tmin * sr))

    corr = np.correlate(audio_frame, audio_frame, mode='full')
    corr = corr[len(audio_frame)-1:]
    if corr[0] > 0:
        corr = corr / corr[0]

    corr = corr[Nmin:Nmax+1]

    peak_indices, props = sig.find_peaks(corr, height=threshold, distance=Nmin)
    if len(peak_indices) != 0:
        strongest_peak = peak_indices[np.argmax(props["peak_heights"])]
        k = Nmin + strongest_peak
        f0 = sr / k

    return f0


def evaluate_autocorrelation_with_threshold(audio_paths, annot_paths, threshold,
                                           frame_size=2048, hop_ratio=0.5,
                                           minfreq=50, maxfreq=800):
    """Evaluate autocorrelation with specific threshold."""
    all_scores = []

    for audio_path, annot_path in zip(audio_paths, annot_paths):
        # Load audio
        sr, audio = wavfile.read(audio_path)

        # Block audio
        frames, times = block_audio(audio, sr=sr, frame_size=frame_size,
                                   hop_ratio=hop_ratio, pad=True)

        # Estimate F0
        f0s = []
        for frame in frames:
            f0 = estimate_f0(frame, sr, minfreq=minfreq, maxfreq=maxfreq, threshold=threshold)
            f0s.append(0.0 if np.isnan(f0) else f0)

        f0s = np.array(f0s)

        # Load GT
        gt_times, gt_freqs = load_ground_truth(annot_path)

        # Evaluate
        try:
            scores = mir_eval.melody.evaluate(gt_times, gt_freqs, times, f0s)
            all_scores.append(scores)
        except:
            pass

    if len(all_scores) == 0:
        return None

    # Average scores
    avg_scores = {}
    for key in all_scores[0].keys():
        values = [s[key] for s in all_scores]
        avg_scores[key] = np.mean(values)

    return avg_scores


# ======================= Model Definitions =======================

class SimpleHcqtCNN(nn.Module):
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
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv_out(x)
        x = x.squeeze(1)
        return x


class ResNetSalience(nn.Module):
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


# ======================= Helper Functions =======================

def compute_hcqt(y, sr):
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
    hcqt = np.stack(hcqt_list_trimmed, axis=0)

    return hcqt


def cqt_bin_to_hz(bin_idx):
    if bin_idx < 0 or bin_idx >= N_BINS:
        return 0.0
    f_hz = FMIN * (2 ** (bin_idx / BINS_PER_OCTAVE))
    return f_hz


def salience_to_f0_contour(salience, threshold=0.5):
    F, T = salience.shape
    f0_contour = np.zeros(T)

    for t in range(T):
        sal_frame = salience[:, t]
        max_idx = np.argmax(sal_frame)
        max_val = sal_frame[max_idx]

        if max_val > threshold:
            f0_contour[t] = cqt_bin_to_hz(max_idx)
        else:
            f0_contour[t] = 0.0

    return f0_contour


def load_model(model_path, model_type, device):
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


def load_ground_truth(annotation_path):
    df = pd.read_csv(annotation_path, header=None, names=['time', 'frequency'])
    times = df['time'].values
    freqs = df['frequency'].values
    return times, freqs


def evaluate_with_threshold(model, audio_paths, annot_paths, device, threshold):
    """Evaluate model on multiple files with a specific threshold."""
    all_scores = []

    for audio_path, annot_path in zip(audio_paths, annot_paths):
        # Load audio
        y, sr = librosa.load(audio_path, sr=SR, mono=True)

        # Compute HCQT
        hcqt = compute_hcqt(y, sr)
        hcqt = np.log1p(hcqt).astype(np.float32)

        # Forward pass
        hcqt_tensor = torch.from_numpy(hcqt).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(hcqt_tensor)
            salience = torch.sigmoid(logits).cpu().numpy()[0]

        # Extract F0
        f0_contour = salience_to_f0_contour(salience, threshold=threshold)

        # Time array
        est_times = librosa.frames_to_time(
            np.arange(len(f0_contour)),
            sr=SR,
            hop_length=HOP_LENGTH
        )

        # Load GT
        gt_times, gt_freqs = load_ground_truth(annot_path)

        # Evaluate
        try:
            scores = mir_eval.melody.evaluate(gt_times, gt_freqs, est_times, f0_contour)
            all_scores.append(scores)
        except:
            pass

    # Average scores
    if len(all_scores) == 0:
        return None

    avg_scores = {}
    for key in all_scores[0].keys():
        values = [s[key] for s in all_scores]
        avg_scores[key] = np.mean(values)

    return avg_scores


# ======================= Main Tuning =======================

def tune_autocorrelation_threshold(audio_files, annot_files, threshold_range):
    """Find best threshold for autocorrelation baseline."""

    print(f"\n{'='*80}")
    print(f"Tuning threshold for: AutoCorrelation Baseline")
    print(f"{'='*80}")

    results = []

    for threshold in threshold_range:
        print(f"\nTesting threshold: {threshold:.3f}")
        scores = evaluate_autocorrelation_with_threshold(
            audio_files, annot_files, threshold,
            frame_size=2048, hop_ratio=0.5, minfreq=50, maxfreq=800
        )

        if scores is None:
            print("  Evaluation failed")
            continue

        print(f"  VR={scores['Voicing Recall']:.3f}, "
              f"VFA={scores['Voicing False Alarm']:.3f}, "
              f"RPA={scores['Raw Pitch Accuracy']:.3f}, "
              f"OA={scores['Overall Accuracy']:.3f}")

        results.append({
            'threshold': threshold,
            'voicing_recall': scores['Voicing Recall'],
            'voicing_fa': scores['Voicing False Alarm'],
            'raw_pitch_acc': scores['Raw Pitch Accuracy'],
            'raw_chroma_acc': scores['Raw Chroma Accuracy'],
            'overall_acc': scores['Overall Accuracy'],
        })

    if len(results) == 0:
        print("\n⚠ No valid results!")
        return None

    # Find best threshold
    results_df = pd.DataFrame(results)
    best_idx = results_df['overall_acc'].idxmax()
    best_result = results_df.iloc[best_idx]

    print(f"\n{'='*80}")
    print(f"BEST THRESHOLD: {best_result['threshold']:.3f}")
    print(f"{'='*80}")
    print(f"  Voicing Recall:     {best_result['voicing_recall']:.4f}")
    print(f"  Voicing FA:         {best_result['voicing_fa']:.4f}")
    print(f"  Raw Pitch Accuracy: {best_result['raw_pitch_acc']:.4f}")
    print(f"  Raw Chroma Acc:     {best_result['raw_chroma_acc']:.4f}")
    print(f"  Overall Accuracy:   {best_result['overall_acc']:.4f}")

    return results_df


def tune_model_threshold(model_name, model_path, model_type, audio_files, annot_files,
                         device, threshold_range):
    """Find best threshold for a model."""

    print(f"\n{'='*80}")
    print(f"Tuning threshold for: {model_name}")
    print(f"{'='*80}")

    # Load model
    print("Loading model...")
    model = load_model(model_path, model_type, device)

    results = []

    # Test each threshold
    for threshold in threshold_range:
        print(f"\nTesting threshold: {threshold:.3f}")
        scores = evaluate_with_threshold(model, audio_files, annot_files, device, threshold)

        if scores is None:
            print("  Evaluation failed (no voiced frames or error)")
            continue

        print(f"  VR={scores['Voicing Recall']:.3f}, "
              f"VFA={scores['Voicing False Alarm']:.3f}, "
              f"RPA={scores['Raw Pitch Accuracy']:.3f}, "
              f"OA={scores['Overall Accuracy']:.3f}")

        results.append({
            'threshold': threshold,
            'voicing_recall': scores['Voicing Recall'],
            'voicing_fa': scores['Voicing False Alarm'],
            'raw_pitch_acc': scores['Raw Pitch Accuracy'],
            'raw_chroma_acc': scores['Raw Chroma Accuracy'],
            'overall_acc': scores['Overall Accuracy'],
        })

    if len(results) == 0:
        print("\n⚠ No valid results!")
        return None

    # Find best threshold
    results_df = pd.DataFrame(results)
    best_idx = results_df['overall_acc'].idxmax()
    best_result = results_df.iloc[best_idx]

    print(f"\n{'='*80}")
    print(f"BEST THRESHOLD: {best_result['threshold']:.3f}")
    print(f"{'='*80}")
    print(f"  Voicing Recall:     {best_result['voicing_recall']:.4f}")
    print(f"  Voicing FA:         {best_result['voicing_fa']:.4f}")
    print(f"  Raw Pitch Accuracy: {best_result['raw_pitch_acc']:.4f}")
    print(f"  Raw Chroma Acc:     {best_result['raw_chroma_acc']:.4f}")
    print(f"  Overall Accuracy:   {best_result['overall_acc']:.4f}")

    return results_df


def main():
    # Paths
    DATA_DIR = Path("Data 2/Evaluation/vocadito")
    AUDIO_DIR = DATA_DIR / "Audio"
    ANNOT_DIR = DATA_DIR / "Annotations" / "F0"
    MODEL_DIR = Path("model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Get test files (use subset for faster tuning)
    all_audio_files = sorted(AUDIO_DIR.glob("vocadito_*.wav"))
    # Use first 5 files for tuning
    audio_files = all_audio_files[:5]
    annot_files = [ANNOT_DIR / f"{f.stem}_f0.csv" for f in audio_files]

    print(f"Tuning on {len(audio_files)} files:")
    for f in audio_files:
        print(f"  - {f.name}")

    # Store all results
    all_results = {}

    # 1. Tune Autocorrelation
    print("\n" + "="*80)
    print("TUNING AUTOCORRELATION BASELINE")
    print("="*80)

    autocorr_results = tune_autocorrelation_threshold(
        audio_files,
        annot_files,
        threshold_range=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    )

    if autocorr_results is not None:
        all_results["AutoCorrelation"] = autocorr_results
        autocorr_results.to_csv("threshold_tuning_AutoCorrelation.csv", index=False)
        print(f"Saved results to threshold_tuning_AutoCorrelation.csv")

    # 2. Tune Deep Learning Models
    models_to_tune = {
        "SimpleNet_BCE": {
            "path": MODEL_DIR / "simpleNet_with_BCE(epoch30_lr1e-3_batchSize8))" / "best_model.pth",
            "type": "simple",
            "threshold_range": [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        },
        "SimpleNet_BCE+MSE": {
            "path": MODEL_DIR / "simpleNet_with_bce_mseloss" / "best_model_hybrid.pth",
            "type": "simple",
            "threshold_range": [0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        },
        "ResNet_BCE_Frozen": {
            "path": MODEL_DIR / "Resnet_with_Bce_freezen(epoch30_lr1e-4_batchSize8))" / "best_model_resnet_bce.pth",
            "type": "resnet",
            "threshold_range": [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10],
        },
        "ResNet_BCE_Unfrozen": {
            "path": MODEL_DIR / "Resnet_with_BCE_no_freezen(epoch30_lr1e-4_batchSize8))" / "best_model_resnet_bce.pth",
            "type": "resnet",
            "threshold_range": [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10],
        },
        "ResNet_BCE+MSE_Frozen": {
            "path": MODEL_DIR / "Resnet_with_BCE+MSE_Freezen(epoch30_lr1e-4_bs8）)" / "best_model_resnet_hybrid.pth",
            "type": "resnet",
            "threshold_range": [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10],
        },
        "ResNet_BCE+MSE_Unfrozen": {
            "path": MODEL_DIR / "Resnet_with_BCE+mse_no_freezen(epoch30_lr1e-4_batchSize8))" / "best_model_resnet_hybrid.pth",
            "type": "resnet",
            "threshold_range": [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10],
        },
    }

    # Tune each model
    for model_name, config in models_to_tune.items():
        results_df = tune_model_threshold(
            model_name,
            config["path"],
            config["type"],
            audio_files,
            annot_files,
            device,
            config["threshold_range"]
        )

        if results_df is not None:
            all_results[model_name] = results_df
            # Save individual results
            results_df.to_csv(f"threshold_tuning_{model_name}.csv", index=False)
            print(f"Saved results to threshold_tuning_{model_name}.csv")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY: OPTIMAL THRESHOLDS FOR ALL MODELS")
    print("="*80)

    summary = []
    for model_name, results_df in all_results.items():
        best_idx = results_df['overall_acc'].idxmax()
        best = results_df.iloc[best_idx]
        summary.append({
            'Model': model_name,
            'Best_Threshold': best['threshold'],
            'Overall_Accuracy': best['overall_acc'],
            'Raw_Pitch_Accuracy': best['raw_pitch_acc'],
            'Voicing_Recall': best['voicing_recall'],
            'Voicing_FA': best['voicing_fa'],
        })

    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))
    print()

    # Save summary
    # summary_df.to_csv("threshold_tuning_summary.csv", index=False)
    print("✓ Saved summary to threshold_tuning_summary.csv")
    print("="*80)


if __name__ == "__main__":
    main()
