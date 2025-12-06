#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data_set_prep.py

Preprocessing MedleyDB-Pitch:
- compute HCQT for each track
- build a frame-aligned F0 salience target with Gaussian blur on the frequency axis
- create train/val split (by track)
- visualize a processed track for sanity check
"""

import argparse
import json
from pathlib import Path

import numpy as np
import librosa
import pandas as pd
import matplotlib.pyplot as plt

# ================== Default config  ==================

DATA_ROOT = Path("PATH/TO/MedleyDB-Pitch")

# Subfolders for audio and annotations 
AUDIO_SUBDIR = "audio"     
ANNOT_SUBDIR = "pitch"

AUDIO_DIR = DATA_ROOT / AUDIO_SUBDIR
ANNOT_DIR = DATA_ROOT / ANNOT_SUBDIR

AUDIO_EXT = ".wav"
ANNOT_EXT = ".csv"

# HCQT parameters
SR = 44100
FMIN_NOTE = "C1"
FMIN = librosa.note_to_hz(FMIN_NOTE)
BINS_PER_OCTAVE = 60
N_OCTAVES = 6
N_BINS = BINS_PER_OCTAVE * N_OCTAVES
HARMONICS = [0.5, 1, 2, 3, 4, 5]  
HOP_LENGTH = 512  # ≈ 11.6 ms

FMAX = FMIN * (2 ** N_OCTAVES)   

# Frequency-axis Gaussian blur
GAUSS_SIGMA_BINS = 1.0 

# Train / val split ratio
TRAIN_RATIO = 0.8
RNG_SEED = 42

# Output directory 
OUT_DIR = Path("preprocessed_hcqt")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ================== Helper functions ==================

def list_track_ids():
    """List all track_ids based on filenames under AUDIO_DIR."""
    track_ids = []
    for audio_path in sorted(AUDIO_DIR.glob(f"*{AUDIO_EXT}")):
        track_ids.append(audio_path.stem)
    return track_ids


def compute_hcqt(y, sr):
    """
    Compute HCQT for one audio clip.

    Returns
    -------
    hcqt : np.ndarray
        Array of shape (H, F, T_common), where
        H = number of harmonics,
        F = number of CQT bins,
        T_common = time frames (trimmed to the shortest among harmonics).
    """
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
        mag = np.abs(cqt).astype(np.float32)  # (F, T_h)
        hcqt_list.append(mag)
        lengths.append(mag.shape[1])

    # Find the minimum time length across all harmonics
    T_min = min(lengths)

    # Trim all harmonics to the same time length
    hcqt_list_trimmed = [h[:, :T_min] for h in hcqt_list]   # each is (F, T_min)

    hcqt = np.stack(hcqt_list_trimmed, axis=0)  # (H, F, T_min)
    return hcqt


def hz_to_cqt_bin(f_hz):
    """Map frequency in Hz to a CQT bin index."""
    if f_hz <= 0 or np.isnan(f_hz):
        return None
    k = BINS_PER_OCTAVE * np.log2(f_hz / FMIN)
    bin_idx = int(np.round(k))
    if bin_idx < 0 or bin_idx >= N_BINS:
        return None
    return bin_idx


def build_gaussian_kernel_1d(sigma, radius=None):
    """
    Build a 1D Gaussian kernel.

    Parameters
    ----------
    sigma : float
        Standard deviation in bins.
    radius : int or None
        Radius of the kernel (if None, use 3 * sigma).

    Returns
    -------
    kernel : np.ndarray
        1D normalized Gaussian kernel.
    """
    if radius is None:
        radius = int(3 * sigma)
    xs = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


GAUSS_KERNEL = build_gaussian_kernel_1d(GAUSS_SIGMA_BINS)
GAUSS_RADIUS = len(GAUSS_KERNEL) // 2


def gaussian_blur_freq_axis(salience):
    """
    Apply 1D Gaussian blur along the frequency axis.

    Parameters
    ----------
    salience : np.ndarray
        Salience map with shape (F, T).

    Returns
    -------
    out : np.ndarray
        Blurred salience map with the same shape (F, T), values clipped to [0, 1].
    """
    F, T = salience.shape
    padded = np.pad(
        salience,
        pad_width=((GAUSS_RADIUS, GAUSS_RADIUS), (0, 0)),
        mode="constant"
    )
    out = np.zeros_like(salience, dtype=np.float32)

    for f in range(F):
        window = padded[f:f + len(GAUSS_KERNEL), :]  # (K, T)
        out[f, :] = (GAUSS_KERNEL[:, None] * window).sum(axis=0)

    # Multiple F0s may overlap; clip everything to [0, 1]
    np.clip(out, 0.0, 1.0, out=out)
    return out


def load_annotations(annot_path):
    """
    Load F0 annotations for a single track (robust version).

    Behavior:
    - Automatically skips malformed lines (on_bad_lines='skip').
    - Only keeps the first two columns: time, freq.
    - Non-numeric / inf / -inf are converted to NaN and dropped.
    - Negative times are removed.
    - Frequencies are clipped to [0, FMAX], freq < 0 treated as silence (0 Hz).
    """
    df = pd.read_csv(
        annot_path,
        header=None,
        comment="#",
        on_bad_lines="skip"
    )

    if df.shape[1] < 2:
        raise RuntimeError(f"Annotation file {annot_path} has <2 columns")

    # Only keep the first two columns
    df = df.iloc[:, :2].copy()
    df.columns = ["time", "freq"]

    # Force numeric; invalid values become NaN
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["freq"] = pd.to_numeric(df["freq"], errors="coerce")

    # Treat inf / -inf as NaN as well
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop any rows with NaN and rows with negative time
    df = df.dropna()
    df = df[df["time"] >= 0]

    # Frequencies below 0 are treated as silence (set to 0)
    df.loc[df["freq"] < 0, "freq"] = 0.0
    # Frequencies above the CQT range are clipped to FMAX
    df.loc[df["freq"] > FMAX, "freq"] = FMAX

    times = df["time"].to_numpy(dtype=np.float32)
    freqs = df["freq"].to_numpy(dtype=np.float32)
    return times, freqs


def build_salience_map(times, freqs, n_frames):
    """
    Build a salience map from annotations.

    Parameters
    ----------
    times : np.ndarray
        1D array of time stamps in seconds.
    freqs : np.ndarray
        1D array of F0 frequencies in Hz.
    n_frames : int
        Number of frames (T) to match HCQT time dimension.

    Returns
    -------
    salience_blur : np.ndarray
        Gaussian-blurred salience map, shape (F, T).
    """
    salience = np.zeros((N_BINS, n_frames), dtype=np.float32)

    for t_sec, f_hz in zip(times, freqs):
        if f_hz <= 0 or np.isnan(f_hz):
            continue  # silent frame
        frame_idx = int(np.round(t_sec * SR / HOP_LENGTH))
        if frame_idx < 0 or frame_idx >= n_frames:
            continue
        bin_idx = hz_to_cqt_bin(f_hz)
        if bin_idx is None:
            continue
        salience[bin_idx, frame_idx] = 1.0

    salience_blur = gaussian_blur_freq_axis(salience)
    return salience_blur


# ================== Main preprocessing loop ==================

def process_all_tracks():
    """
    Process all tracks in AUDIO_DIR:
    - load audio
    - compute HCQT
    - load annotations and build salience map
    - save compressed .npz per track

    Returns
    -------
    processed_ids : list of str
        Track IDs successfully processed and saved.
    """
    track_ids = list_track_ids()
    print(f"Found {len(track_ids)} tracks.")

    processed_ids = []

    for tid in track_ids:
        audio_path = AUDIO_DIR / f"{tid}{AUDIO_EXT}"
        annot_path = ANNOT_DIR / f"{tid}{ANNOT_EXT}"

        if not audio_path.exists():
            print(f"[WARN] Audio not found: {audio_path}")
            continue
        if not annot_path.exists():
            print(f"[WARN] Annotation not found: {annot_path}")
            continue

        print(f"Processing {tid} ...")

        # 1. Load audio
        y, sr = librosa.load(audio_path, sr=SR, mono=True)

        # 2. HCQT
        hcqt = compute_hcqt(y, sr)
        _, _, T = hcqt.shape
        frame_times = (np.arange(T) * HOP_LENGTH / SR).astype(np.float32)

        # 3. Load annotations & build salience
        times, freqs = load_annotations(annot_path)
        salience = build_salience_map(times, freqs, n_frames=T)

        # 4. Save
        out_path = OUT_DIR / f"{tid}.npz"
        np.savez_compressed(
            out_path,
            hcqt=hcqt,
            salience=salience,
            times=frame_times,
            track_id=tid,
        )

        processed_ids.append(tid)

    print(f"Processed {len(processed_ids)} tracks.")
    return processed_ids


def make_train_val_split(track_ids):
    """
    Create a train/val split by track and save as splits.json.

    Parameters
    ----------
    track_ids : list of str
        List of all processed track IDs.
    """
    rng = np.random.default_rng(RNG_SEED)
    ids = np.array(sorted(track_ids))
    rng.shuffle(ids)

    n_train = int(len(ids) * TRAIN_RATIO)
    train_ids = ids[:n_train].tolist()
    val_ids = ids[n_train:].tolist()

    split = {"train": train_ids, "val": val_ids}
    split_path = OUT_DIR / "splits.json"
    with split_path.open("w") as f:
        json.dump(split, f, indent=2)

    print(f"Saved splits to {split_path}")
    print(f"Train: {len(train_ids)} tracks, Val: {len(val_ids)} tracks")


# ================== Visualization: inspect one track ==================

def visualize_track(track_id):
    """
    Visualize a single preprocessed track using the saved .npz and original csv.

    Plots:
    - Top: log HCQT (sum over harmonics)
    - Bottom: salience map with ground-truth F0 curve overlaid

    Parameters
    ----------
    track_id : str
        Track ID (without extension).
    """
    npz_path = OUT_DIR / f"{track_id}.npz"
    annot_path = ANNOT_DIR / f"{track_id}{ANNOT_EXT}"

    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path)
    hcqt = data["hcqt"]         # (H, F, T)
    salience = data["salience"] # (F, T)
    times = data["times"]       # (T,)

    # Simple spectrum: sum over harmonics and convert to dB
    spec = hcqt.sum(axis=0)  # (F, T)
    spec_db = librosa.amplitude_to_db(spec + 1e-8, ref=np.max)

    # Reload GT annotations to plot F0 curve
    gt_times, gt_freqs = load_annotations(annot_path)

    plt.figure(figsize=(12, 8))

    extent = [times[0], times[-1], 0, N_BINS]

    # Top plot: HCQT spectrum
    plt.subplot(2, 1, 1)
    plt.imshow(spec_db[::-1, :], aspect='auto', origin='lower',
               extent=extent, interpolation='nearest')
    plt.colorbar(label="dB")
    plt.ylabel("CQT bin")
    plt.title(f"HCQT (sum over harmonics) - {track_id}")

    # Convert GT F0 to CQT bins and overlay
    gt_bins = [hz_to_cqt_bin(f) for f in gt_freqs]
    gt_bins = np.array([b if b is not None else np.nan for b in gt_bins],
                       dtype=float)
    plt.plot(gt_times, gt_bins, color="lime", linewidth=1.0, label="GT F0 (bin)")
    plt.legend(loc="upper right")

    # Bottom plot: salience map
    plt.subplot(2, 1, 2)
    plt.imshow(salience[::-1, :], aspect='auto', origin='lower',
               extent=extent, interpolation='nearest')
    plt.colorbar(label="salience")
    plt.xlabel("Time (s)")
    plt.ylabel("CQT bin")
    plt.title("Target salience (Gaussian-blurred)")
    plt.tight_layout()
    plt.show()


# ================== CLI entrypoint ==================

def main():
    global DATA_ROOT, AUDIO_DIR, ANNOT_DIR, OUT_DIR, TRAIN_RATIO
    global AUDIO_SUBDIR, ANNOT_SUBDIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None,
                        help="Root folder of MedleyDB-Pitch")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output folder for npz & splits.json")
    parser.add_argument("--split_ratio", type=float, default=None,
                        help="Train/val split ratio (e.g., 0.8)")
    parser.add_argument("--audio_subdir", type=str, default=None,
                        help="Subfolder name for audio (default: 'audio')")
    parser.add_argument("--annot_subdir", type=str, default=None,
                        help="Subfolder name for pitch annotations (default: 'pitch')")
    args = parser.parse_args()

    if args.data_root is not None:
        DATA_ROOT = Path(args.data_root)

    if args.audio_subdir is not None:
        AUDIO_SUBDIR = args.audio_subdir
    if args.annot_subdir is not None:
        ANNOT_SUBDIR = args.annot_subdir

    AUDIO_DIR = DATA_ROOT / AUDIO_SUBDIR
    ANNOT_DIR = DATA_ROOT / ANNOT_SUBDIR

    if args.out_dir is not None:
        OUT_DIR = Path(args.out_dir)
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.split_ratio is not None:
        TRAIN_RATIO = float(args.split_ratio)

    print("DATA_ROOT :", DATA_ROOT)
    print("AUDIO_DIR :", AUDIO_DIR)
    print("ANNOT_DIR :", ANNOT_DIR)
    print("OUT_DIR   :", OUT_DIR)
    print("TRAIN_RATIO:", TRAIN_RATIO)

    ids = process_all_tracks()
    make_train_val_split(ids)
    print("Done.")


if __name__ == "__main__":
    main()
