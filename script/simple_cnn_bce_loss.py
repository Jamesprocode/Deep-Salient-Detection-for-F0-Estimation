#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_model.py

Baseline training script for HCQT -> pitch salience model.

- Reads preprocessed .npz files and splits.json from data_set_prep.py
- Builds a small 2D CNN to map HCQT -> salience map
- Uses BCEWithLogitsLoss
- Logs train/val loss per epoch to CSV
- Saves best model (by val loss)
- Plots loss curves to PNG

Usage (in Colab):
    %cd /content/drive/MyDrive/aca_final_assignment

    !python train_model.py \
        --data_dir /content/drive/MyDrive/aca_final_assignment/hcqt_output \
        --batch_size 8 \
        --epochs 30 \
        --segment_frames 512
"""

import argparse
import json
import os
from pathlib import Path
import csv
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# ======================= Dataset =======================

class HcqtSalienceDataset(Dataset):
    """
    Dataset for HCQT + salience .npz files.

    - data_dir: folder with *.npz and splits.json (from data_set_prep.py)
    - split: 'train' or 'val'
    - segment_frames: fixed T length for each sample (crop or pad)
    - segments_per_track: how many random segments to draw from each track
    - random_crop: True for train, False for val (center crop for val)
    """
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        segment_frames: int = 512,
        segments_per_track: int = 8,
        random_crop: bool = True,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.segment_frames = segment_frames
        self.segments_per_track = segments_per_track
        self.random_crop = random_crop

        # Load split file
        split_path = self.data_dir / "splits.json"
        with split_path.open("r") as f:
            splits = json.load(f)

        if split not in splits:
            raise ValueError(f"Split '{split}' not found in {split_path}")

        self.track_ids = splits[split]
        if len(self.track_ids) == 0:
            raise RuntimeError(f"No tracks in split '{split}'")

        # Cache for npz 
        self._cache = {}

        # Dataset length：Each song is sampled repeatedly using segments_per_track.
        self.num_tracks = len(self.track_ids)
        self._length = self.num_tracks * self.segments_per_track

        print(
            f"[{split}] tracks: {self.num_tracks}, "
            f"segments_per_track: {self.segments_per_track}, "
            f"total samples: {self._length}"
        )

    def __len__(self):
        return self._length

    def _load_npz(self, track_id):
        if track_id in self._cache:
            return self._cache[track_id]

        path = self.data_dir / f"{track_id}.npz"
        if not path.exists():
            raise FileNotFoundError(path)

        data = np.load(path)
        hcqt = data["hcqt"]        # (H, F, T)
        salience = data["salience"]  # (F, T)
        self._cache[track_id] = (hcqt, salience)
        return hcqt, salience

    def _crop_or_pad(self, hcqt, salience):
        """
        hcqt: (H, F, T)
        salience: (F, T)
        -> segment of shape (H, F, segment_frames), (F, segment_frames)
        """
        H, F, T = hcqt.shape
        seg_T = self.segment_frames

        if T >= seg_T:
            if self.random_crop:
                # random crop for train
                start = np.random.randint(0, T - seg_T + 1)
            else:
                # center crop for val
                start = max(0, (T - seg_T) // 2)
            end = start + seg_T
            hcqt_seg = hcqt[:, :, start:end]
            sal_seg = salience[:, start:end]
        else:
            # pad zeros at the end
            pad = seg_T - T
            hcqt_seg = np.pad(
                hcqt,
                pad_width=((0, 0), (0, 0), (0, pad)),
                mode="constant",
            )
            sal_seg = np.pad(
                salience,
                pad_width=((0, 0), (0, pad)),
                mode="constant",
            )
        return hcqt_seg, sal_seg

    def __getitem__(self, idx):
        track_idx = idx // self.segments_per_track
        track_id = self.track_ids[track_idx]

        hcqt, salience = self._load_npz(track_id)   # (H,F,T), (F,T)
        hcqt_seg, sal_seg = self._crop_or_pad(hcqt, salience)

        # log-compress HCQT 
        hcqt_seg = np.log1p(hcqt_seg).astype(np.float32)

        # torch tensor, channel-first
        # input: (H, F, T) -> (C=H, F, T)
        x = torch.from_numpy(hcqt_seg)             # (H, F, T)
        y = torch.from_numpy(sal_seg)              # (F, T)

        return x, y


# ======================= Model =======================

class SimpleHcqtCNN(nn.Module):
    """
    A simple fully-convolutional network:
    Input:  (B, C=H, F, T)
    Output: (B, F, T) logits for salience
    """
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
        """
        x: (B, C=H, F, T)
        returns: logits (B, F, T)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv_out(x)           # (B, 1, F, T)
        x = x.squeeze(1)               # (B, F, T)
        return x


# ======================= Training Utils =======================

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    num_batches = 0

    for x, y in loader:
        x = x.to(device)              # (B, H, F, T)
        y = y.to(device)              # (B, F, T)

        optimizer.zero_grad()
        logits = model(x)             # (B, F, T)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(1, num_batches)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    num_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(1, num_batches)


def plot_losses(train_losses, val_losses, out_path):
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Val loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved loss curve to {out_path}")


def save_loss_csv(train_losses, val_losses, out_path):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, (tr, va) in enumerate(zip(train_losses, val_losses), start=1):
            writer.writerow([i, tr, va])
    print(f"Saved loss log to {out_path}")


# ======================= Main =======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Folder with *.npz and splits.json (from data_set_prep.py)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--segment_frames", type=int, default=512)
    parser.add_argument("--segments_per_track_train", type=int, default=8)
    parser.add_argument("--segments_per_track_val", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Datasets & loaders
    train_ds = HcqtSalienceDataset(
        data_dir=args.data_dir,
        split="train",
        segment_frames=args.segment_frames,
        segments_per_track=args.segments_per_track_train,
        random_crop=True,
    )
    val_ds = HcqtSalienceDataset(
        data_dir=args.data_dir,
        split="val",
        segment_frames=args.segment_frames,
        segments_per_track=args.segments_per_track_val,
        random_crop=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Model
    sample_hcqt, _ = train_ds[0]
    in_channels = sample_hcqt.shape[0]  # H harmonics
    print("Input channels (harmonics):", in_channels)

    model = SimpleHcqtCNN(in_channels=in_channels).to(device)

    # Loss & optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Logging
    train_losses = []
    val_losses = []
    best_val = float("inf")
    best_epoch = -1

    out_dir = Path(args.data_dir)
    ckpt_path = out_dir / "best_model.pth"
    loss_png_path = out_dir / "loss_curve.png"
    loss_csv_path = out_dir / "loss_log.csv"

    # Training loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = eval_one_epoch(model, val_loader, criterion, device)
        t1 = time.time()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:03d} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | "
              f"time={t1 - t0:.1f}s")

        # save the best model
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> New best model saved (epoch {epoch}, val_loss={val_loss:.4f})")

    print(f"Training done. Best val_loss={best_val:.4f} at epoch {best_epoch}.")

    # Plot & save logs
    plot_losses(train_losses, val_losses, loss_png_path)
    save_loss_csv(train_losses, val_losses, loss_csv_path)


if __name__ == "__main__":
    main()
