#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_model_resnet_bce.py

ResNet-18 + BCE baseline for HCQT -> pitch salience.

- reaad .npz + splits.json from data_set_prep.py
- use ResNet-18 backbone (can select freeze)，+ 1x1 Conv output salience logits
- BCEWithLogitsLoss
- log train / val loss
- save best_model_resnet_bce.pth
- plot loss_curve_resnet_bce.png
- save loss_log_resnet_bce.csv

Colab:
    %cd /content/drive/MyDrive/aca_final_assignment

    !python train_model_resnet_bce.py \
        --data_dir /content/drive/MyDrive/aca_final_assignment/hcqt_output \
        --batch_size 8 \
        --epochs 30 \
        --segment_frames 512 \
        --freeze_backbone 1 (yes)\0(no finetuing)
        --pretrained 1(yes,use imagenet)/0(no random ini)
"""

import argparse
import json
from pathlib import Path
import csv
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision.models import resnet18


# ======================= Dataset =======================

class HcqtSalienceDataset(Dataset):
    """
    Dataset for HCQT + salience .npz files.

    - data_dir: folder with *.npz and splits.json (from data_set_prep.py)
    - split: 'train' or 'val'
    - segment_frames: fixed T length for each sample (crop or pad)
    - segments_per_track: how many segments per track
    - random_crop: True for train, False for val
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

        self._cache = {}
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
                start = np.random.randint(0, T - seg_T + 1)
            else:
                start = max(0, (T - seg_T) // 2)
            end = start + seg_T
            hcqt_seg = hcqt[:, :, start:end]
            sal_seg = salience[:, start:end]
        else:
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

        hcqt, salience = self._load_npz(track_id)
        hcqt_seg, sal_seg = self._crop_or_pad(hcqt, salience)

        # log-compress
        hcqt_seg = np.log1p(hcqt_seg).astype(np.float32)

        x = torch.from_numpy(hcqt_seg)   # (H, F, T)
        y = torch.from_numpy(sal_seg)    # (F, T)

        return x, y


# ======================= ResNet Model =======================

class ResNetSalience(nn.Module):
    """
    ResNet-18 backbone + 1x1 Conv for salience.

    Input:  (B, C=H, F, T)
    Output: (B, F, T) logits
    """
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
                    w_mean = w.mean(dim=1, keepdim=True)  # (64,1,7,7)
                    new_weight = w_mean.repeat(1, in_channels, 1, 1)
                    new_conv.weight.data[:, :, :, :] = new_weight[:, :, :, :]
                elif in_channels == 1:
                    w = old_conv.weight.data
                    w_mean = w.mean(dim=1, keepdim=True)
                    new_conv.weight.data = w_mean
                else:
                    w = old_conv.weight.data
                    new_conv.weight.data[:, : in_channels, :, :] = w[:, :in_channels, :, :]
            backbone.conv1 = new_conv

        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.salience_head = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        """
        x: (B, C, F, T)
        returns: logits (B, F, T)
        """
        B, C, F_bins, T = x.shape    
        feat = self.backbone(x)              # (B, 512, F', T')
        logit_small = self.salience_head(feat)  # (B, 1, F', T')
        logit_up = F.interpolate(           
            logit_small,
            size=(F_bins, T),
            mode="bilinear",
            align_corners=False,
        )  # (B, 1, F_bins, T)

        return logit_up.squeeze(1)           # (B, F_bins, T)



# ======================= Training Utils =======================

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x, y in loader:
        x = x.to(device)      # (B, H, F, T)
        y = y.to(device)      # (B, F, T)

        optimizer.zero_grad()
        logits = model(x)     # (B, F, T)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)


def plot_losses(train_losses, val_losses, out_path: Path):
    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train / Val loss (ResNet + BCE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved loss curve to {out_path}")


def save_loss_csv(train_losses, val_losses, out_path: Path):
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for i, (tr, va) in enumerate(zip(train_losses, val_losses), start=1):
            writer.writerow([i, tr, va])
    print(f"Saved loss log to {out_path}")


# ======================= Main =======================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Folder with *.npz and splits.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--segment_frames", type=int, default=512)
    parser.add_argument("--segments_per_track_train", type=int, default=8)
    parser.add_argument("--segments_per_track_val", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--freeze_backbone", type=int, default=0,
                        help="1 to freeze ResNet backbone, 0 to finetune")
    parser.add_argument("--pretrained", type=int, default=1,
                        help="1 to use ImageNet pretrained ResNet-18, 0 for random init")

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Dataset & DataLoader
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
    in_channels = sample_hcqt.shape[0]
    print("Input channels (harmonics):", in_channels)

    model = ResNetSalience(
        in_channels=in_channels,
        pretrained=bool(args.pretrained),
    ).to(device)

    # Freeze backbone if needed
    if args.freeze_backbone == 1:
        print("Freezing ResNet backbone parameters.")
        for p in model.backbone.parameters():
            p.requires_grad = False
    else:
        print("Finetuning full ResNet backbone.")

    # Loss & optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    # Logging
    train_losses, val_losses = [], []
    best_val = float("inf")
    best_epoch = -1

    out_dir = Path(args.data_dir)
    ckpt_path = out_dir / "best_model_resnet_bce.pth"
    loss_png = out_dir / "loss_curve_resnet_bce.png"
    loss_csv = out_dir / "loss_log_resnet_bce.csv"

    # Training loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = eval_one_epoch(model, val_loader, criterion, device)
        t1 = time.time()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"time={t1 - t0:.1f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> New best model saved (epoch {epoch}, val_loss={val_loss:.4f})")

    print(f"Training done. Best val_loss={best_val:.4f} at epoch {best_epoch}.")

    plot_losses(train_losses, val_losses, loss_png)
    save_loss_csv(train_losses, val_losses, loss_csv)


if __name__ == "__main__":
    main()
