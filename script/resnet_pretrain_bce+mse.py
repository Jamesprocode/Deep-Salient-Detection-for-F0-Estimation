#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ResNet-18 + Hybrid Loss (BCE + MSE) for HCQT -> pitch salience.

- read data_set_prep.py  .npz + splits.json
-  ResNet-18 backbone+ 1x1 Conv upsample to (F, T)
- λ_bce * BCEWithLogits + λ_mse * MSE(sigmoid(pred), target)
- train / val  total, bce, mse loss
- save best_model_resnet_hybrid.pth
- output
    - loss_curve_resnet_hybrid.png        
    - loss_components_resnet_hybrid.png   
    - loss_log_resnet_hybrid.csv          

Colab:
    %cd /content/drive/MyDrive/aca_final_assignment

    !python train_model_resnet_hybrid.py \
        --data_dir /content/drive/MyDrive/aca_final_assignment/hcqt_output \
        --batch_size 8 \
        --epochs 30 \
        --segment_frames 512 \
        --freeze_backbone 1 \
        --pretrained 1 \
        --lambda_bce 1.0 \
        --lambda_mse 0.1
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

    - data_dir: folder with *.npz and splits.json
    - split: 'train' or 'val'
    - segment_frames: segment frame
    - segments_per_track: num_of segements
    - random_crop: train=True, val=False 
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

        #  splits.json
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
        hcqt = data["hcqt"]          # (H, F, T)
        salience = data["salience"]  # (F, T)
        self._cache[track_id] = (hcqt, salience)
        return hcqt, salience

    def _crop_or_pad(self, hcqt, salience):
        """
        hcqt: (H, F, T)
        salience: (F, T)
        -> (H, F, segment_frames), (F, segment_frames)
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
                hcqt, ((0, 0), (0, 0), (0, pad)), mode="constant"
            )
            sal_seg = np.pad(
                salience, ((0, 0), (0, pad)), mode="constant"
            )
        return hcqt_seg, sal_seg

    def __getitem__(self, idx):
        track_idx = idx // self.segments_per_track
        track_id = self.track_ids[track_idx]

        hcqt, salience = self._load_npz(track_id)
        hcqt_seg, sal_seg = self._crop_or_pad(hcqt, salience)

        # log 
        hcqt_seg = np.log1p(hcqt_seg).astype(np.float32)

        x = torch.from_numpy(hcqt_seg)   # (H, F, T)
        y = torch.from_numpy(sal_seg)    # (F, T)
        return x, y


# ======================= ResNet Model =======================

class ResNetSalience(nn.Module):
    """
    

    (B, C=H, F, T)
     (B, F, T) logits

    - use resnet18 without avgpool and fc
    - feature map: (B, 512, F', T')
    - 1x1 conv -> (B, 1, F', T')
    - interpolote (F, T)，ensuring alignment with target 
    """
    def __init__(self, in_channels=6, pretrained=True):
        super().__init__()
        
# torchvision old interface: pretrained=True; for newer versions, the weights parameter can be changed.
        backbone = resnet18(pretrained=pretrained)

        # Replace the first layer conv1 to fit in_channels
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
                    w = old_conv.weight.data  # (64,3,7,7)
                    w_mean = w.mean(dim=1, keepdim=True)   # (64,1,7,7)
                    new_weight = w_mean.repeat(1, in_channels, 1, 1)
                    new_conv.weight.data[:, :, :, :] = new_weight
                elif in_channels == 1:
                    w = old_conv.weight.data
                    w_mean = w.mean(dim=1, keepdim=True)
                    new_conv.weight.data = w_mean
                else:
                    w = old_conv.weight.data
                    new_conv.weight.data[:, :in_channels, :, :] = w[:, :in_channels, :, :]
            backbone.conv1 = new_conv

        ## Remove avgpool & fuc from backbone
        self.backbone = nn.Sequential(*list(backbone.children())[:-2]) 
        self.salience_head = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        """
        x: (B, C, F, T)
        returns: (B, F, T) logits
        """
        B, C, F_in, T_in = x.shape
        feat = self.backbone(x)                  # (B, 512, F', T')
        logit_small = self.salience_head(feat)   # (B, 1, F', T')

        # upsample
        logit_up = F.interpolate(
            logit_small,
            size=(F_in, T_in),
            mode="bilinear",
            align_corners=False,
        )  # (B,1,F_in,T_in)

        return logit_up.squeeze(1)               # (B,F_in,T_in)


# ======================= Hybrid Loss =======================

class HybridBCEMSELoss(nn.Module):
    """
    L = λ_bce * BCEWithLogits(logits, target)
      + λ_mse * MSE(sigmoid(logits), target)

    - BCE: Pay attention to details, bin-level presence/absence F0
    - MSE: Grasp the overall shape of the salience after Gaussian blur.
    """
    def __init__(self, lambda_bce=1.0, lambda_mse=0.1):
        super().__init__()
        self.lambda_bce = lambda_bce
        self.lambda_mse = lambda_mse
        self.bce = nn.BCEWithLogitsLoss()
        self.mse = nn.MSELoss()

    def forward(self, logits, target):
        """
        logits: (B, F, T)
        target: (B, F, T)
        """
        loss_bce = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        loss_mse = self.mse(probs, target)

        loss = self.lambda_bce * loss_bce + self.lambda_mse * loss_mse
        return loss, {"bce": loss_bce.detach(), "mse": loss_mse.detach()}


# ======================= Utils =======================

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = total_bce = total_mse = 0.0
    num_batches = 0

    for x, y in loader:
        x = x.to(device)   # (B,H,F,T)
        y = y.to(device)   # (B,F,T)

        optimizer.zero_grad()
        logits = model(x)
        loss, parts = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_bce  += parts["bce"].item()
        total_mse  += parts["mse"].item()
        num_batches += 1

    denom = max(1, num_batches)
    return (
        total_loss / denom,
        total_bce / denom,
        total_mse / denom,
    )


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_bce = total_mse = 0.0
    num_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss, parts = criterion(logits, y)

        total_loss += loss.item()
        total_bce  += parts["bce"].item()
        total_mse  += parts["mse"].item()
        num_batches += 1

    denom = max(1, num_batches)
    return (
        total_loss / denom,
        total_bce / denom,
        total_mse / denom,
    )


def plot_losses(train_tot, val_tot, train_bce, val_bce, train_mse, val_mse, out_dir: Path):
    epochs = np.arange(1, len(train_tot) + 1)

    #  loss 
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_tot, label="Train total")
    plt.plot(epochs, val_tot,   label="Val total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet + Hybrid Loss (Total)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    png_total = out_dir / "loss_curve_resnet_hybrid.png"
    plt.savefig(png_total, dpi=200)
    plt.close()
    print(f"Saved total loss curve to {png_total}")

    # BCE / MSE seprate loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_bce, label="Train BCE")
    plt.plot(epochs, val_bce,   label="Val BCE")
    plt.plot(epochs, train_mse, label="Train MSE")
    plt.plot(epochs, val_mse,   label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ResNet + Hybrid Loss Components")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    png_comp = out_dir / "loss_components_resnet_hybrid.png"
    plt.savefig(png_comp, dpi=200)
    plt.close()
    print(f"Saved loss components curve to {png_comp}")


def save_loss_csv(train_tot, val_tot, train_bce, val_bce, train_mse, val_mse, out_path: Path):
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_total", "val_total",
            "train_bce", "val_bce",
            "train_mse", "val_mse",
        ])
        for i in range(len(train_tot)):
            writer.writerow([
                i + 1,
                train_tot[i], val_tot[i],
                train_bce[i], val_bce[i],
                train_mse[i], val_mse[i],
            ])
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
                        help="1=freeze ResNet backbone, 0=finetune")
    parser.add_argument("--pretrained", type=int, default=1,
                        help="1= ImageNet , 0=ini_random")

    parser.add_argument("--lambda_bce", type=float, default=1.0)
    parser.add_argument("--lambda_mse", type=float, default=0.1)

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

    # Freeze or finetune
    if args.freeze_backbone == 1:
        print("Freezing ResNet backbone.")
        for p in model.backbone.parameters():
            p.requires_grad = False
    else:
        print("Finetuning full ResNet backbone.")

    # Loss & optimizer
    criterion = HybridBCEMSELoss(
        lambda_bce=args.lambda_bce,
        lambda_mse=args.lambda_mse,
    )
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    # Logging
    train_tot, val_tot = [], []
    train_bce, val_bce = [], []
    train_mse, val_mse = [], []

    best_val = float("inf")
    best_epoch = -1
    out_dir = Path(args.data_dir)
    ckpt_path = out_dir / "best_model_resnet_hybrid.pth"
    csv_path = out_dir / "loss_log_resnet_hybrid.csv"

    # Training loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_tot, tr_bce, tr_mse = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_tot, va_bce, va_mse = eval_one_epoch(model, val_loader, criterion, device)
        t1 = time.time()

        train_tot.append(tr_tot)
        val_tot.append(va_tot)
        train_bce.append(tr_bce)
        val_bce.append(va_bce)
        train_mse.append(tr_mse)
        val_mse.append(va_mse)

        print(
            f"Epoch {epoch:03d} | "
            f"train_total={tr_tot:.4f} (bce={tr_bce:.4f}, mse={tr_mse:.4f}) | "
            f"val_total={va_tot:.4f} (bce={va_bce:.4f}, mse={va_mse:.4f}) | "
            f"time={t1 - t0:.1f}s"
        )

        if va_tot < best_val:
            best_val = va_tot
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> New best model saved (epoch {epoch}, val_total={va_tot:.4f})")

    print(f"Training done. Best val_total={best_val:.4f} at epoch {best_epoch}.")

    plot_losses(train_tot, val_tot, train_bce, val_bce, train_mse, val_mse, out_dir)
    save_loss_csv(train_tot, val_tot, train_bce, val_bce, train_mse, val_mse, csv_path)


if __name__ == "__main__":
    main()
