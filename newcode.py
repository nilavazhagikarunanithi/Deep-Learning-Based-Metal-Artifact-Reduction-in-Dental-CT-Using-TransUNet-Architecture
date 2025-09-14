#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CPU-only TransUNet-style training for CBCT projection metal removal (Secondary Capture DICOMs).

- Combines ALL configured series (HA, T1, T2, T3...) into one dataset
- Matches metal <-> no-metal by exact filename, skips mismatches
- Normalizes with robust percentiles (0.5–99.5) per image (no HU)
- Trains at 512x512, saves per-epoch predictions as 1536x1536 DICOMs
- Preserves unsigned 16-bit (PixelRepresentation=0) and MONOCHROME settings
- Default: save predictions on VAL split each epoch (use --save_split all to save for all)
"""

import os
import random
import math
from pathlib import Path
import csv
import argparse
from typing import List, Tuple, Dict, Any

import numpy as np
import pydicom
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

# -----------------------------
# --------- CONFIG ------------
# -----------------------------

# >>>>>> EDIT THESE PATHS (all series are combined) <<<<<<
# Use ALL folders together (metal ↔ no-metal; match by same filename)
SERIES_MAPPINGS = [
    (r"D:\ct data\raw dcm\HA raw dcm",  r"D:\ct data\raw dcm\HA no metal raw dcm",  "HA"),
    (r"D:\ct data\raw dcm\t1 raw dcm",  r"D:\ct data\raw dcm\t1 no metal raw dcm",  "T1"),
    (r"D:\ct data\raw dcm\t2 raw dcm",  r"D:\ct data\raw dcm\t2 no metal raw dcm",  "T2"),
    (r"D:\ct data\raw dcm\t3 raw dcm",  r"D:\ct data\raw dcm\t3 no metal raw dcm",  "T3"),
]


RUNS_DIR = Path("runs")
IMG_TRAIN_SIZE = 512
IMG_SAVE_SIZE  = 1536

# CPU-friendly defaults
BATCH_SIZE    = 2
NUM_EPOCHS    = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-5
VAL_SPLIT     = 0.1
SEED          = 42
NUM_WORKERS   = 0   # CPU safe on Windows
PIN_MEMORY    = False

# Robust percentile window for normalization
P_LOW  = 0.5
P_HIGH = 99.5

# -----------------------------
# ------- UTILITIES -----------
# -----------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def is_dicom_file(fp: Path) -> bool:
    return fp.suffix.lower() == ".dcm"

def get_max_possible(bits_stored: int, pixel_representation: int) -> Tuple[int, int]:
    """
    Returns (min_val, max_val) for the stored pixel type per BitsStored & PixelRepresentation.
    pixel_representation: 0 = unsigned, 1 = signed
    """
    if bits_stored <= 0:
        return (0, 65535)
    if pixel_representation == 0:
        # unsigned: 0 .. (2^bits - 1)
        return (0, (1 << bits_stored) - 1)
    else:
        # signed: -2^(bits-1) .. 2^(bits-1)-1
        return (-(1 << (bits_stored - 1)), (1 << (bits_stored - 1)) - 1)

def read_dicom_pixels(ds: pydicom.dataset.FileDataset) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Reads pixel array and returns (arr_float, meta).
    Handles MONOCHROME1 -> invert to MONOCHROME2 convention for processing.
    No HU conversion (Secondary Capture).
    """
    arr = ds.pixel_array  # as numpy
    bits_stored = int(getattr(ds, "BitsStored", 16))
    pixel_rep   = int(getattr(ds, "PixelRepresentation", 0))
    mono        = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
    vmin, vmax  = get_max_possible(bits_stored, pixel_rep)

    arr = arr.astype(np.float32)

    # If MONOCHROME1, invert to MONOCHROME2 convention for consistent training
    inverted = False
    if mono == "MONOCHROME1":
        arr = (vmax - (arr - vmin))  # reflect around [vmin..vmax]
        inverted = True

    meta = {
        "mono": mono,
        "inverted_for_training": inverted,
        "bits_stored": bits_stored,
        "pixel_rep": pixel_rep,
        "vmin_dtype": vmin,
        "vmax_dtype": vmax,
    }
    return arr, meta

def percentile_normalize(arr: np.ndarray, p_low=P_LOW, p_high=P_HIGH) -> Tuple[np.ndarray, float, float]:
    """
    Normalize image to [0,1] using robust percentiles.
    Returns (normed, lo, hi) so we can invert later.
    """
    lo, hi = np.percentile(arr, [p_low, p_high])
    if hi <= lo:
        # Fallback to min/max
        lo = float(np.min(arr))
        hi = float(np.max(arr)) if np.max(arr) > lo else lo + 1.0
    x = (arr - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    return x, float(lo), float(hi)

def denorm_from_percentiles(x01: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Inverse of percentile_normalize."""
    x = x01 * (hi - lo) + lo
    return x.astype(np.float32)

def save_pred_as_dicom_uint(pred_01: np.ndarray,
                            ref_path: Path,
                            out_path: Path,
                            lo: float,
                            hi: float,
                            meta: Dict[str, Any],
                            save_rows=IMG_SAVE_SIZE,
                            save_cols=IMG_SAVE_SIZE):
    """
    Save prediction (in [0,1]) as DICOM using ref metadata.
    - Inverse percentile normalization via (lo, hi)
    - Preserve PixelRepresentation (unsigned/signed) and Bits settings
    - Respect MONOCHROME1 by reversing the training inversion if needed
    """
    ref_ds = pydicom.dcmread(str(ref_path))

    # Resize to requested save size
    pred_01 = pred_01.astype(np.float32)
    if pred_01.shape != (save_rows, save_cols):
        pred_01 = cv2.resize(pred_01, (save_cols, save_rows), interpolation=cv2.INTER_CUBIC)

    # Denormalize to original intensity scale
    pred_scaled = denorm_from_percentiles(pred_01, lo, hi)

    bits_stored = int(meta["bits_stored"])
    pixel_rep   = int(meta["pixel_rep"])
    vmin_dtype  = int(meta["vmin_dtype"])
    vmax_dtype  = int(meta["vmax_dtype"])

    # If we inverted during training (MONOCHROME1), invert back for saving
    if meta.get("inverted_for_training", False):
        pred_scaled = (vmax_dtype - (pred_scaled - vmin_dtype))

    # Clip to representable range and cast
    pred_scaled = np.clip(pred_scaled, vmin_dtype, vmax_dtype)

    if pixel_rep == 0:
        out_arr = np.round(pred_scaled).astype(np.uint16)
    else:
        out_arr = np.round(pred_scaled).astype(np.int16)

    # Prepare DICOM for writing
    ds = ref_ds.copy()
    ds.Rows, ds.Columns = out_arr.shape

    # Keep original photometric; default to MONOCHROME2 if missing
    mono = meta.get("mono", str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")))
    ds.PhotometricInterpretation = mono

    # Bits settings
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = pixel_rep  # 0=unsigned, 1=signed

    # Ensure single-sample grayscale
    ds.SamplesPerPixel = 1

    # DO NOT set RescaleSlope/Intercept (these SC files don't have HU)
    if "RescaleSlope" in ds: del ds.RescaleSlope
    if "RescaleIntercept" in ds: del ds.RescaleIntercept

    ds.PixelData = out_arr.tobytes()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(out_path), write_like_original=False)

def build_pair_list(series_mappings: List[Tuple[str, str, str]]):
    pairs = []
    for metal_dir, clean_dir, series_name in series_mappings:
        mdir, cdir = Path(metal_dir), Path(clean_dir)
        if not mdir.exists() or not cdir.exists():
            print(f"[WARN] Missing dir: {mdir} or {cdir} -> skip")
            continue
        clean_map = {fp.name: fp for fp in cdir.iterdir() if fp.is_file() and is_dicom_file(fp)}
        total, kept = 0, 0
        for mfp in mdir.iterdir():
            if not (mfp.is_file() and is_dicom_file(mfp)):
                continue
            total += 1
            match = clean_map.get(mfp.name)
            if match is not None:
                pairs.append((mfp, match, series_name))
                kept += 1
        print(f"[{series_name}] matched {kept}/{total} ({mdir.name} -> {cdir.name})")
    return pairs

# -----------------------------
# ---------- DATA -------------
# -----------------------------

class MetalCleanDicomPairs(Dataset):
    def __init__(self, pairs: List[Tuple[Path, Path, str]], img_size=IMG_TRAIN_SIZE):
        self.pairs = pairs
        self.img_size = img_size

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx: int):
        metal_path, clean_path, series = self.pairs[idx]

        # Read input (metal) and target (clean) images with SC-aware handling
        mds = pydicom.dcmread(str(metal_path))
        cds = pydicom.dcmread(str(clean_path))

        m_arr, m_meta = read_dicom_pixels(mds)  # float32
        c_arr, c_meta = read_dicom_pixels(cds)

        # Percentile normalization per image
        m01, m_lo, m_hi = percentile_normalize(m_arr, P_LOW, P_HIGH)
        c01, c_lo, c_hi = percentile_normalize(c_arr, P_LOW, P_HIGH)

        # Resize to training size
        m01r = cv2.resize(m01, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        c01r = cv2.resize(c01, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        # Build tensors
        x = torch.from_numpy(m01r[None, ...])  # (1,H,W)
        y = torch.from_numpy(c01r[None, ...])

        meta = {
            "in_path": str(metal_path),
            "series": series,
            "filename": metal_path.name,
            # For saving, use the INPUT (metal) image's scale/format to inverse-map
            "p_lo": m_lo,
            "p_hi": m_hi,
            "mono": m_meta["mono"],
            "inverted_for_training": m_meta["inverted_for_training"],
            "bits_stored": m_meta["bits_stored"],
            "pixel_rep": m_meta["pixel_rep"],
            "vmin_dtype": m_meta["vmin_dtype"],
            "vmax_dtype": m_meta["vmax_dtype"],
        }
        return x, y, meta

# -----------------------------
# ---------- MODEL ------------
# -----------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.block(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x): return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, 2)
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        dy, dx = skip.size(2) - x.size(2), skip.size(3) - x.size(3)
        if dy or dx:
            x = F.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        return self.conv(torch.cat([skip, x], dim=1))

class TransformerBottleneck(nn.Module):
    def __init__(self, channels: int, num_layers=2, nhead=8, dim_ff=1024):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d_model=channels, nhead=nhead,
                                           dim_feedforward=dim_ff, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pos = None
    def forward(self, x):
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)   # (B,N,C)
        N = H * W
        if (self.pos is None) or self.pos.shape[1] != N or self.pos.shape[2] != C:
            self.pos = nn.Parameter(torch.zeros(1, N, C, device=x.device))
            nn.init.trunc_normal_(self.pos, std=0.02)
        tokens = self.encoder(tokens + self.pos)
        return tokens.transpose(1, 2).reshape(B, C, H, W)

class TransUNetTiny(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=64):
        super().__init__()
        self.inc   = ConvBlock(in_ch, base)
        self.down1 = Down(base, base*2)
        self.down2 = Down(base*2, base*4)
        self.down3 = Down(base*4, base*8)
        self.bottleneck = nn.Sequential(
            ConvBlock(base*8, base*8),
            TransformerBottleneck(base*8, num_layers=2, nhead=8, dim_ff=1024),
            ConvBlock(base*8, base*8),
        )
        self.up3  = Up(base*8, base*4)
        self.up2  = Up(base*4, base*2)
        self.up1  = Up(base*2, base)
        self.outc = nn.Conv2d(base, out_ch, 1)
        self.act  = nn.Sigmoid()
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        xb = self.bottleneck(x4)
        x  = self.up3(xb, x3)
        x  = self.up2(x, x2)
        x  = self.up1(x, x1)
        return self.act(self.outc(x))

# -----------------------------
# ---------- TRAIN ------------
# -----------------------------

def train_one_epoch(model, loader, optim, device, loss_fn):
    model.train()
    running = 0.0
    for x, y, _ in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optim.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optim.step()
        running += loss.item() * x.size(0)
    return running / len(loader.dataset)

@torch.no_grad()
def run_infer_and_save(model, loader, device, epoch_dir: Path):
    """Save predictions as DICOMs (1536x1536) for all batches in loader."""
    model.eval()
    for x, _, metas in tqdm(loader, desc="Save preds", leave=False):
        x = x.to(device)
        pred = model(x).squeeze(1).cpu().numpy()  # (B,H,W) in [0,1]
        # Save each sample using its own per-image scale/format
        for i in range(pred.shape[0]):
            in_path  = Path(metas["in_path"][i])
            series   = metas["series"][i]
            filename = metas["filename"][i]
            out_path = epoch_dir / series / filename
            save_pred_as_dicom_uint(
                pred[i],
                in_path,
                out_path,
                lo=metas["p_lo"][i],
                hi=metas["p_hi"][i],
                meta={k: metas[k][i] for k in ["mono","inverted_for_training","bits_stored","pixel_rep","vmin_dtype","vmax_dtype"]},
                save_rows=IMG_SAVE_SIZE,
                save_cols=IMG_SAVE_SIZE
            )

def collate_meta(batch):
    """
    Default collate stacks tensors; for meta dict of lists, we need to collect per-field.
    This custom collate keeps metas as dict of lists so we can access per-sample values.
    """
    xs, ys, metas = zip(*batch)  # lists
    # Stack tensors
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    # Turn list of dicts into dict of lists
    meta_out = {}
    for d in metas:
        for k, v in d.items():
            meta_out.setdefault(k, []).append(v)
    return x, y, meta_out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--val_split", type=float, default=VAL_SPLIT)
    parser.add_argument("--device", type=str, default="cpu")  # CPU by default
    parser.add_argument("--save_split", type=str, choices=["val","train","all"], default="val",
                        help="which split to save predictions for each epoch")
    args = parser.parse_args()

    set_seed(SEED)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Build combined pairs from ALL folders
    pairs = build_pair_list(SERIES_MAPPINGS)
    if len(pairs) == 0:
        print("No matched pairs found. Check SERIES_MAPPINGS.")
        return

    # Dataset & split
    full_ds = MetalCleanDicomPairs(pairs, img_size=IMG_TRAIN_SIZE)
    val_len = max(1, int(len(full_ds) * args.val_split))
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len],
                                    generator=torch.Generator().manual_seed(SEED))

    # Dataloaders (CPU-friendly) with custom collate to keep meta
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=PIN_MEMORY,
                              drop_last=False, collate_fn=collate_meta)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                              num_workers=args.workers, pin_memory=PIN_MEMORY,
                              drop_last=False, collate_fn=collate_meta)

    # Device/model/opt/loss
    device = torch.device(args.device)
    model = TransUNetTiny(in_ch=1, out_ch=1, base=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.L1Loss()

    # CSV log
    log_path = RUNS_DIR / "train_log.csv"
    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","train_loss","val_loss","checkpoint"])

    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} (CPU) ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, device, loss_fn)

        # Validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, _ in tqdm(val_loader, desc="Val", leave=False):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss += F.l1_loss(pred, y, reduction="sum").item()
        val_loss /= len(val_loader.dataset)

        # Save predictions for requested split
        epoch_dir = RUNS_DIR / f"epoch_{epoch:03d}"
        if args.save_split in ("val","all"):
            run_infer_and_save(model, val_loader, device, epoch_dir)
        if args.save_split in ("train","all"):
            run_infer_and_save(model, train_loader, device, epoch_dir)

        # Checkpoints & log
        ckpt_path = RUNS_DIR / f"model_epoch_{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, ckpt_path)

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", str(ckpt_path)])

        print(f"Epoch {epoch}: train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), RUNS_DIR / "model_best.pt")
            print("  ↳ New best model saved.")

    print("\nDone. Per-epoch DICOM predictions in runs/epoch_XXX/<series>/file.dcm")
    print(f"Log: {log_path}")

if __name__ == "__main__":
    main()
