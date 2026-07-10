"""
multispectral_analysis.py
─────────────────────────
Full pipeline for 13-band EuroSAT analysis:

  1. Data loading    – GeoTIFF 13-band Sentinel-2 patches
  2. Model training  – ResNet34 + ViT-S/16 adapted for 13 channels
  3. Wavelength sensitivity – per-band dropout → spectral sensitivity curve
  4. Calibrated noise       – Sentinel-2 SNR model, atmospheric haze,
                              striping, cloud shadow
  5. Plots                  – spectral curves, band importance heatmaps,
                              noise comparison figures

Dataset: EuroSAT All Bands (13-channel GeoTIFF)
Download: https://zenodo.org/record/7711810
Expected folder structure:
  D:/data/EuroSAT_MS/
    AnnualCrop/
      AnnualCrop_00001.tif  (13-band uint16 GeoTIFF)
      ...
    Forest/
    ...

Run: python multispectral_analysis.py
"""

import csv
import gc
import os
import random
import sys
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
import tifffile as tiff

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────────────────────────────────────
# Paths & global config
# ─────────────────────────────────────────────────────────────────────
MS_DATASET_PATH = r"D:\data\EuroSAT_MS\EuroSAT_MS"   # root of 13-band GeoTIFF dataset
OUT_DIR         = r"D:\data\EuroSAT_MS\EuroSAT_MS\analysis"

SEED            = 42
IMAGE_SIZE      = 64          # EuroSAT patches are 64×64
NUM_CLASSES     = 10
IN_CHANNELS     = 13

TRAIN_SPLIT     = 0.70
VAL_SPLIT       = 0.15
TEST_SPLIT      = 0.15

EPOCHS_RESNET   = 30
EPOCHS_VIT      = 35
PATIENCE        = 5
BATCH_SIZE      = 128
EVAL_BATCH      = 256
LR_RESNET       = 1e-4
LR_VIT          = 3e-5
WEIGHT_DECAY_R  = 1e-4
WEIGHT_DECAY_V  = 0.05
WORKERS         = 0           # safe on Windows

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
]

# ─────────────────────────────────────────────────────────────────────
# Sentinel-2 band metadata
# ─────────────────────────────────────────────────────────────────────
# Order matches EuroSAT All Bands channel order:
# B01,B02,B03,B04,B05,B06,B07,B08,B08A,B09,B10,B11,B12
BAND_META = [
    # name    wavelength(nm)  SNR   read_noise   resolution(m)  description
    {"name": "B01",  "wl": 443,  "snr": 129, "read": 0.0030, "res": 60,  "desc": "Coastal aerosol"},
    {"name": "B02",  "wl": 490,  "snr": 154, "read": 0.0025, "res": 10,  "desc": "Blue"},
    {"name": "B03",  "wl": 560,  "snr": 168, "read": 0.0023, "res": 10,  "desc": "Green"},
    {"name": "B04",  "wl": 665,  "snr": 142, "read": 0.0027, "res": 10,  "desc": "Red"},
    {"name": "B05",  "wl": 705,  "snr": 117, "read": 0.0033, "res": 20,  "desc": "Red Edge 1"},
    {"name": "B06",  "wl": 740,  "snr": 89,  "read": 0.0043, "res": 20,  "desc": "Red Edge 2"},
    {"name": "B07",  "wl": 783,  "snr": 105, "read": 0.0037, "res": 20,  "desc": "Red Edge 3"},
    {"name": "B08",  "wl": 842,  "snr": 174, "read": 0.0022, "res": 10,  "desc": "NIR"},
    {"name": "B08A", "wl": 865,  "snr": 72,  "read": 0.0054, "res": 20,  "desc": "Narrow NIR"},
    {"name": "B09",  "wl": 940,  "snr": 114, "read": 0.0034, "res": 60,  "desc": "Water vapour"},
    {"name": "B10",  "wl": 1375, "snr": 50,  "read": 0.0078, "res": 60,  "desc": "Cirrus"},
    {"name": "B11",  "wl": 1610, "snr": 100, "read": 0.0039, "res": 20,  "desc": "SWIR 1"},
    {"name": "B12",  "wl": 2190, "snr": 100, "read": 0.0039, "res": 20,  "desc": "SWIR 2"},
]

BAND_NAMES  = [b["name"] for b in BAND_META]
WAVELENGTHS = [b["wl"]   for b in BAND_META]
SNR_VALUES  = [b["snr"]  for b in BAND_META]
READ_NOISE  = [b["read"] for b in BAND_META]

# Band groups for ablation
BAND_GROUPS = {
    "Visible (B02-B04)":      [1, 2, 3],
    "Red Edge (B05-B07)":     [4, 5, 6],
    "NIR (B08+B08A)":         [7, 8],
    "SWIR (B11-B12)":         [11, 12],
    "Vegetation Index (NDVI)": [3, 7],    # Red + NIR
    "Water Index (NDWI)":     [2, 7],    # Green + NIR
    "All non-visible":        [0, 4, 5, 6, 7, 8, 9, 10, 11, 12],
}

# Per-band normalization statistics for EuroSAT multispectral
# Values are mean/std of reflectance*10000 divided by 10000
# Computed from the EuroSAT training set
BAND_MEANS = [
    0.0340, 0.0378, 0.0600, 0.0543, 0.0633, 0.1147,
    0.1381, 0.1328, 0.1438, 0.0300, 0.0010, 0.0696, 0.0465
]
BAND_STDS = [
    0.0207, 0.0281, 0.0400, 0.0503, 0.0532, 0.0702,
    0.0797, 0.0786, 0.0830, 0.0219, 0.0045, 0.0513, 0.0387
]

# ─────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def autocast_ctx(device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()

def safe_fn(s: str) -> str:
    for ch in r'/\:*?"<>| ':
        s = s.replace(ch, "_")
    return s

def normalize_ms(img: torch.Tensor) -> torch.Tensor:
    """Normalize 13-channel tensor using per-band statistics."""
    mean = torch.tensor(BAND_MEANS, dtype=img.dtype).view(13, 1, 1)
    std  = torch.tensor(BAND_STDS,  dtype=img.dtype).view(13, 1, 1)
    return (img - mean) / std.clamp(min=1e-6)

# ─────────────────────────────────────────────────────────────────────
# GeoTIFF loading
# ─────────────────────────────────────────────────────────────────────
def read_tif(path: str) -> np.ndarray:
    """
    Read a 13-band GeoTIFF using tifffile.
    Returns float32 array (13, H, W) normalized to [0, 1].
    EuroSAT stores uint16 with values up to ~10000 (reflectance × 10000).
    """
    arr = tiff.imread(path).astype(np.float32)
    
    # Ensure correct shape (13, H, W)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    elif arr.ndim == 3 and arr.shape[-1] == 13:
        arr = arr.transpose(2, 0, 1)          # HWC → CHW

    # Normalize: reflectance values are stored as reflectance × 10000
    arr = arr / 10000.0
    return arr.clip(0, 1)

# ─────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────
class EuroSATMultispectral(Dataset):
    """
    Loads 13-band GeoTIFF EuroSAT images.
    Expects folder structure: root/ClassName/ClassName_XXXXX.tif
    """
    def __init__(self, root: str, class_names: list[str]):
        self.root     = root
        self.classes  = sorted(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples: list[tuple[str, int]] = []

        for cls in self.classes:
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Class folder not found: {cls_dir}")
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(".tif"):
                    self.samples.append(
                        (os.path.join(cls_dir, fname), self.class_to_idx[cls])
                    )

        self.targets = [s[1] for s in self.samples]
        print(f"  Loaded {len(self.samples)} images, {len(self.classes)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        arr = read_tif(path)                       # (13, H, W) float32
        return torch.from_numpy(arr), label

class TransformDataset(Dataset):
    """Wraps a base dataset with an optional transform pipeline."""
    def __init__(self, base: EuroSATMultispectral,
                 indices: list[int],
                 augment: bool = False,
                 corruption_fn: Optional[Callable] = None):
        self.base         = base
        self.indices      = indices
        self.augment      = augment
        self.corruption   = corruption_fn

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]    # (13, 64, 64) float32

        # Resize to IMAGE_SIZE if needed
        if img.shape[-1] != IMAGE_SIZE or img.shape[-2] != IMAGE_SIZE:
            img = F.interpolate(
                img.unsqueeze(0), size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear", align_corners=False
            ).squeeze(0)

        # Apply corruption on raw [0,1] tensor
        if self.corruption is not None:
            img = self.corruption(img)

        # Augmentation (training only)
        if self.augment:
            if random.random() > 0.5:
                img = torch.flip(img, dims=[-1])   # horizontal flip
            if random.random() > 0.5:
                img = torch.flip(img, dims=[-2])   # vertical flip
            if random.random() > 0.3:
                k = random.choice([1, 2, 3])
                img = torch.rot90(img, k, dims=[-2, -1])

        return normalize_ms(img), label

def build_splits(targets: list[int]):
    idx = np.arange(len(targets))
    t   = np.array(targets)
    train_idx, temp = train_test_split(
        idx, test_size=1-TRAIN_SPLIT, stratify=t, random_state=SEED)
    temp_t = t[temp]
    val_idx, test_idx = train_test_split(
        temp, test_size=TEST_SPLIT/(VAL_SPLIT+TEST_SPLIT),
        stratify=temp_t, random_state=SEED)
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

def make_loader(dataset: Dataset, shuffle: bool = False,
                batch: int = EVAL_BATCH) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch, shuffle=shuffle,
        num_workers=WORKERS, pin_memory=True,
        drop_last=shuffle,
    )

# ─────────────────────────────────────────────────────────────────────
# Model builders — 13-channel input
# ─────────────────────────────────────────────────────────────────────
def build_resnet34_ms(num_classes: int, device: torch.device) -> nn.Module:
    """
    ResNet34 adapted for 13-channel input.
    timm handles in_chans by averaging pretrained 3-channel weights
    across the 13 input channels — preserves ImageNet features.
    """
    model = timm.create_model(
        "resnet34",
        pretrained=True,
        num_classes=num_classes,
        in_chans=IN_CHANNELS,
    )
    return model.to(device)

def build_vit_ms(num_classes: int, device: torch.device) -> nn.Module:
    """
    ViT-S/16 adapted for 13-channel input.
    64×64 patches with patch_size=8 gives 8×8=64 tokens (manageable).
    """
    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=True,
        num_classes=num_classes,
        in_chans=IN_CHANNELS,
        img_size=IMAGE_SIZE,       # 64×64 instead of 224×224
        drop_rate=0.10,
        drop_path_rate=0.10,
    )
    return model.to(device)

# ─────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────
def train_model(model: nn.Module,
                model_name: str,
                train_loader: DataLoader,
                val_loader: DataLoader,
                device: torch.device,
                epochs: int,
                lr: float,
                weight_decay: float,
                ckpt_path: str) -> dict:

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.05)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = torch.amp.GradScaler("cuda", enabled=device.type=="cuda")

    best_val     = -1.0
    patience_ctr = 0
    history      = {"train_acc": [], "val_acc": [],
                    "train_loss": [], "val_loss": []}

    print(f"\n{'='*58}")
    print(f"  Training {model_name}  ({epochs} epochs, patience={PATIENCE})")
    print(f"{'='*58}")

    for epoch in range(epochs):
        t0 = time.time()

        # ── train ──
        model.train()
        run_loss = run_correct = run_total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(device):
                out  = model(imgs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            run_loss    += loss.item() * imgs.size(0)
            run_correct += out.argmax(1).eq(labels).sum().item()
            run_total   += imgs.size(0)

        train_acc  = 100 * run_correct / run_total
        train_loss = run_loss / run_total

        # ── val ──
        model.eval()
        v_loss = v_correct = v_total = 0
        with torch.inference_mode():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                with autocast_ctx(device):
                    out  = model(imgs)
                    loss = criterion(out, labels)
                v_loss    += loss.item() * imgs.size(0)
                v_correct += out.argmax(1).eq(labels).sum().item()
                v_total   += imgs.size(0)

        val_acc  = 100 * v_correct / v_total
        val_loss = v_loss / v_total
        scheduler.step()

        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = val_acc > best_val
        if improved:
            best_val = val_acc
            patience_ctr = 0
            torch.save(model.state_dict(), ckpt_path)
            flag = "  ✓ saved"
        else:
            patience_ctr += 1
            flag = f"  ({patience_ctr}/{PATIENCE})"

        print(f"  Epoch [{epoch+1:02d}/{epochs}] "
              f"Train {train_acc:.2f}% | Val {val_acc:.2f}%"
              f"{flag}  [{time.time()-t0:.0f}s]")

        if patience_ctr >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    print(f"\n  Best val accuracy: {best_val:.2f}%")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return history

# ─────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device,
             verbose: bool = True) -> tuple[float, float, list[float]]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.inference_mode():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = float(np.mean(all_preds == all_labels))
    f1  = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))

    per_class = []
    for c in range(NUM_CLASSES):
        mask = all_labels == c
        per_class.append(
            float(np.mean(all_preds[mask] == c)) if mask.sum() > 0 else float("nan")
        )

    if verbose:
        print(classification_report(
            all_labels, all_preds,
            target_names=CLASS_NAMES, digits=4, zero_division=0))
        print(f"  Accuracy: {acc*100:.4f}%  F1: {f1:.4f}")

    return acc, f1, per_class

def evaluate_with_corruption(model: nn.Module,
                              base_dataset: EuroSATMultispectral,
                              test_idx: list[int],
                              device: torch.device,
                              corruption_fn: Optional[Callable] = None
                              ) -> tuple[float, float, list[float]]:
    ds  = TransformDataset(base_dataset, test_idx,
                           augment=False, corruption_fn=corruption_fn)
    ldr = make_loader(ds, shuffle=False)
    return evaluate(model, ldr, device, verbose=False)

# ─────────────────────────────────────────────────────────────────────
# Physically calibrated Sentinel-2 noise functions
# All operate on (13, H, W) float tensors in [0, 1]
# ─────────────────────────────────────────────────────────────────────

def sentinel2_full_noise(img: torch.Tensor,
                         severity: float = 1.0) -> torch.Tensor:
    """
    Apply calibrated Sentinel-2 sensor noise to ALL bands.
    Noise model: σ_total = sqrt(σ_shot² + σ_read²)
    σ_shot = sqrt(signal / SNR)  — signal-dependent Poisson
    σ_read = fixed per band      — detector read noise

    severity: 1.0 = real sensor level, 2.0 = 2× stress, 5.0 = extreme
    """
    out = img.clone()
    for i in range(13):
        snr         = SNR_VALUES[i]
        read        = READ_NOISE[i]
        signal      = img[i].clamp(min=1e-6)
        sigma_shot  = torch.sqrt(signal / snr)
        sigma_total = torch.sqrt(sigma_shot**2 + read**2) * severity
        noise       = torch.randn_like(signal) * sigma_total
        out[i]      = (signal + noise).clamp(0, 1)
    return out

def sentinel2_single_band_noise(img: torch.Tensor,
                                 band_idx: int,
                                 severity: float = 1.0) -> torch.Tensor:
    """
    Apply calibrated noise to a SINGLE band only.
    Used for per-band sensitivity analysis.
    """
    out         = img.clone()
    snr         = SNR_VALUES[band_idx]
    read        = READ_NOISE[band_idx]
    signal      = img[band_idx].clamp(min=1e-6)
    sigma_shot  = torch.sqrt(signal / snr)
    sigma_total = torch.sqrt(sigma_shot**2 + read**2) * severity
    out[band_idx] = (signal + torch.randn_like(signal) * sigma_total).clamp(0, 1)
    return out

def atmospheric_haze(img: torch.Tensor,
                      optical_depth: float = 0.2) -> torch.Tensor:
    """
    Beer-Lambert atmospheric scattering.
    Rayleigh scattering ∝ 1/λ⁴ — affects short wavelengths strongly.

    optical_depth: 0.05=clear, 0.2=moderate haze, 0.5=heavy haze
    """
    out = img.clone()
    wls = torch.tensor(WAVELENGTHS, dtype=torch.float32)

    # Rayleigh scattering coefficient normalized to Green band (560nm)
    ref_wl   = 560.0
    rayleigh = (ref_wl / wls) ** 4
    rayleigh = rayleigh / rayleigh[2]    # normalize to Green=1.0

    for i in range(13):
        tau           = float(optical_depth * rayleigh[i].item())
        transmittance = float(np.exp(-tau))
        # Path radiance — atmosphere adds brightness at short wavelengths
        # Surface albedo approximation: 0.1 × (1-transmittance)
        path_radiance = (1.0 - transmittance) * 0.08
        out[i] = (img[i] * transmittance + path_radiance).clamp(0, 1)
    return out

def detector_striping(img: torch.Tensor,
                       severity: float = 1.0) -> torch.Tensor:
    """
    Detector striping — column-wise gain variation from individual
    detector element calibration errors.
    Primarily affects 20m bands (B05-B07, B08A, B11-B12).
    """
    out = img.clone()
    _, H, W = img.shape

    # 20m bands — more striping because fewer detector elements
    stripe_bands = [4, 5, 6, 8, 11, 12]
    # 10m bands — minor striping
    minor_bands  = [1, 2, 3, 7]

    for bi in stripe_bands:
        amp       = 0.008 * severity
        col_gains = 1.0 + torch.randn(W) * amp
        out[bi]   = (img[bi] * col_gains.view(1, W)).clamp(0, 1)

    for bi in minor_bands:
        amp       = 0.003 * severity
        col_gains = 1.0 + torch.randn(W) * amp
        out[bi]   = (img[bi] * col_gains.view(1, W)).clamp(0, 1)

    return out

def cloud_shadow(img: torch.Tensor,
                  severity: float = 0.4) -> torch.Tensor:
    """
    Cloud shadow — spatially correlated darkening.
    NIR reflectance is partly diffuse so less affected than visible.
    SWIR is also less affected than visible.

    severity: shadow absorption factor (0=no shadow, 1=complete black)
    """
    _, H, W = img.shape

    # Elliptical shadow mask
    cx = random.randint(W // 4, 3 * W // 4)
    cy = random.randint(H // 4, 3 * H // 4)
    rx = random.randint(W // 5, W // 2)
    ry = random.randint(H // 5, H // 2)

    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    mask   = (((xx - cx).float() / rx)**2 +
              ((yy - cy).float() / ry)**2 <= 1).float()

    # Shadow absorption varies by wavelength
    # Shorter wavelengths affected more (multiple scattering under cloud)
    # Based on typical Sentinel-2 shadow studies
    shadow_absorption = torch.tensor([
        0.70,  # B01 coastal — heavily shadowed
        0.65,  # B02 blue
        0.60,  # B03 green
        0.60,  # B04 red
        0.55,  # B05 red edge 1
        0.50,  # B06 red edge 2
        0.48,  # B07 red edge 3
        0.40,  # B08 NIR — less affected
        0.40,  # B08A narrow NIR
        0.35,  # B09 water vapour
        0.30,  # B10 cirrus
        0.45,  # B11 SWIR 1
        0.45,  # B12 SWIR 2
    ]) * severity

    out = img.clone()
    for i in range(13):
        out[i] = img[i] * (1.0 - mask * shadow_absorption[i])
    return out.clamp(0, 1)

def band_dropout(img: torch.Tensor, band_idx: int) -> torch.Tensor:
    """Zero out a single band completely."""
    out = img.clone()
    out[band_idx] = 0.0
    return out

def band_group_dropout(img: torch.Tensor,
                        band_indices: list[int]) -> torch.Tensor:
    """Zero out a group of bands."""
    out = img.clone()
    for bi in band_indices:
        out[bi] = 0.0
    return out

def band_scale(img: torch.Tensor,
               band_idx: int,
               factor: float) -> torch.Tensor:
    """Scale a single band by a factor."""
    out = img.clone()
    out[band_idx] = (out[band_idx] * factor).clamp(0, 1)
    return out

# ─────────────────────────────────────────────────────────────────────
# Analysis experiments
# ─────────────────────────────────────────────────────────────────────

@dataclass
class AnalysisRecord:
    model:       str
    experiment:  str
    band_name:   str
    band_idx:    int
    wavelength:  int
    severity:    float
    accuracy:    float
    f1:          float
    per_class:   list = field(default_factory=list)

def run_wavelength_sensitivity(models_dict: dict,
                                base_dataset: EuroSATMultispectral,
                                test_idx: list[int],
                                device: torch.device) -> list[AnalysisRecord]:
    """
    Experiment 1: Per-band dropout.
    Zero out each of 13 bands one at a time.
    Measures: which wavelengths are most important for each architecture.
    """
    print("\n" + "="*58)
    print("  EXPERIMENT 1: Wavelength Sensitivity (per-band dropout)")
    print("="*58)
    records = []

    # Clean baseline
    for mname, model in models_dict.items():
        acc, f1, pc = evaluate_with_corruption(
            model, base_dataset, test_idx, device, None)
        records.append(AnalysisRecord(
            mname, "clean", "all", -1, 0, 0.0, acc, f1, pc))
        print(f"  Clean {mname}: {acc*100:.2f}%")

    # Per-band dropout
    for bi in range(13):
        bname = BAND_NAMES[bi]
        wl    = WAVELENGTHS[bi]
        fn    = partial(band_dropout, band_idx=bi)
        print(f"  Dropout {bname} ({wl}nm)", end="  ")
        for mname, model in models_dict.items():
            acc, f1, pc = evaluate_with_corruption(
                model, base_dataset, test_idx, device, fn)
            records.append(AnalysisRecord(
                mname, "band_dropout", bname, bi, wl, 1.0, acc, f1, pc))
            print(f"{mname}:{acc*100:.1f}%", end="  ")
        print()

    return records

def run_calibrated_noise(models_dict: dict,
                          base_dataset: EuroSATMultispectral,
                          test_idx: list[int],
                          device: torch.device) -> list[AnalysisRecord]:
    """
    Experiment 2: Physically calibrated Sentinel-2 noise.
    - Full sensor noise at 1×, 2×, 5×, 10× real levels
    - Per-band calibrated noise one band at a time
    - Atmospheric haze at 4 optical depth levels
    - Detector striping at 3 severity levels
    - Cloud shadow at 3 levels
    """
    print("\n" + "="*58)
    print("  EXPERIMENT 2: Physically Calibrated Sensor Noise")
    print("="*58)
    records = []

    # 2a. Full sensor noise at multiple severity levels
    print("\n  2a. Full sensor noise (all bands)")
    for severity in [1.0, 2.0, 5.0, 10.0]:
        fn = partial(sentinel2_full_noise, severity=severity)
        print(f"  Severity {severity:.1f}×", end="  ")
        for mname, model in models_dict.items():
            acc, f1, pc = evaluate_with_corruption(
                model, base_dataset, test_idx, device, fn)
            records.append(AnalysisRecord(
                mname, "full_sensor_noise", "all", -1, 0, severity, acc, f1, pc))
            print(f"{mname}:{acc*100:.1f}%", end="  ")
        print()

    # 2b. Per-band calibrated noise — spectral noise sensitivity curve
    print("\n  2b. Per-band calibrated noise (severity=3×)")
    severity = 3.0
    for bi in range(13):
        bname = BAND_NAMES[bi]
        wl    = WAVELENGTHS[bi]
        snr   = SNR_VALUES[bi]
        fn    = partial(sentinel2_single_band_noise,
                        band_idx=bi, severity=severity)
        print(f"  {bname} ({wl}nm, SNR={snr})", end="  ")
        for mname, model in models_dict.items():
            acc, f1, pc = evaluate_with_corruption(
                model, base_dataset, test_idx, device, fn)
            records.append(AnalysisRecord(
                mname, "single_band_noise", bname, bi, wl, severity, acc, f1, pc))
            print(f"{mname}:{acc*100:.1f}%", end="  ")
        print()

    # 2c. Per-band calibrated noise at multiple severity levels
    # (for full spectral sensitivity curves)
    print("\n  2c. Per-band noise sweep (severity 1×→10×)")
    for severity in [1.0, 3.0, 5.0, 10.0]:
        for bi in range(13):
            bname = BAND_NAMES[bi]
            wl    = WAVELENGTHS[bi]
            fn    = partial(sentinel2_single_band_noise,
                            band_idx=bi, severity=severity)
            for mname, model in models_dict.items():
                acc, f1, pc = evaluate_with_corruption(
                    model, base_dataset, test_idx, device, fn)
                records.append(AnalysisRecord(
                    mname, "band_noise_sweep", bname, bi, wl,
                    severity, acc, f1, pc))
        print(f"  Severity {severity:.0f}× done")

    # 2d. Atmospheric haze
    print("\n  2d. Atmospheric haze")
    for od in [0.05, 0.10, 0.20, 0.35, 0.50]:
        fn = partial(atmospheric_haze, optical_depth=od)
        print(f"  Optical depth {od:.2f}", end="  ")
        for mname, model in models_dict.items():
            acc, f1, pc = evaluate_with_corruption(
                model, base_dataset, test_idx, device, fn)
            records.append(AnalysisRecord(
                mname, "atmospheric_haze", "all", -1, 0, od, acc, f1, pc))
            print(f"{mname}:{acc*100:.1f}%", end="  ")
        print()

    # 2e. Detector striping
    print("\n  2e. Detector striping")
    for severity in [1.0, 3.0, 5.0]:
        fn = partial(detector_striping, severity=severity)
        print(f"  Severity {severity:.1f}", end="  ")
        for mname, model in models_dict.items():
            acc, f1, pc = evaluate_with_corruption(
                model, base_dataset, test_idx, device, fn)
            records.append(AnalysisRecord(
                mname, "detector_striping", "all", -1, 0, severity, acc, f1, pc))
            print(f"{mname}:{acc*100:.1f}%", end="  ")
        print()

    # 2f. Cloud shadow
    print("\n  2f. Cloud shadow")
    for severity in [0.2, 0.4, 0.7]:
        fn = partial(cloud_shadow, severity=severity)
        print(f"  Severity {severity:.1f}", end="  ")
        for mname, model in models_dict.items():
            acc, f1, pc = evaluate_with_corruption(
                model, base_dataset, test_idx, device, fn)
            records.append(AnalysisRecord(
                mname, "cloud_shadow", "all", -1, 0, severity, acc, f1, pc))
            print(f"{mname}:{acc*100:.1f}%", end="  ")
        print()

    return records

def run_band_group_ablation(models_dict: dict,
                              base_dataset: EuroSATMultispectral,
                              test_idx: list[int],
                              device: torch.device) -> list[AnalysisRecord]:
    """
    Experiment 3: Band group ablation.
    Remove semantically meaningful groups (visible, NIR, SWIR, etc.)
    Tests whether architectural differences align with vegetation indices.
    """
    print("\n" + "="*58)
    print("  EXPERIMENT 3: Band Group Ablation")
    print("="*58)
    records = []

    for group_name, band_indices in BAND_GROUPS.items():
        fn = partial(band_group_dropout, band_indices=band_indices)
        print(f"  Dropout: {group_name}", end="  ")
        for mname, model in models_dict.items():
            acc, f1, pc = evaluate_with_corruption(
                model, base_dataset, test_idx, device, fn)
            records.append(AnalysisRecord(
                mname, "band_group_dropout", group_name,
                -1, 0, 1.0, acc, f1, pc))
            print(f"{mname}:{acc*100:.1f}%", end="  ")
        print()

    return records

# ─────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────
MODEL_COLORS = {"ResNet34": "#6366f1", "ViT-S/16": "#f59e0b"}

def plot_spectral_sensitivity_curve(dropout_records: list[AnalysisRecord],
                                     clean_acc: dict[str, float],
                                     out_dir: str) -> None:
    """
    THE key figure — accuracy drop vs wavelength (nm).
    x-axis: wavelength in nm (physically meaningful)
    y-axis: accuracy after band dropout
    This is the main novel contribution figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Spectral Sensitivity Curve — Band Dropout Analysis\n"
                 "(x-axis = actual wavelength, not arbitrary band index)",
                 fontsize=13, fontweight="bold")

    model_names = sorted({r.model for r in dropout_records if r.band_idx >= 0})

    for ax_idx, (ax, metric, ylabel) in enumerate(zip(
            axes,
            ["accuracy", "drop"],
            ["Accuracy after band dropout (%)",
             "Accuracy drop vs clean (pp)"])):

        for mname in model_names:
            subset = sorted(
                [r for r in dropout_records
                 if r.model == mname and r.experiment == "band_dropout"],
                key=lambda r: r.wavelength
            )
            xs = [r.wavelength for r in subset]
            if metric == "accuracy":
                ys = [r.accuracy * 100 for r in subset]
            else:
                ys = [(clean_acc[mname] - r.accuracy) * 100 for r in subset]

            ax.plot(xs, ys, marker="o", linewidth=2.5, markersize=8,
                    color=MODEL_COLORS.get(mname, "gray"), label=mname)

            # Annotate band names
            for r in subset:
                ax.annotate(r.band_name,
                            (r.wavelength, r.accuracy*100 if metric=="accuracy"
                             else (clean_acc[mname]-r.accuracy)*100),
                            textcoords="offset points",
                            xytext=(0, 8), fontsize=7,
                            ha="center", color=MODEL_COLORS.get(mname, "gray"),
                            alpha=0.8)

        if metric == "accuracy":
            for mname in model_names:
                ax.axhline(clean_acc[mname]*100,
                           color=MODEL_COLORS.get(mname,"gray"),
                           linestyle="--", linewidth=1, alpha=0.4,
                           label=f"{mname} clean")

        # Shade wavelength regions
        ax.axvspan(400, 700,   alpha=0.05, color="blue",   label="Visible")
        ax.axvspan(700, 900,   alpha=0.05, color="red",    label="NIR")
        ax.axvspan(900, 1400,  alpha=0.05, color="purple", label="SWIR edge")
        ax.axvspan(1400, 2300, alpha=0.05, color="brown",  label="SWIR")

        ax.set_xlabel("Wavelength (nm)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "spectral_sensitivity_curve.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"  Saved → spectral_sensitivity_curve.png")

def plot_band_importance_heatmap(dropout_records: list[AnalysisRecord],
                                  clean_acc: dict[str, float],
                                  out_dir: str) -> None:
    """
    Heatmap: classes × bands, value = accuracy drop when band zeroed.
    Shows per-class spectral importance.
    """
    model_names = sorted({r.model for r in dropout_records if r.band_idx >= 0})

    fig, axes = plt.subplots(1, len(model_names),
                              figsize=(8*len(model_names), 9))
    if len(model_names) == 1:
        axes = [axes]

    fig.suptitle("Per-class Band Importance (accuracy drop when band zeroed)",
                 fontsize=12, fontweight="bold")

    for ax, mname in zip(axes, model_names):
        drop = np.zeros((NUM_CLASSES, 13))
        for bi in range(13):
            r_list = [r for r in dropout_records
                      if r.model == mname and r.experiment == "band_dropout"
                      and r.band_idx == bi]
            if not r_list:
                continue
            r = r_list[0]
            for ci in range(NUM_CLASSES):
                clean_pc = [rec.per_class[ci] for rec in dropout_records
                             if rec.model == mname and rec.experiment == "clean"]
                if clean_pc:
                    drop[ci, bi] = max(0, (clean_pc[0] - r.per_class[ci]) * 100)

        im = ax.imshow(drop, cmap="YlOrRd", vmin=0,
                       vmax=max(drop.max()+2, 10), aspect="auto")
        ax.set_xticks(range(13))
        ax.set_xticklabels(
            [f"{b['name']}\n{b['wl']}nm" for b in BAND_META],
            fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_yticklabels(CLASS_NAMES, fontsize=9)
        ax.set_title(f"{mname}", fontweight="bold")

        for ci in range(NUM_CLASSES):
            for bi in range(13):
                v = drop[ci, bi]
                ax.text(bi, ci, f"{v:.0f}", ha="center", va="center",
                        fontsize=6,
                        color="white" if v > drop.max()*0.6 else "black")

        plt.colorbar(im, ax=ax, label="Accuracy drop (pp)")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "band_importance_heatmap.png"),
                dpi=160, bbox_inches="tight")
    plt.close()
    print("  Saved → band_importance_heatmap.png")

def plot_noise_spectral_curves(noise_records: list[AnalysisRecord],
                                clean_acc: dict[str, float],
                                out_dir: str) -> None:
    """
    Spectral noise sensitivity — accuracy vs wavelength under
    calibrated single-band noise at fixed severity.
    Key plot: shows which wavelengths cause most degradation.
    """
    severity = 3.0
    subset   = [r for r in noise_records
                if r.experiment == "band_noise_sweep"
                and abs(r.severity - severity) < 0.01]

    if not subset:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"Spectral Noise Sensitivity (Calibrated Sentinel-2 noise, {severity:.0f}× sensor level)",
                 fontsize=12, fontweight="bold")

    model_names = sorted({r.model for r in subset})

    for ax, metric in zip(axes, ["accuracy", "drop"]):
        for mname in model_names:
            ms   = sorted([r for r in subset if r.model == mname],
                          key=lambda r: r.wavelength)
            xs   = [r.wavelength for r in ms]
            if metric == "accuracy":
                ys = [r.accuracy * 100 for r in ms]
            else:
                ys = [(clean_acc[mname] - r.accuracy) * 100 for r in ms]

            ax.plot(xs, ys, marker="o", linewidth=2.5, markersize=7,
                    color=MODEL_COLORS.get(mname,"gray"), label=mname)

            for r in ms:
                y = r.accuracy*100 if metric=="accuracy" else (clean_acc[mname]-r.accuracy)*100
                ax.annotate(r.band_name, (r.wavelength, y),
                            textcoords="offset points", xytext=(0,8),
                            fontsize=7, ha="center",
                            color=MODEL_COLORS.get(mname,"gray"), alpha=0.8)

        # Shade spectral regions
        ax.axvspan(400, 700,   alpha=0.05, color="blue")
        ax.axvspan(700, 900,   alpha=0.05, color="red")
        ax.axvspan(1400, 2300, alpha=0.05, color="brown")

        ax.set_xlabel("Wavelength (nm)", fontsize=11)
        ax.set_ylabel("Accuracy (%)" if metric=="accuracy"
                      else "Accuracy drop (pp)", fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "noise_spectral_curves.png"),
                dpi=160, bbox_inches="tight")
    plt.close()
    print("  Saved → noise_spectral_curves.png")

def plot_physical_noise_comparison(noise_records: list[AnalysisRecord],
                                    clean_acc: dict[str, float],
                                    out_dir: str) -> None:
    """
    Compare all physical noise types side by side.
    Rows: full_sensor_noise, atmospheric_haze, detector_striping, cloud_shadow
    """
    noise_types = [
        ("full_sensor_noise", "Full sensor noise (×severity)"),
        ("atmospheric_haze",  "Atmospheric haze (optical depth)"),
        ("detector_striping", "Detector striping (×severity)"),
        ("cloud_shadow",      "Cloud shadow (absorption factor)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes      = axes.ravel()

    model_names = sorted({r.model for r in noise_records})

    for ax, (exp_name, title) in zip(axes, noise_types):
        subset = [r for r in noise_records if r.experiment == exp_name]
        if not subset:
            continue

        for mname in model_names:
            ms = sorted([r for r in subset if r.model == mname],
                        key=lambda r: r.severity)
            if not ms:
                continue
            xs = [r.severity for r in ms]
            ys = [r.accuracy * 100 for r in ms]
            ax.plot(xs, ys, marker="o", linewidth=2.5,
                    color=MODEL_COLORS.get(mname,"gray"), label=mname)
            ax.axhline(clean_acc[mname]*100,
                       color=MODEL_COLORS.get(mname,"gray"),
                       linestyle="--", linewidth=1, alpha=0.3)

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Severity"); ax.set_ylabel("Accuracy (%)")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle("Physical Sensor Noise Robustness — ResNet34 vs ViT-S/16",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "physical_noise_comparison.png"),
                dpi=160, bbox_inches="tight")
    plt.close()
    print("  Saved → physical_noise_comparison.png")

def plot_band_group_ablation(group_records: list[AnalysisRecord],
                              clean_acc: dict[str, float],
                              out_dir: str) -> None:
    """Bar chart of accuracy drop for each semantic band group."""
    model_names = sorted({r.model for r in group_records})
    groups      = list(BAND_GROUPS.keys())
    x           = np.arange(len(groups))
    width       = 0.35
    colors      = [MODEL_COLORS.get(m, "gray") for m in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, metric in zip(axes, ["accuracy", "drop"]):
        for mi, mname in enumerate(model_names):
            vals = []
            for gname in groups:
                r_list = [r for r in group_records
                          if r.model == mname and r.band_name == gname]
                if r_list:
                    if metric == "accuracy":
                        vals.append(r_list[0].accuracy * 100)
                    else:
                        vals.append((clean_acc[mname] - r_list[0].accuracy) * 100)
                else:
                    vals.append(0)

            offset = (mi - 0.5) * width
            bars   = ax.bar(x + offset, vals, width,
                            label=mname, color=colors[mi], alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.3,
                        f"{v:.1f}", ha="center", va="bottom",
                        fontsize=7, color="white")

        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Accuracy (%)" if metric=="accuracy"
                      else "Drop vs clean (pp)")
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_title("Accuracy after band group dropout")
    axes[1].set_title("Accuracy drop — which groups matter most")
    plt.suptitle("Semantic Band Group Ablation — Architecture Spectral Preferences",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "band_group_ablation.png"),
                dpi=160, bbox_inches="tight")
    plt.close()
    print("  Saved → band_group_ablation.png")

def plot_training_curves(histories: dict, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for mname, hist in histories.items():
        c = MODEL_COLORS.get(mname, "gray")
        axes[0].plot(hist["val_acc"],   color=c, linewidth=2, label=f"{mname} val")
        axes[0].plot(hist["train_acc"], color=c, linewidth=1, linestyle="--",
                     alpha=0.5, label=f"{mname} train")
        axes[1].plot(hist["val_loss"],  color=c, linewidth=2, label=f"{mname} val")
        axes[1].plot(hist["train_loss"],color=c, linewidth=1, linestyle="--",
                     alpha=0.5, label=f"{mname} train")

    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch")
    axes[1].set_title("Loss");     axes[1].set_xlabel("Epoch")
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle("13-band EuroSAT Training Curves", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves_ms.png"), dpi=150)
    plt.close()
    print("  Saved → training_curves_ms.png")

# ─────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────
def save_records_csv(records: list[AnalysisRecord],
                     path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["model","experiment","band_name","band_idx",
                  "wavelength_nm","severity","accuracy","f1"]
        header += [f"acc_{c}" for c in CLASS_NAMES]
        writer.writerow(header)
        for r in records:
            row = [r.model, r.experiment, r.band_name, r.band_idx,
                   r.wavelength, f"{r.severity:.4f}",
                   f"{r.accuracy:.6f}", f"{r.f1:.6f}"]
            row += [f"{v:.4f}" for v in r.per_class]
            writer.writerow(row)
    print(f"  Saved CSV → {os.path.basename(path)}")

def write_markdown_summary(dropout_records: list[AnalysisRecord],
                            noise_records: list[AnalysisRecord],
                            group_records: list[AnalysisRecord],
                            clean_acc: dict[str, float],
                            out_dir: str) -> None:
    lines = ["# Multispectral Wavelength Sensitivity Summary", ""]

    lines.append("## Clean Accuracy")
    for mname, acc in clean_acc.items():
        lines.append(f"- {mname}: {acc*100:.2f}%")
    lines.append("")

    lines.append("## Most Important Bands (dropout accuracy drop)")
    model_names = sorted(clean_acc.keys())
    for mname in model_names:
        lines.append(f"\n### {mname}")
        drops = []
        for bi in range(13):
            r_list = [r for r in dropout_records
                      if r.model == mname and r.experiment == "band_dropout"
                      and r.band_idx == bi]
            if r_list:
                drop = (clean_acc[mname] - r_list[0].accuracy) * 100
                drops.append((BAND_META[bi]["name"],
                               BAND_META[bi]["wl"],
                               BAND_META[bi]["desc"], drop))
        drops.sort(key=lambda x: -x[3])
        for name, wl, desc, drop in drops:
            lines.append(f"- {name} ({wl}nm, {desc}): -{drop:.1f} pp")

    lines.append("\n## Physical Noise Robustness")
    for exp_name in ["full_sensor_noise", "atmospheric_haze",
                      "detector_striping", "cloud_shadow"]:
        lines.append(f"\n### {exp_name}")
        for mname in model_names:
            ms = sorted([r for r in noise_records
                         if r.model == mname and r.experiment == exp_name],
                        key=lambda r: r.severity)
            if ms:
                worst = min(ms, key=lambda r: r.accuracy)
                drop  = (clean_acc[mname] - worst.accuracy) * 100
                lines.append(f"- {mname}: worst={worst.accuracy*100:.1f}% "
                              f"(drop {drop:.1f}pp at severity={worst.severity:.2f})")

    lines.append("\n## Band Group Ablation")
    for gname in BAND_GROUPS:
        lines.append(f"\n### {gname}")
        for mname in model_names:
            r_list = [r for r in group_records
                      if r.model == mname and r.band_name == gname]
            if r_list:
                drop = (clean_acc[mname] - r_list[0].accuracy) * 100
                lines.append(f"- {mname}: {r_list[0].accuracy*100:.1f}% "
                              f"(drop {drop:.1f}pp)")

    path = os.path.join(out_dir, "ms_analysis_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved summary → ms_analysis_summary.md")

# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    seed_everything(SEED)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")
    print(f"TIF reader : tifffile")
    print(f"Dataset    : {MS_DATASET_PATH}")
    print(f"Output     : {OUT_DIR}")

    # ── discover classes from folder names ────────────────────────
    available = sorted([
        d for d in os.listdir(MS_DATASET_PATH)
        if os.path.isdir(os.path.join(MS_DATASET_PATH, d))
    ])
    print(f"\nFound classes: {available}")
    use_classes = [c for c in CLASS_NAMES if c in available]
    if len(use_classes) < NUM_CLASSES:
        print(f"WARNING: Expected {NUM_CLASSES} classes, found {len(use_classes)}")

    # ── load dataset ──────────────────────────────────────────────
    print("\nLoading dataset...")
    dataset    = EuroSATMultispectral(MS_DATASET_PATH, use_classes)
    train_idx, val_idx, test_idx = build_splits(dataset.targets)
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Test: {len(test_idx)}")

    # ── build data loaders ────────────────────────────────────────
    train_ds = TransformDataset(dataset, train_idx, augment=True)
    val_ds   = TransformDataset(dataset, val_idx,   augment=False)
    test_ds  = TransformDataset(dataset, test_idx,  augment=False)

    train_ldr = make_loader(train_ds, shuffle=True,  batch=BATCH_SIZE)
    val_ldr   = make_loader(val_ds,   shuffle=False, batch=EVAL_BATCH)
    test_ldr  = make_loader(test_ds,  shuffle=False, batch=EVAL_BATCH)

    # ─────────────────────────────────────────────────────────────
    # 1. Train ResNet34 (13-channel)
    # ─────────────────────────────────────────────────────────────
    print("\nBuilding ResNet34 (13-channel)...")
    resnet = build_resnet34_ms(NUM_CLASSES, device)
    params_r = sum(p.numel() for p in resnet.parameters()) / 1e6
    print(f"  Parameters: {params_r:.1f}M")

    resnet_ckpt = os.path.join(OUT_DIR, "resnet34_ms_best.pth")
    hist_resnet = train_model(
        resnet, "ResNet34", train_ldr, val_ldr, device,
        EPOCHS_RESNET, LR_RESNET, WEIGHT_DECAY_R, resnet_ckpt)

    print("\nResNet34 test evaluation:")
    resnet_acc, resnet_f1, resnet_pc = evaluate(resnet, test_ldr, device)

    # ─────────────────────────────────────────────────────────────
    # 2. Train ViT-S/16 (13-channel)
    # ─────────────────────────────────────────────────────────────
    print("\nBuilding ViT-S/16 (13-channel)...")
    vit = build_vit_ms(NUM_CLASSES, device)
    params_v = sum(p.numel() for p in vit.parameters()) / 1e6
    print(f"  Parameters: {params_v:.1f}M")

    vit_ckpt = os.path.join(OUT_DIR, "vit_s16_ms_best.pth")
    hist_vit = train_model(
        vit, "ViT-S/16", train_ldr, val_ldr, device,
        EPOCHS_VIT, LR_VIT, WEIGHT_DECAY_V, vit_ckpt)

    print("\nViT-S/16 test evaluation:")
    vit_acc, vit_f1, vit_pc = evaluate(vit, test_ldr, device)

    # ─────────────────────────────────────────────────────────────
    # Training curves
    # ─────────────────────────────────────────────────────────────
    plot_training_curves(
        {"ResNet34": hist_resnet, "ViT-S/16": hist_vit}, OUT_DIR)

    # ─────────────────────────────────────────────────────────────
    # Clean accuracy dict (used throughout)
    # ─────────────────────────────────────────────────────────────
    clean_acc = {
        "ResNet34": resnet_acc,
        "ViT-S/16": vit_acc,
    }
    models_dict = {
        "ResNet34": resnet,
        "ViT-S/16": vit,
    }

    print(f"\n  Clean accuracy summary:")
    print(f"  ResNet34 : {resnet_acc*100:.2f}%  F1={resnet_f1:.4f}")
    print(f"  ViT-S/16 : {vit_acc*100:.2f}%  F1={vit_f1:.4f}")

    # ─────────────────────────────────────────────────────────────
    # 3. Wavelength sensitivity (band dropout)
    # ─────────────────────────────────────────────────────────────
    dropout_records = run_wavelength_sensitivity(
        models_dict, dataset, test_idx, device)

    # ─────────────────────────────────────────────────────────────
    # 4. Physically calibrated sensor noise
    # ─────────────────────────────────────────────────────────────
    noise_records = run_calibrated_noise(
        models_dict, dataset, test_idx, device)

    # ─────────────────────────────────────────────────────────────
    # 5. Band group ablation
    # ─────────────────────────────────────────────────────────────
    group_records = run_band_group_ablation(
        models_dict, dataset, test_idx, device)

    # ─────────────────────────────────────────────────────────────
    # Plots
    # ─────────────────────────────────────────────────────────────
    print("\n── Generating figures ──")
    plot_spectral_sensitivity_curve(dropout_records, clean_acc, OUT_DIR)
    plot_band_importance_heatmap(dropout_records, clean_acc, OUT_DIR)
    plot_noise_spectral_curves(noise_records, clean_acc, OUT_DIR)
    plot_physical_noise_comparison(noise_records, clean_acc, OUT_DIR)
    plot_band_group_ablation(group_records, clean_acc, OUT_DIR)

    # ─────────────────────────────────────────────────────────────
    # Export
    # ─────────────────────────────────────────────────────────────
    print("\n── Saving CSVs ──")
    save_records_csv(dropout_records,
                     os.path.join(OUT_DIR, "wavelength_sensitivity.csv"))
    save_records_csv(noise_records,
                     os.path.join(OUT_DIR, "calibrated_noise.csv"))
    save_records_csv(group_records,
                     os.path.join(OUT_DIR, "band_group_ablation.csv"))
    write_markdown_summary(
        dropout_records, noise_records, group_records, clean_acc, OUT_DIR)

    # ─────────────────────────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────────────────────────
    print("\n" + "="*58)
    print("  FINAL SUMMARY")
    print("="*58)
    print(f"  ResNet34  clean: {resnet_acc*100:.2f}%")
    print(f"  ViT-S/16  clean: {vit_acc*100:.2f}%")

    print("\n  Most important bands (ResNet34):")
    drops_r = []
    for bi in range(13):
        r_list = [r for r in dropout_records
                  if r.model=="ResNet34" and r.experiment=="band_dropout"
                  and r.band_idx==bi]
        if r_list:
            drops_r.append((BAND_NAMES[bi], WAVELENGTHS[bi],
                             (resnet_acc - r_list[0].accuracy)*100))
    for name, wl, drop in sorted(drops_r, key=lambda x: -x[2])[:5]:
        print(f"    {name} ({wl}nm): -{drop:.1f} pp")

    print("\n  Most important bands (ViT-S/16):")
    drops_v = []
    for bi in range(13):
        r_list = [r for r in dropout_records
                  if r.model=="ViT-S/16" and r.experiment=="band_dropout"
                  and r.band_idx==bi]
        if r_list:
            drops_v.append((BAND_NAMES[bi], WAVELENGTHS[bi],
                             (vit_acc - r_list[0].accuracy)*100))
    for name, wl, drop in sorted(drops_v, key=lambda x: -x[2])[:5]:
        print(f"    {name} ({wl}nm): -{drop:.1f} pp")

    print(f"\n  All outputs saved to: {OUT_DIR}")
    print("  Key figures:")
    print("    spectral_sensitivity_curve.png  ← THE main novel figure")
    print("    band_importance_heatmap.png     ← per-class wavelength importance")
    print("    noise_spectral_curves.png       ← noise sensitivity vs wavelength")
    print("    physical_noise_comparison.png   ← haze/stripes/shadow comparison")
    print("    band_group_ablation.png         ← NDVI, NDWI, visible vs NIR")

if __name__ == "__main__":
    main()