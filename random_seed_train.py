"""
multiseed_spectral_robustness.py
─────────────────────────────────
Runs the full spectral + noise robustness evaluation across multiple
random seeds and reports mean ± std + p-values for every metric.
Smart Checkpointing: Skips training if weights already exist on disk.

Run: python multiseed_spectral_robustness.py
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
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.stats import ttest_rel
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

# ─────────────────────────────────────────────────────────────────────
# Config — edit these
# ─────────────────────────────────────────────────────────────────────
USE_MULTISPECTRAL  = True       # True = 13-band, False = RGB EuroSAT

MS_DATASET_PATH    = r"D:\data\EuroSAT_MS\EuroSAT_MS"
RGB_DATASET_PATH   = r"D:\data\EuroSAT\2750"
OUT_DIR            = r"D:\data\EuroSAT_MS\multiseed_analysis"

SEEDS              = [42, 123, 456]              # Reduced to 3 seeds for faster execution
SPLIT_SEED         = 42                          # Locks dataset splits for fair testing
FORCE_RETRAIN      = False                       # Set to True to overwrite existing checkpoints

TRAIN_SPLIT        = 0.70
VAL_SPLIT          = 0.15
TEST_SPLIT         = 0.15

# Training hyperparams
IMAGE_SIZE         = 64 if USE_MULTISPECTRAL else 224
IN_CHANNELS        = 13 if USE_MULTISPECTRAL else 3
NUM_CLASSES        = 10
EPOCHS_RESNET      = 15
EPOCHS_VIT         = 20
PATIENCE           = 5
BATCH_SIZE         = 128
EVAL_BATCH         = 256
LR_RESNET          = 1e-4
LR_VIT             = 3e-5
WD_RESNET          = 1e-4
WD_VIT             = 0.05
WORKERS            = 0          # safe on Windows

MODEL_NAMES        = ["ResNet34", "ViT-S/16"]

CLASS_NAMES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
]

# ── Sentinel-2 band metadata ──────────────────────────────────────────
BAND_META = [
    {"name": "B01",  "wl": 443,  "snr": 129, "read": 0.0030},
    {"name": "B02",  "wl": 490,  "snr": 154, "read": 0.0025},
    {"name": "B03",  "wl": 560,  "snr": 168, "read": 0.0023},
    {"name": "B04",  "wl": 665,  "snr": 142, "read": 0.0027},
    {"name": "B05",  "wl": 705,  "snr": 117, "read": 0.0033},
    {"name": "B06",  "wl": 740,  "snr": 89,  "read": 0.0043},
    {"name": "B07",  "wl": 783,  "snr": 105, "read": 0.0037},
    {"name": "B08",  "wl": 842,  "snr": 174, "read": 0.0022},
    {"name": "B08A", "wl": 865,  "snr": 72,  "read": 0.0054},
    {"name": "B09",  "wl": 940,  "snr": 114, "read": 0.0034},
    {"name": "B10",  "wl": 1375, "snr": 50,  "read": 0.0078},
    {"name": "B11",  "wl": 1610, "snr": 100, "read": 0.0039},
    {"name": "B12",  "wl": 2190, "snr": 100, "read": 0.0039},
]
BAND_NAMES  = [b["name"] for b in BAND_META]
WAVELENGTHS = [b["wl"]   for b in BAND_META]

BAND_GROUPS = {
    "Visible (B02-B04)":      [1, 2, 3],
    "Red Edge (B05-B07)":     [4, 5, 6],
    "NIR (B08+B08A)":         [7, 8],
    "SWIR (B11-B12)":         [11, 12],
    "NDVI (Red+NIR)":         [3, 7],
    "NDWI (Green+NIR)":       [2, 7],
}

# Noise severities
SENSOR_SEVERITIES = [1.0, 2.0, 5.0, 10.0]
HAZE_DEPTHS       = [0.05, 0.10, 0.20, 0.35, 0.50]

# Per-band normalization for 13-band EuroSAT
BAND_MEANS = [0.0340,0.0378,0.0600,0.0543,0.0633,0.1147,
              0.1381,0.1328,0.1438,0.0300,0.0010,0.0696,0.0465]
BAND_STDS  = [0.0207,0.0281,0.0400,0.0503,0.0532,0.0702,
              0.0797,0.0786,0.0830,0.0219,0.0045,0.0513,0.0387]

MASK_CACHE: dict = {}


# ─────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_ctx(device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def normalize_ms(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(BAND_MEANS, dtype=img.dtype).view(IN_CHANNELS, 1, 1)
    std  = torch.tensor(BAND_STDS,  dtype=img.dtype).view(IN_CHANNELS, 1, 1)
    return (img - mean) / std.clamp(min=1e-6)


def normalize_rgb(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=img.dtype).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], dtype=img.dtype).view(3, 1, 1)
    return (img - mean) / std


normalize_fn = normalize_ms if USE_MULTISPECTRAL else normalize_rgb


# ─────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────
def load_image_ms(path: str) -> torch.Tensor:
    try:
        import rasterio
        with rasterio.open(path) as src:
            arr = src.read().astype(np.float32)
    except ImportError:
        import tifffile
        arr = tifffile.imread(path).astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 13:
            arr = arr.transpose(2, 0, 1)
    return torch.from_numpy((arr / 10000.0).clip(0, 1))


def load_image_rgb(path: str) -> torch.Tensor:
    from PIL import Image
    from torchvision import transforms
    img = Image.open(path).convert("RGB")
    t   = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])
    return t(img)


load_image = load_image_ms if USE_MULTISPECTRAL else load_image_rgb


class SatDataset(Dataset):
    """Generic dataset for both MS and RGB EuroSAT."""
    def __init__(self, root: str, class_names: list[str]):
        self.classes = sorted(class_names)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        ext = ".tif" if USE_MULTISPECTRAL else ".jpg"
        self.samples: list[tuple[str, int]] = []

        for cls in self.classes:
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                for d in os.listdir(root):
                    if d.lower() == cls.lower():
                        cls_dir = os.path.join(root, d)
                        break
            for fname in sorted(os.listdir(cls_dir)):
                if fname.lower().endswith(ext) or \
                   (not USE_MULTISPECTRAL and fname.lower().endswith(".jpg")) or \
                   (not USE_MULTISPECTRAL and fname.lower().endswith(".png")):
                    self.samples.append(
                        (os.path.join(cls_dir, fname),
                         self.class_to_idx[cls])
                    )

        self.targets = [s[1] for s in self.samples]
        print(f"  Loaded {len(self.samples)} images ({len(self.classes)} classes)")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return load_image(path), label


class TransformDataset(Dataset):
    def __init__(self, base: SatDataset, indices: list[int],
                 augment: bool = False,
                 corruption_fn: Optional[Callable] = None):
        self.base       = base
        self.indices    = indices
        self.augment    = augment
        self.corruption = corruption_fn

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]

        if img.shape[-1] != IMAGE_SIZE or img.shape[-2] != IMAGE_SIZE:
            img = F.interpolate(
                img.unsqueeze(0),
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear", align_corners=False
            ).squeeze(0)

        if self.corruption is not None:
            img = self.corruption(img)

        if self.augment:
            if random.random() > 0.5:
                img = torch.flip(img, dims=[-1])
            if random.random() > 0.5:
                img = torch.flip(img, dims=[-2])
            if random.random() > 0.3:
                img = torch.rot90(img, random.choice([1,2,3]), dims=[-2,-1])

        return normalize_fn(img), label


def build_splits(targets: list[int]):
    idx = np.arange(len(targets))
    t   = np.array(targets)
    # Locked SPLIT_SEED ensures cross-seed evaluations are fair
    train_idx, temp = train_test_split(
        idx, test_size=1-TRAIN_SPLIT, stratify=t, random_state=SPLIT_SEED)
    temp_t = t[temp]
    val_idx, test_idx = train_test_split(
        temp, test_size=TEST_SPLIT/(VAL_SPLIT+TEST_SPLIT),
        stratify=temp_t, random_state=SPLIT_SEED)
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def make_loader(ds: Dataset, shuffle: bool = False,
                batch: int = EVAL_BATCH) -> DataLoader:
    return DataLoader(ds, batch_size=batch, shuffle=shuffle,
                      num_workers=WORKERS, pin_memory=True,
                      drop_last=shuffle)


# ─────────────────────────────────────────────────────────────────────
# Corruption functions
# ─────────────────────────────────────────────────────────────────────
def sentinel2_full_noise(img: torch.Tensor,
                         severity: float = 1.0) -> torch.Tensor:
    out = img.clone()
    n   = min(IN_CHANNELS, len(BAND_META))
    for i in range(n):
        snr         = BAND_META[i]["snr"]
        read        = BAND_META[i]["read"]
        signal      = img[i].clamp(min=1e-6)
        sigma_shot  = torch.sqrt(signal / snr)
        sigma_total = torch.sqrt(sigma_shot**2 + read**2) * severity
        out[i]      = (signal + torch.randn_like(signal)*sigma_total).clamp(0,1)
    return out


def atmospheric_haze(img: torch.Tensor,
                     optical_depth: float = 0.2) -> torch.Tensor:
    out = img.clone()
    wls = np.array(WAVELENGTHS[:IN_CHANNELS])
    rayleigh = (560.0 / wls) ** 4
    rayleigh = rayleigh / rayleigh[2] if len(rayleigh) > 2 else rayleigh
    for i in range(IN_CHANNELS):
        tau           = optical_depth * rayleigh[i]
        transmittance = float(np.exp(-tau))
        path_radiance = (1.0 - transmittance) * 0.08
        out[i]        = (img[i]*transmittance + path_radiance).clamp(0,1)
    return out


def band_dropout(img: torch.Tensor, band_idx: int) -> torch.Tensor:
    out = img.clone()
    out[band_idx] = 0.0
    return out


def band_group_dropout(img: torch.Tensor,
                       indices: list[int]) -> torch.Tensor:
    out = img.clone()
    for bi in indices:
        if bi < IN_CHANNELS:
            out[bi] = 0.0
    return out


# ─────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────
def build_resnet(device: torch.device) -> nn.Module:
    m = timm.create_model(
        "resnet34", pretrained=True,
        num_classes=NUM_CLASSES, in_chans=IN_CHANNELS)
    return m.to(device)


def build_vit(device: torch.device) -> nn.Module:
    m = timm.create_model(
        "vit_small_patch16_224", pretrained=True,
        num_classes=NUM_CLASSES, in_chans=IN_CHANNELS,
        img_size=IMAGE_SIZE, drop_rate=0.10, drop_path_rate=0.10)
    return m.to(device)


# ─────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────
def train_one_model(model: nn.Module, model_name: str,
                    train_ldr: DataLoader, val_ldr: DataLoader,
                    device: torch.device, epochs: int,
                    lr: float, wd: float, seed: int) -> float:

    ckpt_path = os.path.join(
        OUT_DIR, f"seed{seed}_{model_name.replace('/','_')}.pth")

    # === SMART CHECKPOINTING BYPASS ===
    if not FORCE_RETRAIN and os.path.exists(ckpt_path):
        print(f"\n  [✓] Found existing checkpoint for {model_name} [seed={seed}]. Skipping training!")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        return 0.0  # Return early, exact val accuracy not needed here since test eval happens right after

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr*0.05)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = torch.amp.GradScaler("cuda", enabled=device.type=="cuda")

    best_val     = -1.0
    patience_ctr = 0

    print(f"\n  Training {model_name} [seed={seed}]")

    for epoch in range(epochs):
        model.train()
        correct = total = 0
        for imgs, labels in train_ldr:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            with autocast_ctx(device):
                outputs = model(imgs) 
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            with torch.inference_mode():
                preds    = outputs.argmax(1) 
                correct += preds.eq(labels).sum().item()
                total   += labels.size(0)

        model.eval()
        v_correct = v_total = 0
        with torch.inference_mode():
            for imgs, labels in val_ldr:
                imgs, labels = imgs.to(device), labels.to(device)
                v_correct += model(imgs).argmax(1).eq(labels).sum().item()
                v_total   += labels.size(0)

        val_acc = 100 * v_correct / v_total
        scheduler.step()

        if val_acc > best_val:
            best_val = val_acc
            patience_ctr = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:02d}: val={val_acc:.2f}%  "
                  f"best={best_val:.2f}%")

    print(f"    → Best val: {best_val:.2f}%")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return best_val


# ─────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────
def eval_model(model: nn.Module, base_dataset: SatDataset,
               test_idx: list[int], device: torch.device,
               corruption_fn: Optional[Callable] = None) -> float:
    ds  = TransformDataset(base_dataset, test_idx,
                           augment=False, corruption_fn=corruption_fn)
    ldr = make_loader(ds)
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for imgs, labels in ldr:
            imgs = imgs.to(device)
            correct += model(imgs).argmax(1).cpu().eq(labels).sum().item()
            total   += labels.size(0)
    return correct / total


# ─────────────────────────────────────────────────────────────────────
# Single-seed experiment runner
# ─────────────────────────────────────────────────────────────────────
@dataclass
class SeedResult:
    seed:          int
    model_name:    str
    clean_acc:     float=0.0
    band_dropout:  dict = field(default_factory=dict)
    sensor_noise:  dict = field(default_factory=dict)
    haze:          dict = field(default_factory=dict)
    band_groups:   dict = field(default_factory=dict)


def run_one_seed(seed: int, dataset: SatDataset,
                 device: torch.device) -> list[SeedResult]:
    seed_everything(seed)
    train_idx, val_idx, test_idx = build_splits(dataset.targets) 

    train_ds  = TransformDataset(dataset, train_idx, augment=True)
    val_ds    = TransformDataset(dataset, val_idx,   augment=False)
    train_ldr = make_loader(train_ds, shuffle=True,  batch=BATCH_SIZE)
    val_ldr   = make_loader(val_ds,   shuffle=False, batch=EVAL_BATCH)

    results = []

    # ── ResNet34 ──────────────────────────────────────────────────
    resnet = build_resnet(device)
    train_one_model(resnet, "ResNet34", train_ldr, val_ldr,
                    device, EPOCHS_RESNET, LR_RESNET, WD_RESNET, seed)

    # ── ViT-S/16 ─────────────────────────────────────────────────
    vit = build_vit(device)
    train_one_model(vit, "ViT-S/16", train_ldr, val_ldr,
                    device, EPOCHS_VIT, LR_VIT, WD_VIT, seed)

    # ── Evaluate both ─────────────────────────────────────────────
    for model_name, model in [("ResNet34", resnet), ("ViT-S/16", vit)]:
        r = SeedResult(seed=seed, model_name=model_name)

        r.clean_acc = eval_model(model, dataset, test_idx, device)
        print(f"  [seed={seed}] {model_name} clean: {r.clean_acc*100:.2f}%")

        num_bands = IN_CHANNELS
        for bi in range(num_bands):
            fn = partial(band_dropout, band_idx=bi)
            acc = eval_model(model, dataset, test_idx, device, fn)
            bname = BAND_NAMES[bi] if USE_MULTISPECTRAL else ["R","G","B"][bi]
            r.band_dropout[bname] = acc

        for sev in SENSOR_SEVERITIES:
            fn = partial(sentinel2_full_noise, severity=sev)
            r.sensor_noise[sev] = eval_model(model, dataset, test_idx, device, fn)

        for od in HAZE_DEPTHS:
            fn = partial(atmospheric_haze, optical_depth=od)
            r.haze[od] = eval_model(model, dataset, test_idx, device, fn)

        for gname, bindices in BAND_GROUPS.items():
            valid = [bi for bi in bindices if bi < IN_CHANNELS]
            if not valid:
                continue
            fn = partial(band_group_dropout, indices=valid)
            r.band_groups[gname] = eval_model(model, dataset, test_idx, device, fn)

        results.append(r)

    del resnet, vit
    torch.cuda.empty_cache(); gc.collect()
    return results


# ─────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────
def aggregate(all_results: list[SeedResult]) -> dict:
    agg = {m: {} for m in MODEL_NAMES}

    for model_name in MODEL_NAMES:
        rs = [r for r in all_results if r.model_name == model_name]

        # Clean
        vals = [r.clean_acc for r in rs]
        agg[model_name]["clean"] = {
            "accuracy": {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "vals": vals}
        }

        # Band dropout
        agg[model_name]["band_dropout"] = {}
        all_bands = list(rs[0].band_dropout.keys())
        for bname in all_bands:
            vals = [r.band_dropout[bname] for r in rs]
            agg[model_name]["band_dropout"][bname] = {
                "mean": float(np.mean(vals)), "std": float(np.std(vals)), "vals": vals
            }

        # Sensor noise
        agg[model_name]["sensor_noise"] = {}
        for sev in SENSOR_SEVERITIES:
            vals = [r.sensor_noise[sev] for r in rs]
            agg[model_name]["sensor_noise"][str(sev)] = {
                "mean": float(np.mean(vals)), "std": float(np.std(vals)), "vals": vals
            }

        # Haze
        agg[model_name]["haze"] = {}
        for od in HAZE_DEPTHS:
            vals = [r.haze[od] for r in rs]
            agg[model_name]["haze"][str(od)] = {
                "mean": float(np.mean(vals)), "std": float(np.std(vals)), "vals": vals
            }

        # Band groups
        agg[model_name]["band_groups"] = {}
        for gname in BAND_GROUPS:
            if gname not in rs[0].band_groups: continue
            vals = [r.band_groups[gname] for r in rs]
            agg[model_name]["band_groups"][gname] = {
                "mean": float(np.mean(vals)), "std": float(np.std(vals)), "vals": vals
            }

    return agg


# ─────────────────────────────────────────────────────────────────────
# Table Printer with Statistical Tests
# ─────────────────────────────────────────────────────────────────────
def print_paper_table(agg: dict):
    print("\n" + "="*80)
    print("  PAPER-READY RESULTS TABLE (with paired t-tests)")
    print("="*80)

    print(f"\n  {'Experiment':<30} {'ResNet34':>18} {'ViT-S/16':>18} {'p-value':>10}")
    print(f"  {'-'*80}")

    def fmt(d):
        return f"{d['mean']*100:.2f} ± {d['std']*100:.2f}%"

    def calc_p(vals1, vals2):
        if len(vals1) < 2: return "N/A"
        try:
            _, p_val = ttest_rel(vals1, vals2)
            return f"{p_val:.3f}"
        except:
            return "err"

    # Clean
    r_clean = agg['ResNet34']['clean']['accuracy']
    v_clean = agg['ViT-S/16']['clean']['accuracy']
    p_val   = calc_p(r_clean['vals'], v_clean['vals'])
    print(f"  {'Clean accuracy':<30} {fmt(r_clean):>18} {fmt(v_clean):>18} {p_val:>10}")

    # Key band dropouts
    print(f"\n  {'--- Band dropout (accuracy) ---':<30}")
    for bname in list(agg[MODEL_NAMES[0]]["band_dropout"].keys())[:6]:
        r_d = agg["ResNet34"]["band_dropout"][bname]
        v_d = agg["ViT-S/16"]["band_dropout"][bname]
        p_val = calc_p(r_d['vals'], v_d['vals'])
        print(f"  {'  Dropout '+bname:<30} {fmt(r_d):>18} {fmt(v_d):>18} {p_val:>10}")

    # Noise
    print(f"\n  {'--- Sensor noise (severity) ---':<30}")
    for sev in ["1.0", "5.0"]:
        if sev in agg["ResNet34"]["sensor_noise"]:
            r_d = agg["ResNet34"]["sensor_noise"][sev]
            v_d = agg["ViT-S/16"]["sensor_noise"][sev]
            p_val = calc_p(r_d['vals'], v_d['vals'])
            print(f"  {'  Noise ×'+sev:<30} {fmt(r_d):>18} {fmt(v_d):>18} {p_val:>10}")

    # Haze
    print(f"\n  {'--- Atmospheric haze ---':<30}")
    for od in ["0.2", "0.5"]:
        if od in agg["ResNet34"]["haze"]:
            r_d = agg["ResNet34"]["haze"][od]
            v_d = agg["ViT-S/16"]["haze"][od]
            p_val = calc_p(r_d['vals'], v_d['vals'])
            print(f"  {'  Haze OD='+od:<30} {fmt(r_d):>18} {fmt(v_d):>18} {p_val:>10}")

    # Band groups
    print(f"\n  {'--- Band group dropout ---':<30}")
    for gname in list(agg[MODEL_NAMES[0]]["band_groups"].keys()):
        r_d = agg["ResNet34"]["band_groups"].get(gname, {})
        v_d = agg["ViT-S/16"]["band_groups"].get(gname, {})
        if r_d and v_d:
            short = gname[:28]
            p_val = calc_p(r_d['vals'], v_d['vals'])
            print(f"  {'  '+short:<30} {fmt(r_d):>18} {fmt(v_d):>18} {p_val:>10}")

    print("="*80)


def plot_spectral_sensitivity(agg: dict, out_dir: str) -> None:
    if not USE_MULTISPECTRAL:
        band_labels = ["R (Red)", "G (Green)", "B (Blue)"]
        band_wls    = [665, 560, 490]
    else:
        band_labels = BAND_NAMES
        band_wls    = WAVELENGTHS

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Spectral Sensitivity Curve — Mean ± Std over {len(SEEDS)} seeds\n"
        f"(x-axis = wavelength in nm, shaded = ±1 std)",
        fontsize=12, fontweight="bold")

    for ax, metric in zip(axes, ["accuracy", "drop"]):
        for model_name in MODEL_NAMES:
            clean_mean = agg[model_name]["clean"]["accuracy"]["mean"]
            band_data  = agg[model_name]["band_dropout"]

            xs    = []
            means = []
            stds  = []

            for bname, wl in zip(band_labels, band_wls):
                key = bname.split()[0] if not USE_MULTISPECTRAL else bname
                if key not in band_data:
                    continue
                xs.append(wl)
                m = band_data[key]["mean"]
                s = band_data[key]["std"]
                if metric == "accuracy":
                    means.append(m * 100)
                    stds.append(s * 100)
                else:
                    means.append((clean_mean - m) * 100)
                    stds.append(s * 100)

            order = np.argsort(xs)
            xs    = np.array(xs)[order]
            means = np.array(means)[order]
            stds  = np.array(stds)[order]

            color = {"ResNet34": "#6366f1", "ViT-S/16": "#f59e0b"}.get(model_name, "gray")
            ax.plot(xs, means, marker="o", linewidth=2.5,
                    markersize=7, color=color, label=model_name)
            ax.fill_between(xs, means - stds, means + stds,
                             alpha=0.15, color=color)

            if USE_MULTISPECTRAL:
                for bname, x, m in zip(
                        [b for b in band_labels],
                        xs, means):
                    ax.annotate(bname, (x, m),
                                textcoords="offset points",
                                xytext=(0, 8), fontsize=6,
                                ha="center", color=color, alpha=0.8)

        if USE_MULTISPECTRAL:
            ax.axvspan(400,  700,  alpha=0.05, color="blue",  label="Visible")
            ax.axvspan(700,  900,  alpha=0.05, color="red",   label="NIR")
            ax.axvspan(1400, 2300, alpha=0.05, color="brown", label="SWIR")

        if metric == "accuracy":
            for model_name in MODEL_NAMES:
                m = agg[model_name]["clean"]["accuracy"]["mean"] * 100
                s = agg[model_name]["clean"]["accuracy"]["std"]  * 100
                ax.axhline(m, color={"ResNet34": "#6366f1", "ViT-S/16": "#f59e0b"}[model_name],
                           linestyle="--", linewidth=1, alpha=0.4)
            ax.set_ylabel("Accuracy after band dropout (%)", fontsize=11)
        else:
            ax.set_ylabel("Accuracy drop vs clean (pp)", fontsize=11)

        ax.set_xlabel("Wavelength (nm)", fontsize=11)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "spectral_curve_mean_std.png"),
                dpi=160, bbox_inches="tight")
    plt.close()
    print("  Saved → spectral_curve_mean_std.png")


def plot_noise_robustness(agg: dict, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Physical Noise Robustness — Mean ± Std ({len(SEEDS)} seeds)",
        fontsize=12, fontweight="bold")

    ax = axes[0]
    for model_name in MODEL_NAMES:
        data  = agg[model_name]["sensor_noise"]
        xs    = sorted([float(k) for k in data.keys()])
        means = [data[str(x)]["mean"] * 100 for x in xs]
        stds  = [data[str(x)]["std"]  * 100 for x in xs]
        color = {"ResNet34": "#6366f1", "ViT-S/16": "#f59e0b"}[model_name]
        ax.errorbar(xs, means, yerr=stds, marker="o", linewidth=2,
                    capsize=5, color=color, label=model_name)
        cm = agg[model_name]["clean"]["accuracy"]["mean"] * 100
        ax.axhline(cm, color=color, linestyle="--", linewidth=1, alpha=0.4)

    ax.set_xlabel("Noise severity (× real sensor level)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Calibrated sensor noise (all bands)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for model_name in MODEL_NAMES:
        data  = agg[model_name]["haze"]
        xs    = sorted([float(k) for k in data.keys()])
        means = [data[str(x)]["mean"] * 100 for x in xs]
        stds  = [data[str(x)]["std"]  * 100 for x in xs]
        color = {"ResNet34": "#6366f1", "ViT-S/16": "#f59e0b"}[model_name]
        ax.errorbar(xs, means, yerr=stds, marker="o", linewidth=2,
                    capsize=5, color=color, label=model_name)
        cm = agg[model_name]["clean"]["accuracy"]["mean"] * 100
        ax.axhline(cm, color=color, linestyle="--", linewidth=1, alpha=0.4)

    ax.set_xlabel("Optical depth")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Atmospheric haze (Beer-Lambert)")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "noise_mean_std.png"),
                dpi=160, bbox_inches="tight")
    plt.close()
    print("  Saved → noise_mean_std.png")


def plot_band_groups(agg: dict, out_dir: str) -> None:
    groups = [g for g in BAND_GROUPS
              if g in agg[MODEL_NAMES[0]].get("band_groups", {})]
    if not groups:
        return

    x      = np.arange(len(groups))
    width  = 0.35
    colors = [{"ResNet34": "#6366f1", "ViT-S/16": "#f59e0b"}[m] for m in MODEL_NAMES]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        f"Band Group Ablation — Mean ± Std ({len(SEEDS)} seeds)",
        fontsize=12, fontweight="bold")

    for ax, metric in zip(axes, ["accuracy", "drop"]):
        for mi, model_name in enumerate(MODEL_NAMES):
            clean_mean = agg[model_name]["clean"]["accuracy"]["mean"]
            vals, errs = [], []
            for gname in groups:
                d = agg[model_name]["band_groups"].get(gname, {})
                m = d.get("mean", 0)
                s = d.get("std",  0)
                if metric == "accuracy":
                    vals.append(m * 100)
                    errs.append(s * 100)
                else:
                    vals.append((clean_mean - m) * 100)
                    errs.append(s * 100)

            offset = (mi - 0.5) * width
            bars   = ax.bar(x + offset, vals, width, yerr=errs,
                            label=model_name, color=colors[mi],
                            alpha=0.85, capsize=4)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+0.5,
                        f"{v:.1f}", ha="center", va="bottom",
                        fontsize=7, color="white")

        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(
            "Accuracy (%)" if metric=="accuracy"
            else "Accuracy drop (pp)")
        ax.legend(); ax.grid(True, alpha=0.3, axis="y")

    axes[0].set_title("Accuracy after band group dropout")
    axes[1].set_title("Drop vs clean — which groups matter most")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "band_group_mean_std.png"),
                dpi=160, bbox_inches="tight")
    plt.close()
    print("  Saved → band_group_mean_std.png")


def save_per_seed_csv(all_results: list[SeedResult], out_dir: str):
    path = os.path.join(out_dir, "per_seed_results.csv")
    rows = []
    for r in all_results:
        base = {"seed": r.seed, "model": r.model_name,
                "clean_acc": f"{r.clean_acc:.6f}"}
        for bname, acc in r.band_dropout.items():
            base[f"drop_{bname}"] = f"{acc:.6f}"
        for sev, acc in r.sensor_noise.items():
            base[f"noise_sev{sev}"] = f"{acc:.6f}"
        for od, acc in r.haze.items():
            base[f"haze_od{od}"] = f"{acc:.6f}"
        for gname, acc in r.band_groups.items():
            safe = gname.replace(" ","_").replace("(","").replace(")","")
            base[f"group_{safe}"] = f"{acc:.6f}"
        rows.append(base)

    if rows:
        keys = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    print(f"  Saved → per_seed_results.csv")


def save_summary_csv(agg: dict, out_dir: str):
    path = os.path.join(out_dir, "summary_table.csv")
    rows = []

    for model_name in MODEL_NAMES:
        d = agg[model_name]["clean"]["accuracy"]
        rows.append({
            "model": model_name, "experiment": "clean",
            "key": "accuracy",
            "mean": f"{d['mean']*100:.3f}",
            "std":  f"{d['std']*100:.3f}",
            "mean_pm_std": f"{d['mean']*100:.2f} ± {d['std']*100:.2f}",
        })

        for bname, d in agg[model_name]["band_dropout"].items():
            clean_m = agg[model_name]["clean"]["accuracy"]["mean"]
            drop    = (clean_m - d["mean"]) * 100
            rows.append({
                "model": model_name, "experiment": "band_dropout",
                "key": bname,
                "mean": f"{d['mean']*100:.3f}",
                "std":  f"{d['std']*100:.3f}",
                "mean_pm_std": (f"{d['mean']*100:.2f} ± {d['std']*100:.2f} (drop {drop:.1f}pp)"),
            })

        for sev, d in agg[model_name]["sensor_noise"].items():
            rows.append({
                "model": model_name, "experiment": "sensor_noise",
                "key": f"severity={sev}",
                "mean": f"{d['mean']*100:.3f}",
                "std":  f"{d['std']*100:.3f}",
                "mean_pm_std": f"{d['mean']*100:.2f} ± {d['std']*100:.2f}",
            })

        for od, d in agg[model_name]["haze"].items():
            rows.append({
                "model": model_name, "experiment": "atmospheric_haze",
                "key": f"optical_depth={od}",
                "mean": f"{d['mean']*100:.3f}",
                "std":  f"{d['std']*100:.3f}",
                "mean_pm_std": f"{d['mean']*100:.2f} ± {d['std']*100:.2f}",
            })

        for gname, d in agg[model_name]["band_groups"].items():
            clean_m = agg[model_name]["clean"]["accuracy"]["mean"]
            drop    = (clean_m - d["mean"]) * 100
            rows.append({
                "model": model_name, "experiment": "band_group_dropout",
                "key": gname,
                "mean": f"{d['mean']*100:.3f}",
                "std":  f"{d['std']*100:.3f}",
                "mean_pm_std": (f"{d['mean']*100:.2f} ± {d['std']*100:.2f} (drop {drop:.1f}pp)"),
            })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model","experiment","key",
                           "mean","std","mean_pm_std"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved → summary_table.csv")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode   = "Multispectral (13-band)" if USE_MULTISPECTRAL else "RGB"
    
    print(f"Device  : {device}")
    print(f"Mode    : {mode}")
    print(f"Seeds   : {SEEDS}  ({len(SEEDS)} runs)")
    print(f"Output  : {OUT_DIR}")

    data_path = MS_DATASET_PATH if USE_MULTISPECTRAL else RGB_DATASET_PATH
    print(f"\nLoading dataset from: {data_path}")
    dataset = SatDataset(data_path, CLASS_NAMES)

    all_results: list[SeedResult] = []
    t_start = time.time()

    for si, seed in enumerate(SEEDS):
        print(f"\n{'='*60}")
        print(f"  SEED {seed}  ({si+1}/{len(SEEDS)})")
        print(f"  Elapsed: {(time.time()-t_start)/60:.1f} min")
        print(f"{'='*60}")

        seed_results = run_one_seed(seed, dataset, device)
        all_results.extend(seed_results)
        save_per_seed_csv(all_results, OUT_DIR)

    print(f"\n{'='*60}")
    print(f"  AGGREGATING {len(SEEDS)} SEEDS")
    print(f"{'='*60}")

    agg = aggregate(all_results)
    save_summary_csv(agg, OUT_DIR)
    plot_spectral_sensitivity(agg, OUT_DIR)
    plot_noise_robustness(agg, OUT_DIR)
    plot_band_groups(agg, OUT_DIR)
    print_paper_table(agg)

    total_min = (time.time() - t_start) / 60
    print(f"\nTotal time: {total_min:.1f} minutes")

if __name__ == "__main__":
    main()