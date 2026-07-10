"""
spectral_band_analysis.py
─────────────────────────
Systematically analyses how removing or perturbing individual RGB bands
affects per-class accuracy for ResNet34 and ViT-S/16.

Outputs (saved to checkpoints_best/analysis/spectral_bands/):
  • band_importance_matrix.png   – heatmap: drop per class × band
  • band_scale_curves.png        – accuracy vs scale factor per band
  • band_noise_curves.png        – accuracy vs noise sigma per band
  • band_importance.csv          – raw numbers
  • band_summary.md              – human-readable summary

Run:  python spectral_band_analysis.py
"""

import csv
import os
import sys
import warnings
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
import train as cfg

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
OUT_DIR   = os.path.join(cfg.SAVE_DIR, "analysis", "spectral_bands")
SEED      = cfg.SEED
BATCH     = cfg.EVAL_BATCH_SIZE
WORKERS   = 0           # 0 = safe on Windows
MAX_SAMPLES = None      # set to 500 for a quick pilot

BANDS     = [0, 1, 2]   # R=0, G=1, B=2
BAND_NAMES = ["Red", "Green", "Blue"]
BAND_COLORS = ["#ef4444", "#22c55e", "#3b82f6"]

# Scale factors: 1.0 = clean, 0.0 = channel zeroed
SCALE_LEVELS = [1.0, 0.75, 0.50, 0.25, 0.0]

# Noise sigmas applied to a single band
NOISE_SIGMAS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]

MASK_CACHE: dict = {}


# ─────────────────────────────────────────────
# Records
# ─────────────────────────────────────────────
@dataclass
class BandRecord:
    model:      str
    operation:  str   # "scale" | "noise" | "dropout"
    band_idx:   int
    band_name:  str
    severity:   float
    accuracy:   float
    f1:         float
    per_class_acc: list  # list[float] len=10


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
def base_t():
    return transforms.Compose([
        transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
                          interpolation=InterpolationMode.BILINEAR,
                          antialias=True),
        transforms.ToTensor(),
    ])


def normalize(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(cfg.IMAGENET_MEAN).view(3,1,1)
    std  = torch.tensor(cfg.IMAGENET_STD).view(3,1,1)
    return (img - mean) / std


class BandDataset(Dataset):
    def __init__(self, dataset, indices, band_fn=None):
        self.dataset  = dataset
        self.indices  = list(indices)
        self.to_t     = base_t()
        self.band_fn  = band_fn

    def __len__(self): return len(self.indices)

    def __getitem__(self, i):
        img, label = self.dataset[self.indices[i]]
        img = self.to_t(img)
        if self.band_fn:
            img = self.band_fn(img)
        return normalize(img), label


def make_loader(dataset, device):
    kwargs = dict(dataset=dataset, batch_size=BATCH, shuffle=False,
                  num_workers=WORKERS, pin_memory=device.type=="cuda")
    if WORKERS > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**kwargs)


# ─────────────────────────────────────────────
# Band corruption functions
# ─────────────────────────────────────────────
def scale_band(img: torch.Tensor, band: int, factor: float) -> torch.Tensor:
    out = img.clone()
    out[band] = out[band] * factor
    return out.clamp(0, 1)


def noise_band(img: torch.Tensor, band: int, sigma: float) -> torch.Tensor:
    out = img.clone()
    out[band] = (out[band] + torch.randn_like(out[band]) * sigma).clamp(0, 1)
    return out


def jitter_band(img: torch.Tensor, band: int, shift: float) -> torch.Tensor:
    """Shift a channel's mean by ±shift (simulates sensor calibration drift)."""
    out = img.clone()
    out[band] = (out[band] + shift).clamp(0, 1)
    return out


# ─────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────
def evaluate(model, loader, device, channels_last, num_classes):
    all_preds, all_labels = [], []
    model.eval()
    with torch.inference_mode():
        for imgs, labels in loader:
            if channels_last and imgs.ndim == 4:
                imgs = imgs.to(device, memory_format=torch.channels_last)
            else:
                imgs = imgs.to(device)
            labels = labels.to(device)
            preds  = model(imgs).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = float(np.mean(all_preds == all_labels))
    f1  = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))

    per_class = []
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() == 0:
            per_class.append(float("nan"))
        else:
            per_class.append(float(np.mean(all_preds[mask] == c)))

    return acc, f1, per_class


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
def plot_dropout_heatmap(records, class_names, out_dir):
    """
    Heatmap: rows = classes, cols = bands.
    Value = accuracy when that band is ZEROED OUT.
    Separate figure per model.
    """
    models = sorted({r.model for r in records})
    for model_name in models:
        # filter: scale operation, severity=0 (zeroed), per band
        data = np.zeros((len(class_names), 3))
        clean_acc = None

        for band_idx in BANDS:
            subset = [r for r in records
                      if r.model == model_name
                      and r.operation == "scale"
                      and r.band_idx == band_idx
                      and r.severity == 0.0]
            if not subset:
                continue
            r = subset[0]
            for ci in range(len(class_names)):
                data[ci, band_idx] = r.per_class_acc[ci] * 100

        # also get clean
        clean_r = [r for r in records
                   if r.model == model_name and r.operation == "clean"]
        if clean_r:
            clean_pc = clean_r[0].per_class_acc
        else:
            clean_pc = [1.0] * len(class_names)

        # compute drop = clean - zeroed
        clean_arr = np.array([x * 100 for x in clean_pc])
        drop = clean_arr[:, np.newaxis] - data   # higher = more important

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle(f"{model_name} — Band Importance (accuracy drop when band zeroed)",
                     fontsize=12, fontweight="bold")

        # absolute accuracy when zeroed
        im0 = axes[0].imshow(data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        axes[0].set_xticks(range(3))
        axes[0].set_xticklabels(BAND_NAMES)
        axes[0].set_yticks(range(len(class_names)))
        axes[0].set_yticklabels(class_names, fontsize=9)
        axes[0].set_title("Accuracy when band=0 (%)")
        for ci in range(len(class_names)):
            for bi in range(3):
                v = data[ci, bi]
                axes[0].text(bi, ci, f"{v:.0f}", ha="center", va="center",
                             fontsize=7, color="black" if v > 40 else "white")
        plt.colorbar(im0, ax=axes[0])

        # accuracy drop (importance)
        im1 = axes[1].imshow(drop, cmap="Reds", vmin=0, vmax=drop.max()+5, aspect="auto")
        axes[1].set_xticks(range(3))
        axes[1].set_xticklabels(BAND_NAMES)
        axes[1].set_yticks(range(len(class_names)))
        axes[1].set_yticklabels(class_names, fontsize=9)
        axes[1].set_title("Accuracy DROP vs clean (↑ = more important)")
        for ci in range(len(class_names)):
            for bi in range(3):
                v = drop[ci, bi]
                axes[1].text(bi, ci, f"{v:.0f}", ha="center", va="center",
                             fontsize=7, color="white" if v > drop.max()*0.6 else "black")
        plt.colorbar(im1, ax=axes[1])

        plt.tight_layout()
        safe = model_name.replace("/","_").replace(" ","_")
        plt.savefig(os.path.join(out_dir, f"{safe}_band_importance.png"), dpi=160)
        plt.close()
        print(f"  Saved band importance heatmap for {model_name}")


def plot_scale_curves(records, out_dir):
    models = sorted({r.model for r in records})
    fig, axes = plt.subplots(1, len(models), figsize=(7*len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        for band_idx in BANDS:
            subset = sorted(
                [r for r in records
                 if r.model == model_name and r.operation == "scale"
                 and r.band_idx == band_idx],
                key=lambda r: r.severity
            )
            if not subset:
                continue
            xs = [r.severity for r in subset]
            ys = [r.accuracy * 100 for r in subset]
            ax.plot(xs, ys, marker="o", linewidth=2,
                    color=BAND_COLORS[band_idx],
                    label=BAND_NAMES[band_idx])

        # clean reference line
        clean_r = [r for r in records if r.model == model_name and r.operation == "clean"]
        if clean_r:
            ax.axhline(clean_r[0].accuracy * 100, color="white",
                       linestyle="--", linewidth=1, alpha=0.4, label="Clean")

        ax.set_xlabel("Band scale factor (1.0 = original, 0.0 = zeroed)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{model_name} — Band scaling robustness")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "band_scale_curves.png"), dpi=160)
    plt.close()
    print("  Saved band scale curves")


def plot_noise_curves(records, out_dir):
    models = sorted({r.model for r in records})
    fig, axes = plt.subplots(1, len(models), figsize=(7*len(models), 5))
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        for band_idx in BANDS:
            subset = sorted(
                [r for r in records
                 if r.model == model_name and r.operation == "noise"
                 and r.band_idx == band_idx],
                key=lambda r: r.severity
            )
            if not subset:
                continue
            xs = [r.severity for r in subset]
            ys = [r.accuracy * 100 for r in subset]
            ax.plot(xs, ys, marker="o", linewidth=2,
                    color=BAND_COLORS[band_idx],
                    label=BAND_NAMES[band_idx])

        clean_r = [r for r in records if r.model == model_name and r.operation == "clean"]
        if clean_r:
            ax.axhline(clean_r[0].accuracy * 100, color="white",
                       linestyle="--", linewidth=1, alpha=0.4, label="Clean (all bands)")

        ax.set_xlabel("Noise sigma (applied to single band only)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{model_name} — Single-band noise robustness")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "band_noise_curves.png"), dpi=160)
    plt.close()
    print("  Saved band noise curves")


def plot_comparison(records, out_dir):
    """Side by side bar chart: accuracy drop per band, ResNet vs ViT."""
    models = sorted({r.model for r in records})

    # get overall acc drop when each band is zeroed
    fig, ax = plt.subplots(figsize=(9, 5))
    x      = np.arange(3)
    width  = 0.35
    colors = ["#6366f1", "#f59e0b"]

    for mi, model_name in enumerate(models):
        clean_r = [r for r in records if r.model == model_name and r.operation == "clean"]
        clean_acc = clean_r[0].accuracy if clean_r else 1.0

        drops = []
        for band_idx in BANDS:
            z = [r for r in records
                 if r.model == model_name and r.operation == "scale"
                 and r.band_idx == band_idx and r.severity == 0.0]
            if z:
                drops.append((clean_acc - z[0].accuracy) * 100)
            else:
                drops.append(0)

        offset = (mi - 0.5) * width
        bars = ax.bar(x + offset, drops, width,
                      label=model_name, color=colors[mi], alpha=0.85)
        for bar, val in zip(bars, drops):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom",
                    fontsize=8, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(BAND_NAMES)
    ax.set_ylabel("Accuracy drop (pp) when band zeroed")
    ax.set_title("Band importance comparison: ResNet34 vs ViT-S/16")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "band_importance_comparison.png"), dpi=160)
    plt.close()
    print("  Saved band importance comparison")


def write_csv(records, out_dir):
    path = os.path.join(out_dir, "band_importance.csv")
    class_names = [
        "AnnualCrop","Forest","HerbaceousVegetation","Highway",
        "Industrial","Pasture","PermanentCrop","Residential",
        "River","SeaLake"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["model","operation","band","severity","accuracy","f1"]
        header += [f"acc_{c}" for c in class_names]
        writer.writerow(header)
        for r in records:
            row = [r.model, r.operation, r.band_name,
                   f"{r.severity:.4f}", f"{r.accuracy:.6f}", f"{r.f1:.6f}"]
            row += [f"{v:.6f}" for v in r.per_class_acc]
            writer.writerow(row)
    print(f"  Saved CSV → {path}")


def write_summary(records, out_dir):
    class_names = [
        "AnnualCrop","Forest","HerbaceousVegetation","Highway",
        "Industrial","Pasture","PermanentCrop","Residential",
        "River","SeaLake"
    ]
    models = sorted({r.model for r in records})
    lines  = ["# Spectral Band Importance Summary", ""]

    for model_name in models:
        lines.append(f"## {model_name}")
        clean_r = [r for r in records if r.model == model_name and r.operation == "clean"]
        if clean_r:
            lines.append(f"- Clean accuracy: {clean_r[0].accuracy*100:.2f}%")

        lines.append("\n### Accuracy when band zeroed:")
        for band_idx in BANDS:
            z = [r for r in records if r.model == model_name
                 and r.operation == "scale" and r.band_idx == band_idx
                 and r.severity == 0.0]
            if z:
                drop = (clean_r[0].accuracy - z[0].accuracy) * 100 if clean_r else 0
                lines.append(f"- {BAND_NAMES[band_idx]}: {z[0].accuracy*100:.2f}%  "
                              f"(drop: {drop:.2f} pp)")

        lines.append("\n### Most band-sensitive classes (Red band zeroed):")
        r_zero = [r for r in records if r.model == model_name
                  and r.operation == "scale" and r.band_idx == 0
                  and r.severity == 0.0]
        if r_zero and clean_r:
            drops = [(class_names[i],
                      (clean_r[0].per_class_acc[i] - r_zero[0].per_class_acc[i]) * 100)
                     for i in range(len(class_names))]
            drops.sort(key=lambda x: -x[1])
            for name, d in drops[:5]:
                lines.append(f"  - {name}: -{d:.1f} pp")
        lines.append("")

    path = os.path.join(out_dir, "band_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved summary → {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset     = ImageFolder(root=cfg.DATASET_PATH)
    class_names = dataset.classes
    num_classes = len(class_names)
    _, _, test_idx = cfg.build_splits(dataset.targets)
    test_idx = list(test_idx)

    if MAX_SAMPLES:
        rng = np.random.default_rng(SEED)
        test_idx = rng.choice(test_idx, min(MAX_SAMPLES, len(test_idx)),
                              replace=False).tolist()

    print(f"Test samples: {len(test_idx)}")

    # ── load models ────────────────────────────────────────────────
    resnet = timm.create_model(cfg.RESNET_MODEL_NAME, pretrained=False,
                               num_classes=num_classes)
    resnet.load_state_dict(cfg.safe_load_state_dict(cfg.RESNET_CKPT, device))
    resnet = resnet.to(device).eval()

    vit = timm.create_model(cfg.VIT_MODEL_NAME, pretrained=False,
                            num_classes=num_classes,
                            drop_rate=cfg.VIT_DROP_RATE,
                            drop_path_rate=cfg.VIT_DROP_PATH_RATE)
    vit.load_state_dict(cfg.safe_load_state_dict(cfg.VIT_CKPT, device))
    vit = vit.to(device).eval()

    model_specs = [
        (cfg.RESNET_DISPLAY_NAME, resnet, True),
        (cfg.VIT_DISPLAY_NAME,    vit,    False),
    ]

    records: list[BandRecord] = []

    # ── clean baseline ─────────────────────────────────────────────
    print("\n── Clean baseline ──")
    clean_ds     = BandDataset(dataset, test_idx, band_fn=None)
    clean_loader = make_loader(clean_ds, device)

    for mname, model, cl in model_specs:
        acc, f1, pc = evaluate(model, clean_loader, device, cl, num_classes)
        records.append(BandRecord(mname, "clean", -1, "clean", 0.0, acc, f1, pc))
        print(f"  {mname}: {acc*100:.2f}%")

    # ── band scaling (0 = zeroed, 0.25, 0.5, 0.75, 1.0) ──────────
    print("\n── Band scaling ──")
    for band_idx in BANDS:
        for scale in SCALE_LEVELS:
            print(f"  Band={BAND_NAMES[band_idx]}  scale={scale:.2f}", end="  ")
            fn = lambda img, b=band_idx, s=scale: scale_band(img, b, s)
            ds = BandDataset(dataset, test_idx, band_fn=fn)
            ldr = make_loader(ds, device)
            for mname, model, cl in model_specs:
                acc, f1, pc = evaluate(model, ldr, device, cl, num_classes)
                records.append(BandRecord(
                    mname, "scale", band_idx, BAND_NAMES[band_idx], scale, acc, f1, pc))
                print(f"{mname}:{acc*100:.1f}%", end="  ")
            print()

    # ── single-band noise ──────────────────────────────────────────
    print("\n── Single-band noise ──")
    for band_idx in BANDS:
        for sigma in NOISE_SIGMAS:
            if sigma == 0.0:
                continue
            print(f"  Band={BAND_NAMES[band_idx]}  sigma={sigma:.2f}", end="  ")
            fn = lambda img, b=band_idx, s=sigma: noise_band(img, b, s)
            ds = BandDataset(dataset, test_idx, band_fn=fn)
            ldr = make_loader(ds, device)
            for mname, model, cl in model_specs:
                acc, f1, pc = evaluate(model, ldr, device, cl, num_classes)
                records.append(BandRecord(
                    mname, "noise", band_idx, BAND_NAMES[band_idx], sigma, acc, f1, pc))
                print(f"{mname}:{acc*100:.1f}%", end="  ")
            print()

    # ── jitter (mean shift) ────────────────────────────────────────
    print("\n── Single-band jitter (mean shift) ──")
    jitter_shifts = [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]
    for band_idx in BANDS:
        for shift in jitter_shifts:
            print(f"  Band={BAND_NAMES[band_idx]}  shift={shift:+.1f}", end="  ")
            fn = lambda img, b=band_idx, s=shift: jitter_band(img, b, s)
            ds = BandDataset(dataset, test_idx, band_fn=fn)
            ldr = make_loader(ds, device)
            for mname, model, cl in model_specs:
                acc, f1, pc = evaluate(model, ldr, device, cl, num_classes)
                records.append(BandRecord(
                    mname, "jitter", band_idx, BAND_NAMES[band_idx], shift, acc, f1, pc))
                print(f"{mname}:{acc*100:.1f}%", end="  ")
            print()

    # ── plots & export ────────────────────────────────────────────
    print("\n── Saving outputs ──")
    plot_dropout_heatmap(records, class_names, OUT_DIR)
    plot_scale_curves(records, OUT_DIR)
    plot_noise_curves(records, OUT_DIR)
    plot_comparison(records, OUT_DIR)
    write_csv(records, OUT_DIR)
    write_summary(records, OUT_DIR)

    print(f"\nDone. Outputs saved to:\n  {OUT_DIR}")


if __name__ == "__main__":
    run()