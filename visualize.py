"""
visualize.py  –  Grad-CAM (ResNet34) + Attention-Rollout (ViT-S/16)
Generates heatmaps for:
  • Clean images
  • Gaussian-noise corrupted images (sigma=0.08)
  • Low-pass spectral filtered images (cutoff=0.18)
  • High-pass spectral filtered images (cutoff=0.03)

Output: checkpoints_best/analysis/heatmaps/
"""

import os
import random
import warnings
from contextlib import nullcontext

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import train as cfg

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
HEATMAP_DIR   = os.path.join(cfg.SAVE_DIR, "analysis", "heatmaps")
NUM_SAMPLES   = 20          # total images to visualise
SEED          = cfg.SEED
ALPHA         = 0.55        # heatmap overlay opacity

GAUSS_SIGMA   = 0.08        # Gaussian noise severity shown
LP_CUTOFF     = 0.18        # low-pass  cutoff
HP_CUTOFF     = 0.03        # high-pass cutoff  (ViT still ~74 % here)

IMAGENET_MEAN = cfg.IMAGENET_MEAN
IMAGENET_STD  = cfg.IMAGENET_STD
IMAGE_SIZE    = cfg.IMAGE_SIZE

MASK_CACHE: dict = {}


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────
def to_tensor(pil_img) -> torch.Tensor:
    """PIL → [0,1] float tensor (3,H,W) resized to IMAGE_SIZE."""
    t = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE),
                          interpolation=InterpolationMode.BILINEAR,
                          antialias=True),
        transforms.ToTensor(),
    ])
    return t(pil_img)


def normalize(img: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, dtype=img.dtype).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  dtype=img.dtype).view(3, 1, 1)
    return (img - mean) / std


def denormalize(img: torch.Tensor) -> np.ndarray:
    """Normalized tensor → HWC numpy [0,1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    out  = (img.cpu() * std + mean).clamp(0, 1)
    return out.permute(1, 2, 0).numpy()


# ─────────────────────────────────────────────
# Corruptions (operate on [0,1] tensors)
# ─────────────────────────────────────────────
def _radial_mask(h: int, w: int, cutoff: float, mode: str) -> torch.Tensor:
    key = (h, w, float(cutoff), mode)
    if key not in MASK_CACHE:
        yy, xx = torch.meshgrid(
            torch.linspace(-1., 1., h),
            torch.linspace(-1., 1., w),
            indexing="ij",
        )
        r = torch.sqrt(xx ** 2 + yy ** 2)
        mask = (r <= cutoff).float() if mode == "low" else (r >= cutoff).float()
        MASK_CACHE[key] = mask.unsqueeze(0)          # (1,H,W)
    return MASK_CACHE[key]


def low_pass(img: torch.Tensor, cutoff: float) -> torch.Tensor:
    _, h, w = img.shape
    mask  = _radial_mask(h, w, cutoff, "low")
    freq  = torch.fft.fftshift(torch.fft.fft2(img, dim=(-2, -1)), dim=(-2, -1))
    out   = torch.fft.ifft2(
        torch.fft.ifftshift(freq * mask, dim=(-2, -1)), dim=(-2, -1)
    ).real
    return out.clamp(0., 1.)


def high_pass(img: torch.Tensor, cutoff: float) -> torch.Tensor:
    _, h, w = img.shape
    mask  = _radial_mask(h, w, cutoff, "high")
    freq  = torch.fft.fftshift(torch.fft.fft2(img, dim=(-2, -1)), dim=(-2, -1))
    out   = torch.fft.ifft2(
        torch.fft.ifftshift(freq * mask, dim=(-2, -1)), dim=(-2, -1)
    ).real
    return out.clamp(0., 1.)


def gaussian_noise(img: torch.Tensor, sigma: float) -> torch.Tensor:
    return (img + torch.randn_like(img) * sigma).clamp(0., 1.)


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────
def load_resnet(device: torch.device, num_classes: int) -> nn.Module:
    model = timm.create_model(
        cfg.RESNET_MODEL_NAME, pretrained=False, num_classes=num_classes
    )
    model.load_state_dict(cfg.safe_load_state_dict(cfg.RESNET_CKPT, device))
    model = model.to(device)
    model.eval()
    return model


def load_vit(device: torch.device, num_classes: int) -> nn.Module:
    model = timm.create_model(
        cfg.VIT_MODEL_NAME,
        pretrained=False,
        num_classes=num_classes,
        drop_rate=cfg.VIT_DROP_RATE,
        drop_path_rate=cfg.VIT_DROP_PATH_RATE,
    )
    model.load_state_dict(cfg.safe_load_state_dict(cfg.VIT_CKPT, device))
    model = model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────
# Grad-CAM for ResNet34
# ─────────────────────────────────────────────
class GradCAM:
    """Grad-CAM on the last conv layer of a ResNet."""

    def __init__(self, model: nn.Module) -> None:
        self.model      = model
        self._acts: list  = []
        self._grads: list = []
        # ResNet34: model.layer4[-1].conv2 is the last spatial conv
        target = model.layer4[-1].conv2
        self._fwd = target.register_forward_hook(self._save_acts)
        self._bwd = target.register_full_backward_hook(self._save_grads)

    def _save_acts(self, _m, _i, output) -> None:
        self._acts.append(output.detach())

    def _save_grads(self, _m, _i, grad_out) -> None:
        self._grads.append(grad_out[0].detach())

    def __call__(
        self, x: torch.Tensor, class_idx: int | None = None
    ) -> tuple[np.ndarray, int, float]:
        """
        Returns:
            cam        – (H,W) normalised heatmap in [0,1]
            pred_idx   – predicted class index
            confidence – softmax confidence for pred_idx
        """
        self._acts.clear()
        self._grads.clear()

        self.model.zero_grad()
        logits = self.model(x)                        # (1, C)
        probs  = logits.softmax(dim=1)
        pred   = int(logits.argmax(dim=1).item())
        target = pred if class_idx is None else class_idx

        logits[0, target].backward()

        acts  = self._acts[0].squeeze(0)              # (C, h, w)
        grads = self._grads[0].squeeze(0)             # (C, h, w)
        w     = grads.mean(dim=(1, 2), keepdim=True)  # (C,1,1)
        cam   = F.relu((w * acts).sum(0))             # (h, w)
        cam   = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear", align_corners=False
        ).squeeze().cpu().numpy()
        cam   = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        conf  = float(probs[0, pred].item())
        return cam, pred, conf

    def remove(self) -> None:
        self._fwd.remove()
        self._bwd.remove()


# ─────────────────────────────────────────────
# Attention Rollout for ViT (timm-compatible)
# ─────────────────────────────────────────────
class AttentionRollout:
    """
    Works by re-computing attention weights inside a forward hook
    from Q and K projections — compatible with any timm ViT regardless
    of whether fused_attn is enabled.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model   = model
        self._maps: list[torch.Tensor] = []
        self._hooks: list  = []

        for block in model.blocks:
            h = block.attn.register_forward_hook(self._attn_hook)
            self._hooks.append(h)

    def _attn_hook(self, module, inputs, _output) -> None:
        """Recompute attention scores from Q,K (avoids fused_attn issues)."""
        x = inputs[0]                                 # (B, N, C)
        B, N, C = x.shape
        qkv = module.qkv(x)                          # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)            # (3,B,heads,N,head_dim)
        q, k, _ = qkv.unbind(0)                      # each (B,heads,N,head_dim)
        scale = (C // module.num_heads) ** -0.5
        attn  = (q @ k.transpose(-2, -1)) * scale    # (B,heads,N,N)
        attn  = attn.softmax(dim=-1)                  # (B,heads,N,N)
        self._maps.append(attn.detach().cpu())

    def __call__(
        self, x: torch.Tensor
    ) -> tuple[np.ndarray, int, float]:
        """
        Returns:
            rollout    – (H,W) normalised heatmap in [0,1]
            pred_idx   – predicted class
            confidence – softmax confidence
        """
        self._maps.clear()
        with torch.inference_mode():
            logits = self.model(x)
        probs = logits.softmax(dim=1)
        pred  = int(logits.argmax(dim=1).item())
        conf  = float(probs[0, pred].item())

        if not self._maps:
            raise RuntimeError(
                "No attention maps captured.  "
                "Check that model.blocks exist and are being called."
            )

        # Rollout: accumulate attention through layers
        # Start with identity (each token attends to itself)
        num_tokens = self._maps[0].shape[-1]
        result = torch.eye(num_tokens)                # (N,N)

        for attn in self._maps:
            # average over heads, take first batch item
            a = attn[0].mean(0)                       # (N,N)
            # add residual connection & renormalise
            a = a + torch.eye(num_tokens)
            a = a / a.sum(dim=-1, keepdim=True)
            result = a @ result

        # CLS token (row 0) attends to patch tokens (rows 1..)
        mask = result[0, 1:]                          # (num_patches,)
        n_patches = mask.shape[0]
        h = w = int(n_patches ** 0.5)
        if h * w != n_patches:
            raise ValueError(
                f"Cannot reshape {n_patches} patches into a square grid."
            )
        mask = mask.reshape(h, w).numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

        # Upsample to IMAGE_SIZE × IMAGE_SIZE
        t = torch.tensor(mask).unsqueeze(0).unsqueeze(0)   # (1,1,h,w)
        t = F.interpolate(t, size=(IMAGE_SIZE, IMAGE_SIZE),
                          mode="bilinear", align_corners=False)
        return t.squeeze().numpy(), pred, conf

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()


# ─────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────
CMAP = plt.cm.jet


def overlay(img_np: np.ndarray, heatmap: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Blend a [0,1] HWC image with a jet-coloured heatmap."""
    coloured = CMAP(heatmap)[..., :3]
    return (alpha * coloured + (1.0 - alpha) * img_np).clip(0, 1)


def _axes_off(axes) -> None:
    for ax in np.array(axes).ravel():
        ax.axis("off")


def confidence_str(conf: float) -> str:
    return f"{conf * 100:.1f}%"


# ─────────────────────────────────────────────
# Core figure:  one image × four conditions
# ─────────────────────────────────────────────
def save_heatmap_figure(
    raw_tensor: torch.Tensor,          # (3,H,W) in [0,1], NOT normalized
    true_label: str,
    class_names: list[str],
    grad_cam: GradCAM,
    attn_rollout: AttentionRollout,
    device: torch.device,
    save_path: str,
) -> None:
    """
    Saves a 4-column × 3-row grid:
      rows    → Original | GradCAM heatmap | Attention-Rollout heatmap
      columns → Clean | Gaussian(σ=0.08) | Low-Pass(0.18) | High-Pass(0.03)
    """

    conditions = [
        ("Clean",                  raw_tensor),
        (f"Gaussian σ={GAUSS_SIGMA}", gaussian_noise(raw_tensor, GAUSS_SIGMA)),
        (f"Low-Pass {LP_CUTOFF}",  low_pass(raw_tensor,  LP_CUTOFF)),
        (f"High-Pass {HP_CUTOFF}", high_pass(raw_tensor, HP_CUTOFF)),
    ]

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        f"True class: {true_label}",
        fontsize=15, fontweight="bold", y=1.01
    )

    # 3 rows (image / gradcam-overlay / rollout-overlay), 4 cols
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.08)

    row_labels = ["Image", "GradCAM\n(ResNet34)", "Attn Rollout\n(ViT-S/16)"]

    for col, (cond_name, corrupted_raw) in enumerate(conditions):
        img_np   = corrupted_raw.permute(1, 2, 0).numpy().clip(0, 1)
        inp      = normalize(corrupted_raw).unsqueeze(0).to(device)

        # ── Grad-CAM ──────────────────────────────────
        cam, r_pred, r_conf = grad_cam(inp)

        # ── Attention Rollout ──────────────────────────
        with torch.no_grad():
            roll, v_pred, v_conf = attn_rollout(inp)

        r_name = class_names[r_pred]
        v_name = class_names[v_pred]

        # row 0: original (corrupted) image
        ax0 = fig.add_subplot(gs[0, col])
        ax0.imshow(img_np)
        ax0.set_title(cond_name, fontsize=10, fontweight="bold")
        if col == 0:
            ax0.set_ylabel(row_labels[0], fontsize=9, labelpad=6)
        ax0.axis("off")

        # row 1: GradCAM overlay
        ax1 = fig.add_subplot(gs[1, col])
        ax1.imshow(overlay(img_np, cam))
        ax1.set_title(f"Pred: {r_name}\n{confidence_str(r_conf)}", fontsize=8)
        if col == 0:
            ax1.set_ylabel(row_labels[1], fontsize=9, labelpad=6)
        ax1.axis("off")

        # row 2: Attention-Rollout overlay
        ax2 = fig.add_subplot(gs[2, col])
        ax2.imshow(overlay(img_np, roll))
        ax2.set_title(f"Pred: {v_name}\n{confidence_str(v_conf)}", fontsize=8)
        if col == 0:
            ax2.set_ylabel(row_labels[2], fontsize=9, labelpad=6)
        ax2.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {os.path.basename(save_path)}")


# ─────────────────────────────────────────────
# Comparison figure: ResNet vs ViT side-by-side
# (clean only, multiple samples on one page)
# ─────────────────────────────────────────────
def save_comparison_grid(
    samples: list[tuple],              # list of (raw_tensor, true_label)
    class_names: list[str],
    grad_cam: GradCAM,
    attn_rollout: AttentionRollout,
    device: torch.device,
    save_path: str,
    max_rows: int = 10,
) -> None:
    """
    One row per sample:  Image | GradCAM | Attn-Rollout
    """
    n = min(len(samples), max_rows)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Original", "GradCAM (ResNet34)", "Attention Rollout (ViT-S/16)"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=11, fontweight="bold")

    for i, (raw_tensor, true_label) in enumerate(samples[:n]):
        img_np = raw_tensor.permute(1, 2, 0).numpy().clip(0, 1)
        inp    = normalize(raw_tensor).unsqueeze(0).to(device)

        cam,  r_pred, r_conf = grad_cam(inp)
        roll, v_pred, v_conf = attn_rollout(inp)

        axes[i, 0].imshow(img_np)
        axes[i, 0].set_ylabel(true_label, fontsize=8, rotation=0,
                               labelpad=70, va="center")

        axes[i, 1].imshow(overlay(img_np, cam))
        axes[i, 1].set_title(
            f"{class_names[r_pred]} ({confidence_str(r_conf)})", fontsize=7
        )

        axes[i, 2].imshow(overlay(img_np, roll))
        axes[i, 2].set_title(
            f"{class_names[v_pred]} ({confidence_str(v_conf)})", fontsize=7
        )

    _axes_off(axes)
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {os.path.basename(save_path)}")


# ─────────────────────────────────────────────
# Noise robustness comparison figure
# Shows how heatmap changes across severity
# ─────────────────────────────────────────────
def save_severity_figure(
    raw_tensor: torch.Tensor,
    true_label: str,
    class_names: list[str],
    grad_cam: GradCAM,
    attn_rollout: AttentionRollout,
    device: torch.device,
    save_path: str,
    noise_type: str = "gaussian",   # "gaussian" | "lowpass" | "highpass"
) -> None:
    """
    Shows how GradCAM and Attention-Rollout change as noise severity increases.
    3 rows (image / gradcam / rollout) × N severity columns.
    """
    if noise_type == "gaussian":
        severities = [0.0, 0.02, 0.05, 0.08, 0.12, 0.16]
        title_fn   = lambda s: "Clean" if s == 0.0 else f"σ={s}"
        corrupt_fn = lambda t, s: t if s == 0.0 else gaussian_noise(t, s)
    elif noise_type == "lowpass":
        severities = [0.50, 0.35, 0.25, 0.18, 0.12, 0.08]
        title_fn   = lambda s: f"LP {s}"
        corrupt_fn = lambda t, s: low_pass(t, s)
    elif noise_type == "highpass":
        severities = [0.30, 0.22, 0.15, 0.10, 0.06, 0.03]
        title_fn   = lambda s: f"HP {s}"
        corrupt_fn = lambda t, s: high_pass(t, s)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    ncols = len(severities)
    fig   = plt.figure(figsize=(4 * ncols, 10))
    fig.suptitle(
        f"{true_label}  –  {noise_type} severity sweep",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(3, ncols, figure=fig, hspace=0.4, wspace=0.05)

    for col, sev in enumerate(severities):
        corrupted = corrupt_fn(raw_tensor.clone(), sev)
        img_np    = corrupted.permute(1, 2, 0).numpy().clip(0, 1)
        inp       = normalize(corrupted).unsqueeze(0).to(device)

        cam,  r_pred, r_conf = grad_cam(inp)
        roll, v_pred, v_conf = attn_rollout(inp)

        ax0 = fig.add_subplot(gs[0, col])
        ax0.imshow(img_np)
        ax0.set_title(title_fn(sev), fontsize=9, fontweight="bold")
        if col == 0:
            ax0.set_ylabel("Image", fontsize=9)
        ax0.axis("off")

        ax1 = fig.add_subplot(gs[1, col])
        ax1.imshow(overlay(img_np, cam))
        ax1.set_title(
            f"{class_names[r_pred]}\n{confidence_str(r_conf)}", fontsize=7
        )
        if col == 0:
            ax1.set_ylabel("GradCAM\nResNet34", fontsize=9)
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[2, col])
        ax2.imshow(overlay(img_np, roll))
        ax2.set_title(
            f"{class_names[v_pred]}\n{confidence_str(v_conf)}", fontsize=7
        )
        if col == 0:
            ax2.set_ylabel("Attn Rollout\nViT-S/16", fontsize=9)
        ax2.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {os.path.basename(save_path)}")


# ─────────────────────────────────────────────
# Sample selection
# ─────────────────────────────────────────────
def select_samples(
    dataset: ImageFolder,
    test_indices: list[int],
    resnet: nn.Module,
    vit: nn.Module,
    device: torch.device,
    class_names: list[str],
    n_total: int = NUM_SAMPLES,
) -> list[tuple[torch.Tensor, str, int, str]]:
    """
    Returns a balanced list of (raw_tensor, true_label, true_idx, status)
    where status ∈ {"both_correct", "vit_only", "both_wrong"}.
    Tries to include hard classes (Pasture, PermanentCrop, AnnualCrop).
    """
    HARD_CLASSES = {"Pasture", "PermanentCrop", "AnnualCrop",
                    "HerbaceousVegetation", "Forest"}

    base_t = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE),
                          interpolation=InterpolationMode.BILINEAR,
                          antialias=True),
        transforms.ToTensor(),
    ])

    resnet.eval(); vit.eval()
    buckets: dict[str, list] = {
        "both_correct": [], "vit_only": [], "both_wrong": []
    }

    rng = np.random.default_rng(SEED)
    idx_shuffled = rng.permutation(test_indices).tolist()

    print("Collecting clean predictions for sample selection...")
    with torch.inference_mode():
        for raw_idx in idx_shuffled:
            pil_img, label_idx = dataset[raw_idx]
            raw  = base_t(pil_img)
            inp  = normalize(raw).unsqueeze(0).to(device)

            r_pred = int(resnet(inp).argmax(1).item())
            v_pred = int(vit(inp).argmax(1).item())

            r_ok  = r_pred == label_idx
            v_ok  = v_pred == label_idx
            if r_ok and v_ok:
                status = "both_correct"
            elif v_ok and not r_ok:
                status = "vit_only"
            else:
                status = "both_wrong"

            entry = (raw, class_names[label_idx], label_idx, status)
            buckets[status].append(entry)

            if (len(buckets["both_correct"]) >= n_total * 2 and
                    len(buckets["vit_only"]) >= 4):
                break

    # Build final list: prioritise hard classes & interesting cases
    # Use integer IDs to avoid tensor equality comparison
    selected: list = []
    selected_ids: set[int] = set()   # track by id(entry) not value

    def _add(e) -> bool:
        eid = id(e)
        if eid not in selected_ids and len(selected) < n_total:
            selected.append(e)
            selected_ids.add(eid)
            return True
        return False

    # At least 4 "vit_only" samples (ViT correct, ResNet wrong)
    for e in buckets["vit_only"][:4]:
        _add(e)

    # Fill rest from both_correct, preferring hard classes first
    hard_first = sorted(
        buckets["both_correct"],
        key=lambda e: (e[1] not in HARD_CLASSES, 0)
    )
    for e in hard_first:
        if len(selected) >= n_total:
            break
        _add(e)

    # Pad with both_wrong if still needed
    for e in buckets["both_wrong"]:
        if len(selected) >= n_total:
            break
        _add(e)

    print(f"Selected {len(selected)} samples for heatmap generation.")
    return selected[:n_total]


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def run() -> None:
    os.makedirs(HEATMAP_DIR, exist_ok=True)
    seed_everything(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Dataset path: {cfg.DATASET_PATH}")

    dataset     = ImageFolder(root=cfg.DATASET_PATH)
    class_names = dataset.classes
    num_classes = len(class_names)

    _, _, test_idx = cfg.build_splits(dataset.targets)
    test_idx = list(test_idx)
    print(f"Test samples: {len(test_idx)}")

    resnet = load_resnet(device, num_classes)
    vit    = load_vit(device, num_classes)

    grad_cam     = GradCAM(resnet)
    attn_rollout = AttentionRollout(vit)

    # ── sanity check: verify hooks work on a dummy input ──────────────
    dummy = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    try:
        _ = grad_cam(dummy)
        print("GradCAM hook: OK")
    except Exception as e:
        print(f"GradCAM hook FAILED: {e}")

    try:
        _ = attn_rollout(dummy)
        print("AttentionRollout hook: OK")
    except Exception as e:
        print(f"AttentionRollout hook FAILED: {e}")
        raise

    # ── sample selection ──────────────────────────────────────────────
    samples = select_samples(
        dataset, test_idx, resnet, vit, device, class_names
    )

    # ── per-sample: full 4-condition figure ───────────────────────────
    print("\n--- Generating per-sample heatmap figures ---")
    for i, (raw, true_label, true_idx, status) in enumerate(samples):
        tag = f"{i:02d}_{true_label}_{status}"
        path = os.path.join(HEATMAP_DIR, f"{tag}_all_conditions.png")
        print(f"  [{i+1}/{len(samples)}] {true_label} ({status})")
        save_heatmap_figure(
            raw_tensor  = raw,
            true_label  = true_label,
            class_names = class_names,
            grad_cam    = grad_cam,
            attn_rollout= attn_rollout,
            device      = device,
            save_path   = path,
        )

    # ── comparison grid (clean only, all samples) ─────────────────────
    print("\n--- Generating comparison grid ---")
    save_comparison_grid(
        samples     = [(r, l) for r, l, _, _ in samples],
        class_names = class_names,
        grad_cam    = grad_cam,
        attn_rollout= attn_rollout,
        device      = device,
        save_path   = os.path.join(HEATMAP_DIR, "comparison_grid_clean.png"),
        max_rows    = NUM_SAMPLES,
    )

    # ── severity sweep figures for 3 interesting samples ─────────────
    print("\n--- Generating severity sweep figures ---")
    # Pick first sample from each: vit_only, both_correct (hard), both_correct
    sweep_samples = samples[:3]
    noise_types   = ["gaussian", "lowpass", "highpass"]

    for raw, true_label, _, status in sweep_samples:
        for nt in noise_types:
            tag  = f"{true_label}_{nt}_severity_sweep"
            path = os.path.join(HEATMAP_DIR, f"{tag}.png")
            print(f"  {true_label} × {nt}")
            save_severity_figure(
                raw_tensor   = raw,
                true_label   = true_label,
                class_names  = class_names,
                grad_cam     = grad_cam,
                attn_rollout = attn_rollout,
                device       = device,
                save_path    = path,
                noise_type   = nt,
            )

    # ── clean up hooks ────────────────────────────────────────────────
    grad_cam.remove()
    attn_rollout.remove()

    print(f"\nDone. All outputs saved to:\n  {HEATMAP_DIR}")
    print("\nKey figures for the paper:")
    print("  • *_all_conditions.png  – 4-condition grid per image")
    print("  • comparison_grid_clean.png  – all samples side-by-side")
    print("  • *_severity_sweep.png  – how heatmaps evolve with noise")


if __name__ == "__main__":
    run()
