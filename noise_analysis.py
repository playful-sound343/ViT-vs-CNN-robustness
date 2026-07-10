import csv
import os
import random
import warnings
from dataclasses import dataclass
from functools import partial

import matplotlib
import numpy as np
import timm
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="`huggingface_hub` cache-system uses symlinks")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Ensure your training script is named train.py
import train as train_cfg


ANALYSIS_DIR = os.path.join(train_cfg.SAVE_DIR, "analysis")
METRICS_CSV = os.path.join(ANALYSIS_DIR, "robustness_metrics.csv")
SUMMARY_MD = os.path.join(ANALYSIS_DIR, "analysis_summary.md")

MAX_TEST_SAMPLES = None      # set to 1000 for a quicker pilot run
ANALYSIS_BATCH_SIZE = train_cfg.EVAL_BATCH_SIZE
ANALYSIS_NUM_WORKERS = 2 if os.name == "nt" else 4

SPECTRAL_LOW_PASS_CUTOFFS = [0.08, 0.12, 0.18, 0.25, 0.35, 0.50]
SPECTRAL_HIGH_PASS_CUTOFFS = [0.03, 0.06, 0.10, 0.15, 0.22, 0.30]

GAUSSIAN_NOISE_SIGMAS = [0.02, 0.05, 0.08, 0.12, 0.16]
SPECKLE_NOISE_SIGMAS = [0.02, 0.05, 0.08, 0.12]
SALT_PEPPER_AMOUNTS = [0.01, 0.03, 0.05, 0.08]
POISSON_SCALES = [60, 30, 15, 8]

MASK_CACHE = {}


@dataclass
class RobustnessRecord:
    model_name: str
    category: str
    transform_name: str
    severity: str
    accuracy: float
    precision: float
    recall: float
    f1: float


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_tensor(image):
    mean = torch.tensor(train_cfg.IMAGENET_MEAN, dtype=image.dtype).view(3, 1, 1)
    std = torch.tensor(train_cfg.IMAGENET_STD, dtype=image.dtype).view(3, 1, 1)
    return (image - mean) / std


def base_tensor_transform():
    return transforms.Compose(
        [
            transforms.Resize(
                (train_cfg.IMAGE_SIZE, train_cfg.IMAGE_SIZE),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
        ]
    )


def get_radial_mask(height, width, cutoff_ratio, mode):
    key = (height, width, float(cutoff_ratio), mode)
    if key in MASK_CACHE:
        return MASK_CACHE[key]

    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, steps=height),
        torch.linspace(-1.0, 1.0, steps=width),
        indexing="ij",
    )
    radius = torch.sqrt(xx**2 + yy**2)

    if mode == "low":
        mask = (radius <= cutoff_ratio).float()
    elif mode == "high":
        mask = (radius >= cutoff_ratio).float()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    MASK_CACHE[key] = mask.unsqueeze(0)
    return MASK_CACHE[key]


def low_pass_filter(image, cutoff_ratio):
    _, h, w = image.shape
    mask = get_radial_mask(h, w, cutoff_ratio, "low")
    freq = torch.fft.fftshift(torch.fft.fft2(image, dim=(-2, -1)), dim=(-2, -1))
    filtered = freq * mask
    spatial = torch.fft.ifft2(torch.fft.ifftshift(filtered, dim=(-2, -1)), dim=(-2, -1))
    return spatial.real.clamp(0.0, 1.0)


def high_pass_filter(image, cutoff_ratio):
    _, h, w = image.shape
    mask = get_radial_mask(h, w, cutoff_ratio, "high")
    freq = torch.fft.fftshift(torch.fft.fft2(image, dim=(-2, -1)), dim=(-2, -1))
    filtered = freq * mask
    spatial = torch.fft.ifft2(torch.fft.ifftshift(filtered, dim=(-2, -1)), dim=(-2, -1))
    return spatial.real.clamp(0.0, 1.0)


def add_gaussian_noise(image, sigma):
    return (image + torch.randn_like(image) * sigma).clamp(0.0, 1.0)


def add_speckle_noise(image, sigma):
    return (image + image * torch.randn_like(image) * sigma).clamp(0.0, 1.0)


def add_salt_pepper_noise(image, amount):
    noisy = image.clone().view(3, -1)
    total = noisy.shape[1]
    num_pixels = max(1, int(total * amount))
    salt_idx = torch.randint(0, total, (num_pixels,))
    pepper_idx = torch.randint(0, total, (num_pixels,))
    noisy[:, salt_idx] = 1.0
    noisy[:, pepper_idx] = 0.0
    return noisy.view_as(image)


def add_poisson_noise(image, scale):
    noisy = torch.poisson(torch.clamp(image * scale, min=0.0)) / scale
    return noisy.clamp(0.0, 1.0)


class CorruptionDataset(Dataset):
    def __init__(self, dataset, indices, corruption_fn=None):
        self.dataset = dataset
        self.indices = list(indices)
        self.to_tensor = base_tensor_transform()
        self.corruption_fn = corruption_fn

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        image = self.to_tensor(image)
        if self.corruption_fn is not None:
            image = self.corruption_fn(image)
        image = normalize_tensor(image)
        return image, label


def create_loader(dataset, device):
    kwargs = {
        "dataset": dataset,
        "batch_size": ANALYSIS_BATCH_SIZE,
        "shuffle": False,
        "num_workers": ANALYSIS_NUM_WORKERS,
        "pin_memory": device.type == "cuda",
    }
    if ANALYSIS_NUM_WORKERS > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


def build_test_indices(dataset):
    _, _, test_idx = train_cfg.build_splits(dataset.targets)
    test_idx = list(test_idx)

    if MAX_TEST_SAMPLES is not None:
        rng = np.random.default_rng(train_cfg.SEED)
        rng.shuffle(test_idx)
        test_idx = test_idx[:MAX_TEST_SAMPLES]

    return test_idx


def build_models(device, num_classes):
    resnet = timm.create_model(
        train_cfg.RESNET_MODEL_NAME,
        pretrained=False,
        num_classes=num_classes,
    )
    resnet.load_state_dict(train_cfg.safe_load_state_dict(train_cfg.RESNET_CKPT, device))
    resnet = train_cfg.optimize_model_for_device(resnet, device, channels_last=True)
    resnet.eval()

    vit = timm.create_model(
        train_cfg.VIT_MODEL_NAME,
        pretrained=False,
        num_classes=num_classes,
        drop_rate=train_cfg.VIT_DROP_RATE,
        drop_path_rate=train_cfg.VIT_DROP_PATH_RATE,
    )
    vit.load_state_dict(train_cfg.safe_load_state_dict(train_cfg.VIT_CKPT, device))
    vit = train_cfg.optimize_model_for_device(vit, device, channels_last=False)
    vit.eval()

    return [
        (train_cfg.RESNET_DISPLAY_NAME, resnet, True),
        (train_cfg.VIT_DISPLAY_NAME, vit, False),
    ]


def evaluate_loader(model, loader, device, channels_last):
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in loader:
            images, labels = train_cfg.move_batch_to_device(
                images, labels, device, channels_last=channels_last
            )
            with train_cfg.get_autocast_context(device):
                outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = float(np.mean(all_preds == all_labels))
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return accuracy, precision, recall, f1


def mean_radial_spectrum(dataset, indices):
    to_tensor = base_tensor_transform()
    spectra = []

    for idx in indices:
        image, _ = dataset[idx]
        image = to_tensor(image).mean(dim=0)
        freq = torch.fft.fftshift(torch.fft.fft2(image))
        magnitude = torch.log1p(torch.abs(freq))

        h, w = magnitude.shape
        yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        cy = (h - 1) / 2.0
        cx = (w - 1) / 2.0
        radius = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).long()

        max_radius = min(h, w) // 2
        radius = radius.clamp(max=max_radius - 1)

        bins = torch.zeros(max_radius, dtype=torch.float64)
        counts = torch.zeros(max_radius, dtype=torch.float64)

        bins.scatter_add_(0, radius.flatten(), magnitude.flatten().double())
        counts.scatter_add_(0, radius.flatten(), torch.ones_like(magnitude.flatten(), dtype=torch.float64))

        spectra.append((bins / counts.clamp_min(1.0)).numpy())

    return np.stack(spectra, axis=0).mean(axis=0)


def plot_radial_spectrum(spectrum):
    x = np.linspace(0.0, 1.0, len(spectrum))
    plt.figure(figsize=(8, 5))
    plt.plot(x, spectrum, color="black", linewidth=2)
    plt.xlabel("Normalized radial frequency")
    plt.ylabel("Mean log magnitude")
    plt.title("Mean radial spectrum of the test set")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "mean_radial_spectrum.png"), dpi=160)
    plt.close()


def plot_family_curves(records, category, transform_name, filename):
    plt.figure(figsize=(8, 5))
    model_names = sorted({r.model_name for r in records if r.category == category and r.transform_name == transform_name})

    for model_name in model_names:
        subset = [
            r for r in records
            if r.category == category and r.transform_name == transform_name and r.model_name == model_name
        ]
        subset.sort(key=lambda item: float(item.severity))
        xs = [float(item.severity) for item in subset]
        ys = [item.accuracy * 100.0 for item in subset]
        plt.plot(xs, ys, marker="o", linewidth=2, label=model_name)

    plt.xlabel("Severity")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{category.title()} robustness: {transform_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, filename), dpi=160)
    plt.close()


def write_csv(records):
    with open(METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "category", "transform", "severity", "accuracy", "precision", "recall", "f1"])
        for r in records:
            writer.writerow([
                r.model_name,
                r.category,
                r.transform_name,
                r.severity,
                f"{r.accuracy:.6f}",
                f"{r.precision:.6f}",
                f"{r.recall:.6f}",
                f"{r.f1:.6f}",
            ])


def write_summary(records):
    clean = {
        r.model_name: r
        for r in records
        if r.category == "clean"
    }

    families = [
        ("spectral", "low_pass"),
        ("spectral", "high_pass"),
        ("noise", "gaussian_noise"),
        ("noise", "speckle_noise"),
        ("noise", "salt_pepper"),
        ("noise", "poisson"),
    ]

    lines = ["# Spectral and Noise Robustness Summary", ""]
    for model_name in sorted(clean):
        lines.append(f"## {model_name}")
        lines.append(f"- Clean accuracy: {clean[model_name].accuracy * 100:.2f}%")
        lines.append(f"- Clean F1: {clean[model_name].f1:.4f}")
        for category, transform_name in families:
            subset = [
                r.accuracy
                for r in records
                if r.model_name == model_name and r.category == category and r.transform_name == transform_name
            ]
            if subset:
                mean_acc = float(np.mean(subset))
                worst_acc = float(np.min(subset))
                drop = clean[model_name].accuracy - mean_acc
                lines.append(
                    f"- {transform_name}: mean acc {mean_acc * 100:.2f}%, worst acc {worst_acc * 100:.2f}%, mean drop {drop * 100:.2f} points"
                )
        lines.append("")

    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_analysis():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    seed_everything(train_cfg.SEED)
    train_cfg.enable_fast_runtime()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageFolder(root=train_cfg.DATASET_PATH)
    test_indices = build_test_indices(dataset)

    print(f"Using device: {device}")
    print(f"Dataset path: {train_cfg.DATASET_PATH}")
    print(f"Test samples used: {len(test_indices)}")

    spectrum = mean_radial_spectrum(dataset, test_indices)
    plot_radial_spectrum(spectrum)

    models = build_models(device, num_classes=len(dataset.classes))
    records = []

    # --- 0. CLEAN EVALUATION ---
    clean_dataset = CorruptionDataset(dataset, test_indices, corruption_fn=None)
    clean_loader = create_loader(clean_dataset, device)

    for model_name, model, channels_last in models:
        print(f"\nClean evaluation -> {model_name}")
        acc, prec, rec, f1 = evaluate_loader(model, clean_loader, device, channels_last)
        records.append(RobustnessRecord(model_name, "clean", "clean", "0", acc, prec, rec, f1))
        print(f"  Accuracy: {acc * 100:.2f}% | F1: {f1:.4f}")

    # --- 1. SPECTRAL JOBS ---
    spectral_jobs = []
    for cutoff in SPECTRAL_LOW_PASS_CUTOFFS:
        spectral_jobs.append(("low_pass", cutoff, partial(low_pass_filter, cutoff_ratio=cutoff)))
    for cutoff in SPECTRAL_HIGH_PASS_CUTOFFS:
        spectral_jobs.append(("high_pass", cutoff, partial(high_pass_filter, cutoff_ratio=cutoff)))

    # --- 2. EVALUATE SPECTRAL JOBS ---
    for transform_name, severity, corruption_fn in spectral_jobs:
        print(f"\nSpectral -> {transform_name} severity={severity}")
        corr_dataset = CorruptionDataset(dataset, test_indices, corruption_fn=corruption_fn)
        corr_loader = create_loader(corr_dataset, device)
        for model_name, model, channels_last in models:
            acc, prec, rec, f1 = evaluate_loader(model, corr_loader, device, channels_last)
            records.append(
                RobustnessRecord(
                    model_name, "spectral", transform_name, f"{severity:.4f}", acc, prec, rec, f1
                )
            )
            print(f"  {model_name}: acc={acc * 100:.2f}% | f1={f1:.4f}")

    # =====================================================================
    # NOISE EVALUATIONS COMMENTED OUT TO SAVE TIME
    # =====================================================================
    
    # noise_jobs = []
    # for sigma in GAUSSIAN_NOISE_SIGMAS:
    #     noise_jobs.append(("gaussian_noise", sigma, partial(add_gaussian_noise, sigma=sigma)))
    # for sigma in SPECKLE_NOISE_SIGMAS:
    #     noise_jobs.append(("speckle_noise", sigma, partial(add_speckle_noise, sigma=sigma)))
    # for amount in SALT_PEPPER_AMOUNTS:
    #     noise_jobs.append(("salt_pepper", amount, partial(add_salt_pepper_noise, amount=amount)))
    # for scale in POISSON_SCALES:
    #     noise_jobs.append(("poisson", scale, partial(add_poisson_noise, scale=scale)))

    # for transform_name, severity, corruption_fn in noise_jobs:
    #     print(f"\nNoise -> {transform_name} severity={severity}")
    #     corr_dataset = CorruptionDataset(dataset, test_indices, corruption_fn=corruption_fn)
    #     corr_loader = create_loader(corr_dataset, device)
    #     for model_name, model, channels_last in models:
    #         acc, prec, rec, f1 = evaluate_loader(model, corr_loader, device, channels_last)
    #         records.append(
    #             RobustnessRecord(
    #                 model_name, "noise", transform_name, f"{severity:.4f}", acc, prec, rec, f1
    #             )
    #         )
    #         print(f"  {model_name}: acc={acc * 100:.2f}% | f1={f1:.4f}")

    # =====================================================================

    write_csv(records)
    write_summary(records)

    plot_family_curves(records, "spectral", "low_pass", "spectral_low_pass_accuracy.png")
    plot_family_curves(records, "spectral", "high_pass", "spectral_high_pass_accuracy.png")
    
    # Plotting for noise is also commented out to prevent errors since the data won't exist
    # plot_family_curves(records, "noise", "gaussian_noise", "noise_gaussian_accuracy.png")
    # plot_family_curves(records, "noise", "speckle_noise", "noise_speckle_accuracy.png")
    # plot_family_curves(records, "noise", "salt_pepper", "noise_salt_pepper_accuracy.png")
    # plot_family_curves(records, "noise", "poisson", "noise_poisson_accuracy.png")

    print(f"\nSaved analysis outputs to: {ANALYSIS_DIR}")
    print(f"- CSV: {METRICS_CSV}")
    print(f"- Summary: {SUMMARY_MD}")


if __name__ == "__main__":
    run_analysis()
