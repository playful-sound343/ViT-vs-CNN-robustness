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
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="`huggingface_hub` cache-system uses symlinks")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import train as train_cfg


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
ANALYSIS_DIR = os.path.join(train_cfg.SAVE_DIR, "analysis")
METRICS_CSV = os.path.join(ANALYSIS_DIR, "robustness_metrics.csv")
PER_CLASS_CSV = os.path.join(ANALYSIS_DIR, "per_class_noise_robustness.csv")
SUMMARY_MD = os.path.join(ANALYSIS_DIR, "analysis_summary.md")

FREQ_SENSITIVITY_CSV = os.path.join(ANALYSIS_DIR, "frequency_sensitivity_profile.csv")
SPECTRAL_ENERGY_CSV = os.path.join(ANALYSIS_DIR, "spectral_energy_contribution.csv")

MAX_TEST_SAMPLES = None
ANALYSIS_BATCH_SIZE = train_cfg.EVAL_BATCH_SIZE
ANALYSIS_NUM_WORKERS = 0 if os.name == "nt" else 4

SPECTRAL_LOW_PASS_CUTOFFS = [0.08, 0.12, 0.18, 0.25, 0.35, 0.50]
SPECTRAL_HIGH_PASS_CUTOFFS = [0.03, 0.06, 0.10, 0.15, 0.22, 0.30]

GAUSSIAN_NOISE_SIGMAS = [0.02, 0.05, 0.08, 0.12, 0.16]
SPECKLE_NOISE_SIGMAS = [0.02, 0.05, 0.08, 0.12]
SALT_PEPPER_AMOUNTS = [0.01, 0.03, 0.05, 0.08]
POISSON_SCALES = [60, 30, 15, 8]

FREQ_BANDS = [
    (0.00, 0.08),
    (0.08, 0.16),
    (0.16, 0.24),
    (0.24, 0.32),
    (0.32, 0.40),
    (0.40, 0.50),
    (0.50, 0.65),
    (0.65, 0.85),
]

MASK_CACHE = {}
RADIUS_CACHE = {}


# ---------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------
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


@dataclass
class PerClassRecord:
    model_name: str
    category: str
    transform_name: str
    severity: str
    class_name: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class FrequencySensitivityRecord:
    model_name: str
    band_label: str
    r_low: float
    r_high: float
    retained_accuracy: float
    retained_f1: float
    accuracy_drop: float
    f1_drop: float


@dataclass
class SpectralEnergyContributionRecord:
    model_name: str
    band_label: str
    r_low: float
    r_high: float
    energy_fraction: float
    accuracy_drop: float
    f1_drop: float
    energy_weighted_accuracy_drop: float
    energy_weighted_f1_drop: float


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_tag(text):
    return str(text).strip().lower().replace("-", "_").replace(" ", "_")


def safe_filename(text):
    text = str(text).strip().replace("/", "_").replace("\\", "_")
    text = text.replace(":", "_").replace("*", "_").replace("?", "_")
    text = text.replace('"', "_").replace("<", "_").replace(">", "_").replace("|", "_")
    text = text.replace(" ", "_")
    return text


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


def enable_fast_runtime():
    if hasattr(train_cfg, "enable_fast_runtime"):
        train_cfg.enable_fast_runtime()
    else:
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True


def safe_load_state_dict(path, device):
    if hasattr(train_cfg, "safe_load_state_dict"):
        return train_cfg.safe_load_state_dict(path, device)
    return torch.load(path, map_location=device)


def optimize_model_for_device(model, device, channels_last=False):
    if hasattr(train_cfg, "optimize_model_for_device"):
        try:
            return train_cfg.optimize_model_for_device(model, device, channels_last=channels_last)
        except TypeError:
            return train_cfg.optimize_model_for_device(model, device)
    model = model.to(device)
    if device.type == "cuda" and channels_last:
        model = model.to(memory_format=torch.channels_last)
    return model


def get_autocast_context(device):
    if hasattr(train_cfg, "get_autocast_context"):
        return train_cfg.get_autocast_context(device)
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", enabled=True)
    from contextlib import nullcontext
    return nullcontext()


def move_batch_to_device(images, labels, device, channels_last=False):
    if hasattr(train_cfg, "move_batch_to_device"):
        try:
            return train_cfg.move_batch_to_device(images, labels, device, channels_last=channels_last)
        except TypeError:
            return train_cfg.move_batch_to_device(images, labels, device)

    non_blocking = device.type == "cuda"
    if device.type == "cuda" and channels_last and images.ndim == 4:
        images = images.to(device, non_blocking=non_blocking, memory_format=torch.channels_last)
    else:
        images = images.to(device, non_blocking=non_blocking)
    labels = labels.to(device, non_blocking=non_blocking)
    return images, labels


def build_test_indices(dataset):
    if not hasattr(train_cfg, "build_splits"):
        raise RuntimeError("train.py must expose build_splits(dataset.targets)")
    _, _, test_idx = train_cfg.build_splits(dataset.targets)
    test_idx = list(test_idx)

    if MAX_TEST_SAMPLES is not None:
        rng = np.random.default_rng(train_cfg.SEED)
        rng.shuffle(test_idx)
        test_idx = test_idx[:MAX_TEST_SAMPLES]

    return test_idx


# ---------------------------------------------------------------------
# Spectral / frequency helpers
# ---------------------------------------------------------------------
def get_normalized_radius_map(height, width, device, dtype):
    key = (height, width, str(device), str(dtype))
    if key in RADIUS_CACHE:
        return RADIUS_CACHE[key]

    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius = radius / radius.max().clamp_min(1e-8)

    RADIUS_CACHE[key] = radius
    return radius


def low_pass_filter(image, cutoff_ratio):
    image = image.clone()
    _, h, w = image.shape
    radius = get_normalized_radius_map(h, w, image.device, image.dtype)
    mask = (radius <= cutoff_ratio).to(image.dtype).unsqueeze(0)

    freq = torch.fft.fftshift(torch.fft.fft2(image, dim=(-2, -1)), dim=(-2, -1))
    filtered = freq * mask
    spatial = torch.fft.ifft2(torch.fft.ifftshift(filtered, dim=(-2, -1)), dim=(-2, -1))
    return spatial.real.clamp(0.0, 1.0)


def high_pass_filter(image, cutoff_ratio):
    image = image.clone()
    _, h, w = image.shape
    radius = get_normalized_radius_map(h, w, image.device, image.dtype)
    mask = (radius >= cutoff_ratio).to(image.dtype).unsqueeze(0)

    freq = torch.fft.fftshift(torch.fft.fft2(image, dim=(-2, -1)), dim=(-2, -1))
    filtered = freq * mask
    spatial = torch.fft.ifft2(torch.fft.ifftshift(filtered, dim=(-2, -1)), dim=(-2, -1))
    return spatial.real.clamp(0.0, 1.0)


def band_pass_filter(image, r_low, r_high):
    image = image.clone()
    _, h, w = image.shape
    radius = get_normalized_radius_map(h, w, image.device, image.dtype)
    mask = ((radius >= r_low) & (radius < r_high)).to(image.dtype).unsqueeze(0)

    freq = torch.fft.fftshift(torch.fft.fft2(image, dim=(-2, -1)), dim=(-2, -1))
    filtered = freq * mask
    spatial = torch.fft.ifft2(torch.fft.ifftshift(filtered, dim=(-2, -1)), dim=(-2, -1))
    return spatial.real.clamp(0.0, 1.0)


def band_stop_filter(image, r_low, r_high):
    image = image.clone()
    _, h, w = image.shape
    radius = get_normalized_radius_map(h, w, image.device, image.dtype)
    mask = (~((radius >= r_low) & (radius < r_high))).to(image.dtype).unsqueeze(0)

    freq = torch.fft.fftshift(torch.fft.fft2(image, dim=(-2, -1)), dim=(-2, -1))
    filtered = freq * mask
    spatial = torch.fft.ifft2(torch.fft.ifftshift(filtered, dim=(-2, -1)), dim=(-2, -1))
    return spatial.real.clamp(0.0, 1.0)


# ---------------------------------------------------------------------
# Noise helpers
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Dataset / loader
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
def build_models(device, num_classes):
    resnet = timm.create_model(
        train_cfg.RESNET_MODEL_NAME,
        pretrained=False,
        num_classes=num_classes,
    )
    resnet.load_state_dict(safe_load_state_dict(train_cfg.RESNET_CKPT, device))
    resnet = optimize_model_for_device(resnet, device, channels_last=True)
    resnet.eval()

    vit = timm.create_model(
        train_cfg.VIT_MODEL_NAME,
        pretrained=False,
        num_classes=num_classes,
        drop_rate=train_cfg.VIT_DROP_RATE,
        drop_path_rate=train_cfg.VIT_DROP_PATH_RATE,
    )
    vit.load_state_dict(safe_load_state_dict(train_cfg.VIT_CKPT, device))
    vit = optimize_model_for_device(vit, device, channels_last=False)
    vit.eval()

    return [
        (train_cfg.RESNET_DISPLAY_NAME, resnet, True),
        (train_cfg.VIT_DISPLAY_NAME, vit, False),
    ]


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def evaluate_loader(model, loader, device, channels_last, class_names, category, transform_name, severity, model_name):
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for images, labels in loader:
            images, labels = move_batch_to_device(images, labels, device, channels_last=channels_last)
            with get_autocast_context(device):
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

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    per_class_records = []
    for class_name in class_names:
        class_stats = report[class_name]
        per_class_records.append(
            PerClassRecord(
                model_name=model_name,
                category=normalize_tag(category),
                transform_name=normalize_tag(transform_name),
                severity=str(severity),
                class_name=class_name,
                precision=float(class_stats["precision"]),
                recall=float(class_stats["recall"]),
                f1=float(class_stats["f1-score"]),
                support=int(class_stats["support"]),
            )
        )

    return accuracy, precision, recall, f1, per_class_records


# ---------------------------------------------------------------------
# Spectrum analysis
# ---------------------------------------------------------------------
def mean_radial_spectrum(dataset, indices, num_bins=80):
    to_tensor = base_tensor_transform()
    total_bins = np.zeros(num_bins, dtype=np.float64)
    total_counts = np.zeros(num_bins, dtype=np.float64)

    for idx in indices:
        image, _ = dataset[idx]
        image = to_tensor(image).mean(dim=0)
        freq = torch.fft.fftshift(torch.fft.fft2(image))
        magnitude = torch.log1p(torch.abs(freq))

        radius = get_normalized_radius_map(
            magnitude.shape[0],
            magnitude.shape[1],
            magnitude.device,
            magnitude.dtype,
        )
        bin_idx = torch.clamp((radius * (num_bins - 1)).long(), 0, num_bins - 1)

        flat_idx = bin_idx.flatten().cpu().numpy()
        flat_mag = magnitude.flatten().cpu().numpy()

        total_bins += np.bincount(flat_idx, weights=flat_mag, minlength=num_bins)
        total_counts += np.bincount(flat_idx, minlength=num_bins)

    return total_bins / np.clip(total_counts, 1.0, None)


def compute_dataset_band_energy(dataset, indices, freq_bands):
    to_tensor = base_tensor_transform()
    band_energy = np.zeros(len(freq_bands), dtype=np.float64)

    for idx in indices:
        image, _ = dataset[idx]
        image = to_tensor(image).mean(dim=0)
        freq = torch.fft.fftshift(torch.fft.fft2(image))
        power = torch.abs(freq) ** 2

        radius = get_normalized_radius_map(
            power.shape[0],
            power.shape[1],
            power.device,
            power.dtype,
        )

        total_power = float(power.sum().item())
        total_power = max(total_power, 1e-12)

        for band_idx, (r_low, r_high) in enumerate(freq_bands):
            mask = ((radius >= r_low) & (radius < r_high)).to(power.dtype)
            band_power = float((power * mask).sum().item())
            band_energy[band_idx] += band_power / total_power

    band_energy /= max(1, len(indices))
    return band_energy


# ---------------------------------------------------------------------
# Frequency sensitivity / energy contribution
# ---------------------------------------------------------------------
def run_frequency_sensitivity_profile(models, dataset, test_indices, device, class_names, clean_records):
    clean_lookup = {
        r.model_name: {"acc": r.accuracy, "f1": r.f1}
        for r in clean_records
    }

    freq_records = []

    for r_low, r_high in FREQ_BANDS:
        band_label = f"{r_low:.2f}-{r_high:.2f}"
        print(f"\nFrequency sensitivity -> band-stop [{band_label}]")

        corr_dataset = CorruptionDataset(
            dataset,
            test_indices,
            corruption_fn=partial(band_stop_filter, r_low=r_low, r_high=r_high),
        )
        corr_loader = create_loader(corr_dataset, device)

        for model_name, model, channels_last in models:
            acc, _, _, f1, _ = evaluate_loader(
                model,
                corr_loader,
                device,
                channels_last,
                class_names,
                "frequency_sensitivity",
                "band_stop",
                band_label,
                model_name,
            )

            clean_acc = clean_lookup[model_name]["acc"]
            clean_f1 = clean_lookup[model_name]["f1"]

            freq_records.append(
                FrequencySensitivityRecord(
                    model_name=model_name,
                    band_label=band_label,
                    r_low=r_low,
                    r_high=r_high,
                    retained_accuracy=acc,
                    retained_f1=f1,
                    accuracy_drop=clean_acc - acc,
                    f1_drop=clean_f1 - f1,
                )
            )

            print(
                f"  {model_name}: retained acc={acc * 100:.2f}% | "
                f"acc drop={(clean_acc - acc) * 100:.2f} points"
            )

    return freq_records


def build_spectral_energy_contribution(freq_records, mean_band_energy):
    band_to_energy = {
        f"{r_low:.2f}-{r_high:.2f}": float(mean_band_energy[idx])
        for idx, (r_low, r_high) in enumerate(FREQ_BANDS)
    }

    energy_records = []
    for rec in freq_records:
        energy_fraction = band_to_energy[rec.band_label]
        energy_records.append(
            SpectralEnergyContributionRecord(
                model_name=rec.model_name,
                band_label=rec.band_label,
                r_low=rec.r_low,
                r_high=rec.r_high,
                energy_fraction=energy_fraction,
                accuracy_drop=rec.accuracy_drop,
                f1_drop=rec.f1_drop,
                energy_weighted_accuracy_drop=energy_fraction * rec.accuracy_drop,
                energy_weighted_f1_drop=energy_fraction * rec.f1_drop,
            )
        )

    return energy_records


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
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
    category = normalize_tag(category)
    transform_name = normalize_tag(transform_name)

    subset_all = [
        r for r in records
        if normalize_tag(r.category) == category and normalize_tag(r.transform_name) == transform_name
    ]

    if not subset_all:
        print(f"No data found for {category}/{transform_name}, skipping {filename}")
        return

    plt.figure(figsize=(8, 5))
    model_names = sorted({r.model_name for r in subset_all})

    for model_name in model_names:
        subset = [r for r in subset_all if r.model_name == model_name]
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


def plot_per_class_curves(per_class_records, transform_name, metric_name, filename):
    subset_all = [r for r in per_class_records if normalize_tag(r.transform_name) == normalize_tag(transform_name)]
    if not subset_all:
        print(f"No per-class data found for {transform_name}, skipping {filename}")
        return

    plt.figure(figsize=(11, 7))
    classes = sorted({r.class_name for r in subset_all})

    for class_name in classes:
        subset = [r for r in subset_all if r.class_name == class_name]
        subset.sort(key=lambda item: float(item.severity))
        xs = [float(item.severity) for item in subset]
        ys = [getattr(item, metric_name) * 100.0 for item in subset]
        plt.plot(xs, ys, marker="o", linewidth=1.8, label=class_name)

    plt.xlabel("Severity")
    plt.ylabel(f"{metric_name.title()} (%)")
    plt.title(f"Per-class {metric_name} robustness: {transform_name}")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, filename), dpi=160)
    plt.close()


def plot_frequency_sensitivity_profile(freq_records):
    if not freq_records:
        return

    plt.figure(figsize=(10, 6))
    model_names = sorted({r.model_name for r in freq_records})

    for model_name in model_names:
        subset = [r for r in freq_records if r.model_name == model_name]
        subset.sort(key=lambda x: x.r_low)
        xs = [0.5 * (r.r_low + r.r_high) for r in subset]
        ys = [r.accuracy_drop * 100.0 for r in subset]
        plt.plot(xs, ys, marker="o", linewidth=2, label=model_name)

    plt.xlabel("Normalized radial frequency")
    plt.ylabel("Accuracy drop (points)")
    plt.title("Frequency sensitivity profile (band-stop ablation)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "frequency_sensitivity_profile.png"), dpi=160)
    plt.close()


def plot_dataset_spectral_energy(mean_band_energy):
    labels = [f"{r_low:.2f}-{r_high:.2f}" for r_low, r_high in FREQ_BANDS]
    x = np.arange(len(labels))

    plt.figure(figsize=(11, 5))
    plt.bar(x, mean_band_energy * 100.0, color="steelblue")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Mean energy fraction (%)")
    plt.xlabel("Normalized radial frequency band")
    plt.title("Dataset spectral energy distribution")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "dataset_spectral_energy_distribution.png"), dpi=160)
    plt.close()


def plot_energy_weighted_contribution(energy_records):
    if not energy_records:
        return

    labels = [f"{r_low:.2f}-{r_high:.2f}" for r_low, r_high in FREQ_BANDS]
    x = np.arange(len(labels))
    width = 0.38

    plt.figure(figsize=(11, 5))
    model_names = sorted({r.model_name for r in energy_records})

    for i, model_name in enumerate(model_names):
        subset = [r for r in energy_records if r.model_name == model_name]
        subset.sort(key=lambda item: item.r_low)
        ys = [r.energy_weighted_accuracy_drop * 100.0 for r in subset]
        offset = (i - (len(model_names) - 1) / 2.0) * width
        plt.bar(x + offset, ys, width=width, label=model_name)

    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Energy-weighted accuracy drop")
    plt.xlabel("Normalized radial frequency band")
    plt.title("Spectral energy contribution")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ANALYSIS_DIR, "spectral_energy_contribution.png"), dpi=160)
    plt.close()


# ---------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------
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


def write_per_class_csv(per_class_records):
    with open(PER_CLASS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "category", "transform", "severity", "class_name", "precision", "recall", "f1", "support"])
        for r in per_class_records:
            writer.writerow([
                r.model_name,
                r.category,
                r.transform_name,
                r.severity,
                r.class_name,
                f"{r.precision:.6f}",
                f"{r.recall:.6f}",
                f"{r.f1:.6f}",
                r.support,
            ])


def write_frequency_sensitivity_csv(freq_records):
    with open(FREQ_SENSITIVITY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "band_label",
            "r_low",
            "r_high",
            "retained_accuracy",
            "retained_f1",
            "accuracy_drop",
            "f1_drop",
        ])
        for r in freq_records:
            writer.writerow([
                r.model_name,
                r.band_label,
                f"{r.r_low:.4f}",
                f"{r.r_high:.4f}",
                f"{r.retained_accuracy:.6f}",
                f"{r.retained_f1:.6f}",
                f"{r.accuracy_drop:.6f}",
                f"{r.f1_drop:.6f}",
            ])


def write_spectral_energy_csv(energy_records):
    with open(SPECTRAL_ENERGY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "band_label",
            "r_low",
            "r_high",
            "energy_fraction",
            "accuracy_drop",
            "f1_drop",
            "energy_weighted_accuracy_drop",
            "energy_weighted_f1_drop",
        ])
        for r in energy_records:
            writer.writerow([
                r.model_name,
                r.band_label,
                f"{r.r_low:.4f}",
                f"{r.r_high:.4f}",
                f"{r.energy_fraction:.6f}",
                f"{r.accuracy_drop:.6f}",
                f"{r.f1_drop:.6f}",
                f"{r.energy_weighted_accuracy_drop:.6f}",
                f"{r.energy_weighted_f1_drop:.6f}",
            ])


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
def append_frequency_summary(lines, freq_records, energy_records):
    lines.append("## Frequency Sensitivity Profile")
    lines.append("")

    for model_name in sorted({r.model_name for r in freq_records}):
        freq_subset = [r for r in freq_records if r.model_name == model_name]
        energy_subset = [r for r in energy_records if r.model_name == model_name]

        if not freq_subset or not energy_subset:
            continue

        most_sensitive = max(freq_subset, key=lambda x: x.accuracy_drop)
        most_contributing = max(energy_subset, key=lambda x: x.energy_weighted_accuracy_drop)

        lines.append(f"### {model_name}")
        lines.append(
            f"- Most sensitive band: {most_sensitive.band_label} "
            f"(accuracy drop {most_sensitive.accuracy_drop * 100:.2f} points)"
        )
        lines.append(
            f"- Highest energy-weighted contribution: {most_contributing.band_label} "
            f"(score {most_contributing.energy_weighted_accuracy_drop * 100:.2f})"
        )
        lines.append("")


def write_summary(records, freq_records=None, energy_records=None):
    clean = {r.model_name: r for r in records if normalize_tag(r.category) == "clean"}
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
                if r.model_name == model_name
                and normalize_tag(r.category) == normalize_tag(category)
                and normalize_tag(r.transform_name) == normalize_tag(transform_name)
            ]
            if subset:
                mean_acc = float(np.mean(subset))
                worst_acc = float(np.min(subset))
                drop = clean[model_name].accuracy - mean_acc
                lines.append(
                    f"- {transform_name}: mean acc {mean_acc * 100:.2f}%, "
                    f"worst acc {worst_acc * 100:.2f}%, mean drop {drop * 100:.2f} points"
                )
        lines.append("")

    if freq_records is not None and energy_records is not None:
        append_frequency_summary(lines, freq_records, energy_records)

    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------
# Job builders
# ---------------------------------------------------------------------
def build_spectral_jobs():
    jobs = []
    for cutoff in SPECTRAL_LOW_PASS_CUTOFFS:
        jobs.append(("low_pass", cutoff, partial(low_pass_filter, cutoff_ratio=cutoff)))
    for cutoff in SPECTRAL_HIGH_PASS_CUTOFFS:
        jobs.append(("high_pass", cutoff, partial(high_pass_filter, cutoff_ratio=cutoff)))
    return jobs


def build_noise_jobs():
    jobs = []
    for sigma in GAUSSIAN_NOISE_SIGMAS:
        jobs.append(("gaussian_noise", sigma, partial(add_gaussian_noise, sigma=sigma)))
    for sigma in SPECKLE_NOISE_SIGMAS:
        jobs.append(("speckle_noise", sigma, partial(add_speckle_noise, sigma=sigma)))
    for amount in SALT_PEPPER_AMOUNTS:
        jobs.append(("salt_pepper", amount, partial(add_salt_pepper_noise, amount=amount)))
    for scale in POISSON_SCALES:
        jobs.append(("poisson", scale, partial(add_poisson_noise, scale=scale)))
    return jobs


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def run_analysis():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    seed_everything(train_cfg.SEED)
    enable_fast_runtime()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ImageFolder(root=train_cfg.DATASET_PATH)
    class_names = dataset.classes
    test_indices = build_test_indices(dataset)

    print(f"Using device: {device}")
    print(f"Dataset path: {train_cfg.DATASET_PATH}")
    print(f"Test samples used: {len(test_indices)}")

    spectrum = mean_radial_spectrum(dataset, test_indices)
    plot_radial_spectrum(spectrum)

    models = build_models(device, num_classes=len(class_names))
    records = []
    per_class_records = []

    # Clean
    clean_dataset = CorruptionDataset(dataset, test_indices, corruption_fn=None)
    clean_loader = create_loader(clean_dataset, device)

    for model_name, model, channels_last in models:
        print(f"\nClean evaluation -> {model_name}")
        acc, prec, rec, f1, class_records = evaluate_loader(
            model,
            clean_loader,
            device,
            channels_last,
            class_names,
            "clean",
            "clean",
            "0",
            model_name,
        )
        records.append(
            RobustnessRecord(
                model_name=model_name,
                category="clean",
                transform_name="clean",
                severity="0",
                accuracy=acc,
                precision=prec,
                recall=rec,
                f1=f1,
            )
        )
        per_class_records.extend(class_records)
        print(f"  Accuracy: {acc * 100:.2f}% | F1: {f1:.4f}")

    # Spectral robustness
    spectral_jobs = build_spectral_jobs()
    for transform_name, severity, corruption_fn in spectral_jobs:
        print(f"\nSpectral -> {transform_name} severity={severity}")
        corr_dataset = CorruptionDataset(dataset, test_indices, corruption_fn=corruption_fn)
        corr_loader = create_loader(corr_dataset, device)
        for model_name, model, channels_last in models:
            acc, prec, rec, f1, class_records = evaluate_loader(
                model,
                corr_loader,
                device,
                channels_last,
                class_names,
                "spectral",
                transform_name,
                f"{severity:.4f}",
                model_name,
            )
            records.append(
                RobustnessRecord(
                    model_name=model_name,
                    category="spectral",
                    transform_name=transform_name,
                    severity=f"{severity:.4f}",
                    accuracy=acc,
                    precision=prec,
                    recall=rec,
                    f1=f1,
                )
            )
            per_class_records.extend(class_records)
            print(f"  {model_name}: acc={acc * 100:.2f}% | f1={f1:.4f}")

    # Noise robustness
    noise_jobs = build_noise_jobs()
    for transform_name, severity, corruption_fn in noise_jobs:
        print(f"\nNoise -> {transform_name} severity={severity}")
        corr_dataset = CorruptionDataset(dataset, test_indices, corruption_fn=corruption_fn)
        corr_loader = create_loader(corr_dataset, device)
        for model_name, model, channels_last in models:
            acc, prec, rec, f1, class_records = evaluate_loader(
                model,
                corr_loader,
                device,
                channels_last,
                class_names,
                "noise",
                transform_name,
                f"{severity:.4f}",
                model_name,
            )
            records.append(
                RobustnessRecord(
                    model_name=model_name,
                    category="noise",
                    transform_name=transform_name,
                    severity=f"{severity:.4f}",
                    accuracy=acc,
                    precision=prec,
                    recall=rec,
                    f1=f1,
                )
            )
            per_class_records.extend(class_records)
            print(f"  {model_name}: acc={acc * 100:.2f}% | f1={f1:.4f}")

    # Frequency sensitivity profile
    clean_records = [r for r in records if normalize_tag(r.category) == "clean"]

    freq_records = run_frequency_sensitivity_profile(
        models=models,
        dataset=dataset,
        test_indices=test_indices,
        device=device,
        class_names=class_names,
        clean_records=clean_records,
    )

    # Spectral energy contribution
    mean_band_energy = compute_dataset_band_energy(
        dataset=dataset,
        indices=test_indices,
        freq_bands=FREQ_BANDS,
    )

    energy_records = build_spectral_energy_contribution(
        freq_records=freq_records,
        mean_band_energy=mean_band_energy,
    )

    print("\nRecorded categories/transforms:")
    print(sorted(set((r.category, r.transform_name) for r in records)))

    write_csv(records)
    write_per_class_csv(per_class_records)
    write_frequency_sensitivity_csv(freq_records)
    write_spectral_energy_csv(energy_records)
    write_summary(records, freq_records=freq_records, energy_records=energy_records)

    plot_family_curves(records, "spectral", "low_pass", "spectral_low_pass_accuracy.png")
    plot_family_curves(records, "spectral", "high_pass", "spectral_high_pass_accuracy.png")
    plot_family_curves(records, "noise", "gaussian_noise", "noise_gaussian_accuracy.png")
    plot_family_curves(records, "noise", "speckle_noise", "noise_speckle_accuracy.png")
    plot_family_curves(records, "noise", "salt_pepper", "noise_salt_pepper_accuracy.png")
    plot_family_curves(records, "noise", "poisson", "noise_poisson_accuracy.png")

    for model_name in [train_cfg.RESNET_DISPLAY_NAME, train_cfg.VIT_DISPLAY_NAME]:
        for transform_name in ["gaussian_noise", "speckle_noise", "salt_pepper", "poisson"]:
            subset = [
                r
                for r in per_class_records
                if r.model_name == model_name
                and normalize_tag(r.category) == "noise"
                and normalize_tag(r.transform_name) == normalize_tag(transform_name)
            ]
            if subset:
                safe_model = safe_filename(model_name)
                plot_per_class_curves(
                    subset,
                    transform_name,
                    "recall",
                    f"{safe_model}_{transform_name}_per_class_recall.png",
                )
                plot_per_class_curves(
                    subset,
                    transform_name,
                    "f1",
                    f"{safe_model}_{transform_name}_per_class_f1.png",
                )

    plot_frequency_sensitivity_profile(freq_records)
    plot_dataset_spectral_energy(mean_band_energy)
    plot_energy_weighted_contribution(energy_records)

    print(f"\nSaved analysis outputs to: {ANALYSIS_DIR}")
    print(f"- Overall CSV: {METRICS_CSV}")
    print(f"- Per-class CSV: {PER_CLASS_CSV}")
    print(f"- Frequency sensitivity CSV: {FREQ_SENSITIVITY_CSV}")
    print(f"- Spectral energy CSV: {SPECTRAL_ENERGY_CSV}")
    print(f"- Summary: {SUMMARY_MD}")


if __name__ == "__main__":
    run_analysis()
