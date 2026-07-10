import gc
import os
import random
import time
from contextlib import nullcontext

import matplotlib
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import InterpolationMode

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Hardware-tuned config for Ryzen 7 7840HS + RTX 4050 6GB + 16GB RAM
DATASET_PATH = r"D:\data\EuroSAT\2750"
SAVE_DIR = "./checkpoints_best"
MODE = "balanced"

SEED = 42
IMAGE_SIZE = 224
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

ENABLE_COMPILE = False
REUSE_RESNET_CHECKPOINT = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MODE_CONFIGS = {
    "fast": {
        "resnet_model_name": "resnet18",
        "vit_model_name": "vit_tiny_patch16_224",
        "resnet_display_name": "ResNet18",
        "vit_display_name": "ViT-T/16",
        "epochs_resnet": 10,
        "epochs_vit": 12,
        "patience": 3,
        "batch_size_resnet": 128,
        "batch_size_vit": 96,
        "eval_batch_size": 128,
        "num_workers": 2,
        "prefetch_factor": 2,
        "lr_resnet": 1e-4,
        "lr_vit": 4e-5,
        "weight_decay_resnet": 1e-4,
        "weight_decay_vit": 0.03,
        "mixup_alpha": 0.1,
        "vit_drop_rate": 0.05,
        "vit_drop_path_rate": 0.05,
        "transform_profile": "fast",
        "val_every": 2,
        "print_every": 40,
    },
    "balanced": {
        "resnet_model_name": "resnet34",
        "vit_model_name": "vit_small_patch16_224",
        "resnet_display_name": "ResNet34",
        "vit_display_name": "ViT-S/16",
        "epochs_resnet": 12,
        "epochs_vit": 16,
        "patience": 4,
        "batch_size_resnet": 128,
        "batch_size_vit": 64,
        "eval_batch_size": 128,
        "num_workers": 2,
        "prefetch_factor": 2,
        "lr_resnet": 1e-4,
        "lr_vit": 3e-5,
        "weight_decay_resnet": 1e-4,
        "weight_decay_vit": 0.05,
        "mixup_alpha": 0.2,
        "vit_drop_rate": 0.10,
        "vit_drop_path_rate": 0.10,
        "transform_profile": "balanced",
        "val_every": 2,
        "print_every": 40,
    },
    "final": {
        "resnet_model_name": "resnet50",
        "vit_model_name": "vit_small_patch16_224",
        "resnet_display_name": "ResNet50",
        "vit_display_name": "ViT-S/16",
        "epochs_resnet": 15,
        "epochs_vit": 18,
        "patience": 4,
        "batch_size_resnet": 96,
        "batch_size_vit": 64,
        "eval_batch_size": 128,
        "num_workers": 2,
        "prefetch_factor": 2,
        "lr_resnet": 1e-4,
        "lr_vit": 3e-5,
        "weight_decay_resnet": 1e-4,
        "weight_decay_vit": 0.05,
        "mixup_alpha": 0.2,
        "vit_drop_rate": 0.10,
        "vit_drop_path_rate": 0.10,
        "transform_profile": "final",
        "val_every": 1,
        "print_every": 40,
    },
}

if MODE not in MODE_CONFIGS:
    raise ValueError(f"Unsupported MODE '{MODE}'. Choose from: {sorted(MODE_CONFIGS)}")

ACTIVE = MODE_CONFIGS[MODE]

RESNET_MODEL_NAME = ACTIVE["resnet_model_name"]
VIT_MODEL_NAME = ACTIVE["vit_model_name"]
RESNET_DISPLAY_NAME = ACTIVE["resnet_display_name"]
VIT_DISPLAY_NAME = ACTIVE["vit_display_name"]
EPOCHS_RESNET = ACTIVE["epochs_resnet"]
EPOCHS_VIT = ACTIVE["epochs_vit"]
PATIENCE = ACTIVE["patience"]
BATCH_SIZE_RESNET = ACTIVE["batch_size_resnet"]
BATCH_SIZE_VIT = ACTIVE["batch_size_vit"]
EVAL_BATCH_SIZE = ACTIVE["eval_batch_size"]
NUM_WORKERS = ACTIVE["num_workers"]
PREFETCH_FACTOR = ACTIVE["prefetch_factor"]
LR_RESNET = ACTIVE["lr_resnet"]
LR_VIT = ACTIVE["lr_vit"]
WEIGHT_DECAY_RESNET = ACTIVE["weight_decay_resnet"]
WEIGHT_DECAY_VIT = ACTIVE["weight_decay_vit"]
MIXUP_ALPHA = ACTIVE["mixup_alpha"]
VIT_DROP_RATE = ACTIVE["vit_drop_rate"]
VIT_DROP_PATH_RATE = ACTIVE["vit_drop_path_rate"]
TRANSFORM_PROFILE = ACTIVE["transform_profile"]
VAL_EVERY = ACTIVE["val_every"]
PRINT_EVERY = ACTIVE["print_every"]

RESNET_CKPT = os.path.join(SAVE_DIR, f"best_{MODE}_resnet.pth")
VIT_CKPT = os.path.join(SAVE_DIR, f"best_{MODE}_vit.pth")

os.makedirs(SAVE_DIR, exist_ok=True)


class TransformSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform
        self.classes = getattr(dataset, "classes", None)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, label = self.dataset[self.indices[idx]]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def validate_config():
    if not os.path.isdir(DATASET_PATH):
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_PATH}")

    total_split = TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT
    if not np.isclose(total_split, 1.0):
        raise ValueError(
            f"TRAIN_SPLIT + VAL_SPLIT + TEST_SPLIT must equal 1.0, got {total_split:.3f}."
        )


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def enable_fast_runtime():
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = True

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True


def get_autocast_context(device):
    if device.type != "cuda":
        return nullcontext()

    if hasattr(torch, "autocast"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    return torch.cuda.amp.autocast(dtype=torch.float16)


def make_grad_scaler(device):
    enabled = device.type == "cuda"

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)

    return torch.cuda.amp.GradScaler(enabled=enabled)


def make_adamw(params, lr, weight_decay, device):
    kwargs = {"lr": lr, "weight_decay": weight_decay}
    if device.type == "cuda":
        try:
            return optim.AdamW(params, fused=True, **kwargs)
        except TypeError:
            pass
    return optim.AdamW(params, **kwargs)


def build_train_transform():
    resize = transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )

    if TRANSFORM_PROFILE == "fast":
        return transforms.Compose(
            [
                resize,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    if TRANSFORM_PROFILE == "balanced":
        return transforms.Compose(
            [
                resize,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply(
                    [transforms.RandomRotation(degrees=10)],
                    p=0.20,
                ),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.10,
                            contrast=0.10,
                            saturation=0.08,
                            hue=0.02,
                        )
                    ],
                    p=0.20,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                transforms.RandomErasing(
                    p=0.05,
                    scale=(0.02, 0.08),
                    ratio=(0.3, 3.3),
                    value="random",
                ),
            ]
        )

    return transforms.Compose(
        [
            resize,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.RandomRotation(degrees=15)],
                p=0.30,
            ),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.15,
                        contrast=0.15,
                        saturation=0.10,
                        hue=0.03,
                    )
                ],
                p=0.25,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            transforms.RandomErasing(
                p=0.10,
                scale=(0.02, 0.10),
                ratio=(0.3, 3.3),
                value="random",
            ),
        ]
    )


def build_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_splits(targets):
    indices = np.arange(len(targets))
    targets = np.array(targets)

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(1.0 - TRAIN_SPLIT),
        stratify=targets,
        random_state=SEED,
    )

    temp_targets = targets[temp_idx]
    test_fraction_of_temp = TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_fraction_of_temp,
        stratify=temp_targets,
        random_state=SEED,
    )

    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def create_dataloader(dataset, batch_size, shuffle, num_workers, device):
    generator = torch.Generator()
    generator.manual_seed(SEED)

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker if num_workers > 0 else None,
        "generator": generator,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    return DataLoader(**loader_kwargs)


def move_batch_to_device(images, labels, device, channels_last=False):
    non_blocking = device.type == "cuda"
    if channels_last and images.ndim == 4:
        images = images.to(
            device,
            non_blocking=non_blocking,
            memory_format=torch.channels_last,
        )
    else:
        images = images.to(device, non_blocking=non_blocking)

    labels = labels.to(device, non_blocking=non_blocking)
    return images, labels


def mixup_data(images, labels, alpha):
    if alpha <= 0:
        return images, labels, labels, 1.0

    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[index]
    return mixed_images, labels, labels[index], lam


def mixup_loss(criterion, predictions, target_a, target_b, lam):
    return lam * criterion(predictions, target_a) + (1.0 - lam) * criterion(
        predictions, target_b
    )


def optimize_model_for_device(model, device, channels_last=False):
    model = model.to(device)
    if channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    if ENABLE_COMPILE and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            print("`torch.compile` enabled.")
        except Exception as exc:
            print(f"`torch.compile` skipped: {exc}")

    return model


def is_worker_failure(exc):
    text = str(exc)
    markers = [
        "DataLoader worker",
        "worker exited unexpectedly",
        "BrokenPipeError",
        "EOFError",
        "Ran out of input",
        "pickle",
    ]
    return any(marker in text for marker in markers) or isinstance(
        exc, (EOFError, BrokenPipeError)
    )


def safe_load_state_dict(path, device):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def safe_filename(name):
    unsafe_chars = '<>:"/\\|?*'
    cleaned = ''.join('_' if ch in unsafe_chars else ch for ch in name)
    return cleaned.replace(' ', '_')
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    criterion,
    channels_last=False,
    use_mixup=False,
):
    model.train()
    running_loss = 0.0
    running_correct = 0.0
    running_total = 0
    epoch_start = time.time()

    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images, labels = move_batch_to_device(
            images, labels, device, channels_last=channels_last
        )

        if use_mixup:
            images, target_a, target_b, lam = mixup_data(images, labels, MIXUP_ALPHA)

        optimizer.zero_grad(set_to_none=True)

        with get_autocast_context(device):
            predictions = model(images)
            if use_mixup:
                loss = mixup_loss(criterion, predictions, target_a, target_b, lam)
            else:
                loss = criterion(predictions, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        predicted = predictions.argmax(dim=1)
        if use_mixup:
            running_correct += lam * predicted.eq(target_a).sum().item()
            running_correct += (1.0 - lam) * predicted.eq(target_b).sum().item()
        else:
            running_correct += predicted.eq(labels).sum().item()
        running_total += labels.size(0)

        if batch_idx % PRINT_EVERY == 0 or batch_idx == len(loader):
            elapsed = time.time() - epoch_start
            print(
                f"  Batch {batch_idx:03d}/{len(loader)} | "
                f"Loss {running_loss / max(1, running_total):.4f} | "
                f"Acc {100.0 * running_correct / max(1, running_total):.2f}% | "
                f"Elapsed {elapsed:.1f}s",
                flush=True,
            )

    avg_loss = running_loss / max(1, running_total)
    avg_acc = 100.0 * running_correct / max(1, running_total)
    return avg_loss, avg_acc


def validate_one_epoch(model, loader, device, criterion, channels_last=False):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    running_total = 0

    with torch.inference_mode():
        for images, labels in loader:
            images, labels = move_batch_to_device(
                images, labels, device, channels_last=channels_last
            )
            with get_autocast_context(device):
                predictions = model(images)
                loss = criterion(predictions, labels)

            running_loss += loss.item() * images.size(0)
            running_correct += predictions.argmax(dim=1).eq(labels).sum().item()
            running_total += labels.size(0)

    avg_loss = running_loss / max(1, running_total)
    avg_acc = 100.0 * running_correct / max(1, running_total)
    return avg_loss, avg_acc


def train_model(
    model,
    model_name,
    train_loader,
    val_loader,
    device,
    epochs,
    lr,
    weight_decay,
    checkpoint_path,
    train_criterion,
    val_criterion,
    channels_last=False,
    use_mixup=False,
    val_every=1,
):
    optimizer = make_adamw(model.parameters(), lr, weight_decay, device)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)
    scaler = make_grad_scaler(device)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = -1.0
    epochs_without_improvement = 0

    print("\n" + "=" * 60)
    print(f"Training {model_name} (max {epochs} epochs, patience={PATIENCE})")
    print("=" * 60)

    for epoch in range(epochs):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            train_criterion,
            channels_last=channels_last,
            use_mixup=use_mixup,
        )
        should_validate = (
            epoch == 0 or (epoch + 1) % val_every == 0 or (epoch + 1) == epochs
        )
        if should_validate:
            val_loss, val_acc = validate_one_epoch(
                model,
                val_loader,
                device,
                val_criterion,
                channels_last=channels_last,
            )
        else:
            val_loss, val_acc = np.nan, np.nan
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if should_validate:
            print(
                f"Epoch [{epoch + 1:02d}/{epochs}] "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                f"Time: {time.time() - epoch_start:.1f}s"
            )
        else:
            print(
                f"Epoch [{epoch + 1:02d}/{epochs}] "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                f"Val skipped this epoch | "
                f"Time: {time.time() - epoch_start:.1f}s"
            )

        if should_validate:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  Saved best checkpoint ({best_val_acc:.2f}%)")
            else:
                epochs_without_improvement += 1
                print(f"  No improvement ({epochs_without_improvement}/{PATIENCE})")
                if epochs_without_improvement >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

    print(f"\nBest {model_name} validation accuracy: {best_val_acc:.2f}%")
    return history


def evaluate_model(model, model_name, loader, device, class_names, channels_last=False):
    num_classes = len(class_names)
    model_name_file = safe_filename(model_name)
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.inference_mode():
        for images, labels in loader:
            images, labels = move_batch_to_device(
                images, labels, device, channels_last=channels_last
            )
            with get_autocast_context(device):
                outputs = model(images)
            probs = torch.softmax(outputs.float(), dim=1)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    print("\n" + "=" * 60)
    print(f"Full Evaluation - {model_name}")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    precision = precision_score(
        all_labels, all_preds, average="weighted", zero_division=0
    )
    accuracy = float(np.mean(all_preds == all_labels))

    print(f"Accuracy : {accuracy * 100:.4f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=True, xticks_rotation=45)
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"confusion_matrix_{model_name_file}.png"), dpi=150)
    plt.close()

    labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
    fpr = {}
    tpr = {}
    roc_auc = {}
    valid_auc_indices = []

    for i in range(num_classes):
        positives = labels_bin[:, i].sum()
        negatives = len(labels_bin) - positives
        if positives == 0 or negatives == 0:
            roc_auc[i] = float("nan")
            continue
        fpr[i], tpr[i], _ = roc_curve(labels_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        valid_auc_indices.append(i)

    if valid_auc_indices:
        all_fpr = np.unique(np.concatenate([fpr[i] for i in valid_auc_indices]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in valid_auc_indices:
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= len(valid_auc_indices)
        macro_auc = auc(all_fpr, mean_tpr)

        fig, ax = plt.subplots(figsize=(12, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
        for i, color in enumerate(colors):
            if i not in valid_auc_indices:
                continue
            ax.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=1.5,
                label=f"{class_names[i]} (AUC = {roc_auc[i]:.4f})",
            )

        ax.plot(
            all_fpr,
            mean_tpr,
            color="black",
            lw=2.5,
            linestyle="--",
            label=f"Macro Average (AUC = {macro_auc:.4f})",
        )
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves - {model_name}")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, f"roc_curve_{model_name_file}.png"), dpi=150)
        plt.close()
    else:
        macro_auc = float("nan")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": macro_auc,
    }


def plot_history(history_resnet, history_vit):
    epochs_resnet = len(history_resnet["train_acc"])
    epochs_vit = len(history_vit["train_acc"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0][0].plot(range(1, epochs_resnet + 1), history_resnet["train_acc"], label="Train")
    axes[0][0].plot(range(1, epochs_resnet + 1), history_resnet["val_acc"], label="Val")
    axes[0][0].set_title(f"{RESNET_DISPLAY_NAME} Accuracy")
    axes[0][0].set_xlabel("Epoch")
    axes[0][0].set_ylabel("Accuracy (%)")
    axes[0][0].legend()
    axes[0][0].grid(True)

    axes[0][1].plot(
        range(1, epochs_resnet + 1), history_resnet["train_loss"], label="Train"
    )
    axes[0][1].plot(range(1, epochs_resnet + 1), history_resnet["val_loss"], label="Val")
    axes[0][1].set_title(f"{RESNET_DISPLAY_NAME} Loss")
    axes[0][1].set_xlabel("Epoch")
    axes[0][1].set_ylabel("Loss")
    axes[0][1].legend()
    axes[0][1].grid(True)

    axes[1][0].plot(range(1, epochs_vit + 1), history_vit["train_acc"], label="Train")
    axes[1][0].plot(range(1, epochs_vit + 1), history_vit["val_acc"], label="Val")
    axes[1][0].set_title(f"{VIT_DISPLAY_NAME} Accuracy")
    axes[1][0].set_xlabel("Epoch")
    axes[1][0].set_ylabel("Accuracy (%)")
    axes[1][0].legend()
    axes[1][0].grid(True)

    axes[1][1].plot(range(1, epochs_vit + 1), history_vit["train_loss"], label="Train")
    axes[1][1].plot(range(1, epochs_vit + 1), history_vit["val_loss"], label="Val")
    axes[1][1].set_title(f"{VIT_DISPLAY_NAME} Loss")
    axes[1][1].set_xlabel("Epoch")
    axes[1][1].set_ylabel("Loss")
    axes[1][1].legend()
    axes[1][1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_curves_all.png"), dpi=150)
    plt.close()


def build_loaders(base_dataset, train_idx, val_idx, test_idx, device, num_workers):
    train_transform = build_train_transform()
    eval_transform = build_eval_transform()

    train_set = TransformSubset(base_dataset, train_idx, transform=train_transform)
    val_set = TransformSubset(base_dataset, val_idx, transform=eval_transform)
    test_set = TransformSubset(base_dataset, test_idx, transform=eval_transform)

    loaders = {
        "resnet_train": create_dataloader(
            train_set,
            batch_size=BATCH_SIZE_RESNET,
            shuffle=True,
            num_workers=num_workers,
            device=device,
        ),
        "vit_train": create_dataloader(
            train_set,
            batch_size=BATCH_SIZE_VIT,
            shuffle=True,
            num_workers=num_workers,
            device=device,
        ),
        "eval": create_dataloader(
            val_set,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            device=device,
        ),
        "test": create_dataloader(
            test_set,
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            device=device,
        ),
    }

    return train_set, val_set, test_set, loaders


def train_suite(base_dataset, class_names, num_classes, device, num_workers):
    train_idx, val_idx, test_idx = build_splits(base_dataset.targets)
    train_set, val_set, test_set, loaders = build_loaders(
        base_dataset, train_idx, val_idx, test_idx, device, num_workers
    )

    print(
        f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)} | Workers: {num_workers}"
    )

    print(f"\nInitializing {RESNET_DISPLAY_NAME}...")
    resnet50 = optimize_model_for_device(
        timm.create_model(RESNET_MODEL_NAME, pretrained=True, num_classes=num_classes),
        device,
        channels_last=True,
    )
    if REUSE_RESNET_CHECKPOINT and os.path.exists(RESNET_CKPT):
        print(f"Using existing {RESNET_DISPLAY_NAME} checkpoint: {RESNET_CKPT}")
        history_resnet = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    else:
        history_resnet = train_model(
            resnet50,
            model_name=RESNET_DISPLAY_NAME,
            train_loader=loaders["resnet_train"],
            val_loader=loaders["eval"],
            device=device,
            epochs=EPOCHS_RESNET,
            lr=LR_RESNET,
            weight_decay=WEIGHT_DECAY_RESNET,
            checkpoint_path=RESNET_CKPT,
            train_criterion=nn.CrossEntropyLoss(),
            val_criterion=nn.CrossEntropyLoss(),
            channels_last=True,
            use_mixup=False,
            val_every=VAL_EVERY,
        )

    resnet50.load_state_dict(safe_load_state_dict(RESNET_CKPT, device))
    metrics_resnet = evaluate_model(
        resnet50,
        RESNET_DISPLAY_NAME,
        loaders["test"],
        device,
        class_names,
        channels_last=True,
    )

    del resnet50
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\nInitializing {VIT_DISPLAY_NAME}...")
    vit = optimize_model_for_device(
        timm.create_model(
            VIT_MODEL_NAME,
            pretrained=True,
            num_classes=num_classes,
            drop_rate=VIT_DROP_RATE,
            drop_path_rate=VIT_DROP_PATH_RATE,
        ),
        device,
        channels_last=False,
    )
    history_vit = train_model(
        vit,
        model_name=VIT_DISPLAY_NAME,
        train_loader=loaders["vit_train"],
        val_loader=loaders["eval"],
        device=device,
        epochs=EPOCHS_VIT,
        lr=LR_VIT,
        weight_decay=WEIGHT_DECAY_VIT,
        checkpoint_path=VIT_CKPT,
        train_criterion=nn.CrossEntropyLoss(label_smoothing=0.1),
        val_criterion=nn.CrossEntropyLoss(),
        channels_last=False,
        use_mixup=True,
        val_every=VAL_EVERY,
    )

    vit.load_state_dict(safe_load_state_dict(VIT_CKPT, device))
    metrics_vit = evaluate_model(
        vit, VIT_DISPLAY_NAME, loaders["test"], device, class_names
    )

    del vit
    torch.cuda.empty_cache()
    gc.collect()

    plot_history(history_resnet, history_vit)
    return metrics_resnet, metrics_vit


def print_final_comparison(metrics_resnet, metrics_vit):
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<15} {RESNET_DISPLAY_NAME:>12} {VIT_DISPLAY_NAME:>12}")
    print(f"  {'-' * 42}")
    print(
        f"  {'Accuracy':<15} {metrics_resnet['accuracy'] * 100:>11.4f}% {metrics_vit['accuracy'] * 100:>11.4f}%"
    )
    print(
        f"  {'Precision':<15} {metrics_resnet['precision']:>12.4f} {metrics_vit['precision']:>12.4f}"
    )
    print(
        f"  {'Recall':<15} {metrics_resnet['recall']:>12.4f} {metrics_vit['recall']:>12.4f}"
    )
    print(
        f"  {'F1 Score':<15} {metrics_resnet['f1']:>12.4f} {metrics_vit['f1']:>12.4f}"
    )
    print(
        f"  {'Macro AUC':<15} {metrics_resnet['auc']:>12.4f} {metrics_vit['auc']:>12.4f}"
    )
    print("=" * 60)
    print(f"\nSaved outputs to {SAVE_DIR}")


def main():
    torch.multiprocessing.freeze_support()
    validate_config()
    seed_everything(SEED)
    enable_fast_runtime()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested_workers = NUM_WORKERS

    print(f"Using device: {device}")
    print(f"Mode: {MODE}")
    print(f"Loading dataset from: {DATASET_PATH}")

    base_dataset = ImageFolder(root=DATASET_PATH)
    class_names = base_dataset.classes
    num_classes = len(class_names)

    if num_classes == 0:
        raise RuntimeError(f"No classes were found in: {DATASET_PATH}")

    print(f"Found {len(base_dataset)} images across {num_classes} classes.")
    print(
        "Optimizations: TF32, AMP, pinned memory, persistent workers, batch progress logs, separate batch sizes for ResNet and ViT, and validation every "
        f"{VAL_EVERY} epoch(s) in this mode."
    )

    try:
        metrics_resnet, metrics_vit = train_suite(
            base_dataset, class_names, num_classes, device, requested_workers
        )
    except Exception as exc:
        if requested_workers > 0 and is_worker_failure(exc):
            print(
                f"\nWorker-based loading failed ({exc}). Retrying automatically with num_workers=0."
            )
            metrics_resnet, metrics_vit = train_suite(
                base_dataset, class_names, num_classes, device, 0
            )
        else:
            raise

    print_final_comparison(metrics_resnet, metrics_vit)


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        print("\nCUDA out of memory. Lower `BATCH_SIZE_VIT` to 48 or 32 and try again.")
        raise
    except Exception as exc:
        print(f"\nFatal error: {exc}")
        raise





