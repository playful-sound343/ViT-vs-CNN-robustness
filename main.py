import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms
from datasets import load_dataset
from PIL import Image

# Load directly from HuggingFace (no manual download needed)
hf_dataset = load_dataset("torchgeo/eurosat", split="train")

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Wrapper to make it compatible with PyTorch
class EuroSATWrapper(Dataset):
    def __init__(self, hf_data, transform=None):
        self.data = hf_data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]['image'].convert('RGB')
        label = self.data[idx]['label']
        if self.transform:
            image = self.transform(image)
        return image, label

# Wrap it
dataset = EuroSATWrapper(hf_dataset, transform=transform)

# Split 70/15/15
total = len(dataset)
train_size = int(0.7 * total)
val_size   = int(0.15 * total)
test_size  = total - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

# Dataloaders
train_loader = DataLoader(train_set, batch_size=32, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

num_classes = 10  # EuroSAT has 10 land-use classes

# ResNet50 - pretrained on ImageNet, fine-tune for EuroSAT
resnet50 = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)
resnet50 = resnet50.to(device)

# ViT-B/16 - pretrained on ImageNet, fine-tune for EuroSAT
vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
vit = vit.to(device)

print("ResNet50 ready ✓")
print("ViT-B/16 ready ✓")
dummy = torch.randn(1, 3, 224, 224).to(device)

with torch.no_grad():
    r_out = resnet50(dummy)
    v_out = vit(dummy)

print("ResNet50 output shape:", r_out.shape)  # should be [1, 10]
print("ViT output shape:", v_out.shape)        # should be [1, 10]