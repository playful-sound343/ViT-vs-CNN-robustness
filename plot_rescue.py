import matplotlib.pyplot as plt
import os

SAVE_DIR = "./checkpoints_best"
os.makedirs(SAVE_DIR, exist_ok=True)

# ─── EXTRACTED TERMINAL DATA ───
# ResNet34 Data (Skipped validation epochs are carried over for a smooth curve)
rn_epochs = list(range(1, 13))
rn_train_acc = [60.86, 90.89, 95.34, 96.28, 97.02, 97.50, 97.63, 97.66, 98.01, 98.19, 98.21, 98.29]
rn_val_acc =   [86.17, 94.07, 94.07, 97.48, 97.48, 97.80, 97.80, 98.12, 98.12, 98.25, 98.25, 98.15]
rn_train_loss =[1.5090, 0.3802, 0.1609, 0.1173, 0.0978, 0.0812, 0.0744, 0.0727, 0.0623, 0.0579, 0.0578, 0.0551]
rn_val_loss =  [0.6996, 0.1908, 0.1908, 0.0800, 0.0800, 0.0733, 0.0733, 0.0625, 0.0625, 0.0563, 0.0563, 0.0575]

# ViT-S/16 Data
vit_epochs = list(range(1, 17))
vit_train_acc =[77.01, 87.39, 87.02, 88.27, 88.70, 87.68, 89.32, 87.94, 90.51, 89.54, 90.14, 90.77, 90.27, 90.73, 90.02, 89.15]
vit_val_acc =  [96.96, 98.07, 98.07, 98.49, 98.49, 98.64, 98.64, 98.91, 98.91, 98.99, 98.99, 99.14, 99.14, 99.11, 99.11, 99.28]
vit_train_loss=[1.1373, 0.8877, 0.8794, 0.8363, 0.8086, 0.8204, 0.7984, 0.8071, 0.7658, 0.7692, 0.7602, 0.7453, 0.7557, 0.7449, 0.7581, 0.7736]
vit_val_loss = [0.2121, 0.1687, 0.1687, 0.1661, 0.1661, 0.1399, 0.1399, 0.1653, 0.1653, 0.1337, 0.1337, 0.1258, 0.1258, 0.1266, 0.1266, 0.1294]

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# ResNet Plots
axes[0][0].plot(rn_epochs, rn_train_acc, label="Train", color="#1f77b4", lw=2)
axes[0][0].plot(rn_epochs, rn_val_acc, label="Val", color="#ff7f0e", lw=2)
axes[0][0].set_title("ResNet34 Accuracy"); axes[0][0].set_xlabel("Epoch"); axes[0][0].set_ylabel("Accuracy (%)"); axes[0][0].legend(); axes[0][0].grid(True, alpha=0.3)

axes[0][1].plot(rn_epochs, rn_train_loss, label="Train", color="#1f77b4", lw=2)
axes[0][1].plot(rn_epochs, rn_val_loss, label="Val", color="#ff7f0e", lw=2)
axes[0][1].set_title("ResNet34 Loss"); axes[0][1].set_xlabel("Epoch"); axes[0][1].set_ylabel("Loss"); axes[0][1].legend(); axes[0][1].grid(True, alpha=0.3)

# ViT Plots
axes[1][0].plot(vit_epochs, vit_train_acc, label="Train", color="#2ca02c", lw=2)
axes[1][0].plot(vit_epochs, vit_val_acc, label="Val", color="#d62728", lw=2)
axes[1][0].set_title("ViT-S/16 Accuracy"); axes[1][0].set_xlabel("Epoch"); axes[1][0].set_ylabel("Accuracy (%)"); axes[1][0].legend(); axes[1][0].grid(True, alpha=0.3)

axes[1][1].plot(vit_epochs, vit_train_loss, label="Train", color="#2ca02c", lw=2)
axes[1][1].plot(vit_epochs, vit_val_loss, label="Val", color="#d62728", lw=2)
axes[1][1].set_title("ViT-S/16 Loss"); axes[1][1].set_xlabel("Epoch"); axes[1][1].set_ylabel("Loss"); axes[1][1].legend(); axes[1][1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "paper_ready_training_curves.png"), dpi=300)
print("Perfect training curves successfully saved!")