# Remote Sensing Model Robustness Research

This repository contains experiments for evaluating deep learning robustness on remote sensing imagery. The work compares ResNet and Vision Transformer models across EuroSAT RGB, multispectral Sentinel-2-style band analysis, and UC Merced transfer-learning experiments.

## Highlights

- Built PyTorch/timm training and evaluation pipelines for land-use classification.
- Evaluated ResNet and Vision Transformer models under clean, noisy, spectral-filtered, and band-dropout conditions.
- Analyzed spectral band sensitivity and model robustness across RGB and multispectral remote sensing data.
- Generated statistical comparisons, confidence intervals, effect sizes, and final robustness summaries.

## Repository Structure

```text
.
├── data/
│   ├── EuroSAT.zip
│   └── README.md
├── checkpoints/
│   └── README.md
├── checkpoints_best/
│   └── README.md
├── train.py
├── multispectral_analysis.py
├── spectral_band_analysis.py
├── noise_analysis.py
├── noise2_analysis.py
├── random_seed_train.py
├── ucemerced_analysis.py
├── visualize.py
├── data_analysis.py
├── final_stats.csv
├── Figure_1.png
├── requirements.txt
└── .gitignore
```

## Main Scripts

- `train.py` - trains ResNet and ViT models on EuroSAT RGB imagery.
- `multispectral_analysis.py` - analyzes multispectral model behavior and band sensitivity.
- `spectral_band_analysis.py` - evaluates the effect of spectral band removal and grouped band perturbations.
- `noise_analysis.py` and `noise2_analysis.py` - run noise robustness experiments.
- `random_seed_train.py` - repeats training/evaluation across seeds for more reliable metrics.
- `ucemerced_analysis.py` - evaluates transfer learning on UC Merced.
- `visualize.py` and `data_analysis.py` - generate plots and summary analysis.

## Dataset Setup

`data/EuroSAT.zip` is included as the compact dataset archive. Before running local training, unzip it:

```bash
unzip data/EuroSAT.zip -d data/
```

Some scripts currently reference a local Windows path such as:

```text
D:\data\EuroSAT\2750
```

If your dataset is inside this repository, update the script path to:

```text
data/EuroSAT/2750
```

## Installation

Create a clean Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version from the official PyTorch instructions.

## Running Experiments

Example:

```bash
python train.py
python noise_analysis.py
python spectral_band_analysis.py
python ucemerced_analysis.py
```

## Notes on Large Files

Model checkpoint folders are not included in the GitHub-ready copy because they are very large and should be stored with GitHub Releases, Google Drive, Hugging Face, Kaggle, or another artifact store.

The placeholder folders `checkpoints/` and `checkpoints_best/` are included so the expected output locations are clear.

## Results

- `final_stats.csv` contains the consolidated robustness/statistical summary.
- `Figure_1.png` contains one generated figure from the analysis workflow.

## License

Add your preferred license before publishing publicly. If this is academic coursework or research-in-progress, confirm dataset and model license requirements before making the repository public.
