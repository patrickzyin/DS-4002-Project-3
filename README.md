# DS 4002 — Project 3: Image Classification of Landfill Waste


## Section 0: Project Overview

This project applies deep learning image classification to automatically categorize landfill waste into nine material types using the RealWaste dataset (4,752 images). We compare three pretrained convolutional neural network architectures — MobileNetV3-Small, EfficientNet-B0, and ResNet-50 — fine-tuned via transfer learning, and evaluate each on accuracy, macro-averaged F1 score, and inference speed.

**Hypothesis:** ResNet-50 will perform the best out of all models in terms of both speed and accuracy.

**Research Question:** Can an image classification model accurately predict the correct disposal category for waste based on visual features?

**Goal:** Achieve at least 50% accuracy on a nine-class waste classification task to support the efficiency of automated recycling systems.

---

## Section 1: Software and Platform

**Language:** Python 3.9+

Install all required packages before running the notebook:

```
pip install torch torchvision matplotlib seaborn scikit-learn tqdm Pillow
```

| Package | Purpose |
|---|---|
| torch / torchvision | Model training, transfer learning, data loading |
| matplotlib | Training curve and comparison plots |
| seaborn | Confusion matrix heatmaps |
| scikit-learn | F1 score, classification report, train/test split |
| tqdm | Training progress bars |
| Pillow | Image loading |

The notebook automatically uses a GPU if one is available (`cuda`), and falls back to CPU otherwise.

---

## Section 2: Map of Documentation

```
DS-4002-Project-3/
│
├── README.md                         — Project overview, reproduction instructions, and results summary
│
├── DATA/
│   └── realwaste-main/
│       └── RealWaste/                — Root folder of the RealWaste dataset
│           ├── Cardboard/
│           ├── Food Organics/
│           ├── Glass/
│           ├── Metal/
│           ├── Miscellaneous Trash/
│           ├── Paper/
│           ├── Plastic/
│           ├── Textile Trash/
│           └── Vegetation/
│
├── SCRIPTS/
│   └── realwaste_classification.ipynb — Single notebook containing all steps:
│                                         data preparation, EDA, class weighting,
│                                         model training, evaluation, and comparison
│
└── OUTPUTS/
    ├── class_distribution.png        — Bar chart of image counts per waste category
    ├── sample_images.png             — 3×3 grid of one sample image per class
    ├── training_curves.png           — Train/val loss and accuracy curves for all three models
    ├── confusion_matrices.png        — Side-by-side normalized confusion matrices (test set)
    ├── model_comparison.png          — Bar chart comparing accuracy, macro F1, and inference speed
    ├── classification_reports.txt    — Per-class precision, recall, and F1 for each model
    └── summary_results.txt           — Final summary table and hypothesis check
```

**Dataset details:**

| Field | Value |
|---|---|
| Source | RealWaste (Single, Iranmanesh & Raad, 2023 — UCI ML Repository) |
| Total images | 4,752 |
| Classes | 9 (Cardboard, Food Organics, Glass, Metal, Miscellaneous Trash, Paper, Plastic, Textile Trash, Vegetation) |
| Largest class | Plastic (921 images) |
| Smallest class | Textile Trash (318 images) |

---

## Section 3: Reproducing the Results

Follow these steps in order. All work runs from within the single notebook `SCRIPTS/realwaste_classification.ipynb`.

### Step 1 — Set up your environment

Clone or download the repository and navigate into the project folder:

```bash
cd DS-4002-Project-3
pip install torch torchvision matplotlib seaborn scikit-learn tqdm Pillow
```

A GPU is strongly recommended for reasonable training times.

### Step 2 — Download and place the dataset

Download the RealWaste dataset from the UCI Machine Learning Repository:  
[https://archive.ics.uci.edu/dataset/908/realwaste](https://archive.ics.uci.edu/dataset/908/realwaste)

Extract the archive so that the nine class folders sit at:

```
DATA/realwaste-main/RealWaste/Cardboard/
DATA/realwaste-main/RealWaste/Food Organics/
... (and so on for all nine classes)
```

If your folder structure differs, update `DATA_DIR` before running.

### Step 3 — Run the notebook

Open the notebook and run all cells from top to bottom:

```
SCRIPTS/realwaste_classification.ipynb
```

The notebook is organized into the following sequential sections:

| Section | Description |
|---|---|
| 0. Dependencies | Install and import all packages; set random seeds; detect device |
| 1. Configuration | Set paths, image size, batch size, training hyperparameters, class names |
| 2. Data Preparation | Resize and normalize images; stratified 70/15/15 train/val/test split; apply augmentation to training set only |
| 3. EDA | Class distribution bar chart; sample image grid |
| 4. Class Imbalance | Compute inverse-frequency class weights for weighted cross-entropy loss |
| 5. Model Definitions | Define MobileNetV3-Small, EfficientNet-B0, and ResNet-50 with replaced classification heads |
| 6. Training Utilities | `freeze_backbone`, `unfreeze_top_layers`, `run_epoch`, `train_model` helper functions |
| 7. Train All Models | Two-phase training loop for all three architectures; saves best checkpoint per model |
| 8. Training Curves | Loss and accuracy plots with fine-tuning phase marker |
| 9. Evaluation | Per-class classification reports, normalized confusion matrices, accuracy/F1/speed comparison chart |
| 10. Summary | Final results table printed to console and saved to `OUTPUTS/summary_results.txt` |

