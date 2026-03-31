# 🚗 Real-Time Drivable Space Segmentation

## 📌 Project Overview

Autonomous vehicles must understand **where they can safely drive**, not just follow 
predefined lanes. In complex urban environments, lane markings may be absent, faded, 
or misleading.

This project focuses on **real-time semantic segmentation of drivable space**, 
identifying **free space** where a vehicle can physically move. The system classifies 
each pixel into:

- ✅ **Drivable Area**
- ❌ **Non-Drivable Area** (curbs, sidewalks, barriers, vegetation, etc.)

The model is designed with **real-time constraints**, making it suitable for deployment 
in **Level 4 autonomous driving systems**.

> ⚠️ **No pre-trained models are used. The entire model is trained from scratch.**

---

## 🎯 Problem Motivation

Traditional perception systems rely heavily on lane detection, which fails in:
- Unstructured roads
- Construction zones
- Poor weather conditions
- Occluded or missing lane markings

This project solves that by directly predicting **drivable regions**, enabling safer 
and more robust navigation.

---

## 🧠 Model Architecture — DrivableSegNet

We use a custom **Encoder–Decoder architecture** optimized for real-time performance, 
built entirely from scratch using PyTorch.

### 🔹 Encoder — MobileNetV3-Small (from scratch)
- **Stem:** Stride-2 conv → [16, H/2, W/2]
- **Stage 1:** Stride-2 → [16, H/4, W/4]
- **Stage 2:** Stride-2 → [24, H/8, W/8]
- **Stage 3:** Stride-2 → [48, H/16, W/16]
- **Stage 4:** Stride-2 → [96, H/32, W/32]
- Uses **Inverted Residual blocks** with depthwise separable convolutions
- **Squeeze-and-Excitation (SE)** blocks for channel attention
- **HardSwish / HardSigmoid** activations for efficiency

### 🔹 Decoder — Lightweight U-Net
- 4 bilinear upsample + skip-connection stages
- Skip channels fused: [16, 24, 48, 96]
- Final 1×1 conv → single-channel logit map

### 🔹 Model Stats
| Component | Parameters |
|---|---|
| Encoder (MobileNetV3-Small) | 862,072 |
| Decoder (U-Net) | 134,705 |
| **Total** | **996,777 (~1M)** |

- Input: `(1, 3, 288, 512)` → Output: `(1, 1, 288, 512)`
- Weight initialization: **Kaiming He (fan-out)**

---

## 📊 Dataset

- **Dataset:** nuScenes v1.0-mini (public)
- **Scenes:** 10 | **Key-frames:** 404
- **Cameras used:** All 6 — `CAM_FRONT`, `CAM_FRONT_LEFT`, `CAM_FRONT_RIGHT`, 
  `CAM_BACK`, `CAM_BACK_LEFT`, `CAM_BACK_RIGHT`
- **Total samples after ×6 cameras:** ~2,424 image–mask pairs

### 🔹 Mask Generation (LiDAR-based, No Manual Annotation)
Ground-truth binary masks are generated automatically from **LiDAR point clouds**:
1. Load LiDAR point cloud from LIDAR_TOP
2. Filter ground-level points: `-2.0m < z < 0.3m` in ego frame
3. Project ground points onto camera image using calibrated extrinsics + intrinsics
4. Rasterize projected points into a binary mask
5. Morphological closing (`kernel=25`) to fill sparse LiDAR gaps
6. Obstacle mask subtracted to clean boundaries

### 🔹 Class Distribution
| Class | Percentage |
|---|---|
| Non-Drivable | ~79.2% |
| Drivable | ~20.8% |

### 🔹 Dataset Split
| Split | Key-frames | Samples (×6 cameras) |
|---|---|---|
| Train | 70% (~283) | ~1,696 |
| Val | 15% (~61) | ~363 |
| Test | 15% (~60) | ~364 |

---

## ⚙️ Methodology

### 🔹 Loss Function — Combined BCE + Dice
```python
Loss = 0.5 × BCE + 0.5 × Dice
```
- **BCE** handles pixel-level calibration
- **Dice** directly optimizes IoU-like overlap
- Equal weighting (α=0.5) chosen to balance class imbalance

### 🔹 Data Augmentation
- Horizontal flipping (p=0.5)
- Color jitter (brightness, contrast, saturation, hue)
- Gaussian blur (p=0.3)
- Random rotation ±10°
- Random resized crop (scale 0.8–1.0)

### 🔹 Preprocessing
- Resize: 1600×900 → **512×288** (16:9 preserved)
- Normalization: ImageNet mean/std ([0.485, 0.456, 0.406] / [0.229, 0.224, 0.225])

---

## 🏋️ Training Details

| Parameter | Value |
|---|---|
| Epochs | 50 |
| Batch Size | 8 |
| Optimizer | **AdamW** |
| Learning Rate | **3e-4** |
| Weight Decay | 1e-4 |
| LR Schedule | Linear Warmup (5 epochs) + Cosine Annealing |
| Early Stopping | Patience = 15 epochs |
| Gradient Clipping | max norm = 5.0 |
| Input Size | **512×288** |
| Seed | 42 |

---

## 📈 Results

### Test Set Metrics
| Metric | Value |
|---|---|
| **mIoU** | **88.74%** |
| IoU — Drivable | 82.87% |
| IoU — Non-Drivable | 94.60% |
| Dice Score | 0.9063 |
| Precision | 0.8807 |
| Recall | 0.9335 |
| Pixel Accuracy | **95.72%** |

## 🛠️ Setup & Installation
```bash
# Clone the repository
git clone https://github.com/Banasri007/Drivable-Space-Segmentation.git
cd Drivable-Space-Segmentation

# Install dependencies
pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
pip install nuscenes-devkit==1.1.11 scikit-learn matplotlib \
  seaborn tqdm scipy pyquaternion Pillow
```

### requirements.txt
```
torch==2.2.0
torchvision==0.17.0
nuscenes-devkit==1.1.11
scikit-learn==1.3.2
scipy==1.11.4
numpy==1.24.4
matplotlib
seaborn
tqdm
pyquaternion
Pillow
```

---

## ▶️ How to Run

1. **Download nuScenes v1.0-mini** from [nuScenes](https://www.nuscenes.org/download) 
   and note its path.

2. **Open the notebook** `Drivable_Space_Segmentation.ipynb` on Kaggle or locally.

3. **Update the dataset path** in Cell 5:
```python
CFG = {
    'DATAROOT': '/path/to/your/v1.0-mini',
    ...
}
```

4. **Run Cell 1 first**, then **restart the kernel**, then run all remaining cells 
   sequentially.

5. Trained model checkpoint saves automatically to `outputs/best_model.pth`.
   Output plots (training curves, confusion matrix, qualitative results) save to 
   the `outputs/` folder.

---

## 📁 Output Files

| File | Description |<img width="1789" height="985" alt="005d4d0e-d32c-4697-a631-213d1e88f97d" src="https://github.com/user-attachments/assets/b828bc62-abf1-4703-91ba-1c6d090d86bf" />

|---|---|
| `01_data_samples.png` | Sample images with LiDAR-generated masks |
| `02_training_curves.png` | Loss and mIoU curves over epochs |
| `03_confusion_matrix.png` | Pixel-level confusion matrix on test set |
| `04_qualitative_results.png` | Predicted masks vs ground truth |
| `05_performance_dashboard.png` | Full metrics dashboard |
| `best_model.pth` | Best checkpoint (by val mIoU) |

