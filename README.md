# 🚗 Real-Time Drivable Space Segmentation

## 📌 Project Overview
Autonomous vehicles must understand **where they can safely drive**, not just follow predefined lanes. In complex urban environments, lane markings may be absent, faded, or misleading.

This project focuses on **real-time semantic segmentation of drivable space**, identifying **free space** where a vehicle can physically move. The system classifies each pixel into:
- **Drivable Area**
- **Non-Drivable Area** (curbs, sidewalks, barriers, vegetation, etc.)

The model is designed with **real-time constraints**, making it suitable for deployment in **Level 4 autonomous driving systems**.

---

## 🎯 Problem Motivation
Traditional perception systems rely heavily on lane detection, which fails in:
- Unstructured roads  
- Construction zones  
- Poor weather conditions  
- Occluded or missing lane markings  

This project solves that by directly predicting **drivable regions**, enabling safer and more robust navigation.

---

## ⚠️ Key Challenges
- **Class Imbalance** → Drivable pixels dominate  
- **Ambiguous Boundaries** → Road vs grass, puddles  
- **Real-Time Constraints** → High FPS required  
- **Urban Complexity** → Dynamic obstacles and irregular layouts  

---

## 🧠 Model Architecture

We use an **Encoder–Decoder architecture** optimized for real-time performance.

### 🔹 Encoder (Feature Extraction)
- Lightweight backbone (trained from scratch):
  - MobileNet-style architecture  
  - EfficientNet-inspired blocks  
- Depthwise separable convolutions for efficiency  

### 🔹 Decoder (Segmentation Head)
- U-Net style skip connections  
- Multi-scale feature fusion  
- Upsampling layers for pixel-wise prediction  

### 🔹 Design Goals
- Low latency  
- High accuracy  
- Efficient memory usage  

---

## ⚙️ Methodology

### 🔹 Loss Functions
To handle class imbalance and improve segmentation quality:
- Binary Cross Entropy (BCE)  
- Dice Loss  
- Focal Loss  

### 🔹 Data Augmentation
- Horizontal flipping  
- Random cropping  
- Scaling  
- Brightness & contrast adjustments  

### 🔹 Optimization
- Optimizer: Adam  
- Learning rate scheduling  
- Early stopping (optional)  

---

## 📊 Dataset Used

- **Dataset:** nuScenes  
- Diverse real-world driving scenarios:
  - Urban environments  
  - Day/night conditions  
  - Weather variations  

### 🔹 Preprocessing
- Image resizing  
- Normalization  
- Mask generation (binary or multi-class)  

---

## 🏋️ Training Details

| Parameter        | Value            |
|----------------|-----------------|
| Epochs         | 50 (adjustable) |
| Batch Size     | 8               |
| Optimizer      | Adam            |
| Learning Rate  | 0.001           |
| Input Size     | 512×512         |

### 🔹 Hardware
- GPU recommended (NVIDIA CUDA-enabled)  
- Can run on CPU (slower inference)  

---

## 🛠️ Setup & Installation

```bash
# Clone the repository
git clone https://github.com/your-username/drivable-space-segmentation.git

# Navigate to project folder
cd drivable-space-segmentation

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
