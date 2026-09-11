# 🍎 Deep Learning-Based Fruit Freshness Assessment with ResNet34 and Explainable Visualizations

An end-to-end deep learning pipeline that classifies fruit images (apples, bananas, oranges) as fresh or rotten using transfer learning, with a full evaluation suite (cross-validation, learning curves, Cohen's Kappa) and Grad-CAM explainability to visualize what the model is "looking at" when it makes a prediction.

---

## 🎯 Objectives

The main objectives of this project are to:

1. Build an automated fruit-freshness classification pipeline.
2. Compare multiple modern deep-learning architectures.
3. Use a reproducible **80/10/10 stratified train/validation/test split**.
4. Evaluate image preprocessing using **PSNR** and **SSIM**.
5. Measure classification performance using accuracy, precision, recall, F1-score, AUC, Cohen's kappa, confidence intervals, and inference time.
6. Improve model transparency using **Grad-CAM**.
7. Validate the best-performing model using **5-fold cross-validation**.

---

## 🗂️ Dataset

The project uses the following Kaggle dataset:

🔗 **Fruits Fresh and Rotten for Classification**  
https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification

The dataset contains **13,599 RGB images** covering three fruit types in fresh and rotten conditions.

### Classes

| Class | Description |
|---|---|
| `freshapples` | Fresh apples |
| `freshbanana` | Fresh bananas |
| `freshoranges` | Fresh oranges |
| `rottenapples` | Rotten apples |
| `rottenbanana` | Rotten bananas |
| `rottenoranges` | Rotten oranges |

### Dataset Split

A stratified **80% / 10% / 10%** split is used:

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| Fresh Apples | 1,670 | 209 | 209 |
| Fresh Bananas | 1,570 | 196 | 196 |
| Fresh Oranges | 1,483 | 186 | 185 |
| Rotten Apples | 2,355 | 294 | 294 |
| Rotten Bananas | 2,203 | 275 | 276 |
| Rotten Oranges | 1,598 | 200 | 200 |
| **Total** | **10,879** | **1,360** | **1,360** |

---

## 🧠 Models Compared

Six architectures are evaluated under a unified training setup:

1. **ResNet34**
2. **MobileViT-S**
3. **LeViT-128S**
4. **DeiT-Tiny Patch16-224**
5. **EfficientNet-B0**
6. **TinyViT-5M-224**

### Test Performance

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---:|---:|---:|---:|---:|
| **ResNet34** | **99.93%** | **99.91%** | **99.92%** | **99.91%** | **1.0000** |
| MobileViT-S | 99.71% | 99.71% | 99.70% | 99.71% | 1.0000 |
| EfficientNet-B0 | 99.49% | 99.40% | 99.38% | 99.39% | 0.9998 |
| TinyViT-5M-224 | 99.41% | 99.40% | 99.45% | 99.42% | 0.9999 |
| DeiT-Tiny Patch16-224 | 99.26% | 99.15% | 99.18% | 99.16% | 0.9999 |
| LeViT-128S | 98.16% | 98.28% | 98.04% | 98.14% | 0.9992 |

**ResNet34 achieved the highest overall test performance and was selected as the proposed model.**

---

# 🍎 Deep Learning-Based Fruit Freshness Assessment with ResNet34 and Explainable Visualizations

An end-to-end deep learning pipeline that classifies fruit images (apples, bananas, oranges) as **fresh** or **rotten** using transfer learning, with a full evaluation suite (cross-validation, learning curves, Cohen's Kappa) and **Grad-CAM** explainability to visualize what the model is "looking at" when it makes a prediction.

> 📄 Companion research paper: *"Deep Learning-Based Fruit Freshness Assessment with ResNet34 and Explainable Visualizations"* — Tabassum Talukder, Md. Abdulla Hasan, Md. Ehsanul Haque (Dept. of CSE / MPS, East West University, Dhaka, Bangladesh). Published in IEEE Xplore: [ieeexplore.ieee.org/document/11502516](https://ieeexplore.ieee.org/document/11502516). See [`CRC.pdf`](./CRC.pdf) for the camera-ready write-up.

---

## 🏆 Highlights

- **6-class classification**: Fresh/Rotten × {Apples, Bananas, Oranges}
- **6 architectures benchmarked**: ResNet34, MobileViT-S, LeViT-128S, DeiT-Tiny (Patch16-224), EfficientNet-B0, TinyViT-5M-224
- **Best model — ResNet34**: **99.93% test accuracy**, Cohen's Kappa **0.9991**, AUC **1.0000**
- **5-fold cross-validation** mean accuracy of **99.61%**, confirming generalization
- **Grad-CAM** visualizations confirming the model attends to freshness-relevant regions (texture, color, decay)
- Rigorous preprocessing pipeline: resize → CLAHE contrast enhancement → Non-Local Means denoising → normalization, validated quantitatively with **PSNR** and **SSIM**

---

## 📊 Results at a Glance

### Model Comparison (Test Set)

| Model                  | Accuracy | Precision | Recall | F1-Score | AUC    |
|-------------------------|:--------:|:---------:|:------:|:--------:|:------:|
| **ResNet34 (proposed)** | **0.9993** | **0.9991** | **0.9992** | **0.9991** | **1.0000** |
| MobileViT-S             | 0.9978   | 0.9977    | 0.9975 | 0.9976   | 0.9998 |
| EfficientNet-B0         | 0.9949   | 0.9940    | 0.9938 | 0.9939   | 0.9998 |
| TinyViT-5M-224          | 0.9941   | 0.9940    | 0.9945 | 0.9942   | 0.9999 |
| DeiT-Tiny (Patch16-224) | 0.9897   | 0.9884    | 0.9897 | 0.9889   | 0.9981 |
| LeViT-128S              | 0.9816   | 0.9828    | 0.9804 | 0.9814   | 0.9992 |

### Cohen's Kappa & Confidence Intervals

| Model                  | Kappa  | 95% CI              |
|-------------------------|:------:|:--------------------:|
| ResNet34                | 0.9991 | [0.9978, 1.0000]     |
| MobileViT-S              | 0.9973 | [0.9953, 1.0000]     |
| EfficientNet-B0          | 0.9938 | [0.9910, 0.9987]     |
| TinyViT-5M-224           | 0.9929 | [0.9901, 0.9982]     |
| DeiT-Tiny (Patch16-224)  | 0.9876 | [0.9843, 0.9951]     |
| LeViT-128S               | 0.9778 | [0.9745, 0.9888]     |

### 5-Fold Cross-Validation (ResNet34)

| Fold | Accuracy |
|------|:--------:|
| 1    | 0.9904   |
| 2    | 0.9996   |
| 3    | 0.9915   |
| 4    | 0.9996   |
| 5    | 0.9993   |
| **Mean** | **0.9961** |

### Efficiency Comparison

| Model            | Inference (ms) | Training (s) | GPU (GB) | RAM (GB) |
|-------------------|:---------------:|:------------:|:--------:|:--------:|
| **ResNet34**      | 5.45            | 391.20       | 1.56     | 2.56     |
| DeiT-Tiny         | 10.48           | 307.92       | 0.72     | 2.71     |
| LeViT-128S        | 11.34            | 370.84       | 0.68     | 3.36     |
| TinyViT-5M-224    | 12.00           | 1007.02      | 3.26     | 3.45     |
| EfficientNet-B0   | 36.84           | 277.37       | 3.07     | 3.42     |
| MobileViT-S       | 172.15          | 481.22       | 2.97     | 3.07     |

ResNet34 offers the best trade-off between accuracy, stability, and inference speed, making it the recommended model for real-world deployment.

---

## 🧠 Methodology Overview

```
Raw Images (13,599 RGB, 6 classes)
        │
        ▼
Preprocessing
  ├─ Resize → 224×224
  ├─ CLAHE contrast enhancement
  ├─ Non-Local Means denoising
  └─ Normalization [0, 1]
        │
        ▼
Stratified Split (80% train / 10% val / 10% test)
        │
        ▼
Train 6 Architectures
  ResNet34 · MobileViT-S · LeViT-128S · DeiT-Tiny · EfficientNet-B0 · TinyViT-5M
        │
        ▼
Evaluation (all 6 models)
  Accuracy · Precision · Recall · F1 · Cohen's Kappa · AUC
        │
        ▼
Best Model → ResNet34
        │
        ├─► Explainability: Grad-CAM heatmaps on final convolutional layer
        │
        └─► 5-Fold Cross-Validation (ResNet34 only)
```
---

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- TIMM
- NumPy
- Pandas
- OpenCV
- Pillow
- Scikit-learn
- Scikit-image
- Matplotlib
- Seaborn
- Joblib
- PSNR / SSIM
- Grad-CAM
- Kaggle GPU environment

---

## ⚙️ Installation

Install the main dependencies with:

```bash
pip install numpy pandas pillow opencv-python matplotlib seaborn scikit-learn scikit-image torch torchvision timm psutil joblib
```

For the easiest reproduction, run the notebook in **Kaggle with a GPU runtime**, since the notebook uses Kaggle dataset paths and CUDA-enabled training.

---

## ▶️ How to Run

### Option 1 — Kaggle

1. Open the notebook in Kaggle.
2. Add the dataset:
   `sriramr/fruits-fresh-and-rotten-for-classification`
3. Enable a GPU runtime.
4. Run the notebook cells in order.

### Option 2 — Local / Other Notebook Environment

Download the Kaggle dataset and update the dataset paths used in the notebook, for example:

```python
DATA_ROOT = "/path/to/fruits-fresh-and-rotten-for-classification/dataset"
```

The notebook currently uses Kaggle-specific paths such as:

```text
/kaggle/input/fruits-fresh-and-rotten-for-classification/dataset
/kaggle/working/split_dataset
```

These should be changed when running outside Kaggle.

---

## 📁 Suggested Repository Structure

```text
fruit-freshness-and-rotten-for-classification/
│
├── README.md
├── fruit-freshness-and-rotten-for-classification.ipynb
│
├── results/
│   ├── confusion_matrix/
│   ├── gradcam/
│   ├── learning_curves/
│   ├── roc_curves/
│   └── pr_curves/
│
└── models/
    └── resnet34_best.pth
```

> The original Kaggle dataset should generally **not be committed to GitHub**. Keep the dataset external and link to the original Kaggle source instead.

---

## 📈 Key Findings

- **ResNet34** performed best among the six evaluated architectures.
- The proposed model achieved **99.93% test accuracy**.
- 5-fold cross-validation produced a mean accuracy of **99.61%**.
- The model achieved a Cohen's kappa of **0.9991**.
- Grad-CAM showed that predictions were based on meaningful freshness-related image regions.
- The evaluation also considers computational efficiency through inference time, GPU memory usage, RAM consumption, and training time.
- The results indicate that ResNet34 is a strong candidate for automated fruit-freshness assessment.

---

## ⚠️ Limitations

The dataset covers only three fruit types:

- Apples
- Bananas
- Oranges

Therefore, the results may not directly generalize to every fruit or to real-world conditions such as:

- Different lighting environments
- Occlusion
- Background clutter
- Camera variations
- Different stages of decay
- Additional fruit varieties
- Real-time market or warehouse conditions

The accompanying paper recommends expanding the dataset and evaluating the system under more diverse real-world conditions.

---

## 🚀 Future Work

Potential extensions include:

- Expanding the dataset to additional fruit categories.
- Collecting images under varied environmental and lighting conditions.
- Testing the system on real-world market or warehouse images.
- Building a real-time fruit freshness detection application.
- Deploying the trained model in automated sorting or quality-control systems.
- Developing a user-friendly software interface for practical use.

---

## 📚 Research Paper

**Deep Learning-Based Fruit Freshness Assessment with ResNet34 and Explainable Visualizations**

**Authors:**
- Tabassum Talukder
- Md. Abdulla Hasan
- Md. Ehsanul Haque

The paper presents the motivation, methodology, model comparison, preprocessing evaluation, cross-validation results, Grad-CAM analysis, and conclusions associated with this project.

---

## 📖 References

### Dataset

S. R. Kalluri, *Fruits Fresh and Rotten for Classification*, Kaggle, 2018.  
https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification

### Related Work

- Y. Yuan et al., *An innovative approach to detecting the freshness of fruits and vegetables through the integration of convolutional neural networks and bidirectional long short-term memory network*, 2024.
- J. F. Martínez Pazos et al., *Freshnets: Highly accurate and efficient food freshness assessment based on deep convolutional neural networks*, 2024.
- Y. Shu et al., *Fruit freshness classification and detection based on the ResNet-101 network and non-local attention mechanism*, 2025.
- Y. Gulzar, *Fruit image classification model based on MobileNetV2 with deep transfer learning technique*, 2023.
- M. S. Morshed et al., *Fruit quality assessment with densely connected convolutional neural network*, 2022.

---

## 📄 Citation

If you use this project, dataset, or methodology in academic work, please cite the accompanying research paper and the original Kaggle dataset.

---

## 👨‍💻 Project

**Fruit Freshness Classification using Deep Learning, Transfer Learning, and Explainable AI**

Built with **PyTorch + TIMM + Scikit-learn + OpenCV** and evaluated using **ResNet34, MobileViT-S, LeViT-128S, DeiT-Tiny, EfficientNet-B0, and TinyViT-5M-224**.
