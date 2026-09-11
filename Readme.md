# 🍎 Fruit Freshness Classification using Deep Learning

A deep learning project for **automatic fruit freshness classification** using the Kaggle *Fruits Fresh and Rotten for Classification* dataset.

The project compares six transfer-learning / modern deep-learning architectures and identifies **ResNet34** as the best-performing model. It also includes image preprocessing analysis, PSNR/SSIM-based image-quality evaluation, stratified dataset splitting, detailed model evaluation, 5-fold cross-validation, and **Grad-CAM** visualizations for model interpretability.

> **Best model:** ResNet34  
> **Test Accuracy:** **99.93%**  
> **5-Fold Cross-Validation Accuracy:** **99.61% ± 0.42%**

---

## 📌 Project Overview

Manual fruit-quality inspection can be time-consuming, subjective, and inconsistent. This project investigates an automated computer-vision approach for classifying fruit freshness from RGB images.

The system classifies images into six categories:

- Fresh Apples
- Fresh Bananas
- Fresh Oranges
- Rotten Apples
- Rotten Bananas
- Rotten Oranges

The accompanying research paper describes the methodology and experimental findings in detail, while the provided Jupyter Notebook contains the implementation and experiments.

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

## 🔬 Methodology

### 1. Image Preparation

Images are standardized to **224 × 224** pixels.

The project also includes preprocessing analysis using:

- Image resizing
- CLAHE-based contrast enhancement
- Non-Local Means denoising analysis
- Pixel normalization
- PSNR evaluation
- SSIM evaluation

The notebook quantitatively compares image quality before and after preprocessing.

### 2. Stratified Dataset Splitting

The complete dataset is reorganized into:

```text
Train       → 80%
Validation  → 10%
Test        → 10%
```

A fixed random seed of **42** is used for reproducibility.

### 3. Data Augmentation

For model training, the ResNet34 pipeline includes:

- Resize to 224 × 224
- Random resized crop
- Random rotation
- Random horizontal flip
- Color jitter
- Tensor conversion
- ImageNet normalization

Validation and test images use deterministic resizing/cropping followed by ImageNet normalization.

### 4. Model Training

The main training configuration is:

| Setting | Value |
|---|---|
| Input Size | 224 × 224 |
| Batch Size | 32 |
| Maximum Epochs | 40 |
| Optimizer | AdamW |
| Learning Rate | `3 × 10⁻⁴` |
| Weight Decay | `1 × 10⁻⁴` |
| Loss | Cross-Entropy |
| Dropout | 0.5 |
| Early Stopping Patience | 3 |
| Random Seed | 42 |
| Data Loader Workers | 4 |

The experiments were performed with GPU acceleration.

---

## 📊 ResNet34 Results

### Test Set

ResNet34 achieved:

- **Accuracy:** 99.93%
- **Precision:** 99.91%
- **Recall:** 99.92%
- **F1-score:** 99.91%
- **ROC-AUC:** 1.0000
- **Cohen's Kappa:** 0.9991
- **95% Accuracy CI:** [0.9978, 1.0000]
- **Inference time:** approximately 2.71 ms/image in the notebook evaluation

The test confusion matrix shows only a single misclassification: one **rotten orange** was classified as a **fresh orange**.

---

## 🔁 5-Fold Cross-Validation

The selected ResNet34 model was additionally evaluated using 5-fold stratified cross-validation.

| Fold | Accuracy |
|---|---:|
| Fold 1 | 99.04% |
| Fold 2 | 99.96% |
| Fold 3 | 99.15% |
| Fold 4 | 99.96% |
| Fold 5 | 99.93% |
| **Mean** | **99.61%** |
| **Std. Dev.** | **0.42%** |

The consistently high scores across folds indicate strong stability and generalization across different data partitions.

---

## 🔍 Explainable AI with Grad-CAM

To make the ResNet34 predictions more interpretable, the project uses **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

Grad-CAM heatmaps highlight image regions that contribute most strongly to the model's predictions. In the accompanying analysis, the model focuses on meaningful visual characteristics such as:

- Fruit texture
- Color changes
- Visible decay
- Rotten regions
- Other freshness-related visual patterns

This provides a visual explanation of *where* the model is looking when making a freshness prediction.

---

## 🖼️ Image Quality Evaluation

The notebook evaluates preprocessing with two widely used image-quality metrics:

- **PSNR — Peak Signal-to-Noise Ratio**
- **SSIM — Structural Similarity Index**

The paper reports improvements after preprocessing, including higher PSNR and SSIM values for the evaluated fruit classes.

| Class | PSNR Before | PSNR After | SSIM Before | SSIM After |
|---|---:|---:|---:|---:|
| Fresh Apples | 20.94 | 26.11 | 0.3153 | 0.7184 |
| Fresh Bananas | 21.58 | 27.09 | 0.3257 | 0.8479 |
| Fresh Oranges | 21.59 | 30.08 | 0.2104 | 0.9168 |
| Rotten Apples | 21.55 | 24.16 | 0.3051 | 0.6025 |
| Rotten Bananas | 22.42 | 24.54 | 0.2853 | 0.6734 |
| Rotten Oranges | 21.03 | 27.20 | 0.2720 | 0.6861 |

---

## 📓 Notebook

The main notebook is:

```text
fruit-freshness-and-rotten-for-classification.ipynb
```

The notebook is organized into the following major stages:

```text
Dataset Preparation
        ↓
Class Distribution Analysis
        ↓
Preprocessing Visualization
        ↓
PSNR / SSIM Evaluation
        ↓
Stratified 80/10/10 Split
        ↓
ResNet34 Training
        ↓
Grad-CAM Visualization
        ↓
MobileViT-S Training
        ↓
LeViT-128S Training
        ↓
DeiT-Tiny Training
        ↓
EfficientNet-B0 Training
        ↓
TinyViT-5M Training
        ↓
5-Fold Cross-Validation
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
