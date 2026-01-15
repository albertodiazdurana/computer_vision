# Computer Vision Portfolio

> Deep learning projects demonstrating computer vision techniques for manufacturing quality control and process optimization.

**[Live Demo: Steel Defect Segmentation Dashboard](https://steel-defect-segmentation.streamlit.app/)**

## Projects

### 1. Image Classification (CIFAR-10)

Transfer learning with ResNet50 for multi-class image classification.

| Metric | Value |
|--------|-------|
| Test Accuracy | 48.9% |
| Architecture | ResNet50 + custom head |
| Training | Two-phase (frozen → fine-tuned) |

**Manufacturing relevance:** Product categorization, defect classification

![Grad-CAM Samples](outputs/figures/gradcam_samples.png)

---

### 2. Defect Detection (Casting Products)

Anomaly detection using convolutional autoencoder trained only on "good" samples.

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.869 |
| Accuracy | 77.5% |
| Precision (Defect) | 91% |

**Manufacturing relevance:** Quality inspection, surface defect detection

![ROC Curve](outputs/figures/defect_roc_curve.png)

**Key insight:** No labeled defect data required for training - model learns "what good looks like" and flags anomalies.

![Error Distribution](outputs/figures/defect_error_distribution.png)

---

### 3. Steel Defect Segmentation

U-Net semantic segmentation for pixel-level defect detection on steel surfaces.

| Metric | Value |
|--------|-------|
| Dice Coefficient | 0.42 |
| IoU | 0.28 |
| Architecture | U-Net (487K params) |
| Dataset | Severstal Steel (4,000 images) |

**Manufacturing relevance:** Automated quality inspection, surface defect localization

![Steel Sample Images with Masks](outputs/figures/steel_sample_images_masks.png)

**Key insight:** Combined BCE + Dice loss enables stable training on severely imbalanced data (~3% defect pixels).

**[Try the Interactive Dashboard](https://steel-defect-segmentation.streamlit.app/)**

---

## Skills Demonstrated

- **Deep Learning:** CNN, Transfer Learning, Autoencoders, U-Net, Semantic Segmentation
- **Python:** TensorFlow/Keras, scikit-learn, matplotlib, Streamlit
- **MLOps:** MLflow experiment tracking, modular package design, cloud deployment
- **Data Analysis:** ROC curves, confusion matrices, Dice/IoU metrics
- **Interpretability:** Grad-CAM, reconstruction error maps, mask overlays

---

## Project Structure

```
computer_vision/
├── app/                     # Streamlit dashboard
│   ├── steel_segmentation_dashboard.py
│   ├── steel_unet.keras     # Trained model
│   └── sample_data/         # Demo images
├── src/cv_toolkit/          # Modular Python package
│   ├── config.py            # Configuration management
│   ├── data.py              # Data loading
│   ├── model.py             # Model architectures
│   ├── training.py          # Training with MLflow
│   ├── evaluation.py        # Metrics and visualization
│   └── gradcam.py           # Model interpretability
├── notebooks/
│   ├── 01_cifar10_classification.ipynb
│   ├── 02_defect_detection.ipynb
│   └── 03_steel_defect_segmentation.ipynb
├── tests/                   # Unit tests (pytest)
├── config/                  # YAML configuration
└── outputs/figures/         # Generated visualizations
```

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/albertodiazdurana/computer_vision.git
cd computer_vision

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install package
pip install -e .

# Run notebooks
jupyter notebook notebooks/
```

---

## Requirements

```
tensorflow>=2.10.0
mlflow>=2.0.0
streamlit>=1.28.0
scikit-learn
matplotlib
seaborn
numpy
pandas
pyyaml
```

---

## Author

**Alberto Diaz Durana**

[![GitHub](https://img.shields.io/badge/GitHub-albertodiazdurana-181717?style=flat&logo=github)](https://github.com/albertodiazdurana)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-albertodiazdurana-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/albertodiazdurana/)

---

*Last updated: 2026-01-15*
