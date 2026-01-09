# Day 2 Checkpoint - Defect Detection Notebook

**Date:** 2026-01-08
**Sprint:** 2
**Day:** 2

---

## Completed Tasks

- [x] Downloaded Casting Product Defects dataset (Kaggle)
- [x] Created `notebooks/02_defect_detection.ipynb`
- [x] Implemented convolutional autoencoder for anomaly detection
- [x] Trained on "good" samples only (2,875 images)
- [x] Evaluated with ROC-AUC (0.869), confusion matrix
- [x] Visualized reconstructions & error maps
- [x] Saved 4 result figures

---

## Dataset

| Split | OK | Defect | Total |
|-------|-----|--------|-------|
| Train | 2,875 | 3,758 | 6,633 |
| Test | 262 | 453 | 715 |

Source: Casting Product Defects (Kaggle)
Image size: 128x128 RGB

---

## Model Architecture

**Convolutional Autoencoder** (333,955 parameters)

Encoder:
- Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Conv2D(128) → MaxPool
- Output: 16x16x128

Decoder:
- Conv2DTranspose(128) → UpSample → Conv2DTranspose(64) → UpSample → Conv2DTranspose(32) → UpSample → Conv2D(3)
- Output: 128x128x3

---

## Key Decisions

1. **Anomaly detection approach**: Train only on OK samples
   - Why: Don't need labeled defects, can detect unknown defect types
   
2. **Image size 128x128**: Balance between detail and memory
   - Why: Faster training, fits in GPU memory

3. **MSE loss**: Reconstruction error
   - Why: Simple, effective for pixel-wise comparison

4. **Optimal threshold via Youden's J**: 0.001572
   - Why: Balances true positive rate and false positive rate

---

## Results

| Metric | Value | Target |
|--------|-------|--------|
| ROC-AUC | 0.869 | > 0.8 |
| Accuracy | 77.5% | - |
| Precision (Defect) | 91% | - |
| Recall (Defect) | 72% | - |

---

## Artifacts

| File | Purpose |
|------|---------|
| notebooks/02_defect_detection.ipynb | Main notebook |
| outputs/figures/defect_training_history.png | Loss curves |
| outputs/figures/defect_error_distribution.png | OK vs Defect errors |
| outputs/figures/defect_roc_curve.png | ROC curve |
| outputs/figures/defect_confusion_matrix.png | Confusion matrix |

---

## Git Status

- Branch: sprint2_learning
- Commit: 6bcfa4a - "Add defect detection notebook with autoencoder approach"
- Pushed to origin

---

## Manufacturing Relevance

- **Quality inspection**: Detect surface defects on casting products
- **Error maps**: Localize where defects occur
- **Unsupervised**: No labeled defect data required for training
- **Explainable**: Visual output shows model reasoning

---

## Next Steps

1. Image Segmentation notebook (Day 4)
2. Manufacturing Dashboard (Days 5-6)
