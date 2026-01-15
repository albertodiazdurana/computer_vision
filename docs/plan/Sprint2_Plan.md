# Sprint 2 Plan: Portfolio Enhancement for Deltia Interview

## Overview

| Aspect | Detail |
|--------|--------|
| **Duration** | 9 days (Interview: 2026-01-17) |
| **Goal** | Manufacturing-relevant portfolio |
| **Success Metric** | Repo showcases defect detection, segmentation, dashboard |

---

## Completed (Day 1)

- [x] Modular package structure (`cv_toolkit`)
- [x] Config management (YAML + dataclasses)
- [x] MLflow integration
- [x] Grad-CAM implementation
- [x] Unit tests (4 passing)

---

## Completed (Days 2-3): Defect Detection Notebook

**Why**: Directly relevant to Deltia's quality inspection use case

### Tasks:
- [x] Download Casting Product Defects dataset (Kaggle)
- [x] Create `notebooks/02_defect_detection.ipynb`
- [x] Implement autoencoder-based anomaly detection
- [x] Train on "good" samples only
- [x] Evaluate with ROC-AUC (0.869), confusion matrix
- [x] Visualize reconstructions & error maps

---

## Day 4: Steel Defect Segmentation Notebook

**Why**: Shows pixel-level defect localization - directly relevant to manufacturing quality inspection

**Dataset**: Severstal Steel Defect Detection (Kaggle)
- 12,568 train images of steel sheets
- 4 defect classes with pixel-level segmentation masks
- Real manufacturing data from steel production

### Tasks:
- [x] Download Severstal Steel Defect dataset (Kaggle)
- [x] Create `notebooks/03_steel_defect_segmentation.ipynb`
- [x] Implement U-Net architecture for multi-class segmentation
- [x] Evaluate with IoU and Dice metrics
- [x] Visualize predictions vs ground truth masks

---

## Completed (Days 5-6): Steel Defect Segmentation Dashboard

**Why**: Interactive demo of ML model - directly showcases the segmentation work

**Pivot**: Changed from generic manufacturing KPIs to Steel Defect Segmentation dashboard

### Tasks:
- [x] Create `app/steel_segmentation_dashboard.py`
- [x] Load trained U-Net model with Streamlit caching
- [x] Image selection with Dice score filtering
- [x] Display: Input | Ground Truth | Prediction side-by-side
- [x] Per-image metrics (Dice, IoU, defect pixels)
- [x] Overlay visualization (prediction on original)
- [x] Deploy to Streamlit Community Cloud

**Live Demo**: https://steel-defect-segmentation.streamlit.app/

---

## Completed (Days 7-8): Portfolio Polish + README Update

### Tasks:
- [x] Update README.md with manufacturing focus
- [x] Add requirements.txt (app/requirements.txt for Streamlit Cloud)
- [x] Add live dashboard link to README
- [x] Push final changes to GitHub

---

## Completed (Day 9): Final Review

- [x] Test dashboard on Streamlit Cloud
- [x] Verify model loads and predictions work
- [x] Final GitHub push with updated README

---

## Files Created

| File | Purpose | Status |
|------|---------|--------|
| `notebooks/02_defect_detection.ipynb` | Anomaly detection | Done |
| `notebooks/03_steel_defect_segmentation.ipynb` | Steel defect segmentation | Done |
| `app/steel_segmentation_dashboard.py` | Interactive Streamlit dashboard | Done |
| `app/steel_unet.keras` | Trained U-Net model (487K params) | Done |
| `app/sample_data/` | 20 sample images for cloud deployment | Done |
| `app/requirements.txt` | Streamlit Cloud dependencies | Done |
| `README.md` | Updated with manufacturing focus + live demo | Done |

---

## Success Criteria

| Criteria | Target | Achieved |
|----------|--------|----------|
| Defect detection works | AUC-ROC > 0.8 | 0.869 |
| Segmentation works | mIoU > 0.5 | 0.28 (IoU), 0.42 (Dice) |
| Dashboard runs | Streamlit Cloud | https://steel-defect-segmentation.streamlit.app/ |
| README updated | Manufacturing highlighted | Done |

**Note on Segmentation**: IoU target was ambitious for this dataset. The 0.42 Dice score demonstrates the core workflow with limited GPU memory (2.2GB VRAM). Competition winners achieved ~0.90 Dice with pretrained encoders and extensive augmentation.
