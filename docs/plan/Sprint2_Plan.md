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

## Days 2-3: Defect Detection Notebook

**Why**: Directly relevant to Deltia's quality inspection use case

### Tasks:
- [ ] Download Casting Product Defects dataset (Kaggle)
- [ ] Create `notebooks/02_defect_detection.ipynb`
- [ ] Implement autoencoder-based anomaly detection
- [ ] Train on "good" samples only
- [ ] Evaluate with ROC-AUC, confusion matrix
- [ ] Visualize reconstructions & error maps

---

## Day 4: Image Segmentation Notebook

**Why**: Shows pixel-level analysis capability

### Tasks:
- [ ] Create `notebooks/03_image_segmentation.ipynb`
- [ ] Use Oxford-IIIT Pet dataset (tensorflow_datasets)
- [ ] Implement U-Net architecture
- [ ] Evaluate with IoU and Dice metrics
- [ ] Visualize predictions vs ground truth

---

## Days 5-6: Manufacturing Dashboard (Streamlit)

**Why**: Demonstrates data visualization & communication skills

### Tasks:
- [ ] Create `app/manufacturing_dashboard.py`
- [ ] Generate synthetic production data
- [ ] KPI cards (OEE, Defect Rate, Cycle Time)
- [ ] Cycle time trend chart
- [ ] Defect Pareto chart
- [ ] Station comparison

---

## Days 7-8: Portfolio Polish + README Update

### Tasks:
- [ ] Update README.md with manufacturing focus
- [ ] Add requirements.txt
- [ ] Review all notebooks for clarity
- [ ] Push final changes to GitHub

---

## Day 9: Buffer / Final Review

- [ ] Test all notebooks run end-to-end
- [ ] Verify Streamlit dashboard works
- [ ] Final GitHub push

---

## Files to Create

| File | Purpose |
|------|---------|
| `notebooks/02_defect_detection.ipynb` | Anomaly detection |
| `notebooks/03_image_segmentation.ipynb` | U-Net segmentation |
| `app/manufacturing_dashboard.py` | Streamlit dashboard |
| `requirements.txt` | Dependencies |
| `README.md` | Updated with manufacturing focus |

---

## Success Criteria

| Criteria | Verification |
|----------|--------------|
| Defect detection works | AUC-ROC > 0.8 |
| Segmentation works | mIoU > 0.5 |
| Dashboard runs | `streamlit run app/manufacturing_dashboard.py` |
| README updated | Manufacturing applications highlighted |
