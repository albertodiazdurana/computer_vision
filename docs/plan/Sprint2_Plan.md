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
| `notebooks/03_steel_defect_segmentation.ipynb` | Steel defect segmentation |
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
