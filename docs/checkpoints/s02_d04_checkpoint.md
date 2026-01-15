# Day 4 Checkpoint - Steel Defect Segmentation

**Date:** 2026-01-09
**Sprint:** 2
**Day:** 4

---

## Completed Tasks

- [x] Downloaded Severstal Steel Defect dataset (full 12,568 images via Kaggle CLI)
- [x] Created `notebooks/03_steel_defect_segmentation.ipynb`
- [x] Implemented U-Net architecture for binary segmentation
- [x] Trained model with combined BCE + Dice loss (stable training)
- [x] Achieved Dice: 0.42, IoU: 0.28 on validation set
- [x] Visualized predictions vs ground truth masks
- [x] Saved figures for README

---

## Results Summary

| Metric | Value |
|--------|-------|
| Dataset | Severstal Steel Defect (4,000 of 12,568 images) |
| Architecture | U-Net (487K params) |
| Task | Binary Segmentation |
| Training Images | 3,200 |
| Validation Images | 800 |
| Best Epoch | 47 |
| **Validation Dice** | **0.42** |
| **Validation IoU** | **0.28** |
| Training Time | ~15 min |

---

## Performance Comparison

| Run | Images | Dice | IoU | Notes |
|-----|--------|------|-----|-------|
| Initial | 605 | 0.21 | 0.12 | Pure Dice loss, collapsed at epoch 9 |
| **Final** | **4,000** | **0.42** | **0.28** | Combined BCE+Dice, stable 50 epochs |

**Improvement: 2x better Dice, 2.3x better IoU**

---

## Key Decisions

### DEC-004: Binary vs Multi-class Segmentation
- **Context:** 4 defect classes with severe imbalance
- **Decision:** Use binary segmentation (defect vs background)
- **Rationale:** Simpler model, demonstrates core segmentation technique

### DEC-005: Combined Loss Function
- **Context:** Pure Dice loss caused training collapse
- **Decision:** Use BCE + Dice combined loss
- **Rationale:** BCE provides stable gradients, Dice handles class imbalance

### DEC-006: Cloud Deployment Size Adaptation
- **Context:** Streamlit Community Cloud has repository size limits; full Severstal dataset is ~2GB (12,568 images)
- **Decision:** Bundle 20 sample images (2.1MB) + trained model (5.7MB) = 7.8MB total app size
- **Rationale:**
  - Demonstrates full segmentation workflow within cloud deployment constraints
  - App auto-detects sample vs full data at runtime (checks if `sample_data/` exists)
  - Local development still uses full dataset for comprehensive testing
  - Sample images selected from defect-containing subset to ensure meaningful demonstrations

---

## Technical Notes

### What Worked
- Combined BCE + Dice loss for stable training
- EarlyStopping monitoring `val_dice_coef` with `mode='max'`
- ReduceLROnPlateau triggered at epoch 42 for final boost
- 4,000 image subset (memory efficient, still significant improvement)

### Memory Management
- Image size: 128x128 (down from 256x1600 original)
- Batch size: 8 (to fit in 2.2GB VRAM)
- MAX_IMAGES: 4,000 (limited from 12,568 to avoid OOM)

---

## Figures Generated

| Figure | Location |
|--------|----------|
| Sample images with masks | `outputs/figures/steel_sample_images_masks.png` |
| Predictions vs ground truth | `outputs/figures/steel_segmentation_predictions.png` |
| Training history | `outputs/figures/steel_training_history.png` |

---

## Notebook Structure (15 cells)

| Cell | Content |
|------|---------|
| 1 | Markdown header |
| 1.5 | Kaggle CLI download (optional) |
| 2 | Imports |
| 3 | Configuration |
| 4 | Load train.csv, explore dataset |
| 5 | RLE decoding function |
| 6 | Visualize samples with masks |
| 7 | Data loading (4,000 images) |
| 8 | Train/validation split |
| 9 | Build U-Net model |
| 10 | Combined loss + training |
| 11 | Evaluate model |
| 12 | Visualize predictions |
| 13 | Plot training history |
| 14 | Conclusions |

---

## Next Steps

1. Commit Day 4 work
2. Update README with segmentation results
3. Continue to Days 5-6 (manufacturing dashboard)

---

## Git Status

- Branch: sprint2_learning
- Files modified: notebooks/03_steel_defect_segmentation.ipynb
- New figures: 3 in outputs/figures/
