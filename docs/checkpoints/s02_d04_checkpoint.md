# Day 4 Checkpoint - Steel Defect Segmentation

**Date:** 2026-01-09
**Sprint:** 2
**Day:** 4

---

## Completed Tasks

- [x] Downloaded Severstal Steel Defect dataset (605 images subset)
- [x] Created `notebooks/03_steel_defect_segmentation.ipynb`
- [x] Implemented U-Net architecture for binary segmentation
- [x] Trained model with Dice loss
- [x] Handled training collapse with EarlyStopping
- [x] Evaluated with IoU and Dice metrics
- [x] Visualized predictions vs ground truth masks
- [x] Saved figures for README

---

## Results Summary

| Metric | Value |
|--------|-------|
| Dataset | Severstal Steel Defect (605 images) |
| Architecture | U-Net (487K params) |
| Task | Binary Segmentation |
| Best Epoch | 7 |
| Validation Dice | 0.21 |
| Validation IoU | 0.12 |
| Training Time | ~2 min |

---

## Key Decisions

### DEC-004: Binary vs Multi-class Segmentation
- **Context:** Only 605 images available (expected 12,568)
- **Decision:** Use binary segmentation instead of 4-class
- **Rationale:** Limited data, class imbalance, simpler model less prone to overfitting

---

## Technical Notes

### Training Collapse Issue
- Model collapsed after epoch 9 (all-zero predictions, dice_coef → 0)
- **Root cause:** Pure Dice loss can become unstable when predictions approach zero
- **Solution:** EarlyStopping with `restore_best_weights=True` recovered epoch 7 weights
- **Future improvement:** Use combined loss (BCE + Dice) for stability

### Data Loading Optimizations
- Reduced image size from 256x256 to 128x128 for memory efficiency
- Added `gc.collect()` every 100 images during loading
- Batch size: 8 (small dataset)

---

## Figures Generated

| Figure | Location |
|--------|----------|
| Sample images with masks | `outputs/figures/steel_defect_samples.png` |
| Predictions vs ground truth | `outputs/figures/steel_segmentation_predictions.png` |
| Training history | `outputs/figures/steel_training_history.png` |

---

## Notebook Structure

| Cell | Content |
|------|---------|
| 1 | Markdown header |
| 2 | Imports |
| 3 | Configuration |
| 4 | Load train.csv, explore dataset |
| 5 | Parse ImageId/ClassId, class distribution |
| 6 | RLE decoding function |
| 7 | Visualize samples with masks |
| 8 | Data loading function |
| 9 | Train/validation split |
| 10 | Build U-Net model |
| 11 | Compile and train with Dice loss |
| 12 | Evaluate model |
| 13 | Visualize predictions |
| 14 | Plot training history |
| 15 | Conclusions |

---

## Performance vs Target

| Metric | Target | Achieved | Status |
|--------|--------|----------|--------|
| mIoU | 0.4-0.5 | 0.12 | Below target |
| Dice | - | 0.21 | Baseline established |

**Note:** Below-target performance is expected with 605 images (5% of full dataset). The notebook demonstrates the approach and would improve significantly with full data.

---

## Next Steps

1. Update README with segmentation project
2. Start manufacturing dashboard (Days 5-6)
3. Optional: Try combined BCE+Dice loss for better training stability

---

## Git Status

- Branch: sprint2_learning
- Files modified: notebooks/03_steel_defect_segmentation.ipynb
- New figures: 3 in outputs/figures/
