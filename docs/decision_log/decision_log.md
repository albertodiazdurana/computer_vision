# Decision Log - CIFAR-10 Classification Project

**Project:** CIFAR-10 Image Classification with ResNet50 Transfer Learning
**Created:** 2025-12-14

---

## Decision Log Format

Each decision follows this format:
- **ID**: DEC-XXX
- **Date**: YYYY-MM-DD
- **Context**: Why this decision was needed
- **Decision**: What was decided
- **Rationale**: Why this option was chosen
- **Alternatives Considered**: Other options evaluated
- **Impact**: Effect on project

---

## Decisions

### DEC-001: Model Architecture Selection
**Date:** 2025-12-14
**Context:** Need to select pre-trained CNN for transfer learning on CIFAR-10
**Decision:** Use ResNet50 with ImageNet weights
**Rationale:**
- Specified in project requirements
- Good balance of depth and performance
- Well-documented for transfer learning
**Alternatives Considered:** EfficientNetB0 (used in reference notebook), VGG16
**Impact:** Determines input preprocessing and fine-tuning strategy

### DEC-002: Validation Strategy
**Date:** 2025-12-14
**Context:** Need to split limited training data for validation while preserving test set
**Decision:** 80/20 stratified split of 10K training subset; use CIFAR-10 test set as holdout
**Rationale:**
- Preserves class distribution
- Maintains separate holdout for final evaluation
- 8K training samples sufficient for transfer learning
**Alternatives Considered:** K-fold CV (too slow), no validation split (risk overfitting)
**Impact:** Training: 8K, Validation: 2K, Test: 10K (holdout)

### DEC-003: Batch Size Selection
**Date:** 2025-12-14
**Context:** Limited VRAM (2.2GB available on Quadro T1000)
**Decision:** Start with batch size 32, reduce to 16 if OOM errors occur
**Rationale:**
- Balance between training speed and memory usage
- ResNet50 has significant memory footprint
- Can adjust based on actual GPU utilization
**Alternatives Considered:** Batch size 64 (likely OOM), batch size 8 (too slow)
**Impact:** Training time and memory usage

### DEC-004: Steel Defect Segmentation - Binary vs Multi-class
**Date:** 2026-01-09
**Context:** Severstal dataset downloaded with only 605 images (subset), highly imbalanced classes
**Decision:** Use binary segmentation (defect vs background) instead of 4-class segmentation
**Rationale:**
- Only 605 images available (expected 12,568)
- Class imbalance: Class 3 has 5,150 annotations, Class 2 only 247
- Binary classification provides stronger learning signal with limited data
- Simpler model less prone to overfitting on small dataset
**Alternatives Considered:**
- 4-class segmentation (rejected: too little data per class)
- Download full dataset (rejected: time constraint for interview prep)
**Impact:**
- Adjusted model: lighter U-Net, more augmentation
- Target mIoU: 0.4-0.5 (realistic for 605 images)
- Training split: 80% train (~480), 20% val (~125)

---

## Pending Decisions

- [ ] Learning rate for fine-tuning phase
- [ ] Data augmentation parameters (rotation range, zoom range)
- [ ] Custom head architecture (number of dense layers, neurons)
