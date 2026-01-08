# Day 2 Checkpoint - CIFAR-10 Classification Project

**Date:** 2025-12-16
**Sprint:** 1
**Day:** 2

---

## Completed Tasks

- [x] Phase 1 training (frozen base) - failed as expected (10.3% accuracy)
- [x] Diagnosed issue: ResNet50 frozen features don't transfer to CIFAR-10
- [x] Added resize layer (32x32 → 224x224) to match ResNet50 input
- [x] Phase 2 training (fine-tuned conv5 block) - 30 epochs, 52 minutes
- [x] Model evaluation on test set
- [x] Confusion matrix and classification report
- [x] Training curves visualization

---

## Key Results

| Metric | Phase 1 (Frozen) | Phase 2 (Fine-tuned) |
|--------|------------------|----------------------|
| Best val_accuracy | 10.3% | 49.7% |
| Test accuracy | N/A | 48.6% |
| Training time | 16 min | 52 min |
| Epochs | 12 (early stop) | 30 (full) |

---

## Key Outputs

| Output | Location |
|--------|----------|
| Phase 1 training curves | outputs/figures/phase1_training_curves.png |
| Phase 2 training curves | outputs/figures/phase2_training_curves.png |
| Confusion matrix | outputs/figures/confusion_matrix.png |
| Best model (frozen) | models/resnet50_224_frozen_best.keras |
| Best model (fine-tuned) | models/resnet50_224_finetuned_best.keras |

---

## Model Summary

- **Architecture:** ResNet50 (conv5 unfrozen) + resize layer + custom head
- **Trainable params:** 15,503,114 (conv5 block + head)
- **Non-trainable params:** 8,611,712 (conv1-conv4 blocks)
- **Input:** 32x32 → resized to 224x224

---

## Per-Class Performance

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| airplane | 0.52 | 0.59 | 0.55 |
| automobile | 0.65 | 0.59 | 0.62 |
| bird | 0.33 | 0.21 | 0.26 |
| cat | 0.29 | 0.18 | 0.23 |
| deer | 0.46 | 0.25 | 0.32 |
| dog | 0.40 | 0.58 | 0.47 |
| frog | 0.49 | 0.71 | 0.58 |
| horse | 0.50 | 0.61 | 0.55 |
| ship | 0.68 | 0.51 | 0.59 |
| truck | 0.48 | 0.61 | 0.54 |

**Best:** Ship (68% precision), Automobile (62% F1)
**Worst:** Cat (23% F1), Bird (26% F1)

---

## Decisions Made

- DEC-004: Added Resizing layer to upscale 32x32 → 224x224 for ResNet50 compatibility
- DEC-005: Unfroze conv5 block only (balance between learning capacity and overfitting)
- DEC-006: Used lower learning rate (1e-4) for fine-tuning to preserve pretrained weights

---

## Observations

1. **Frozen transfer learning failed** - ImageNet features don't discriminate CIFAR-10 classes
2. **Fine-tuning essential** - 5x improvement after unfreezing conv5 block
3. **Validation variance high** - oscillated between 30-50%, indicates overfitting on small dataset
4. **Class confusion patterns** - Cat/Dog, Bird/Deer, Truck/Automobile commonly confused

---

## Gap Analysis

| Target | Achieved | Gap |
|--------|----------|-----|
| 65% test accuracy | 48.6% | -16.4% |

**Root causes:**
- Small training set (8K vs 50K full CIFAR-10)
- Architecture mismatch (ResNet50 designed for 224x224, not upscaled 32x32)
- Interpolation artifacts from 7x upscaling

---

## Tomorrow's Priority

1. Implement Grad-CAM visualization (Section 10)
2. Analyze failure cases
3. Document findings for presentation

---

## Notes

- Training time: Phase 1 (16 min) + Phase 2 (52 min) = 68 min total
- GPU: Quadro T1000, ~93% utilization during training
- Model weights restored from epoch 28 (best validation accuracy)
