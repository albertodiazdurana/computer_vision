# Sprint 2 Improvements - CIFAR-10 Classification Project

**Purpose:** Document methodological improvements identified during Sprint 1 for implementation in Sprint 2 (extracurricular exercise).

---

## Day 1 Critical Assessment

### Architecture Concerns

| Issue | Sprint 1 Approach | Sprint 2 Improvement |
|-------|-------------------|----------------------|
| ResNet50 overkill | Use ResNet50 (course requirement) | Use CIFAR-appropriate architecture (ResNet20/32, EfficientNetB0) |
| Input size mismatch | Native 32x32 (suboptimal for ResNet50) | Either resize to 224x224 or use architecture designed for 32x32 |
| No baseline | Direct transfer learning | Add simple CNN baseline for comparison |

### Data Utilization

| Issue | Sprint 1 Approach | Sprint 2 Improvement |
|-------|-------------------|----------------------|
| Underutilized data | 10K subset (8K train, 2K val) | Use full 50K training set (40K train, 10K val) |
| Justification | Per course specification | Justify data size or use full dataset |

### Evaluation Gaps

| Issue | Sprint 1 Approach | Sprint 2 Improvement |
|-------|-------------------|----------------------|
| No baselines | Single model evaluation | Compare against simple CNN baseline |
| No ablations | Standard pipeline | Ablation study for augmentation impact |
| Statistical rigor | Single run | Multiple runs with confidence intervals |

### Interpretability Limitations

| Issue | Sprint 1 Approach | Sprint 2 Improvement |
|-------|-------------------|----------------------|
| Grad-CAM resolution | Apply to 32x32 (coarse) | Acknowledge limitations; add failure case analysis |
| Depth of analysis | Basic visualization | Class-wise comparison, training phase comparison |

---

## Sprint 2 Proposed Structure

1. **Baseline CNN** - Simple 3-4 layer CNN trained from scratch
2. **CIFAR-appropriate architecture** - ResNet20 or EfficientNetB0
3. **Full dataset utilization** - 50K training images
4. **Ablation studies** - Augmentation on/off, architecture comparison
5. **Statistical robustness** - Multiple seeds, confidence intervals
6. **Enhanced interpretability** - Failure case analysis, class-wise Grad-CAM

---

## Day 2 Critical Assessment

*(To be added after Day 2)*

---

## Day 3 Critical Assessment

*(To be added after Day 3)*

---

## Day 4 Critical Assessment

*(To be added after Day 4)*

---

## Day 5 Critical Assessment

*(To be added after Day 5)*
