# Sprint 2 Improvements - CIFAR-10 Classification Project

**Purpose:** Document methodological improvements identified during Sprint 1 for implementation in Sprint 2 (extracurricular exercise).

**Sprint 2 Primary Objective:** Production-ready workflow

---

## Production Readiness Goals

- [ ] Modular code structure (separate .py files for data, model, training, evaluation)
- [ ] Configuration management (YAML/JSON config files instead of hardcoded values)
- [ ] Reproducibility (seed management, environment pinning)
- [ ] Logging framework (replace print statements with proper logging)
- [ ] Unit tests for critical functions
- [ ] CI/CD pipeline considerations
- [ ] Model versioning and artifact management
- [ ] Documentation (docstrings, README, API docs)

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

### Training Strategy

| Issue | Sprint 1 Approach | Sprint 2 Improvement |
|-------|-------------------|----------------------|
| Frozen transfer failed | Assumed ImageNet features transfer | Validate transfer assumption before committing; consider from-scratch training |
| Single fine-tuning config | Unfroze conv5 only | Try unfreezing more layers (conv4+conv5), or progressive unfreezing |
| High validation variance | 30-50% oscillation | Add regularization (more dropout, weight decay), or use larger batch with accumulation |

### Learning Rate Strategy

| Issue | Sprint 1 Approach | Sprint 2 Improvement |
|-------|-------------------|----------------------|
| Fixed initial LR | Started at 1e-4 | Use learning rate finder to determine optimal starting LR |
| Aggressive reduction | factor=0.5, patience=3 | Consider cosine annealing or warmup + decay schedule |
| No differential LR | Same LR for all layers | Lower LR for early layers, higher for classifier head |

### Class Imbalance in Performance

| Issue | Sprint 1 Approach | Sprint 2 Improvement |
|-------|-------------------|----------------------|
| Poor cat/bird performance | No class-specific handling | Analyze failure cases; consider class-weighted loss |
| Confusion pairs | Cat↔Dog, Bird↔Deer | Add hard negative mining or focal loss |
| No per-class analysis | Single accuracy metric | Track per-class metrics during training |

### Target Accuracy Gap

| Issue | Sprint 1 Approach | Sprint 2 Improvement |
|-------|-------------------|----------------------|
| 65% target not met | Achieved 48.6% | Revise target based on constraints OR use full dataset + appropriate architecture |
| Upscaling artifacts | 32→224 bilinear interpolation | Use architecture designed for 32x32 (ResNet20-CIFAR, WideResNet) |
| Limited training data | 8K samples | Use full 50K CIFAR-10 training set |

### Key Learnings

1. **Frozen transfer learning does not work for CIFAR-10** - ImageNet features are too high-level for 32x32 images
2. **Fine-tuning is essential** - 5x improvement (10% → 50%) after unfreezing conv5
3. **Architecture matters more than tricks** - ResNet50 is fundamentally mismatched for this task
4. **Small dataset = high variance** - Need more data or stronger regularization

---

## Day 3 Critical Assessment

### Grad-CAM Implementation Challenges

| Issue | Sprint 1 Approach | Sprint 2 Improvement |
|-------|-------------------|----------------------|
| Keras 3.x nested models | Custom `make_gradcam_heatmap_v3()` with sublayer access | Use tf-keras-vis or captum library for standardized implementation |
| Coarse heatmap (7x7) | Upscale to 32x32 for visualization | Use earlier conv layers for finer resolution, or guided backprop |
| Single-class focus | Only visualize predicted class | Compare heatmaps for predicted vs true class on misclassifications |

### Interpretability Depth

| Issue | Sprint 1 Approach | Sprint 2 Improvement |
|-------|-------------------|----------------------|
| Limited confusion analysis | Visualized 2 samples per confused pair | Aggregate analysis: mean heatmap per class, attention statistics |
| No quantitative metrics | Visual inspection only | Add localization accuracy, IoU with object regions |
| Static analysis | Post-training only | Track attention evolution during training |

### Notebook Workflow

| Issue | Sprint 1 Approach | Sprint 2 Improvement |
|-------|-------------------|----------------------|
| Path resolution fragile | Fallback chain with hardcoded path | Use `__file__` or config-based paths; package structure |
| State restoration manual | Separate cell for loading saved state | Automatic checkpoint detection and loading |
| No version control for experiments | Single notebook | MLflow/Weights&Biases for experiment tracking |

### Code Quality

| Issue | Sprint 1 Approach | Sprint 2 Improvement |
|-------|-------------------|----------------------|
| Functions in notebook | All code in cells | Extract to `src/` modules (gradcam.py, visualization.py) |
| No error handling | Assumes success | Add try/except, validation, informative error messages |
| Magic numbers | Hardcoded batch size, epochs | Config file (YAML) for all hyperparameters |

### Key Learnings

1. **Keras 3.x breaking changes** - Nested model layer access requires `base_model.get_layer()` not `model.get_layer()`
2. **State management critical** - Saving model + predictions + history enables session recovery
3. **Path handling varies by execution context** - Notebook cwd differs from script execution
4. **Grad-CAM confirms model behavior** - Vehicle classes have cleaner attention than animal classes

---

## Day 4 Critical Assessment

### Training Reproducibility

| Issue | Sprint 1 Approach | Sprint 2 Improvement |
|-------|-------------------|----------------------|
| Result variance between runs | 44-49% test accuracy depending on run | Fix all random seeds (numpy, tf, python), use deterministic operations |
| No run logging | Manual observation of metrics | Log all hyperparameters and metrics per run to file/database |
| Training not resumable | Restart from scratch if interrupted | Implement proper checkpointing with optimizer state |

### Skip Logic Implementation

| Issue | Sprint 1 Approach | Sprint 2 Improvement |
|-------|-------------------|----------------------|
| Long re-run times (~80 min) | Added file-existence checks for saved models | Use proper experiment tracking (MLflow) with automatic caching |
| History object mocking | Created `HistoryMock` class for compatibility | Serialize full Keras callback objects or use standardized format |
| Scattered save logic | Save in training cell, restore in separate cell | Centralized checkpoint manager class |

### Git/Version Control

| Issue | Sprint 1 Approach | Sprint 2 Improvement |
|-------|-------------------|----------------------|
| Large files blocked push | 117 MB evaluation_state.npz exceeded GitHub limit | Use Git LFS from project start, or DVC for data versioning |
| Models in repo | Added models/ to .gitignore after issue | Plan artifact storage strategy upfront (cloud storage, LFS) |
| No experiment tracking | Manual checkpoint files | MLflow/W&B for automatic artifact versioning |

### Presentation Readiness

| Issue | Sprint 1 Approach | Sprint 2 Improvement |
|-------|-------------------|----------------------|
| Manual demo preparation | Created PRESENTATION_GUIDE.md with talking points | Automated demo script with checkpoints at each section |
| Emergency recovery manual | Documented restore steps | Single-command recovery script |
| No backup strategy | Manual file copy to external folder | Automated backup to cloud storage |

### Key Learnings

1. **Reproducibility requires explicit seeding** - Same code can produce 44-49% accuracy on different runs
2. **Large file management needs upfront planning** - GitHub's 100MB limit caught us after committing
3. **Skip logic improves iteration speed** - Re-running notebook takes seconds instead of 80 min
4. **Presentation preparation is project work** - Guide and recovery procedures are deliverables

---

## Day 5 Critical Assessment

*(To be added after Day 5)*
