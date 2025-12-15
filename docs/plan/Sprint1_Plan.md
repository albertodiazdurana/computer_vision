# CIFAR-10 Image Classification Project Plan
## ResNet50 Transfer Learning

**Project:** Computer Vision with Deep Learning - CIFAR-10 Classification
**Duration:** 5 Days
**Author:** Alberto Diaz Durana
**Created:** 2025-12-14

---

## 1. Purpose

**Objective:** Build an image classification model for CIFAR-10 dataset using ResNet50 with transfer learning, incorporating advanced CV techniques.

**Business Value:** Demonstrate proficiency in:
- Transfer learning with pre-trained CNNs
- Fine-tuning strategies for image classification
- Data augmentation for improved generalization
- Model interpretability (Grad-CAM)
- Comprehensive model evaluation

**Deliverable:** Single Google Colab-compatible Jupyter notebook developed incrementally throughout the project.

**Resources:**
- Development: Local machine (WSL2 Ubuntu 22.04)
- GPU: NVIDIA Quadro T1000 (4GB VRAM, ~2.2GB available)
- Time: ~4-6 hours/day over 5 days

---

## 2. Inputs & Dependencies

### Dataset
| Attribute | Value |
|-----------|-------|
| Name | CIFAR-10 |
| Total Images | 60,000 (32x32 RGB) |
| Classes | 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) |
| Training Subset | 10,000 images (project limit) |
| Test Set | 10,000 images |

### Technical Environment
| Component | Specification |
|-----------|---------------|
| OS | WSL2 Ubuntu 22.04 |
| Python | 3.11 |
| TensorFlow | 2.20.0 (GPU) |
| GPU | NVIDIA Quadro T1000 |
| VRAM | 4GB (2.2GB available) |
| RAM | 16GB |

### Key Dependencies
```python
tensorflow>=2.20.0
numpy
matplotlib
scikit-learn  # For classification metrics
```

### Constraints
- Limited VRAM: Batch size optimization required (recommend 16-32)
- Notebook must be Colab-compatible (no hardcoded local paths)
- Training limited to 10 epochs per phase (frozen + fine-tuning)

### Validation Strategy

**Approach:** Train/Val/Test Split (stratified)

**Data Allocation:**
- Training: 8,000 images (80% of 10K subset)
- Validation: 2,000 images (20% of 10K subset)
- Test (Holdout): 10,000 images (CIFAR-10 provided test set)

**Stratification:** Yes (maintain class distribution across splits)

**Random Seed:** 42 (for reproducibility)

**Rationale:**
- CIFAR-10 provides a separate test set - use as final holdout
- Split training subset for validation during training
- Stratification ensures all 10 classes represented proportionally
- 80/20 split balances training data volume with validation reliability

**Implementation:**
```python
from sklearn.model_selection import train_test_split

# CIFAR-10 provides test set separately
(X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()

# Subset to 10,000 training images
X_train_full = X_train_full[:10000]
y_train_full = y_train_full[:10000]

# Split for validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    stratify=y_train_full,
    random_state=42
)

# Final shapes:
# X_train: (8000, 32, 32, 3) - for training
# X_val: (2000, 32, 32, 3) - for validation during training
# X_test: (10000, 32, 32, 3) - holdout for final evaluation only
```

---

## 3. Execution Timeline

**Sprint Configuration:** 5-day project mapped to 4 methodology phases

| Day | Phase | Focus | Hours |
|-----|-------|-------|-------|
| 1 | 0-1 | Setup + Data Exploration | 4-5h |
| 2 | 2 | Preprocessing + Data Augmentation + Model Architecture | 4-5h |
| 3 | 3a | Training (Frozen Layers) with LR Scheduling | 4-5h |
| 4 | 3b | Fine-tuning + Evaluation + Grad-CAM | 4-5h |
| 5 | 4 | Documentation + Presentation | 4-5h |

**Total Estimated:** 20-25 hours

### Advanced Techniques Integration
| Technique | Day | Integration Point |
|-----------|-----|-------------------|
| Data Augmentation | 2 | Preprocessing layer in model |
| Learning Rate Scheduling | 3-4 | Training callbacks |
| Confusion Matrix | 4 | Evaluation section |
| Class-wise Metrics | 4 | Evaluation section |
| Grad-CAM | 4 | Interpretability section |

---

## 4. Detailed Deliverables

### Day 1 - Setup & Data Exploration (Phase 0-1)
**Goal:** Environment ready, data loaded and understood

#### Part 0: Environment Setup (1h)
**Objective:** Configure local development environment
**Activities:**
- Create virtual environment with Python 3.11
- Install TensorFlow 2.20.0 with GPU support
- Verify GPU detection
- Create notebook file
**Deliverables:**
- Working environment with GPU access
- `cifar10_classification.ipynb` created

#### Part 1: Data Loading (1h)
**Objective:** Load and subset CIFAR-10 data
**Activities:**
- Import CIFAR-10 from `tensorflow.keras.datasets`
- Limit training data to 10,000 samples
- Verify data shapes and types
**Deliverables:**
- Data loaded into notebook
- Shape verification: `(10000, 32, 32, 3)` training

#### Part 2: Data Exploration (2h)
**Objective:** Understand dataset characteristics
**Activities:**
- Visualize sample images per class
- Check class distribution
- Document image properties (RGB, 32x32)
**Deliverables:**
- Class distribution visualization
- Sample image grid (10 classes)

---

### Day 2 - Preprocessing, Augmentation & Model Architecture (Phase 2)
**Goal:** Data preprocessed with augmentation, model architecture defined

#### Part 0: Data Preprocessing (1h)
**Objective:** Prepare data for ResNet50
**Activities:**
- Normalize images to [0, 1] range
- Convert labels to one-hot encoding
- Verify input shape compatibility (32x32x3)
**Deliverables:**
- Preprocessed train/test arrays
- Labels one-hot encoded (10 classes)

#### Part 1: Data Augmentation Layer (1h)
**Objective:** Add augmentation for better generalization
**Activities:**
- Implement augmentation using `tf.keras.layers`:
  - `RandomFlip("horizontal")`
  - `RandomRotation(0.1)`
  - `RandomZoom(0.1)`
- Add as preprocessing layers in model
**Deliverables:**
- Data augmentation pipeline integrated
- Augmentation examples visualized

#### Part 2: Base Model Setup (1h)
**Objective:** Configure ResNet50 with frozen weights
**Activities:**
- Load ResNet50 with ImageNet weights
- Set `include_top=False`, `input_shape=(32, 32, 3)`
- Freeze all base model layers
**Deliverables:**
- ResNet50 base model loaded
- Layers frozen (`base_model.trainable = False`)

#### Part 3: Custom Head Architecture (1h)
**Objective:** Build classification head with augmentation
**Activities:**
- Build Sequential model:
  - Input layer
  - Data augmentation layers
  - ResNet50 base
  - GlobalAveragePooling2D
  - Dense(128, relu) -> Dense(64, relu)
  - Output(10, softmax)
- Compile model (Adam optimizer, categorical crossentropy)
**Deliverables:**
- Complete model architecture with augmentation
- Model summary documented

---

### Day 3 - Training Frozen Layers (Phase 3a)
**Goal:** Train custom head with frozen base, LR scheduling

#### Part 0: Training Configuration (45min)
**Objective:** Set up training parameters and callbacks
**Activities:**
- Configure batch size (16-32 for VRAM constraints)
- Set up callbacks:
  - `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)`
  - Optional: `EarlyStopping`
- Define validation split or use test set
**Deliverables:**
- Training configuration documented
- Callbacks defined

#### Part 1: Initial Training (2.5h)
**Objective:** Train model head for 10 epochs
**Activities:**
- Train with frozen base layers
- Monitor loss and accuracy per epoch
- Observe LR scheduling behavior
- Track training/validation metrics
**Deliverables:**
- Training history (10 epochs)
- Metrics: accuracy, loss curves
- LR schedule visualization

#### Part 2: Initial Evaluation (1h)
**Objective:** Assess frozen-layer performance
**Activities:**
- Evaluate on test set
- Document baseline accuracy
- Identify potential issues
**Deliverables:**
- Baseline test accuracy recorded
- Training curves plotted

---

### Day 4 - Fine-tuning, Evaluation & Interpretability (Phase 3b)
**Goal:** Fine-tune model, comprehensive evaluation with Grad-CAM

#### Part 0: Unfreeze & Fine-tune (30min)
**Objective:** Prepare for fine-tuning
**Activities:**
- Set `base_model.trainable = True`
- Recompile with lower learning rate (1e-5)
- Keep `ReduceLROnPlateau` callback
**Deliverables:**
- Model ready for fine-tuning

#### Part 1: Fine-tuning Training (2h)
**Objective:** Train entire model for 10 epochs
**Activities:**
- Fine-tune all layers
- Monitor for overfitting
- Track metric improvements
**Deliverables:**
- Fine-tuning history (10 epochs)
- Combined training curves

#### Part 2: Comprehensive Evaluation (1.5h)
**Objective:** Full model assessment with metrics
**Activities:**
- Final test set evaluation
- Generate confusion matrix visualization
- Calculate class-wise metrics:
  - Precision, Recall, F1-score per class
  - Classification report
- Identify best/worst performing classes
**Deliverables:**
- Final test accuracy
- Confusion matrix heatmap
- Classification report table
- Performance comparison (frozen vs fine-tuned)

#### Part 3: Grad-CAM Visualization (1h)
**Objective:** Model interpretability
**Activities:**
- Implement Grad-CAM for last conv layer
- Generate heatmaps for sample predictions
- Visualize correct and incorrect predictions
- Show what model "sees" per class
**Deliverables:**
- Grad-CAM implementation
- Visualization grid (6-10 samples)
- Interpretation notes

---

### Day 5 - Documentation & Presentation (Phase 4)
**Goal:** Notebook polished, presentation ready

#### Part 0: Notebook Cleanup (2h)
**Objective:** Professional notebook quality
**Activities:**
- Add markdown headers and explanations
- Remove debug/test cells
- Ensure sequential execution works
- Verify Colab compatibility
- Add table of contents
**Deliverables:**
- Clean, well-documented notebook

#### Part 1: Results Documentation (1.5h)
**Objective:** Document findings and decisions
**Activities:**
- Summarize model architecture choices
- Document training results
- Explain augmentation impact
- Note challenges and solutions
- Highlight Grad-CAM insights
**Deliverables:**
- Results section in notebook
- Key findings documented

#### Part 2: Presentation Preparation (1.5h)
**Objective:** Prepare for live presentation
**Activities:**
- Outline presentation flow:
  1. Problem & Dataset
  2. Model Architecture (ResNet50 + augmentation)
  3. Training Process (frozen -> fine-tuned)
  4. Results (confusion matrix, class metrics)
  5. Interpretability (Grad-CAM)
  6. Conclusions
- Identify key visualizations to highlight
**Deliverables:**
- Presentation outline
- Ready for live demo

---

## 5. Readiness Checklist

### Phase 1 -> Phase 2 (Day 1 -> Day 2)
- [ ] GPU detected and functional
- [ ] CIFAR-10 data loaded (10,000 train, 10,000 test)
- [ ] Data shapes verified
- [ ] Sample visualizations complete

### Phase 2 -> Phase 3 (Day 2 -> Day 3)
- [ ] Data normalized to [0, 1]
- [ ] Labels one-hot encoded
- [ ] Data augmentation layers implemented
- [ ] ResNet50 loaded with frozen layers
- [ ] Custom head attached
- [ ] Model compiles without errors

### Phase 3a -> Phase 3b (Day 3 -> Day 4)
- [ ] 10 epochs training complete (frozen)
- [ ] Baseline accuracy recorded
- [ ] LR scheduling working
- [ ] No memory errors during training
- [ ] Training curves saved

### Phase 3b -> Phase 4 (Day 4 -> Day 5)
- [ ] Fine-tuning complete (10 epochs)
- [ ] Final accuracy recorded
- [ ] Confusion matrix generated
- [ ] Class-wise metrics calculated
- [ ] Grad-CAM visualizations complete

---

## 6. Success Criteria

### Quantitative
| Metric | Target | Minimum |
|--------|--------|---------|
| Test Accuracy (frozen) | >50% | >40% |
| Test Accuracy (fine-tuned) | >65% | >55% |
| Training completes | 20 epochs total | 20 epochs |
| Notebook cells | All execute sequentially | No errors |

### Qualitative
- [ ] Notebook readable by non-author
- [ ] Architecture decisions explained
- [ ] Augmentation impact discussed
- [ ] Grad-CAM insights documented
- [ ] Results interpretable
- [ ] Colab-compatible (no local path dependencies)

### Technical
- [ ] No VRAM overflow errors
- [ ] Reproducible (random seeds set)
- [ ] Clean code (no debug cells)
- [ ] All visualizations render correctly

---

## 7. Documentation & Ownership

### Primary Deliverable
**File:** `cifar10_classification.ipynb`
**Location:** Project root (Colab-compatible)
**Development:** Incremental over 5 days

### Notebook Structure (Target)
1. Introduction & Objectives
2. Environment Setup & Imports
3. Data Loading & Exploration
4. Data Preprocessing
5. Data Augmentation
6. Model Architecture (ResNet50 + Custom Head)
7. Training Phase 1 (Frozen Layers + LR Scheduling)
8. Training Phase 2 (Fine-tuning)
9. Evaluation & Metrics
   - Confusion Matrix
   - Classification Report (Precision/Recall/F1)
10. Model Interpretability (Grad-CAM)
11. Conclusions

### Version Control
- Daily commits recommended
- Commit messages: `Day X: [description]`

### Decision Log
Document key decisions in notebook markdown cells:
- Batch size selection (VRAM constraint)
- Learning rate choices
- Augmentation parameters
- Architecture decisions (hidden layer sizes)

---

## 8. Daily Documentation Protocol

Reference: DSM_1.0 Section 6.1.4

### End of Notebook Checklist
Before closing the notebook each day:

**Data Outputs:**
- [ ] All processed data/models saved to appropriate location
- [ ] Key metrics printed for verification

**Code Quality:**
- [ ] All cells execute without errors (Kernel > Restart & Run All)
- [ ] No hardcoded paths
- [ ] Temporary/debugging cells removed or marked

**Documentation:**
- [ ] Summary section completed with key findings
- [ ] Next steps clearly stated
- [ ] Any new decisions logged (DEC-XXX format)

**Validation:**
- [ ] Results make sense (accuracy reasonable, no anomalies)
- [ ] Checkpoint saved

### Notebook Summary Cell Template
Add as final cell after each working session:

```python
# ============================================================
# NOTEBOOK SUMMARY - Day X
# ============================================================
#
# Completed: [Brief description of what was accomplished]
#
# Key Outputs:
# - [model/metrics]: [description]
#
# Key Findings:
# - [Finding 1]
# - [Finding 2]
#
# Decisions Made:
# - DEC-XXX: [Brief description]
#
# Next Steps:
# - [Next task]
#
# ============================================================
```

### Daily Checkpoint Files
**Location:** `docs/checkpoints/s01_dXX_checkpoint.md`

```markdown
# Daily Checkpoint - Sprint 1 Day X

**Date:** YYYY-MM-DD
**Hours Worked:** Xh
**Cumulative Sprint Hours:** Xh

## Completed Today
- [ ] [Task 1]
- [ ] [Task 2]

## Notebook Progress
| Section | Status | Key Output |
|---------|--------|------------|
| [Section name] | Complete/In Progress | [output] |

## Decisions Made
- DEC-XXX: [Brief summary]

## Blockers/Issues
- [Issue and resolution or status]
- None (if no blockers)

## Tomorrow's Priority
1. [First priority task]
2. [Second priority task]

## Notes
[Any context needed for continuity]
```

### End of Day Protocol (5 min)
1. **Save all work** - Notebook saved, git commit if applicable
2. **Update checkpoint file** - Create `docs/checkpoints/s01_d0X_checkpoint.md`
3. **Update decision log** - Add any new DEC-XXX entries
4. **Prepare next day** - Identify first task for tomorrow

---

## Quick Reference

### Key Imports
```python
# Core
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Evaluation
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Grad-CAM
import numpy as np
import matplotlib.pyplot as plt
```

### Data Augmentation Layer
```python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])
```

### LR Scheduling Callback
```python
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)
```

### Grad-CAM Template
```python
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()
```

### VRAM Management
- Batch size: Start with 32, reduce to 16 if OOM
- Clear session between experiments: `tf.keras.backend.clear_session()`

### Colab Compatibility
- Use relative paths only
- Include `!pip install` cells for dependencies
- Test full notebook execution before submission

### Class Names Reference
```python
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
```
