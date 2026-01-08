# Sprint 2 Plan: Production-Ready MLOps

## Overview

| Aspect | Detail |
|--------|--------|
| **Duration** | 2 weeks |
| **Goal** | Production-ready experiment tracking |
| **Primary Tool** | MLflow |
| **Success Metric** | All experiments reproducible and tracked |

---

## Week 1: MLflow Foundation

### Day 1-2: Setup & Configuration

**Tasks:**
- [ ] Install MLflow locally (`pip install mlflow`)
- [ ] Create experiment structure in notebook
- [ ] Define tracking schema (params, metrics, artifacts)
- [ ] Add MLflow to requirements in README

**Files to modify:**
- `notebooks/cifar10_classification.ipynb` - Add Cell 1a after imports
- `README.md` - Add mlflow to requirements

**Config schema to implement:**
```python
CONFIG = {
    # Data
    "train_size": 8000,
    "val_size": 2000,
    "batch_size": 32,
    "random_seed": 42,

    # Model
    "base_model": "ResNet50",
    "input_shape": (32, 32, 3),
    "num_classes": 10,
    "dropout_rate": 0.3,
    "dense_units": 256,

    # Phase 1
    "phase1_epochs": 20,
    "phase1_lr": 1e-3,

    # Phase 2
    "phase2_epochs": 30,
    "phase2_lr": 1e-4,
    "unfreeze_from": "conv5"
}
```

---

### Day 3-4: Notebook Integration

**Tasks:**
- [ ] Add MLflow imports to Cell 2
- [ ] Create experiment initialization cell
- [ ] Modify `load_or_train()` to log params/metrics
- [ ] Add MLflow callback for epoch-level metrics

**Integration points:**
| Cell | Notebook Idx | Integration |
|------|--------------|-------------|
| Cell 2 | 4 | Add MLflow imports, modify `load_or_train()` |
| Cell 12 | 24 | Replace hardcoded values with CONFIG dict |
| Cell 13 | 25 | Wrap Phase 1 initial in MLflow run |
| Cell 15 | 29 | Wrap Phase 1 retry in MLflow run |
| Cell 18 | 36 | Wrap Phase 2 in MLflow run |
| Cell 20 | 41 | Log test metrics |

**Code pattern:**
```python
import mlflow
import mlflow.keras

mlflow.set_experiment("cifar10-classification")

def load_or_train_with_mlflow(model_path, history_path, train_fn, config, description):
    with mlflow.start_run(run_name=description):
        mlflow.log_params(config)
        model, history, training_time = load_or_train(...)
        mlflow.log_metrics({
            "best_val_accuracy": max(history.history['val_accuracy']),
            "final_val_loss": history.history['val_loss'][-1],
            "training_time_min": training_time / 60
        })
        return model, history, training_time
```

---

### Day 5: Artifact Logging

**Tasks:**
- [ ] Log model in SavedModel format (portable)
- [ ] Log all figures as artifacts
- [ ] Log training history as JSON
- [ ] Test artifact retrieval

**Artifact structure:**
```
mlruns/
└── <experiment_id>/
    └── <run_id>/
        ├── params/
        ├── metrics/
        └── artifacts/
            ├── model/
            ├── figures/
            └── history.json
```

---

## Week 2: Production Hardening

### Day 6-7: Model Versioning & Portability

**Tasks:**
- [ ] Save model in SavedModel format (replaces .keras)
- [ ] Test loading in fresh Python environment
- [ ] Test loading in Colab
- [ ] Update `load_or_train()` to use SavedModel

**Code change:**
```python
# Instead of: model.save(path / 'model.keras')
tf.saved_model.save(model, str(path / 'saved_model'))

# Load with:
model = tf.saved_model.load(str(path / 'saved_model'))
```

---

### Day 8-9: Reproducibility & Config Management

**Tasks:**
- [ ] Create CONFIG dataclass at top of notebook
- [ ] Replace all hardcoded values with CONFIG references
- [ ] Add comprehensive seeding (numpy, tf, python random)
- [ ] Log environment info (TF version, GPU)

**Seed management:**
```python
import random
import numpy as np
import tensorflow as tf

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds(CONFIG["random_seed"])
```

---

### Day 10: Documentation & Testing

**Tasks:**
- [ ] Update README with MLflow usage section
- [ ] Document how to view experiments (`mlflow ui`)
- [ ] Test full pipeline from scratch
- [ ] Verify Colab compatibility

---

## Success Criteria

| Criteria | Verification |
|----------|--------------|
| All runs tracked | `mlflow ui` shows experiments |
| Params logged | Can filter runs by hyperparameters |
| Metrics logged | Can compare val_accuracy across runs |
| Artifacts saved | Can download model, figures, history |
| Model portable | Loads successfully in Colab |
| Reproducible | Same config → same results (±1%) |

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `notebooks/cifar10_classification.ipynb` | Add MLflow integration (Cells 2, 12, 13, 15, 18, 20) |
| `README.md` | Add mlflow to requirements, add MLflow section |
| `.gitignore` | Add `mlruns/` |

---

## Sprint 3 Preview

After Sprint 2, ready for:
- **Streamlit app** for interactive classification
- Image upload/selection UI
- Real-time prediction with Grad-CAM overlay
- Model loaded from MLflow registry or SavedModel
