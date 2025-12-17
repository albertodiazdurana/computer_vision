# CIFAR-10 Classification - Presentation Guide

**Duration:** ~15-20 minutes
**Format:** Live notebook walkthrough

---

## Pre-Presentation Checklist

- [ ] Kernel restarted and cleared
- [ ] GPU available (check first cell output)
- [ ] All saved figures in `outputs/figures/`
- [ ] Model files in `models/` (for quick restore if needed)

---

## Presentation Flow

### Opening (2 min)
**Cell:** Header markdown
- Project objective: Transfer learning with ResNet50 on CIFAR-10
- Key challenge: 32x32 images vs 224x224 pretrained model
- Success criteria: 65% target accuracy

### Section 1-2: Setup & Data (3 min)
**Key talking points:**
- GPU memory growth prevents OOM errors
- CIFAR-10: 10 classes, 32x32 RGB images
- Using 10K subset (course requirement)
- Stratified split maintains class balance

**Show:** Sample images grid, class distribution chart

### Section 3-4: Preprocessing & Augmentation (2 min)
**Key talking points:**
- Normalization to [0,1] for neural network stability
- Augmentation: horizontal flip, rotation, zoom
- Augmentation only during training (not inference)

**Show:** Augmentation examples figure

### Section 5: Model Architecture (2 min)
**Key talking points:**
- ResNet50 pretrained on ImageNet (1000 classes, 224x224)
- Custom head: Dropout -> Dense(256) -> Dropout -> Dense(10)
- Initially frozen: only 527K trainable params (2 MB)

**Show:** Model summary output

### Section 6: Phase 1 - Frozen Training (2 min)
**Key talking points:**
- **CRITICAL MOMENT:** Training failed! ~10% accuracy = random guessing
- Diagnosis: ResNet50 features don't transfer to 32x32 images
- Solution: Add Resizing layer (32->224)
- This is a real-world debugging scenario

**Show:** Phase 1 training curves (flat lines)

### Section 7: Phase 2 - Fine-tuning (3 min)
**Key talking points:**
- Unfreeze conv5 block: now 15.5M trainable params
- Lower LR (1e-4) to preserve pretrained weights
- 4.7x improvement: 10% -> 49% accuracy
- Still below 65% target (architecture mismatch)

**Show:** Phase 2 training curves (learning happening!)

### Section 8: Evaluation (2 min)
**Key talking points:**
- Test accuracy 48.9% (close to validation 49.0%)
- Consistent generalization
- Best classes: Ship (65% precision), Automobile (62% precision)
- Worst classes: Cat (21% F1), Bird (26% F1)

**Show:** Confusion matrix heatmap

### Section 9: Grad-CAM Interpretability (3 min)
**Key talking points:**
- Grad-CAM shows WHERE model looks to make decisions
- Correct predictions: focuses on object body
- Confused pairs: Cat/Dog similar shapes, Bird/Deer similar silhouettes
- Model is not a black box - we can explain its behavior

**Show:** Grad-CAM samples, confusion analysis

### Section 10: Conclusions (2 min)
**Key talking points:**
- Transfer learning requires input size compatibility
- Fine-tuning essential for domain adaptation
- Architecture choice matters more than hyperparameter tuning
- Clear path for improvement (different architecture, more data)

---

## Key Numbers to Remember

| Metric | Value |
|--------|-------|
| Test Accuracy | 48.9% |
| Target | 65% |
| Gap | 16.1 pp |
| Phase 1 Accuracy | 10.5% (random) |
| Phase 2 Improvement | 4.7x |
| Training Time | 77 min total |
| Model Parameters | 24.1M |

---

## Potential Questions & Answers

**Q: Why didn't you reach 65%?**
A: Architecture mismatch. ResNet50 expects 224x224 input; upscaling 32x32 introduces artifacts. A CIFAR-specific architecture (ResNet-20) would perform better.

**Q: Why use only 10K images?**
A: Course specification. Full 50K dataset would likely improve results.

**Q: What would you do differently?**
A: Use EfficientNetB0 or ResNet-20 designed for smaller images, or train from scratch on CIFAR-10.

**Q: Why is Cat/Dog confusion so high?**
A: Similar body shapes at 32x32 resolution. The model relies on texture/color which overlaps between these classes.

**Q: What is Grad-CAM actually computing?**
A: Gradients of the predicted class score with respect to the last convolutional layer, showing which regions influenced the prediction most.

---

## Emergency Recovery

If kernel crashes during demo:

1. Run cells 1-4 (setup, paths, data load, split)
2. Run cell 5 (normalize)
3. Skip to Section 9.1 and run the restore cell
4. Continue from Grad-CAM visualization

This skips training and loads saved state.

---

## Colab Compatibility Notes

The notebook auto-detects environment:
- **Local:** Uses `PROJECT_ROOT = computer_vision/`
- **Colab:** Uses `PROJECT_ROOT = /content/`

For Colab demo:
1. Upload notebook to Colab
2. Upload `models/` folder if showing restore functionality
3. Figures will be generated fresh (no need to upload)
