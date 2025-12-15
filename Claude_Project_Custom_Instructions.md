# Project: CIFAR-10 Image Classification
Domain: Computer Vision

## Framework Documents
- **PM Guidelines** (Project Knowledge): Project planning structure
- **Collaboration Methodology v1.1.1** (Project Knowledge): Execution workflow
- **DSM_0 Complete Guide** (Project Knowledge): System overview

## Project Planning Context

### Scope
- **Purpose**: Build CIFAR-10 classifier using ResNet50 transfer learning with advanced CV techniques
- **Resources**: 5 days, single developer, local GPU (NVIDIA Quadro T1000, 2.2GB available VRAM)
- **Success Criteria**:
  - Quantitative: Test accuracy >65% (fine-tuned), training completes without OOM
  - Qualitative: Clear demonstration of transfer learning workflow
  - Technical: Single Colab-compatible notebook, reproducible execution

### Data & Dependencies
- **Primary dataset**: CIFAR-10 (10,000 train subset, 10,000 test), 32x32 RGB
- **Dependencies**: TensorFlow 2.20.0 (GPU), scikit-learn
- **Data quality**: Clean dataset, no preprocessing issues expected

### Validation Strategy
- **Approach**: Train/Val/Test Split (stratified)
- **Training**: 8,000 images (80% of 10K subset)
- **Validation**: 2,000 images (20% of 10K subset)
- **Test (Holdout)**: 10,000 images (CIFAR-10 provided test set)
- **Random Seed**: 42 (reproducibility)

### Stakeholders
- **Primary**: Course instructors - need working notebook + live presentation
- **Communication**: Final presentation (live demo)

## Execution Context

### Timeline & Phases
- **Duration**: 5 days (single sprint)
- **Day 1 (Phase 0-1)**: Setup + Data Exploration
- **Day 2 (Phase 2)**: Preprocessing + Data Augmentation + Model Architecture
- **Day 3 (Phase 3a)**: Training frozen layers + LR scheduling
- **Day 4 (Phase 3b)**: Fine-tuning + Evaluation + Grad-CAM
- **Day 5 (Phase 4)**: Documentation + Presentation prep

### Deliverables
- [x] Notebook: 1 (cifar10_classification.ipynb, Colab-compatible)
- [ ] Daily checkpoints: docs/checkpoints/s01_d0X_checkpoint.md
- [ ] Presentation: Live demo walkthrough

## Domain Adaptations

### Key Techniques (Reference: Appendix D.3 Computer Vision)
- Transfer learning with ResNet50 (ImageNet weights)
- Data augmentation (RandomFlip, RandomRotation, RandomZoom)
- Learning rate scheduling (ReduceLROnPlateau)
- Grad-CAM for model interpretability
- Confusion matrix + class-wise metrics (Precision/Recall/F1)

### Known Challenges
- Limited VRAM (2.2GB) -> Batch size 16-32, clear session between experiments
- Small input size (32x32) vs ResNet expectations -> Use native size, no resize

### Solved Challenges
- [Update as project progresses]

## Advanced Practices

- [ ] Experiment Tracking
- [ ] Hypothesis Management
- [x] Performance Baseline (frozen vs fine-tuned comparison)
- [ ] Ethics & Bias Review
- [ ] Testing Strategy
- [ ] Data Versioning
- [ ] Technical Debt Register
- [ ] Scalability Considerations
- [ ] Literature Review
- [ ] Risk Management

## Communication & Style

### Artifact Generation
- Confirm understanding before generating artifacts
- Progressive execution: cell-by-cell development
- Single notebook structure (~400 lines target per day, 11 sections)
- Follow Daily Documentation Protocol (Section 6.1.4)

### Environment
- OS: WSL2 Ubuntu 22.04
- Python: 3.11
- TensorFlow: 2.20.0 (GPU)
- IDE: VS Code with Jupyter extension

### Standards (CRITICAL)
- No emojis - use "OK:", "WARNING:", "ERROR:"
- Show actual values (shapes, accuracy, loss)
- Markdown before each code cell
- Each cell must show visible output

### Language & Formatting
- Primary language: English
- Numbers: US format (1,234.56)
- Dates: YYYY-MM-DD

## Project-Specific Requirements
- Notebook must be Colab-compatible (no hardcoded local paths)
- Memory management critical (batch size, session clearing)
- Include Grad-CAM visualizations for presentation impact
- Daily checkpoint files in docs/checkpoints/