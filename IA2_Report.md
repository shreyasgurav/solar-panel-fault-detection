# Deep Learning Internal Assessment 2 — Report
## Solar Panel Fault Detection using Deep Learning

---

### Student Details

| | Name | Roll No |
|---|---|---|
| Student 1 | [NAME] | [ROLL NO] |
| Student 2 | [NAME] | [ROLL NO] |

**Course:** Deep Learning | **Semester:** 6 | **Branch:** AI & Data Science

**Submission Date:** April 15, 2026

---

## 1. Problem Statement

Solar panels are a vital renewable energy source, but their efficiency degrades due to environmental and physical factors such as dust accumulation, bird droppings, snow coverage, electrical faults, and physical damage. Manual inspection of large solar farms is time-consuming, expensive, and error-prone.

**Objective:** Build a deep learning-based image classifier that automatically detects and classifies solar panel faults into 6 categories:
1. Clean
2. Dusty
3. Bird-drop
4. Electrical-damage
5. Physical-Damage
6. Snow-Covered

This enables automated, scalable monitoring of solar installations to reduce maintenance costs and maximize energy output.

---

## 2. Dataset Description

- **Source:** [Kaggle — Solar Panel Images](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images)
- **Total Images:** ~876
- **Classes:** 6

| Class | Number of Images | Description |
|---|---|---|
| Bird-drop | ~198 | Bird droppings causing localized shading |
| Clean | ~193 | Normal, functioning panels |
| Dusty | ~190 | Dust-covered panels reducing light absorption |
| Snow-Covered | ~123 | Snow blocking sunlight completely |
| Electrical-damage | ~103 | Hot spots, burnt cells, wiring issues |
| Physical-Damage | ~69 | Cracks, broken glass, structural damage |

**Key Challenge:** Class imbalance — Physical-Damage has nearly 3x fewer samples than Bird-drop.

---

## 3. Preprocessing & Augmentation

### Preprocessing Steps
| Step | Details | Justification |
|---|---|---|
| Resize | 224×224 pixels | Standard input size for CNN and transfer learning models |
| Normalize | ImageNet mean/std (μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225]) | EfficientNetB0 was pretrained on ImageNet; matching statistics ensures better feature extraction |
| Split | 70% train / 15% val / 15% test (stratified) | Stratified split maintains class proportions in each subset |
| Class Weights | Inversely proportional to class frequency | Penalizes misclassification of minority classes more during training |
| Weighted Sampling | Oversamples minority classes in training | Ensures balanced class representation in each training batch |

### Data Augmentation (Training Set Only)
| Augmentation | Parameters | Justification |
|---|---|---|
| Random Horizontal Flip | p=0.5 | Panels can appear from either direction; doesn't change fault type |
| Random Vertical Flip | p=0.3 | Different camera orientations; adds rotational invariance |
| Random Rotation | ±20° | Simulates slight camera angle variations in real-world captures |
| Random Resized Crop | scale=(0.8, 1.0) | Simulates zoom variation and different framing |
| Color Jitter | brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1 | Accounts for different lighting, weather, and time of day |
| Random Affine | translate=(0.1, 0.1), scale=(0.9, 1.1) | Simulates camera movement and varying distances |

---

## 4. Model Architecture

We implemented and compared **two models**:

### Model A: Custom CNN (Built from Scratch)

```
Input (3×224×224)
    ↓
Conv2D(32) → BatchNorm → ReLU → MaxPool2D  →  32×112×112
    ↓
Conv2D(64) → BatchNorm → ReLU → MaxPool2D  →  64×56×56
    ↓
Conv2D(128) → BatchNorm → ReLU → MaxPool2D →  128×28×28
    ↓
Conv2D(256) → BatchNorm → ReLU → MaxPool2D →  256×14×14
    ↓
Global Average Pooling → 256
    ↓
FC(512) → ReLU → Dropout(0.5)
    ↓
FC(256) → ReLU → Dropout(0.5)
    ↓
FC(6) → Output (6 classes)
```

**Design Rationale:**
- Increasing filter sizes (32→64→128→256) capture features at multiple scales
- BatchNorm stabilizes training and acts as mild regularization
- Global Average Pooling reduces parameters compared to flattening
- Dropout (0.5) provides strong regularization for our small dataset

### Model B: EfficientNetB0 (Transfer Learning)

```
Input (3×224×224)
    ↓
EfficientNetB0 Backbone (pretrained on ImageNet, 1.2M images)
    ↓
Global Average Pooling → 1280
    ↓
Dropout(0.4) → FC(256) → ReLU → Dropout(0.4)
    ↓
FC(6) → Output (6 classes)
```

**Training Strategy:**
- **Phase 1 (Feature Extraction):** Freeze all backbone layers, train only the classifier head (10 epochs, lr=0.001)
- **Phase 2 (Fine-Tuning):** Unfreeze top 3 blocks, continue training with lower learning rate (15 epochs, lr=0.0001)

**Architecture Diagram:** See `architecture_diagram.png`

---

## 5. Training Configuration

| Parameter | Value | Purpose |
|---|---|---|
| Loss Function | CrossEntropyLoss (weighted) | Multi-class classification with class imbalance handling |
| Optimizer | Adam (weight_decay=1e-4) | Adaptive learning rate + L2 regularization |
| Learning Rate | 0.001 (CNN), 0.001→0.0001 (EfficientNet) | Standard for Adam; lower for fine-tuning |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3) | Reduces LR when validation loss plateaus |
| Early Stopping | patience=5 | Prevents overfitting by stopping when no improvement |
| Batch Size | 32 | Balance between gradient stability and memory |
| Regularization | Dropout (0.5/0.4), L2 (weight_decay=1e-4), BatchNorm | Multiple regularization techniques to combat overfitting |

---

## 6. Results

### Overall Performance Comparison

| Metric | Custom CNN | EfficientNetB0 |
|---|---|---|
| Accuracy (%) | 54.96 | **84.73** |
| Precision — Macro (%) | 59.87 | **85.07** |
| Recall — Macro (%) | 56.35 | **86.32** |
| F1-Score — Macro (%) | 53.27 | **85.34** |


### Confusion Matrix
See `outputs/plots/confusion_matrix_comparison.png` for side-by-side confusion matrices of both models.

### Training Curves
- Loss curves: `outputs/plots/custom_cnn_loss_curve.png`, `outputs/plots/efficientnet_combined_curves.png`
- Accuracy curves: `outputs/plots/custom_cnn_accuracy_curve.png`

---

## 7. Conclusion

### Key Findings
1. **Transfer Learning significantly outperformed Custom CNN** — EfficientNetB0 achieved **84.73% accuracy** vs Custom CNN's **54.96%**, a ~30% improvement. Pretrained ImageNet features transferred effectively to solar panel fault detection.
2. **Class imbalance handling was critical** — Using class weights + weighted sampling improved performance on minority classes (Physical-Damage, Electrical-damage). Without these, the model would have ignored rare fault types.
3. **Two-phase fine-tuning improved further** — Unfreezing top 3 EfficientNet blocks and training with a 10x lower LR (0.0001) adapted high-level features to our domain while preserving pretrained knowledge.

### Limitations
1. Small dataset size (~876 images) limits generalization
2. Class imbalance (Physical-Damage has only ~69 images)
3. Visual similarity between some classes (Dusty vs Clean)
4. Varied image quality from internet-scraped data
5. Single-label classification (real panels may have multiple faults)

### Future Work
1. Collect more data, especially for underrepresented classes
2. GAN-based augmentation to synthesize realistic minority class images
3. Ensemble models combining predictions from multiple architectures
4. Object detection (YOLO/Faster R-CNN) to localize faults on panels
5. Multi-label classification for detecting concurrent faults
6. Attention mechanisms to focus on fault-specific regions
7. Deploy as a web application or mobile app for real-time inspection

---

## GitHub Repository

[INSERT GITHUB LINK HERE]

---
