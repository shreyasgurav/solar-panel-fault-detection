# Solar Panel Fault Detection - Image Classifier

## Deep Learning Internal Assessment 2 | Semester 6 | AI & Data Science

A deep learning project that classifies solar panel images into 6 fault categories using both a **Custom CNN** and **EfficientNetB0 (Transfer Learning)**.

---

## Classes
1. **Clean** — Normal, functioning panels
2. **Dusty** — Dust-covered panels
3. **Bird-drop** — Bird droppings on panels
4. **Electrical-damage** — Electrically damaged panels
5. **Physical-Damage** — Physically damaged panels
6. **Snow-Covered** — Snow-covered panels

---

## Project Structure

```
solar_panel_classifier/
├── data/
│   └── Faulty_solar_panel/    ← Dataset (6 class folders)
├── notebooks/
│   └── solar_panel_classifier.ipynb   ← Main Jupyter notebook
├── src/
│   ├── __init__.py
│   ├── preprocess.py      ← Preprocessing & augmentation
│   ├── model.py           ← Model architectures (Custom CNN + EfficientNetB0)
│   ├── train.py           ← Training loop with early stopping
│   └── evaluate.py        ← Evaluation & metrics
├── outputs/
│   ├── plots/             ← All graphs and charts
│   └── models/            ← Saved model weights
├── architecture_diagram.png
├── requirements.txt
├── README.md
└── IA2_Report.md
```

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone [INSERT GITHUB LINK HERE]
cd solar_panel_classifier
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
Download from [Kaggle](https://www.kaggle.com/datasets/pythonafroz/solar-panel-images) and place the `Faulty_solar_panel` folder inside `data/`:
```
data/
└── Faulty_solar_panel/
    ├── Bird-drop/
    ├── Clean/
    ├── Dusty/
    ├── Electrical-damage/
    ├── Physical-Damage/
    └── Snow-Covered/
```

Alternatively, if the dataset is already at the project's parent directory, the notebook will auto-detect it.

### 4. Run the Notebook
```bash
cd notebooks
jupyter notebook solar_panel_classifier.ipynb
```

Run all cells in order. The notebook will:
1. Load and preprocess the dataset
2. Train both models (Custom CNN + EfficientNetB0)
3. Evaluate on the test set
4. Generate all plots and save results

---

## Models

### Model A: Custom CNN (from scratch)
- 4 Conv2D blocks (32→64→128→256 filters)
- Each block: Conv2D → BatchNorm → ReLU → MaxPool2D
- Global Average Pooling → FC(512) → Dropout(0.5) → FC(256) → Dropout(0.5) → Output(6)

### Model B: EfficientNetB0 (Transfer Learning)
- Pretrained on ImageNet (1.2M images)
- Phase 1: Frozen backbone, train classifier head
- Phase 2: Unfreeze top 3 blocks, fine-tune with lower LR
- Custom head: Dropout(0.4) → FC(256) → ReLU → Dropout(0.4) → Output(6)

---

## Training Configuration
- **Loss:** CrossEntropyLoss with class weights (handles imbalance)
- **Optimizer:** Adam (lr=0.001, weight_decay=1e-4)
- **Scheduler:** ReduceLROnPlateau (factor=0.5, patience=3)
- **Early Stopping:** patience=5
- **Batch Size:** 32
- **Augmentation:** Random flip, rotation, crop, color jitter, affine transforms

---

## Requirements
- Python 3.8+
- PyTorch 2.0+
- See `requirements.txt` for full list

---

## Authors
- [NAME] — [ROLL NO]
- [NAME] — [ROLL NO]

## GitHub
[INSERT GITHUB LINK HERE]
