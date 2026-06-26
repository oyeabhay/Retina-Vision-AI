# Retina-Vision-AI
Automated Detection of Retinal Diseases from OCT Images Using MobileNetV2 Transfer Learning
# 👁️ Human Eye Disease Prediction System

A deep learning system that classifies **retinal OCT scans** into 4 categories using **MobileNetV2 transfer learning**, deployed as a **Streamlit** web application.

> Based on the [SPOTLESS TECH YouTube playlist](https://youtube.com/playlist?list=PLvz5lCwTgdXD2X0Y9wkq02DbMEZyJ5kav)

---

## 🔬 Detectable Conditions

| Code | Full Name | Description |
|------|-----------|-------------|
| **CNV** | Choroidal Neovascularization | Abnormal blood vessel growth beneath the retina |
| **DME** | Diabetic Macular Edema | Fluid accumulation in the macula due to diabetes |
| **DRUSEN** | Drusen (Early AMD) | Yellow deposits under the retinal pigment epithelium |
| **NORMAL** | Normal Retina | No signs of retinal disease |

---

## 🗂️ Project Structure

```
eye_disease_project/
├── app.py                  ← Streamlit web application
├── train_model.py          ← MobileNetV2 training script (2-phase)
├── download_dataset.py     ← Kaggle dataset downloader
├── requirements.txt        ← Python dependencies
├── model/                  ← (created after training)
│   ├── best_model.keras
│   └── class_indices.json
└── OCT2017/                ← (created after download)
    ├── train/
    │   ├── CNV/
    │   ├── DME/
    │   ├── DRUSEN/
    │   └── NORMAL/
    └── test/
        ├── CNV/
        ├── DME/
        ├── DRUSEN/
        └── NORMAL/
```

---

## ⚙️ Setup

### 1. Clone / copy the project
```bash
cd eye_disease_project
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📥 Download the Dataset

**Option A – Automatic (Kaggle CLI)**
```bash
# Install kaggle CLI and set up credentials
pip install kaggle
# Place ~/.kaggle/kaggle.json (from kaggle.com → Account → API)

python download_dataset.py
```

**Option B – Manual**
1. Go to: https://www.kaggle.com/datasets/anirudhcv/labeled-optical-coherence-tomography-oct
2. Download and extract `OCT2017.zip` into the project root.

---

## 🏋️ Train the Model

### Quick start (default settings)
```bash
python train_model.py --data_dir OCT2017
```

### Full options
```bash
python train_model.py \
    --data_dir         OCT2017 \
    --output_dir       model \
    --epochs           10 \
    --fine_tune_epochs 5
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `OCT2017` | Root of the dataset |
| `--output_dir` | `model` | Where to save model & logs |
| `--epochs` | `10` | Phase-1 (head training) epochs |
| `--fine_tune_epochs` | `5` | Phase-2 (fine-tuning) epochs; `0` to skip |

Training outputs saved to `model/`:
- `best_model.keras` — final best checkpoint
- `class_indices.json` — index → class name mapping
- `confusion_matrix.png` — test set confusion matrix
- `training_history_phase1/2.png` — accuracy & loss curves
- `log_phase1/2.csv` — per-epoch metrics

---

## 🚀 Run the Web App

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

**How to use:**
1. Upload a retinal OCT image (JPG / PNG / BMP / TIFF)
2. The model predicts the disease class and confidence
3. Read the diagnosis details, symptoms, and treatment info

---

## 🧠 Model Architecture

```
Input (224×224×3)
    ↓
MobileNetV2 (ImageNet pre-trained, frozen in Phase 1)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(256, ReLU) → Dropout(0.4)
    ↓
Dense(128, ReLU) → Dropout(0.3)
    ↓
Dense(4, Softmax)   ← CNV / DME / DRUSEN / NORMAL
```

### Two-Phase Training Strategy

| Phase | Layers Trained | LR | Purpose |
|-------|---------------|-----|---------|
| **1 – Head** | Custom top layers only | 1e-4 | Learn task-specific features |
| **2 – Fine-tune** | Top ~54 MobileNetV2 layers + head | 1e-5 | Refine low-level features |

---

## 📊 Expected Performance

Typical results on the OCT2017 test set:

| Metric | Value |
|--------|-------|
| Test Accuracy | ~97–99% |
| Macro F1 | ~0.97 |
| Inference speed | < 50 ms/image |

---

## 💡 Tips

- **GPU training**: TensorFlow automatically uses a CUDA GPU if available. Install `tensorflow-gpu` for older TF versions.
- **Kaggle Notebook**: You can also train directly on Kaggle using a free GPU — just copy `train_model.py` into a notebook.
- **Batch size**: Reduce `BATCH_SIZE` in `train_model.py` if you hit OOM errors.
- **Image quality**: The model works best on real OCT B-scans. Avoid natural photos.

---

## 📜 License

For educational and research purposes. Dataset credits to the original authors of the [Kaggle OCT2017 dataset](https://www.kaggle.com/datasets/anirudhcv/labeled-optical-coherence-tomography-oct).
