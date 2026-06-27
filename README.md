# Retina-Vision-AI
Automated Detection of Retinal Diseases from OCT Images Using MobileNetV2 Transfer Learning
# 👁️ Human Eye Disease Prediction System

A deep learning system that classifies **retinal OCT scans** into 4 categories using **MobileNetV2 transfer learning**, deployed as a **Streamlit** web application.


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

### 1. Open the project
```bash
cd Retina-Vision-AI
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

https://github.com/user-attachments/assets/2ab92027-8ad8-4c93-b16e-f7a085e884a2

<img width="1920" height="1724" alt="Image" src="https://github.com/user-attachments/assets/a90db932-2b46-4706-b9b0-68934a288d8f" />

<img width="1920" height="1754" alt="Image" src="https://github.com/user-attachments/assets/f5af268b-7d17-4716-815a-6e87818bc2e6" />

<img width="1920" height="1002" alt="Image" src="https://github.com/user-attachments/assets/9786de88-855d-4c64-9446-42ada974b4fe" />

<img width="1916" height="1007" alt="Image" src="https://github.com/user-attachments/assets/21f3d051-209c-446d-8d29-309c68a72bff" />

<img width="1171" height="782" alt="Image" src="https://github.com/user-attachments/assets/fe70f873-e442-40e7-96e9-53b8cd7fa4ff" />

<img width="1207" height="532" alt="Image" src="https://github.com/user-attachments/assets/8be33e4c-f4f3-4081-a9e9-8c8a96038f73" />

<img width="1920" height="1590" alt="Image" src="https://github.com/user-attachments/assets/6ee35152-d2c6-4519-96fc-77ccd3ac5ce9" />

<img width="1766" height="807" alt="Image" src="https://github.com/user-attachments/assets/9d5afc99-4d52-4a27-9868-8558f7e222f8" />

<img width="1068" height="627" alt="Image" src="https://github.com/user-attachments/assets/10deb8f1-58b2-4f59-b94d-d789ce5c8218" />

<img width="1713" height="627" alt="Image" src="https://github.com/user-attachments/assets/d70af58f-9e44-44c5-a563-3348205bd75a" />
