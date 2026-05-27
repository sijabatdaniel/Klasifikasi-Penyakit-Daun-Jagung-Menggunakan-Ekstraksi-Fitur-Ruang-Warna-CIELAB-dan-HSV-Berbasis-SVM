# 🌽 Corn Leaf Disease Classification

Sistem klasifikasi penyakit daun jagung menggunakan **Hybrid Color Features (CIELAB + HSV)** dan **Support Vector Machine (SVM)** untuk mengidentifikasi empat kelas penyakit daun jagung.

## 📋 Daftar Isi

- [Fitur](#fitur)
- [Dataset](#dataset)
- [Struktur Proyek](#struktur-proyek)
- [Instalasi](#instalasi)
- [Penggunaan](#penggunaan)
- [Model & Performa](#model--performa)
- [Hasil](#hasil)

## ✨ Fitur

- **3 Metode Ekstraksi Fitur Warna:**
  - **CIELAB**: Perceptual color space (54 fitur)
  - **HSV**: Hue-Saturation-Value (54 fitur)
  - **Hybrid**: Kombinasi CIELAB + HSV (108 fitur)

- **Ekstraksi Fitur:**
  - Mean dan Standard Deviation per channel
  - Normalized histogram dengan 16 bins per channel

- **Klasifikasi:** SVM dengan kernel RBF dan StandardScaler preprocessing

- **Evaluasi Komprehensif:**
  - Classification Report (Precision, Recall, F1-Score)
  - Confusion Matrix
  - Per-class metrics visualization
  - Model comparison dashboard (HTML)

- **Aplikasi Web:** Streamlit untuk prediksi real-time

## 📊 Dataset

Dataset berisi **4 kelas penyakit daun jagung:**
- Sumber : **https://www.kaggle.com/datasets/andril22/daun-jagung-dataset/data?select=data**

1. **Blight** - Southern Corn Leaf Blight
2. **Common Rust** - Common Corn Rust
3. **Gray Leaf Spot** - Gray Leaf Spot
4. **Healthy** - Daun jagung sehat

Struktur folder:
```
data/
├── Blight/
├── Common_Rust/
├── Gray_Leaf_Spot/
└── Healthy/
```

Setiap folder berisi gambar daun jagung dalam format `.jpg`, `.png`, atau `.bmp`.

## 📁 Struktur Proyek

```
tubes/
├── README.md                          # Dokumentasi proyek
├── .gitignore                         # File yang tidak di-track
├── tubes.py                           # Main training & evaluation script
├── sapp.py                            # Streamlit web application
├── data/                              # Dataset folder
│   ├── Blight/
│   ├── Common_Rust/
│   ├── Gray_Leaf_Spot/
│   └── Healthy/
├── scripts/                           # Python training scripts
│   ├── train_cielab.py               # Training untuk CIELAB features
│   ├── train_hsv.py                  # Training untuk HSV features
│   ├── train_combined.py             # Training untuk Hybrid features
│   ├── train_all_and_compare.py      # Training semua model & comparison
│   ├── run_all_training.py           # Automated training runner
│   ├── quick_train_compare.py        # Quick comparison script
│   ├── visual.py                     # Visualization utilities
│   └── tubes2.py                     # Alternative training script
├── models/                            # Trained model files (joblib)
│   ├── cielab_svm.joblib
│   ├── hsv_svm.joblib
│   └── combined_svm.joblib
├── results/                           # Output results
│   ├── features/                      # Feature extraction CSV
│   │   ├── cielab_features.csv
│   │   ├── hsv_features.csv
│   │   └── combined_features.csv
│   ├── metrics/                       # Performance metrics CSV
│   │   ├── detailed_metrics_per_class_cielab.csv
│   │   ├── detailed_metrics_per_class_hsv.csv
│   │   └── detailed_metrics_per_class_combined.csv
│   ├── visualizations/                # Generated plots & charts
│   │   ├── confusion_matrix_*.png
│   │   ├── overall_metrics_*.png
│   │   ├── metrics_per_class_visualization_*.png
│   │   └── ...
│   ├── tabel_perbandingan_metode.html # Model comparison table
│   └── coba/                          # Example visualizations
```

## 🔧 Instalasi

### Requirements
- Python 3.8+
- pip atau conda

### 1. Clone Repository
```bash
git clone <repository-url>
cd tubes
```

### 2. Setup Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

Atau install secara manual:
```bash
pip install numpy pandas scikit-learn opencv-python joblib matplotlib seaborn streamlit
```

### 4. Prepare Dataset
- Tempatkan dataset di folder `data/`
- Pastikan struktur folder sesuai:
  ```
  data/
  ├── Blight/
  ├── Common_Rust/
  ├── Gray_Leaf_Spot/
  └── Healthy/
  ```

## 🚀 Penggunaan

### 1. Training Models
Jalankan script utama untuk melatih semua model:

```bash
python tubes.py
```

Script ini akan:
- Memproses dataset dari folder `data/`
- Melatih 3 model (CIELAB, HSV, Hybrid)
- Menyimpan model ke folder `models/`
- Menghasilkan visualisasi dan metrik ke folder `results/`

Opsi lain untuk training:

```bash
# Training individual feature types
python scripts/train_cielab.py
python scripts/train_hsv.py
python scripts/train_combined.py

# Training dan perbandingan
python scripts/train_all_and_compare.py
python scripts/quick_train_compare.py
```

### 2. Streamlit Web Application
Jalankan aplikasi web untuk prediksi real-time:

```bash
streamlit run sapp.py
```

Kemudian buka browser ke `http://localhost:8501`

**Fitur:**
- Upload gambar daun jagung
- Prediksi penyakit secara real-time
- Visualisasi probabilitas prediksi
- Lihat informasi model dan metrik performa

### 3. Menggunakan Model Terlatih

```python
import joblib
from pathlib import Path
import cv2
from tubes import extract_combined_features, HIST_BINS

# Load model
model_path = Path("models/combined_svm.joblib")
payload = joblib.load(model_path)

# Baca gambar
image_bgr = cv2.imread("path/to/leaf_image.jpg")

# Ekstrak fitur dan prediksi
features = extract_combined_features(image_bgr, hist_bins=HIST_BINS)
prediction = payload["model"].predict([features])[0]
probability = payload["model"].predict_proba([features])[0]

print(f"Prediksi: {prediction}")
print(f"Probabilitas: {probability}")
```

## 📈 Model & Performa

### Feature Engineering

Setiap gambar diekstrak menjadi fitur numerik:

| Metode | Features | Deskripsi |
|--------|----------|-----------|
| **CIELAB** | 54 | L, a, b channels (mean + std + 16-bin histogram) |
| **HSV** | 54 | H, S, V channels (mean + std + 16-bin histogram) |
| **Hybrid** | 108 | CIELAB + HSV kombinasi |

### Klasifier

- **Algorithm:** Support Vector Machine (SVM)
- **Kernel:** Radial Basis Function (RBF)
- **Preprocessing:** StandardScaler normalization
- **Framework:** scikit-learn Pipeline

### Expected Performance

Akurasi model hybrid pada test set biasanya mencapai:
- Blight: ~90-95%
- Common Rust: ~85-90%
- Gray Leaf Spot: ~85-90%
- Healthy: ~95-98%

*(Nilai sebenarnya bergantung pada dataset dan pembagian train-test)*

## 📊 Hasil

Setelah menjalankan `tubes.py`, hasil akan tersimpan di folder `results/`:

### Visualisasi
- **Confusion Matrix:** `results/visualizations/confusion_matrix_combined.png`
- **Overall Metrics:** `results/visualizations/overall_metrics_combined.png`
- **Per-Class Metrics:** `results/visualizations/metrics_per_class_visualization_combined.png`

### Data
- **Features CSV:** `results/features/combined_features.csv`
- **Metrics CSV:** `results/metrics/detailed_metrics_per_class_combined.csv`
- **Model Comparison:** `results/tabel_perbandingan_metode.html`

### Model
- **Combined Model:** `models/combined_svm.joblib`
- **CIELAB Model:** `models/cielab_svm.joblib`
- **HSV Model:** `models/hsv_svm.joblib`

## 🛠️ Troubleshooting

### "Dataset folder not found"
- Pastikan folder `data/` ada di direktori proyek
- Periksa struktur subfolder: `Blight/`, `Common_Rust/`, `Gray_Leaf_Spot/`, `Healthy/`

### "Model belum ada"
- Jalankan `python tubes.py` terlebih dahulu untuk melatih model
- Pastikan ada file di folder `models/`

### Import Error untuk visual.py
- Pastikan file `visual.py` ada di folder `scripts/`
- Periksa bahwa folder `scripts/` dapat diakses

### Streamlit tidak menemukan model
```bash
# Pastikan menjalankan dari direktori proyek
cd /path/to/tubes
streamlit run sapp.py
```

## 📝 File Penting

| File | Deskripsi |
|------|-----------|
| `tubes.py` | Script utama untuk training semua model |
| `sapp.py` | Aplikasi Streamlit untuk prediksi |
| `scripts/visual.py` | Fungsi visualisasi (plotting) |
| `requirements.txt` | Dependency Python |

## 📚 Referensi

- **CIELAB Color Space:** Perceptual color model yang lebih mendekati persepsi manusia
- **HSV Color Space:** Model warna berbasis hue, saturation, value
- **SVM Classification:** Non-parametric supervised learning model
- **scikit-learn:** Machine learning library untuk Python

## 👨‍💻 Author

Tugas Besar Pemrosesan Citra Digital - S1 Informatika
- Haura Gazeilla Falestin
- Abdullah Luthfi
- Karen anggelia pasaribu
- Daniel Febrian Sijabat

## 📄 License

MIT License

---

**Last Updated:** May 2026  
**Status:** Production Ready
