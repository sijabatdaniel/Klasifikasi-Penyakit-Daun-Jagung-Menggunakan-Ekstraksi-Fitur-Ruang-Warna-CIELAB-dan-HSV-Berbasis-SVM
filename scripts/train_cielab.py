"""
train_cielab.py - Training dan evaluasi menggunakan fitur warna CIELAB
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Constants
HIST_BINS = 16
CIELAB_DIM = 54  # 3 channels * (2 mean/std + 16 bins)

def extract_cielab_features(image_bgr, hist_bins=16):
    """Ekstrak fitur CIELAB: mean, std, dan histogram per channel (L, a, b)."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    channels = cv2.split(lab)
    
    features = []
    for ch in channels:
        features += [float(np.mean(ch)), float(np.std(ch))]
        hist = cv2.calcHist([ch], [0], None, [hist_bins], [0, 256])
        hist = hist.flatten().astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()
        features += hist.tolist()
    
    return np.array(features, dtype=np.float32)

def load_dataset(data_dir, hist_bins=16):
    """Load dataset dan ekstrak fitur CIELAB."""
    feature_names = []
    for ch in ['L', 'a', 'b']:
        feature_names += [f"{ch}_mean", f"{ch}_std"]
        feature_names += [f"{ch}_hist_{i}" for i in range(hist_bins)]
    
    rows = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for class_dir in Path(data_dir).iterdir():
        if not class_dir.is_dir():
            continue
        
        label = class_dir.name
        for img_file in class_dir.iterdir():
            if img_file.suffix.lower() not in image_extensions:
                continue
            
            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                
                features = extract_cielab_features(img, hist_bins)
                row = {'label': label, 'path': str(img_file)}
                row.update({name: val for name, val in zip(feature_names, features)})
                rows.append(row)
            except Exception as e:
                print(f"Error loading {img_file}: {e}")
                continue
    
    return pd.DataFrame(rows)

def main():
    print("=" * 80)
    print("TRAINING MENGGUNAKAN FITUR WARNA CIELAB")
    print("=" * 80)
    print()
    
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    
    if not data_dir.exists():
        print(f"Error: Dataset folder not found at {data_dir}")
        return
    
    # ── Load Dataset ────────────────────────────────────────────────────────
    print("[1] Loading dataset...")
    start_load = time.time()
    df = load_dataset(data_dir, HIST_BINS)
    time_load = time.time() - start_load
    
    if df.empty:
        print("Error: No images found in dataset")
        return
    
    print(f"    ✓ Loaded {len(df)} images in {time_load:.2f}s")
    print(f"    Classes: {df['label'].unique()}")
    print(f"    Distribution:\n{df['label'].value_counts().to_string()}")
    print()
    
    # ── Prepare Data ────────────────────────────────────────────────────────
    print("[2] Preparing data...")
    feature_cols = [col for col in df.columns if col not in ['label', 'path']]
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df['label'].to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"    ✓ Train set: {len(X_train)} samples")
    print(f"    ✓ Test set:  {len(X_test)} samples")
    print(f"    ✓ Feature dimension: {X.shape[1]}")
    print()
    
    # ── Training ────────────────────────────────────────────────────────────
    print("[3] Training SVM model...")
    start_train = time.time()
    
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=10.0, gamma='scale', probability=True))
    ])
    model.fit(X_train, y_train)
    
    time_train = time.time() - start_train
    print(f"    ✓ Training completed in {time_train:.2f}s")
    print()
    
    # ── Evaluation ──────────────────────────────────────────────────────────
    print("[4] Evaluating model...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    
    print(f"    ✓ Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"    ✓ F1-Score (weighted): {f1_weighted:.4f}")
    print(f"    ✓ F1-Score (macro):    {f1_macro:.4f}")
    print()
    
    # ── Classification Report ───────────────────────────────────────────────
    print("[5] Classification Report:")
    print("-" * 80)
    print(classification_report(y_test, y_pred))
    print()
    
    # ── Summary ─────────────────────────────────────────────────────────────
    print("=" * 80)
    print("SUMMARY - FITUR CIELAB")
    print("=" * 80)
    print(f"Akurasi:              {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"F1-Score:             {f1_weighted:.4f}")
    print(f"Waktu Komputasi:      {time_train:.2f} detik")
    print(f"Dimensi Fitur:        {CIELAB_DIM}")
    print("=" * 80)

if __name__ == "__main__":
    main()
