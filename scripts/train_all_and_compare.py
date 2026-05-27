"""
train_all_and_compare.py - Training semua metode dan tampilkan perbandingan hasil
Ini adalah master script yang menjalankan ketiga training dan menampilkan ringkasan.
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
from sklearn.metrics import accuracy_score, f1_score

# Constants
HIST_BINS = 16

# Feature extractors
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

def extract_hsv_features(image_bgr, hist_bins=16):
    """Ekstrak fitur HSV: mean, std, dan histogram per channel (H, S, V)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)
    
    channel_ranges = [(h_chan, (0, 180)), (s_chan, (0, 256)), (v_chan, (0, 256))]
    
    features = []
    for ch, hist_range in channel_ranges:
        features += [float(np.mean(ch)), float(np.std(ch))]
        hist = cv2.calcHist([ch], [0], None, [hist_bins], list(hist_range))
        hist = hist.flatten().astype(np.float32)
        if hist.sum() > 0:
            hist /= hist.sum()
        features += hist.tolist()
    
    return np.array(features, dtype=np.float32)

def extract_combined_features(image_bgr, hist_bins=16):
    """Gabungkan fitur CIELAB dan HSV (hybrid approach)."""
    cielab_feat = extract_cielab_features(image_bgr, hist_bins)
    hsv_feat = extract_hsv_features(image_bgr, hist_bins)
    return np.concatenate([cielab_feat, hsv_feat], dtype=np.float32)

# Dataset loaders
def build_feature_names_cielab(hist_bins=16):
    names = []
    for ch in ['L', 'a', 'b']:
        names += [f"{ch}_mean", f"{ch}_std"]
        names += [f"{ch}_hist_{i}" for i in range(hist_bins)]
    return names

def build_feature_names_hsv(hist_bins=16):
    names = []
    for ch in ['H', 'S', 'V']:
        names += [f"{ch}_mean", f"{ch}_std"]
        names += [f"{ch}_hist_{i}" for i in range(hist_bins)]
    return names

def build_feature_names_combined(hist_bins=16):
    return build_feature_names_cielab(hist_bins) + build_feature_names_hsv(hist_bins)

def load_dataset(data_dir, extractor, feature_names, hist_bins=16):
    """Load dataset dan ekstrak fitur."""
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
                
                features = extractor(img, hist_bins)
                row = {'label': label, 'path': str(img_file)}
                row.update({name: val for name, val in zip(feature_names, features)})
                rows.append(row)
            except Exception:
                continue
    
    return pd.DataFrame(rows)

def train_and_evaluate(df, feature_cols, method_name):
    """Training dan evaluasi model."""
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df['label'].to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Training
    start_train = time.time()
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=10.0, gamma='scale', probability=True))
    ])
    model.fit(X_train, y_train)
    time_train = time.time() - start_train
    
    # Evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    return {
        'method': method_name,
        'accuracy': accuracy,
        'f1_score': f1_weighted,
        'time': time_train,
        'features': len(feature_cols)
    }

def main():
    print("\n" + "="*100)
    print(" "*30 + "TRAINING SEMUA METODE EKSTRAKSI CIRI")
    print("="*100 + "\n")
    
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    
    if not data_dir.exists():
        print(f"Error: Dataset folder not found at {data_dir}")
        return
    
    results = []
    methods = [
        ("CIELAB", extract_cielab_features, build_feature_names_cielab),
        ("HSV", extract_hsv_features, build_feature_names_hsv),
        ("Hybrid CIELAB + HSV", extract_combined_features, build_feature_names_combined),
    ]
    
    total_start = time.time()
    
    for method_name, extractor, feature_builder in methods:
        print(f"\n{'─'*100}")
        print(f"Training: {method_name}")
        print(f"{'─'*100}")
        
        try:
            # Build feature names
            feature_names = feature_builder(HIST_BINS)
            print(f"[1] Loading dataset... (Features: {len(feature_names)})")
            
            # Load dataset
            df = load_dataset(data_dir, extractor, feature_names, HIST_BINS)
            if df.empty:
                print(f"    Error: No images found for {method_name}")
                continue
            
            print(f"    ✓ Loaded {len(df)} images")
            print(f"    ✓ Classes: {list(df['label'].unique())}")
            
            # Train and evaluate
            print(f"[2] Training SVM model...")
            result = train_and_evaluate(df, feature_names, method_name)
            results.append(result)
            
            print(f"    ✓ Accuracy:        {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
            print(f"    ✓ F1-Score:        {result['f1_score']:.4f}")
            print(f"    ✓ Training time:   {result['time']:.2f} detik")
            print(f"    ✓ Feature dims:    {result['features']}")
        
        except Exception as e:
            print(f"    Error: {e}")
            continue
    
    total_time = time.time() - total_start
    
    # Display comparison table
    if results:
        print("\n\n" + "="*100)
        print(" "*35 + "PERBANDINGAN HASIL TRAINING")
        print("="*100)
        
        df_results = pd.DataFrame(results)
        
        # Find best method
        best_idx = df_results['accuracy'].idxmax()
        
        print(f"\n{'Metode':<30} {'Akurasi (%)':<15} {'F1-Score':<15} {'Waktu (detik)':<15} {'Dim. Fitur':<15}")
        print("─"*100)
        
        for idx, row in df_results.iterrows():
            marker = " ← TERBAIK ⭐" if idx == best_idx else ""
            print(f"{row['method']:<30} {row['accuracy']*100:>13.2f}% {row['f1_score']:>14.4f} {row['time']:>14.2f} {row['features']:>14}{marker}")
        
        print("="*100)
        print(f"\nTotal waktu training: {total_time:.2f} detik")
        print("="*100 + "\n")
        
        # Detail summary
        best_method = df_results.loc[best_idx]
        print("\n📊 RINGKASAN TERBAIK:")
        print(f"   Metode: {best_method['method']}")
        print(f"   Akurasi: {best_method['accuracy']*100:.2f}%")
        print(f"   F1-Score: {best_method['f1_score']:.4f}")
        print(f"   Waktu: {best_method['time']:.2f} detik")
        print(f"   Dimensi Fitur: {best_method['features']}\n")

if __name__ == "__main__":
    main()
