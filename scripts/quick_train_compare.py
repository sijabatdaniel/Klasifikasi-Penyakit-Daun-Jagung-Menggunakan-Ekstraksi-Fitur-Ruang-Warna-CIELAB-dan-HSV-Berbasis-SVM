"""
quick_train_compare.py - Training cepat menggunakan CSV files yang sudah ada
"""

import pandas as pd
import time
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

def train_and_evaluate(csv_file, method_name, feature_dim):
    """Load CSV dan training model."""
    print(f"\n{'─'*100}")
    print(f"Training: {method_name}")
    print(f"{'─'*100}")
    
    try:
        # Load CSV
        print(f"[1] Loading dataset from CSV...")
        df = pd.read_csv(csv_file)
        
        if df.empty:
            print(f"    Error: CSV file is empty")
            return None
        
        print(f"    ✓ Loaded {len(df)} samples")
        print(f"    ✓ Feature dimension: {feature_dim}")
        
        # Prepare data
        print(f"[2] Preparing data...")
        feature_cols = [col for col in df.columns if col not in ['label', 'path']]
        X = df[feature_cols].to_numpy(dtype='float32')
        y = df['label'].to_numpy()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"    ✓ Train set: {len(X_train)} samples")
        print(f"    ✓ Test set:  {len(X_test)} samples")
        
        # Training
        print(f"[3] Training SVM model...")
        start_train = time.time()
        
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', C=10.0, gamma='scale', probability=True))
        ])
        model.fit(X_train, y_train)
        
        time_train = time.time() - start_train
        print(f"    ✓ Training completed in {time_train:.2f}s")
        
        # Evaluation
        print(f"[4] Evaluating...")
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        
        print(f"    ✓ Accuracy:   {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"    ✓ F1-Score:   {f1_weighted:.4f}")
        print(f"    ✓ Time:       {time_train:.2f}s")
        
        return {
            'method': method_name,
            'accuracy': accuracy,
            'f1_score': f1_weighted,
            'time': time_train,
            'features': feature_dim
        }
    
    except Exception as e:
        print(f"    Error: {e}")
        return None

def main():
    print("\n" + "="*100)
    print(" "*25 + "QUICK TRAINING COMPARISON DARI CSV FILES")
    print("="*100 + "\n")
    
    base_dir = Path(__file__).resolve().parent
    
    # Define methods
    methods = [
        (base_dir / "cielab_features.csv", "Menggunakan fitur warna CIELAB", 54),
        (base_dir / "hsv_features.csv", "Menggunakan fitur warna HSV", 54),
        (base_dir / "combined_features.csv", "Menggunakan fitur Hybrid CIELAB + HSV", 108),
    ]
    
    results = []
    total_start = time.time()
    
    for csv_file, method_name, feature_dim in methods:
        if not csv_file.exists():
            print(f"\n⚠️  Warning: {csv_file.name} not found")
            continue
        
        result = train_and_evaluate(csv_file, method_name, feature_dim)
        if result:
            results.append(result)
    
    total_time = time.time() - total_start
    
    # Display comparison table
    if results:
        print("\n\n" + "="*100)
        print(" "*30 + "📊 PERBANDINGAN HASIL TRAINING")
        print("="*100)
        
        df_results = pd.DataFrame(results)
        best_idx = df_results['accuracy'].idxmax()
        
        print(f"\n{'Metode':<40} {'Akurasi':<12} {'F1-Score':<12} {'Waktu (s)':<12} {'Fitur':<8}")
        print("─"*100)
        
        for idx, row in df_results.iterrows():
            marker = " ⭐ TERBAIK" if idx == best_idx else ""
            print(f"{row['method']:<40} {row['accuracy']*100:>10.2f}% {row['f1_score']:>11.4f} {row['time']:>11.2f} {row['features']:>7}{marker}")
        
        print("="*100)
        print(f"\n⏱️  Total waktu training: {total_time:.2f} detik")
        
        best_method = df_results.loc[best_idx]
        print(f"\n✅ HASIL TERBAIK:")
        print(f"   Metode:       {best_method['method']}")
        print(f"   Akurasi:      {best_method['accuracy']*100:.2f}%")
        print(f"   F1-Score:     {best_method['f1_score']:.4f}")
        print(f"   Waktu:        {best_method['time']:.2f} detik")
        print(f"   Dim. Fitur:   {best_method['features']}")
        print("="*100 + "\n")

if __name__ == "__main__":
    main()
