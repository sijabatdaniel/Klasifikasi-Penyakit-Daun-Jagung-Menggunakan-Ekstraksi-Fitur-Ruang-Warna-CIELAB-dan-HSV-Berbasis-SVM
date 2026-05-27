"""
run_all_training.py - Menjalankan training untuk ketiga metode dan menampilkan perbandingan
"""

import subprocess
import sys
import time
from pathlib import Path

def run_training_script(script_name):
    """Jalankan training script dan capture outputnya."""
    print(f"\n{'='*80}")
    print(f"Running: {script_name}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_name)],
            capture_output=False,
            text=True,
            cwd=Path(__file__).resolve().parent
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error running {script_name}: {e}")
        return False

def main():
    base_dir = Path(__file__).resolve().parent
    scripts = [
        base_dir / "train_cielab.py",
        base_dir / "train_hsv.py",
        base_dir / "train_combined.py"
    ]
    
    print("\n" + "="*80)
    print("TRAINING UNTUK KETIGA METODE EKSTRAKSI CIRI")
    print("="*80)
    
    start_total = time.time()
    
    for script in scripts:
        if not script.exists():
            print(f"Warning: {script.name} not found")
            continue
        
        run_training_script(script)
        print()
    
    total_time = time.time() - start_total
    
    print("\n" + "="*80)
    print("TRAINING SELESAI")
    print("="*80)
    print(f"Total waktu keseluruhan: {total_time:.2f} detik")
    print("\nSilakan lihat output di atas untuk detail hasil training setiap metode.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
