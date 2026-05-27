"""
tubes.py — Klasifikasi citra menggunakan fitur warna CIELAB, HSV, dan kombinasi keduanya.

Pipeline:
  1. Ekstraksi fitur (mean, std, histogram) dari setiap channel warna.
  2. Training SVM dengan StandardScaler via sklearn Pipeline.
  3. Evaluasi lengkap: classification report, confusion matrix, dan visualisasi.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]
    sns = None  # type: ignore[assignment]

try:
    import sys
    from pathlib import Path as PathlibPath
    _scripts_dir = PathlibPath(__file__).resolve().parent / "scripts"
    if _scripts_dir not in sys.path:
        sys.path.insert(0, str(_scripts_dir))
    from visual import (
        visualize_cielab_conversion,
        visualize_cielab_features,
        visualize_histogram_comparison,
        visualize_hsv_features,
    )
except ImportError:
    visualize_cielab_conversion = None
    visualize_cielab_features = None
    visualize_histogram_comparison = None
    visualize_hsv_features = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HIST_BINS: int = 16
MODEL_FILENAME: str = "models/combined_svm.joblib"
FeatureType = Literal["cielab", "hsv", "combined"]

# Mapping nama file output per feature_type
_FILENAMES: dict[str, dict[str, str]] = {
    "metrics_viz":         {"cielab": "results/visualizations/metrics_per_class_visualization_cielab.png",    "hsv": "results/visualizations/metrics_per_class_visualization_hsv.png",    "combined": "results/visualizations/metrics_per_class_visualization_combined.png"},
    "confusion_details":   {"cielab": "results/visualizations/confusion_details_per_class_cielab.png",        "hsv": "results/visualizations/confusion_details_per_class_hsv.png",        "combined": "results/visualizations/confusion_details_per_class_combined.png"},
    "confusion_matrix":    {"cielab": "results/visualizations/confusion_matrix_cielab.png",                   "hsv": "results/visualizations/confusion_matrix_hsv.png",                   "combined": "results/visualizations/confusion_matrix_combined.png"},
    "confusion_per_class": {"cielab": "results/visualizations/confusion_matrix_per_class_cielab.png",         "hsv": "results/visualizations/confusion_matrix_per_class_hsv.png",         "combined": "results/visualizations/confusion_matrix_per_class_combined.png"},
    "overall_metrics":     {"cielab": "results/visualizations/overall_metrics_cielab.png",                    "hsv": "results/visualizations/overall_metrics_hsv.png",                    "combined": "results/visualizations/overall_metrics_combined.png"},
    "detailed_metrics":    {"cielab": "results/metrics/detailed_metrics_per_class_cielab.csv",         "hsv": "results/metrics/detailed_metrics_per_class_hsv.csv",         "combined": "results/metrics/detailed_metrics_per_class_combined.csv"},
}

_TITLE_MAP: dict[str, str] = {
    "cielab": "CIELAB Features",
    "hsv": "HSV Features",
    "combined": "Hybrid CIELAB + HSV Features",
}

# ---------------------------------------------------------------------------
# Feature name builders
# ---------------------------------------------------------------------------

def build_feature_names(hist_bins: int, feature_type: FeatureType = "combined") -> list[str]:
    """Bangun daftar nama fitur sesuai tipe yang dipilih.

    Args:
        hist_bins:    Jumlah bin histogram.
        feature_type: "cielab", "hsv", atau "combined".

    Returns:
        List nama kolom fitur.
    """
    names: list[str] = []

    if feature_type in ("cielab", "combined"):
        for ch in ("L", "a", "b"):
            names += [f"{ch}_mean", f"{ch}_std"]
            names += [f"{ch}_hist_{i}" for i in range(hist_bins)]

    if feature_type in ("hsv", "combined"):
        for ch in ("H", "S", "V"):
            names += [f"{ch}_mean", f"{ch}_std"]
            names += [f"{ch}_hist_{i}" for i in range(hist_bins)]

    return names


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

def _normalized_hist(channel: np.ndarray, hist_bins: int, hist_range: tuple[int, int]) -> list[float]:
    """Hitung histogram yang sudah dinormalisasi (sum=1)."""
    hist = cv2.calcHist([channel], [0], None, [hist_bins], list(hist_range))
    hist = hist.flatten().astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist.tolist()


def extract_cielab_features(image_bgr: np.ndarray, hist_bins: int = 16) -> np.ndarray:
    """Ekstrak fitur CIELAB: mean, std, dan histogram per channel (L, a, b)."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    channels = cv2.split(lab)

    features: list[float] = []
    for ch in channels:
        features += [float(np.mean(ch)), float(np.std(ch))]
        features += _normalized_hist(ch, hist_bins, (0, 256))

    return np.array(features, dtype=np.float32)


def extract_hsv_features(image_bgr: np.ndarray, hist_bins: int = 16) -> np.ndarray:
    """Ekstrak fitur HSV: mean, std, dan histogram per channel (H, S, V)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_chan, s_chan, v_chan = cv2.split(hsv)

    # H: 0–180, S & V: 0–255
    channel_ranges = [(h_chan, (0, 180)), (s_chan, (0, 256)), (v_chan, (0, 256))]

    features: list[float] = []
    for ch, hist_range in channel_ranges:
        features += [float(np.mean(ch)), float(np.std(ch))]
        features += _normalized_hist(ch, hist_bins, hist_range)

    return np.array(features, dtype=np.float32)


def extract_combined_features(image_bgr: np.ndarray, hist_bins: int = 16) -> np.ndarray:
    """Gabungkan fitur CIELAB dan HSV (hybrid approach)."""
    return np.concatenate([
        extract_cielab_features(image_bgr, hist_bins=hist_bins),
        extract_hsv_features(image_bgr, hist_bins=hist_bins),
    ], dtype=np.float32)


# Dispatcher sederhana agar tidak ada if/elif berulang
_EXTRACTORS = {
    "cielab":    extract_cielab_features,
    "hsv":       extract_hsv_features,
    "combined":  extract_combined_features,
}

# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset(
    data_dir: Path,
    hist_bins: int = 16,
    feature_type: FeatureType = "combined",
) -> pd.DataFrame:
    """Muat dataset dari folder, ekstrak fitur, dan kembalikan sebagai DataFrame.

    Struktur folder yang diharapkan::

        data_dir/
            kelas_a/
                img1.jpg
                img2.png
            kelas_b/
                ...

    Args:
        data_dir:     Path ke folder dataset.
        hist_bins:    Jumlah bin histogram.
        feature_type: Tipe fitur yang diekstrak.

    Returns:
        DataFrame dengan kolom: label, path, <feature_names...>.
    """
    extractor = _EXTRACTORS[feature_type]
    feature_names = build_feature_names(hist_bins, feature_type=feature_type)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    rows: list[dict] = []
    visualized_classes: set[str] = set()

    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        label = class_dir.name
        image_paths = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in image_extensions)

        for idx, image_path in enumerate(image_paths, start=1):
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"[WARN] Gagal membaca gambar: {image_path}")
                continue

            # Visualisasi satu sampel per kelas (hanya mode combined)
            if (
                feature_type == "combined"
                and label not in visualized_classes
                and visualize_cielab_conversion is not None
            ):
                print(f"Visualizing sample from class: {label}")
                visualize_cielab_conversion(image, str(image_path))
                visualize_cielab_features(image, hist_bins=hist_bins, image_path=str(image_path))
                visualize_histogram_comparison(image, hist_bins=hist_bins, image_path=str(image_path))
                visualize_hsv_features(image, hist_bins=hist_bins, image_path=str(image_path))
                visualized_classes.add(label)

            features = extractor(image, hist_bins=hist_bins)
            row = {"label": label, "path": str(image_path)}
            row.update(dict(zip(feature_names, features.tolist())))
            rows.append(row)

            if idx % 250 == 0:
                print(f"  Loaded {idx} images from '{label}'...")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def _save_fig(fig, output_path: Path) -> None:
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def _plot_metrics_per_class(
    labels: np.ndarray,
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    width = 0.25

    bar_groups = [
        (x - width, precision, "Precision", "#3498db"),
        (x,         recall,    "Recall",    "#e74c3c"),
        (x + width, f1,        "F1-Score",  "#2ecc71"),
    ]
    for pos, values, label_text, color in bar_groups:
        bars = ax.bar(pos, values, width, label=label_text, color=color)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set(xlabel="Kelas", ylabel="Score",
           title="Evaluasi Metrics per Kelas (Precision, Recall, F1-Score)",
           xticks=x, xticklabels=labels, ylim=[0, 1.1])
    ax.set_xlabel("Kelas", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Evaluasi Metrics per Kelas (Precision, Recall, F1-Score)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, output_path)


def _plot_confusion_details(
    labels: np.ndarray,
    tp: np.ndarray, fp: np.ndarray, fn: np.ndarray, tn: np.ndarray,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))
    width = 0.2

    bar_groups = [
        (x - 1.5 * width, tp, "True Positive",  "#27ae60"),
        (x - 0.5 * width, fp, "False Positive", "#e74c3c"),
        (x + 0.5 * width, fn, "False Negative", "#f39c12"),
        (x + 1.5 * width, tn, "True Negative",  "#3498db"),
    ]
    for pos, values, label_text, color in bar_groups:
        bars = ax.bar(pos, values, width, label=label_text, color=color)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, str(int(h)),
                    ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xlabel("Kelas", fontsize=12, fontweight="bold")
    ax.set_ylabel("Jumlah", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix Details per Kelas (TP, FP, FN, TN)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, output_path)


def _plot_confusion_matrix(
    cm: np.ndarray, labels: np.ndarray, feature_type: FeatureType, output_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        cbar_kws={"label": "Count"}, ax=ax,
        annot_kws={"size": 12, "weight": "bold", "color": "black"},
    )
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
    ax.set_title(f"Confusion Matrix — {_TITLE_MAP[feature_type]}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, output_path)


def _plot_confusion_matrix_per_class(
    labels: np.ndarray, y_test: np.ndarray, y_pred: np.ndarray,
    feature_type: FeatureType, output_path: Path
) -> None:
    n_classes = len(labels)
    n_cols = 2
    n_rows = (n_classes + 1) // 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes_flat = np.array(axes).flatten()

    for idx, lbl in enumerate(labels):
        y_bin_test = (y_test == lbl).astype(int)
        y_bin_pred = (y_pred == lbl).astype(int)
        cm_bin = confusion_matrix(y_bin_test, y_bin_pred)
        tick_labels = [f"Not {lbl}", lbl]

        sns.heatmap(
            cm_bin, annot=True, fmt="d", cmap="Greens",
            xticklabels=tick_labels, yticklabels=tick_labels,
            cbar=False, ax=axes_flat[idx],
            annot_kws={"size": 14, "weight": "bold", "color": "black"},
        )
        axes_flat[idx].set_xlabel("Predicted", fontsize=10, fontweight="bold")
        axes_flat[idx].set_ylabel("Actual", fontsize=10, fontweight="bold")
        axes_flat[idx].set_title(f"Confusion Matrix: {lbl}", fontsize=12, fontweight="bold")

    for idx in range(n_classes, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    _save_fig(fig, output_path)


def _plot_overall_metrics(
    acc: float,
    y_test: np.ndarray, y_pred: np.ndarray, labels: np.ndarray,
    output_path: Path,
) -> None:
    overall_precision = precision_score(y_test, y_pred, labels=labels, average="weighted")
    overall_recall    = recall_score(y_test, y_pred, labels=labels, average="weighted")
    overall_f1        = f1_score(y_test, y_pred, labels=labels, average="weighted")

    names  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    values = [acc, overall_precision, overall_recall, overall_f1]
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=2)
    for bar, value in zip(bars, values):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{value:.4f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Overall Model Metrics", fontsize=14, fontweight="bold")
    ax.set_ylim([0, 1.1])
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, output_path)


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(
    df: pd.DataFrame,
    hist_bins: int = 16,
    feature_type: FeatureType = "combined",
    save_path: Path | None = None,
    output_dir: Path | None = None,
) -> Pipeline:
    """Latih SVM dan evaluasi hasilnya, kemudian simpan model dan visualisasi.

    Args:
        df:           DataFrame hasil load_dataset.
        hist_bins:    Jumlah bin histogram (harus sama saat load_dataset).
        feature_type: Tipe fitur yang digunakan.
        save_path:    Path untuk menyimpan model (.joblib). Opsional.
        output_dir:   Folder output visualisasi. Default: direktori skrip.

    Returns:
        Pipeline sklearn yang sudah dilatih.
    """
    base_dir = output_dir or Path(__file__).resolve().parent

    # ── Persiapan data ──────────────────────────────────────────────────────
    feature_names = build_feature_names(hist_bins, feature_type=feature_type)  # FIX: pakai feature_type
    x = df[feature_names].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Training ────────────────────────────────────────────────────────────
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True)),
    ])
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    # ── Metrics dasar ───────────────────────────────────────────────────────
    acc = accuracy_score(y_test, y_pred)
    labels = np.unique(y)

    print(f"\nAccuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # ── Metrics per kelas ───────────────────────────────────────────────────
    cm        = confusion_matrix(y_test, y_pred, labels=labels)
    precision = precision_score(y_test, y_pred, labels=labels, average=None)
    recall    = recall_score(y_test, y_pred, labels=labels, average=None)
    f1        = f1_score(y_test, y_pred, labels=labels, average=None)

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (tp + fp + fn)

    detailed_metrics = pd.DataFrame({
        "Class":          labels,
        "True Positive":  tp,
        "False Positive": fp,
        "True Negative":  tn,
        "False Negative": fn,
        "Precision":      precision,
        "Recall":         recall,
        "F1-Score":       f1,
    })

    # FIX: nama CSV dibedakan per feature_type agar tidak tertimpa
    csv_path = base_dir / _FILENAMES["detailed_metrics"][feature_type]
    detailed_metrics.to_csv(csv_path, index=False)

    sep = "=" * 100
    print(f"\n{sep}")
    print("DETAILED METRICS PER KELAS:")
    print(sep)
    print(detailed_metrics.to_string(index=False))
    print(f"{sep}\n")
    print(f"Saved detailed metrics to: {csv_path}\n")

    # ── Visualisasi ─────────────────────────────────────────────────────────
    if plt is not None:
        _plot_metrics_per_class(
            labels, precision, recall, f1,
            base_dir / _FILENAMES["metrics_viz"][feature_type],
        )
        _plot_confusion_details(
            labels, tp, fp, fn, tn,
            base_dir / _FILENAMES["confusion_details"][feature_type],
        )
        _plot_overall_metrics(
            acc, y_test, y_pred, labels,
            base_dir / _FILENAMES["overall_metrics"][feature_type],
        )

    if plt is not None and sns is not None:
        _plot_confusion_matrix(
            cm, labels, feature_type,
            base_dir / _FILENAMES["confusion_matrix"][feature_type],
        )
        _plot_confusion_matrix_per_class(
            labels, y_test, y_pred, feature_type,
            base_dir / _FILENAMES["confusion_per_class"][feature_type],
        )

    # ── Simpan model ────────────────────────────────────────────────────────
    if save_path is not None:
        payload = {
            "model":          model,
            "hist_bins":      hist_bins,
            "feature_type":   feature_type,
            "feature_names":  feature_names,
            "labels":         labels.tolist(),
            "confusion_matrix": cm.tolist(),
        }
        joblib.dump(payload, save_path)
        print(f"Saved model to: {save_path}")

    return model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    feature_types: list[FeatureType] = ["cielab", "hsv", "combined"]
    models: dict[str, Pipeline] = {}

    for feature_type in feature_types:
        sep = "=" * 80
        print(f"\n{sep}")
        print(f"  Processing: {feature_type.upper()} Features")
        print(f"{sep}\n")

        df = load_dataset(data_dir, hist_bins=HIST_BINS, feature_type=feature_type)
        if df.empty:
            raise RuntimeError(
                "No images found. Pastikan struktur folder dataset benar:\n"
                "  data/<nama_kelas>/<file_gambar>"
            )

        # Simpan CSV fitur mentah
        csv_path = base_dir / "results/features" / f"{feature_type}_features.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved features CSV : {csv_path}")

        print("\nClass distribution:")
        print(df["label"].value_counts().to_string())

        models[feature_type] = train_and_evaluate(
            df,
            hist_bins=HIST_BINS,
            feature_type=feature_type,
            save_path=base_dir / "models" / f"{feature_type}_svm.joblib",
            output_dir=base_dir,
        )

    sep = "=" * 80
    print(f"\n{sep}")
    print("  All models trained successfully!")
    print(f"{sep}")


if __name__ == "__main__":
    main()