from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
	import matplotlib.pyplot as plt
	import seaborn as sns
except Exception:  # pragma: no cover - plotting is optional
	plt = None
	sns = None

try:
	from visual import (
		visualize_cielab_conversion,
		visualize_cielab_features,
		visualize_histogram_comparison,
		visualize_hsv_features,
	)
except ImportError:
	# Jika visual.py tidak tersedia, visualisasi tidak akan ditampilkan
	visualize_cielab_conversion = None
	visualize_cielab_features = None
	visualize_histogram_comparison = None
	visualize_hsv_features = None


HIST_BINS = 16
MODEL_FILENAME = "cielab_svm.joblib"


def build_feature_names(hist_bins: int) -> list:
	names = [
		"L_mean",
		"L_std",
		"a_mean",
		"a_std",
		"b_mean",
		"b_std",
	]
	for channel in ("L", "a", "b"):
		for i in range(hist_bins):
			names.append(f"{channel}_hist_{i}")
	
	# HSV features
	names.extend([
		"H_mean",
		"H_std",
		"S_mean",
		"S_std",
		"V_mean",
		"V_std",
	])
	for channel in ("H", "S", "V"):
		for i in range(hist_bins):
			names.append(f"{channel}_hist_{i}")
	
	return names


def extract_cielab_features(image_bgr: np.ndarray, hist_bins: int = 16) -> np.ndarray:
	lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
	l_chan, a_chan, b_chan = cv2.split(lab)

	means = [np.mean(l_chan), np.mean(a_chan), np.mean(b_chan)]
	stds = [np.std(l_chan), np.std(a_chan), np.std(b_chan)]

	hist_features = []
	for channel in (l_chan, a_chan, b_chan):
		hist = cv2.calcHist([channel], [0], None, [hist_bins], [0, 256])
		hist = hist.flatten().astype(np.float32)
		hist_sum = np.sum(hist)
		if hist_sum > 0:
			hist = hist / hist_sum
		hist_features.extend(hist.tolist())

	return np.array(means + stds + hist_features, dtype=np.float32)


def extract_hsv_features(image_bgr: np.ndarray, hist_bins: int = 16) -> np.ndarray:
	"""Extract HSV features: mean, std, dan histogram untuk setiap channel."""
	hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
	h_chan, s_chan, v_chan = cv2.split(hsv)

	means = [np.mean(h_chan), np.mean(s_chan), np.mean(v_chan)]
	stds = [np.std(h_chan), np.std(s_chan), np.std(v_chan)]

	hist_features = []
	# H: 0-180, S: 0-255, V: 0-255
	hist_ranges = [(0, 180), (0, 256), (0, 256)]
	for idx, (channel, max_val) in enumerate(zip((h_chan, s_chan, v_chan), hist_ranges)):
		hist = cv2.calcHist([channel], [0], None, [hist_bins], hist_ranges[idx])
		hist = hist.flatten().astype(np.float32)
		hist_sum = np.sum(hist)
		if hist_sum > 0:
			hist = hist / hist_sum
		hist_features.extend(hist.tolist())

	return np.array(means + stds + hist_features, dtype=np.float32)


def extract_combined_features(image_bgr: np.ndarray, hist_bins: int = 16) -> np.ndarray:
	"""Extract combined CIELAB + HSV features untuk hybrid approach."""
	cielab_features = extract_cielab_features(image_bgr, hist_bins=hist_bins)
	hsv_features = extract_hsv_features(image_bgr, hist_bins=hist_bins)
	return np.concatenate([cielab_features, hsv_features], dtype=np.float32)


def load_dataset(data_dir: Path, hist_bins: int = 16) -> pd.DataFrame:
	rows = []
	feature_names = build_feature_names(hist_bins)
	class_dirs = [p for p in sorted(data_dir.iterdir()) if p.is_dir()]
	visualized_classes = set()

	for class_dir in class_dirs:
		label = class_dir.name
		image_paths = sorted(
			p
			for p in class_dir.iterdir()
			if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
		)

		for idx, image_path in enumerate(image_paths, start=1):
			image = cv2.imread(str(image_path))
			if image is None:
				continue

			# Visualisasi satu gambar dari setiap kelas (CIELAB + HSV hybrid)
			if label not in visualized_classes and visualize_cielab_conversion is not None:
				print(f"Visualizing sample from class: {label}")
				visualize_cielab_conversion(image, str(image_path))
				visualize_cielab_features(image, hist_bins=hist_bins, image_path=str(image_path))
				visualize_histogram_comparison(image, hist_bins=hist_bins, image_path=str(image_path))
				visualize_hsv_features(image, hist_bins=hist_bins, image_path=str(image_path))
				visualized_classes.add(label)

			features = extract_combined_features(image, hist_bins=hist_bins)
			row = {
				"label": label,
				"path": str(image_path),
			}
			row.update({name: value for name, value in zip(feature_names, features)})
			rows.append(row)

			if idx % 250 == 0:
				print(f"Loaded {idx} images from {label}...")

	return pd.DataFrame(rows)


def train_and_evaluate(df: pd.DataFrame, hist_bins: int = 16, save_path: Path | None = None) -> Pipeline:
	feature_names = build_feature_names(hist_bins)
	x = df[feature_names].to_numpy(dtype=np.float32)
	y = df["label"].to_numpy()

	x_train, x_test, y_train, y_test = train_test_split(
		x, y, test_size=0.2, random_state=42, stratify=y
	)

	model = Pipeline(
		[
			("scaler", StandardScaler()),
			("svm", SVC(kernel="rbf", C=10.0, gamma="scale", probability=True)),
		]
	)

	model.fit(x_train, y_train)
	y_pred = model.predict(x_test)

	acc = accuracy_score(y_test, y_pred)
	print("Accuracy:", acc)
	print("Classification Report:\n", classification_report(y_test, y_pred))

	labels = np.unique(y)
	cm = confusion_matrix(y_test, y_pred, labels=labels)
	
	# Hitung metrics per kelas
	from sklearn.metrics import precision_score, recall_score, f1_score
	precision = precision_score(y_test, y_pred, labels=labels, average=None)
	recall = recall_score(y_test, y_pred, labels=labels, average=None)
	f1 = f1_score(y_test, y_pred, labels=labels, average=None)
	
	# Buat visualization untuk confusion matrix
	if plt is not None and sns is not None:
		fig, ax = plt.subplots(figsize=(8, 6))
		sns.heatmap(
			cm,
			annot=True,
			fmt="d",
			cmap="Blues",
			xticklabels=labels,
			yticklabels=labels,
			cbar_kws={"label": "Count"},
			ax=ax,
			annot_kws={"size": 12, "weight": "bold", "color": "black"}
		)
		ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
		ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
		ax.set_title("Confusion Matrix - CIELAB Features", fontsize=14, fontweight="bold")
		fig.tight_layout()
		output_path = Path(__file__).resolve().parent / "confusion_matrix_cielab.png"
		fig.savefig(output_path, dpi=150)
		plt.close(fig)
	
	# Buat visualization untuk metrics (Precision, Recall, F1-Score)
	if plt is not None:
		fig, ax = plt.subplots(figsize=(10, 6))
		x = np.arange(len(labels))
		width = 0.25
		
		bars1 = ax.bar(x - width, precision, width, label="Precision", color="#3498db")
		bars2 = ax.bar(x, recall, width, label="Recall", color="#e74c3c")
		bars3 = ax.bar(x + width, f1, width, label="F1-Score", color="#2ecc71")
		
		# Tambahkan nilai di atas setiap bar
		for bars in [bars1, bars2, bars3]:
			for bar in bars:
				height = bar.get_height()
				ax.text(bar.get_x() + bar.get_width()/2., height,
					f"{height:.3f}",
					ha="center", va="bottom", fontsize=9, fontweight="bold")
		
		ax.set_xlabel("Kelas", fontsize=12, fontweight="bold")
		ax.set_ylabel("Score", fontsize=12, fontweight="bold")
		ax.set_title("Metrics per Kelas (Precision, Recall, F1-Score)", fontsize=14, fontweight="bold")
		ax.set_xticks(x)
		ax.set_xticklabels(labels)
		ax.legend()
		ax.set_ylim([0, 1.1])
		ax.grid(axis="y", alpha=0.3)
		
		fig.tight_layout()
		output_path = Path(__file__).resolve().parent / "metrics_cielab.png"
		fig.savefig(output_path, dpi=150)
		plt.close(fig)
	
	# Buat visualization untuk overall metrics (Accuracy)
	if plt is not None:
		fig, ax = plt.subplots(figsize=(8, 6))
		
		# Hitung overall metrics
		overall_precision = precision_score(y_test, y_pred, labels=labels, average="weighted")
		overall_recall = recall_score(y_test, y_pred, labels=labels, average="weighted")
		overall_f1 = f1_score(y_test, y_pred, labels=labels, average="weighted")
		
		metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
		metrics_values = [acc, overall_precision, overall_recall, overall_f1]
		colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
		
		bars = ax.bar(metrics_names, metrics_values, color=colors, edgecolor="black", linewidth=2)
		
		# Tambahkan nilai di atas bar
		for bar, value in zip(bars, metrics_values):
			height = bar.get_height()
			ax.text(bar.get_x() + bar.get_width()/2., height,
				f"{value:.4f}",
				ha="center", va="bottom", fontsize=12, fontweight="bold")
		
		ax.set_ylabel("Score", fontsize=12, fontweight="bold")
		ax.set_title("Overall Model Metrics", fontsize=14, fontweight="bold")
		ax.set_ylim([0, 1.1])
		ax.grid(axis="y", alpha=0.3)
		
		fig.tight_layout()
		output_path = Path(__file__).resolve().parent / "overall_metrics_cielab.png"
		fig.savefig(output_path, dpi=150)
		plt.close(fig)

	if save_path is not None:
		payload = {
			"model": model,
			"hist_bins": hist_bins,
			"feature_names": feature_names,
			"labels": labels.tolist(),
			"confusion_matrix": cm.tolist(),
		}
		joblib.dump(payload, save_path)
		print(f"Saved model to: {save_path}")

	return model


def main() -> None:
	base_dir = Path(__file__).resolve().parent
	data_dir = base_dir / "data"
	if not data_dir.exists():
		raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

	hist_bins = HIST_BINS
	df = load_dataset(data_dir, hist_bins=hist_bins)
	if df.empty:
		raise RuntimeError("No images found. Check dataset folder structure.")

	output_csv = base_dir / "cielab_features.csv"
	df.to_csv(output_csv, index=False)
	print(f"Saved features to: {output_csv}")

	print("Class distribution:")
	print(df["label"].value_counts())

	model_path = base_dir / MODEL_FILENAME
	train_and_evaluate(df, hist_bins=hist_bins, save_path=model_path)


if __name__ == "__main__":
	main()
