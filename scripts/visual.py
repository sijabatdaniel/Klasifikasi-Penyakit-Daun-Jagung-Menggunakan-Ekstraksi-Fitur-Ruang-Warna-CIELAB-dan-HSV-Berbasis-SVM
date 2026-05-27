from pathlib import Path

import cv2
import numpy as np

try:
	import matplotlib.pyplot as plt
	import seaborn as sns
except Exception:
	plt = None
	sns = None


def visualize_cielab_conversion(image_bgr: np.ndarray, image_path: str | None = None) -> None:
	"""Visualisasi perbandingan BGR original vs CIELAB channels."""
	if plt is None:
		print("Matplotlib not available for visualization")
		return
	
	lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
	l_chan, a_chan, b_chan = cv2.split(lab)
	
	# Konversi BGR ke RGB untuk tampilan
	image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
	
	fig, axes = plt.subplots(2, 2, figsize=(12, 10))
	
	# Original (RGB)
	axes[0, 0].imshow(image_rgb)
	axes[0, 0].set_title("Original (RGB)", fontsize=12, fontweight="bold")
	axes[0, 0].axis("off")
	
	# L Channel
	axes[0, 1].imshow(l_chan, cmap="gray")
	axes[0, 1].set_title(f"L Channel (Lightness)\nMean: {np.mean(l_chan):.2f}, Std: {np.std(l_chan):.2f}", fontsize=12, fontweight="bold")
	axes[0, 1].axis("off")
	
	# a Channel
	axes[1, 0].imshow(a_chan, cmap="RdYlGn_r")
	axes[1, 0].set_title(f"a Channel (Green-Red)\nMean: {np.mean(a_chan):.2f}, Std: {np.std(a_chan):.2f}", fontsize=12, fontweight="bold")
	axes[1, 0].axis("off")
	
	# b Channel
	axes[1, 1].imshow(b_chan, cmap="YlGnBu_r")
	axes[1, 1].set_title(f"b Channel (Blue-Yellow)\nMean: {np.mean(b_chan):.2f}, Std: {np.std(b_chan):.2f}", fontsize=12, fontweight="bold")
	axes[1, 1].axis("off")
	
	fig.tight_layout()
	
	# Simpan dengan nama dari image path jika ada
	if image_path:
		output_name = Path(image_path).stem + "_cielab_conversion.png"
	else:
		output_name = "cielab_conversion_sample.png"
	
	output_path = Path(__file__).resolve().parent / output_name
	fig.savefig(output_path, dpi=150, bbox_inches="tight")
	print(f"Visualisasi CIELAB disimpan: {output_path}")
	plt.close(fig)


def visualize_cielab_features(image_bgr: np.ndarray, hist_bins: int = 16, image_path: str | None = None) -> None:
	"""Visualisasi semua fitur CIELAB yang diekstrak (mean, std, histogram)."""
	if plt is None:
		print("Matplotlib not available for visualization")
		return
	
	lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
	l_chan, a_chan, b_chan = cv2.split(lab)
	channels = {"L": l_chan, "a": a_chan, "b": b_chan}
	channel_colors = {"L": "gray", "a": "#FF6B6B", "b": "#4ECDC4"}
	
	fig = plt.figure(figsize=(15, 10))
	gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
	
	for row, (ch_name, ch_data) in enumerate(channels.items()):
		mean_val = np.mean(ch_data)
		std_val = np.std(ch_data)
		
		# Kolom 1: Channel visualization
		ax1 = fig.add_subplot(gs[row, 0])
		im = ax1.imshow(ch_data, cmap="gray" if ch_name == "L" else "viridis")
		ax1.set_title(f"{ch_name} Channel\nMean: {mean_val:.2f}, Std: {std_val:.2f}", 
					 fontsize=11, fontweight="bold")
		ax1.axis("off")
		plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
		
		# Kolom 2: Histogram
		ax2 = fig.add_subplot(gs[row, 1])
		hist = cv2.calcHist([ch_data], [0], None, [hist_bins], [0, 256])
		hist = hist.flatten()
		ax2.bar(range(hist_bins), hist, color=channel_colors[ch_name], alpha=0.7, edgecolor="black")
		ax2.set_title(f"{ch_name} Channel Histogram ({hist_bins} bins)", fontsize=11, fontweight="bold")
		ax2.set_xlabel("Bin")
		ax2.set_ylabel("Frequency")
		ax2.grid(axis="y", alpha=0.3)
		
		# Kolom 3: Statistik detail
		ax3 = fig.add_subplot(gs[row, 2])
		ax3.axis("off")
		
		# Hitung statistik lengkap
		min_val = np.min(ch_data)
		max_val = np.max(ch_data)
		median_val = np.median(ch_data)
		q25_val = np.percentile(ch_data, 25)
		q75_val = np.percentile(ch_data, 75)
		
		stats_text = f"""
		{ch_name} Channel Statistics:
		
		Mean:     {mean_val:.4f}
		Std Dev:  {std_val:.4f}
		Min:      {min_val:.4f}
		Max:      {max_val:.4f}
		Median:   {median_val:.4f}
		Q25:      {q25_val:.4f}
		Q75:      {q75_val:.4f}
		IQR:      {q75_val - q25_val:.4f}
		"""
		
		ax3.text(0.1, 0.5, stats_text, fontsize=10, family="monospace",
				verticalalignment="center", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
	
	fig.suptitle("CIELAB Features Extraction - Complete Visualization", fontsize=14, fontweight="bold", y=0.995)
	
	# Simpan
	if image_path:
		output_name = Path(image_path).stem + "_cielab_features.png"
	else:
		output_name = "cielab_features_sample.png"
	
	output_path = Path(__file__).resolve().parent / output_name
	fig.savefig(output_path, dpi=150, bbox_inches="tight")
	print(f"Visualisasi fitur CIELAB disimpan: {output_path}")
	plt.close(fig)


def visualize_histogram_comparison(image_bgr: np.ndarray, hist_bins: int = 16, image_path: str | None = None) -> None:
	"""Visualisasi histogram perbandingan RGB vs CIELAB dengan detail."""
	if plt is None:
		print("Matplotlib not available for visualization")
		return
	
	# RGB histogram
	image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
	b_hist = cv2.calcHist([image_bgr], [0], None, [hist_bins], [0, 256])
	g_hist = cv2.calcHist([image_bgr], [1], None, [hist_bins], [0, 256])
	r_hist = cv2.calcHist([image_bgr], [2], None, [hist_bins], [0, 256])
	
	# CIELAB histogram
	lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
	l_chan, a_chan, b_chan = cv2.split(lab)
	l_hist = cv2.calcHist([l_chan], [0], None, [hist_bins], [0, 256])
	a_hist = cv2.calcHist([a_chan], [0], None, [hist_bins], [0, 256])
	b_hist_lab = cv2.calcHist([b_chan], [0], None, [hist_bins], [0, 256])
	
	# Normalisasi histogram
	b_hist = cv2.normalize(b_hist, b_hist).flatten()
	g_hist = cv2.normalize(g_hist, g_hist).flatten()
	r_hist = cv2.normalize(r_hist, r_hist).flatten()
	l_hist = cv2.normalize(l_hist, l_hist).flatten()
	a_hist = cv2.normalize(a_hist, a_hist).flatten()
	b_hist_lab = cv2.normalize(b_hist_lab, b_hist_lab).flatten()
	
	fig = plt.figure(figsize=(16, 10))
	gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.3)
	
	# RGB histograms
	ax_rgb = fig.add_subplot(gs[0, 0])
	ax_rgb.imshow(image_rgb)
	ax_rgb.set_title("Original Image (RGB)", fontsize=12, fontweight="bold")
	ax_rgb.axis("off")
	
	ax_b = fig.add_subplot(gs[0, 1])
	ax_b.plot(b_hist, color='blue', linewidth=2)
	ax_b.fill_between(range(len(b_hist)), b_hist, alpha=0.3, color='blue')
	ax_b.set_title("Blue Channel Histogram", fontsize=12, fontweight="bold")
	ax_b.set_xlim([0, hist_bins])
	ax_b.grid(axis="y", alpha=0.3)
	
	ax_g = fig.add_subplot(gs[0, 2])
	ax_g.plot(g_hist, color='green', linewidth=2)
	ax_g.fill_between(range(len(g_hist)), g_hist, alpha=0.3, color='green')
	ax_g.set_title("Green Channel Histogram", fontsize=12, fontweight="bold")
	ax_g.set_xlim([0, hist_bins])
	ax_g.grid(axis="y", alpha=0.3)
	
	ax_r = fig.add_subplot(gs[0, 3])
	ax_r.plot(r_hist, color='red', linewidth=2)
	ax_r.fill_between(range(len(r_hist)), r_hist, alpha=0.3, color='red')
	ax_r.set_title("Red Channel Histogram", fontsize=12, fontweight="bold")
	ax_r.set_xlim([0, hist_bins])
	ax_r.grid(axis="y", alpha=0.3)
	
	# CIELAB histograms
	lab_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
	ax_lab = fig.add_subplot(gs[1, 0])
	ax_lab.imshow(lab_rgb)
	ax_lab.set_title("Image in CIELAB Space", fontsize=12, fontweight="bold")
	ax_lab.axis("off")
	
	ax_l = fig.add_subplot(gs[1, 1])
	ax_l.plot(l_hist, color='gray', linewidth=2)
	ax_l.fill_between(range(len(l_hist)), l_hist, alpha=0.3, color='gray')
	ax_l.set_title("L Channel Histogram (Lightness)", fontsize=12, fontweight="bold")
	ax_l.set_xlim([0, hist_bins])
	ax_l.grid(axis="y", alpha=0.3)
	
	ax_a = fig.add_subplot(gs[1, 2])
	ax_a.plot(a_hist, color='#FF6B6B', linewidth=2)
	ax_a.fill_between(range(len(a_hist)), a_hist, alpha=0.3, color='#FF6B6B')
	ax_a.set_title("a Channel Histogram (Green-Red)", fontsize=12, fontweight="bold")
	ax_a.set_xlim([0, hist_bins])
	ax_a.grid(axis="y", alpha=0.3)
	
	ax_b_lab = fig.add_subplot(gs[1, 3])
	ax_b_lab.plot(b_hist_lab, color='#4ECDC4', linewidth=2)
	ax_b_lab.fill_between(range(len(b_hist_lab)), b_hist_lab, alpha=0.3, color='#4ECDC4')
	ax_b_lab.set_title("b Channel Histogram (Blue-Yellow)", fontsize=12, fontweight="bold")
	ax_b_lab.set_xlim([0, hist_bins])
	ax_b_lab.grid(axis="y", alpha=0.3)
	
	fig.suptitle("Histogram Comparison: RGB vs CIELAB", fontsize=14, fontweight="bold", y=0.995)
	
	# Simpan
	if image_path:
		output_name = Path(image_path).stem + "_histogram_comparison.png"
	else:
		output_name = "histogram_comparison_sample.png"
	
	output_path = Path(__file__).resolve().parent / output_name
	fig.savefig(output_path, dpi=150, bbox_inches="tight")
	print(f"Visualisasi histogram disimpan: {output_path}")
	plt.close(fig)


def visualize_hsv_features(image_bgr: np.ndarray, hist_bins: int = 16, image_path: str | None = None) -> None:
	"""Visualisasi fitur HSV (Hue, Saturation, Value) dengan histogram dan statistik."""
	if plt is None:
		print("Matplotlib not available for visualization")
		return
	
	hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
	h_chan, s_chan, v_chan = cv2.split(hsv)
	channels = {"H": h_chan, "S": s_chan, "V": v_chan}
	channel_colors = {"H": "#FF9500", "S": "#FF6B6B", "V": "#4ECDC4"}
	
	fig = plt.figure(figsize=(15, 10))
	gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
	
	for row, (ch_name, ch_data) in enumerate(channels.items()):
		mean_val = np.mean(ch_data)
		std_val = np.std(ch_data)
		
		# Kolom 1: Channel visualization
		ax1 = fig.add_subplot(gs[row, 0])
		im = ax1.imshow(ch_data, cmap="hsv" if ch_name == "H" else "gray")
		ch_desc = "Hue (Color)" if ch_name == "H" else "Saturation (Purity)" if ch_name == "S" else "Value (Brightness)"
		ax1.set_title(f"{ch_name} Channel ({ch_desc})\nMean: {mean_val:.2f}, Std: {std_val:.2f}", 
					 fontsize=11, fontweight="bold")
		ax1.axis("off")
		plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
		
		# Kolom 2: Histogram
		ax2 = fig.add_subplot(gs[row, 1])
		hist = cv2.calcHist([ch_data], [0], None, [hist_bins], 
							[0, 180] if ch_name == "H" else [0, 256])
		hist = hist.flatten()
		ax2.bar(range(hist_bins), hist, color=channel_colors[ch_name], alpha=0.7, edgecolor="black")
		ax2.set_title(f"{ch_name} Channel Histogram ({hist_bins} bins)", fontsize=11, fontweight="bold")
		ax2.set_xlabel("Bin")
		ax2.set_ylabel("Frequency")
		ax2.grid(axis="y", alpha=0.3)
		
		# Kolom 3: Statistik detail
		ax3 = fig.add_subplot(gs[row, 2])
		ax3.axis("off")
		
		# Hitung statistik lengkap
		min_val = np.min(ch_data)
		max_val = np.max(ch_data)
		median_val = np.median(ch_data)
		q25_val = np.percentile(ch_data, 25)
		q75_val = np.percentile(ch_data, 75)
		
		stats_text = f"""
		{ch_name} Channel Statistics:
		
		Mean:     {mean_val:.4f}
		Std Dev:  {std_val:.4f}
		Min:      {min_val:.4f}
		Max:      {max_val:.4f}
		Median:   {median_val:.4f}
		Q25:      {q25_val:.4f}
		Q75:      {q75_val:.4f}
		IQR:      {q75_val - q25_val:.4f}
		"""
		
		ax3.text(0.1, 0.5, stats_text, fontsize=10, family="monospace",
				verticalalignment="center", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
	
	fig.suptitle("HSV Features Extraction - Complete Visualization", fontsize=14, fontweight="bold", y=0.995)
	
	# Simpan
	if image_path:
		output_name = Path(image_path).stem + "_hsv_features.png"
	else:
		output_name = "hsv_features_sample.png"
	
	output_path = Path(__file__).resolve().parent / output_name
	fig.savefig(output_path, dpi=150, bbox_inches="tight")
	print(f"Visualisasi fitur HSV disimpan: {output_path}")
	plt.close(fig)


def main(data_folder: str = "coba"):
	"""Test visualisasi dengan sample image dari data folder.
	
	Args:
		data_folder: Nama folder data. Default: "coba", fallback: "data"
	"""
	base_dir = Path(__file__).resolve().parent
	data_dir = base_dir / data_folder
	
	# Fallback ke folder "data" jika folder utama tidak ada
	if not data_dir.exists():
		data_dir = base_dir / "data"
	
	if not data_dir.exists():
		print(f"❌ Data folder tidak ditemukan: {base_dir / data_folder}")
		print("\nBuat folder structure seperti ini:")
		print(f"""
{data_folder}/ atau data/
  ├── blight/
  │   ├── image1.jpg
  │   └── ...
  ├── common_rust/
  │   ├── image1.jpg
  │   └── ...
  ├── gray_leaf/
  │   ├── image1.jpg
  │   └── ...
  └── healthy/
      ├── image1.jpg
      └── ...
		""")
		return
		return
	
	# Cari gambar pertama dari setiap kelas
	class_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
	
	print(f"✅ Data folder ditemukan: {data_dir}")
	print(f"📁 Ditemukan {len(class_dirs)} kelas: {[d.name for d in class_dirs]}\n")
	
	visualized_count = 0
	
	for class_dir in class_dirs:
		label = class_dir.name
		image_paths = sorted(
			p for p in class_dir.iterdir()
			if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
		)
		
		if not image_paths:
			print(f"⚠️  Kelas '{label}' tidak memiliki gambar")
			continue
		
		# Baca gambar pertama
		image_path = image_paths[0]
		image = cv2.imread(str(image_path))
		
		if image is None:
			print(f"❌ Gagal membaca: {image_path}")
			continue
		
		print(f"🔍 Memproses kelas: {label}")
		print(f"   Gambar: {image_path.name}")
		
		# Jalankan visualisasi
		try:
			visualize_cielab_conversion(image, str(image_path))
			visualize_cielab_features(image, hist_bins=16, image_path=str(image_path))
			visualize_histogram_comparison(image, hist_bins=16, image_path=str(image_path))
			visualize_hsv_features(image, hist_bins=16, image_path=str(image_path))
			visualized_count += 1
			print(f"   ✅ Visualisasi berhasil\n")
		except Exception as e:
			print(f"   ❌ Error: {e}\n")
	
	print(f"\n{'='*50}")
	print(f"✅ Selesai! {visualized_count} kelas divisualisasikan")
	print(f"{'='*50}")


if __name__ == "__main__":
	import sys
	# Gunakan argument pertama sebagai nama folder, atau default "coba"
	folder = sys.argv[1] if len(sys.argv) > 1 else "coba"
	main(data_folder=folder)
