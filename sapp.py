from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from tubes import extract_combined_features, HIST_BINS, MODEL_FILENAME


@st.cache_resource
def load_model(model_path: Path) -> dict:
    return joblib.load(model_path)


def predict_image(image_bgr: np.ndarray, payload: dict) -> tuple[str, pd.DataFrame]:
    hist_bins = payload["hist_bins"]
    feature_names = payload["feature_names"]
    labels = payload["labels"]
    model = payload["model"]

    features = extract_combined_features(image_bgr, hist_bins=hist_bins)
    features_df = pd.DataFrame([features], columns=feature_names)
    pred_label = model.predict(features_df)[0]

    probs = model.predict_proba(features_df)[0]
    probs_df = pd.DataFrame({"label": labels, "probability": probs}).sort_values(
        "probability", ascending=False
    )

    return pred_label, probs_df


st.set_page_config(page_title="Deteksi Penyakit Daun Jagung", layout="wide")

st.title("🌽 Deteksi Penyakit Daun Jagung")
st.markdown("Prediksi penyakit daun jagung menggunakan fitur warna **CIELAB + HSV (Hybrid)** dan **SVM Classification**")

st.divider()

base_dir = Path(__file__).resolve().parent
model_path = base_dir / MODEL_FILENAME

if not model_path.exists():
    st.error("Model belum ada. Jalankan dulu: python3 tubes.py")
    st.stop()

payload = load_model(model_path)

# Buat tabs
tab1, tab2 = st.tabs(["🔮 Prediksi", "📈 Model Info"])

# ===== TAB 1: PREDIKSI =====
with tab1:
    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        st.subheader("📸 Upload Gambar")
        st.markdown("Pilih gambar daun jagung dengan pencahayaan yang cukup baik.")
        
        uploaded_file = st.file_uploader(
            "Upload gambar", 
            type=["jpg", "jpeg", "png", "bmp"],
            label_visibility="collapsed"
        )
        
        st.write("")
        run_prediction = st.button("🔍 Prediksi", use_container_width=True, type="primary")

    image_bgr = None
    image_rgb = None

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_bgr is None:
            st.error("Gagal membaca gambar. Coba file lain.")
            st.stop()

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with right_col:
        st.subheader("📊 Hasil Prediksi")
        
        if image_rgb is not None:
            st.image(image_rgb, caption="Gambar input", use_container_width=True)
        
        if run_prediction and image_bgr is not None:
            with st.spinner("⏳ Memproses gambar..."):
                pred_label, probs_df = predict_image(image_bgr, payload)
            
            confidence = float(probs_df.iloc[0]["probability"])
            
            st.divider()
            
            col_pred, col_conf = st.columns(2)
            with col_pred:
                st.metric(label="✅ Prediksi", value=pred_label)
            with col_conf:
                st.metric(label="📈 Confidence", value=f"{confidence:.1%}")
            
            st.divider()
            
            st.write("**Probabilitas Semua Kelas:**")
            
            chart_data = probs_df.set_index("label")["probability"]
            st.bar_chart(chart_data, use_container_width=True)
            
            st.write("")
            st.dataframe(
                probs_df.rename(columns={
                    "label": "Kelas Penyakit",
                    "probability": "Probabilitas"
                }).assign(Probabilitas=lambda x: x["Probabilitas"].apply(lambda y: f"{y:.2%}")),
                use_container_width=True,
                hide_index=True
            )
        
        elif uploaded_file and not run_prediction:
            st.info("👆 Klik tombol **Prediksi** untuk mendapatkan hasil analisis")

# ===== TAB 2: MODEL INFO =====
with tab2:
    st.subheader("Confusion Matrix (Test Set)")
    if "confusion_matrix" in payload:
        cm_image_path = base_dir / "results/visualizations/confusion_matrix_combined.png"
        if cm_image_path.exists():
            st.image(str(cm_image_path), use_container_width=True)
        else:
            st.warning("File confusion matrix belum ada. Jalankan: python3 tubes.py")
    
    st.divider()
    
    st.subheader("Overall Metrics")
    overall_metrics_path = base_dir / "results/visualizations/overall_metrics_combined.png"
    if overall_metrics_path.exists():
        st.image(str(overall_metrics_path), use_container_width=True)
    else:
        st.warning("File metrics belum ada. Jalankan: python3 tubes.py")
    
    st.divider()
    
    st.subheader("Metrics per Kelas")
    metrics_path = base_dir / "results/visualizations/metrics_per_class_visualization_combined.png"
    if metrics_path.exists():
        st.image(str(metrics_path), use_container_width=True)
    else:
        st.warning("File metrics per kelas belum ada. Jalankan: python3 tubes.py")
    
    st.divider()
    
    st.subheader("Informasi Model")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Fitur Warna", "CIELAB+HSV")
    with col2:
        st.metric("Statistik", "Mean+Std")
    with col3:
        st.metric("Histogram Bins", HIST_BINS)
    with col4:
        st.metric("Klasifier", "SVM RBF")
    
    st.divider()
    
    st.markdown("**Kelas yang Diprediksi:**")
    for label in payload["labels"]:
        st.write(f"• {label}")

st.divider()
st.caption("Fitur: CIELAB (54 fitur) + HSV (54 fitur) = 108 fitur hybrid | Built with Streamlit")
