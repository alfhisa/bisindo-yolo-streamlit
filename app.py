import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os

# === Setup Streamlit ===
st.set_page_config(page_title="YOLOv11 BISINDO Detection", layout="wide")
st.title("🖐️ BISINDO Detection using YOLOv11")

# === Load Model ===
model_path = 'best.pt'
if not os.path.exists(model_path):
    st.error(f"❌ Model file '{model_path}' tidak ditemukan. Harap unggah atau pastikan path-nya benar.")
    st.stop()

with st.spinner('🔄 Memuat model YOLOv11...'):
    model = YOLO(model_path)

# === Sidebar Controls ===
st.sidebar.header("🎛️ Kontrol")
confidence = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.25, 0.01)
start_button = st.sidebar.button("▶️ Start Webcam")
stop_button = st.sidebar.button("⏹️ Stop Webcam")

# === Webcam State ===
if 'webcam_active' not in st.session_state:
    st.session_state['webcam_active'] = False

if start_button:
    st.session_state['webcam_active'] = True
if stop_button:
    st.session_state['webcam_active'] = False

# === Video Stream Placeholder ===
stframe = st.empty()

# === Webcam Loop ===
if st.session_state['webcam_active']:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Gagal mengakses webcam. Pastikan tidak sedang digunakan oleh aplikasi lain.")
        st.session_state['webcam_active'] = False
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while st.session_state['webcam_active']:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Gagal membaca frame dari webcam.")
                break

            # Prediksi
            results = model.predict(source=frame, conf=confidence, imgsz=640, verbose=False)
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Tampilkan frame
            stframe.image(annotated_frame, channels="RGB", use_column_width=True)

        cap.release()
        stframe.empty()
        st.success("✅ Webcam stopped.")

# === Footer ===
st.markdown("---")
st.caption("🔗 Visit: [alfhisa.github.io](https://alfhisa.github.io)")