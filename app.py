import streamlit as st
import cv2
import torch
from ultralytics import YOLO
import numpy as np
import os

# === Load YOLO model ===
model_path = 'best.pt'
if not os.path.exists(model_path):
    st.error(f"Model file '{model_path}' tidak ditemukan. Silakan unggah atau periksa path-nya.")
    st.stop()

with st.spinner('🔄 Memuat model YOLOv11...'):
    model = YOLO(model_path)

# === Streamlit Setup ===
st.set_page_config(page_title="YOLOv11 BISINDO Detection", layout="wide")
st.title("🖐️ BISINDO Detection using YOLOv11")

# === Inisialisasi session_state ===
if 'webcam_active' not in st.session_state:
    st.session_state['webcam_active'] = False

# Sidebar Settings
confidence = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.25, 0.01)

# Tombol Start dan Stop
start_button = st.sidebar.button("Start Webcam Detection")
stop_button = st.sidebar.button("Stop Detection")

# Kontrol session_state
if start_button:
    st.session_state['webcam_active'] = True
if stop_button:
    st.session_state['webcam_active'] = False

# Placeholder untuk stream
stframe = st.empty()

# Webcam Detection
if st.session_state['webcam_active']:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while st.session_state['webcam_active'] and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Gagal membaca frame dari webcam.")
            break

        # Predict
        results = model.predict(source=frame, conf=confidence, imgsz=640, verbose=False)
        annotated_frame = results[0].plot()

        # Convert BGR ke RGB
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        # Display frame
        stframe.image(annotated_frame, channels="RGB", use_column_width=True)

        # Cek lagi apakah user tekan STOP
        if not st.session_state['webcam_active']:
            break

    cap.release()
    st.success("✅ Webcam stopped.")

# Footer
st.markdown("---")
st.caption("alfhisa.github.io")
