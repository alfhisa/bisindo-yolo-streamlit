import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import os

# === Setup Streamlit ===
st.set_page_config(page_title="YOLOv11 BISINDO Detection", layout="wide")
st.title("🖐️ BISINDO Detection with YOLOv11")

# === Load YOLO Model ===
model_path = 'best.pt'
if not os.path.exists(model_path):
    st.error(f"❌ Model file '{model_path}' tidak ditemukan. Harap unggah atau pastikan path-nya benar.")
    st.stop()

with st.spinner("🔄 Memuat model YOLOv11..."):
    model = YOLO(model_path)

# === Sidebar Settings ===
st.sidebar.header("🎛️ Kontrol")
confidence = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.25, 0.01)

# === Webcam VideoTransformer ===
class YOLOTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(source=img, conf=confidence, imgsz=640, verbose=False)
        annotated = results[0].plot()
        return annotated

# === WebRTC Streamer ===
webrtc_streamer(
    key="yolo-bisindo",
    video_transformer_factory=YOLOTransformer,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)


# === Footer ===
st.markdown("---")
st.caption("🔗 Visit: [alfhisa.github.io](https://alfhisa.github.io)")
