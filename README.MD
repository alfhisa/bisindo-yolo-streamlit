# 🖐️ BISINDO Detection using YOLOv11

Real-time Indonesian Sign Language (BISINDO) recognition powered by [YOLOv11](https://github.com/ultralytics/ultralytics) and deployed with [Streamlit](https://streamlit.io/). This app runs directly in the browser using WebRTC webcam integration.

![YOLOv11 BISINDO Detection Screenshot](https://github.com/alfhisa/bisindo-yolo-streamlit/raw/main/screenshot.png)

---

## 🔧 Features

- ✅ Real-time detection of BISINDO signs from webcam
- ⚡ Built with state-of-the-art YOLOv11 object detection model
- 🌐 Deployable in web browser via Streamlit + WebRTC
- 📆 Lightweight and interactive web app

---

## 🚀 Live Demo

📍 [Try it on Streamlit Cloud](https://bisindo-yolo.streamlit.app/)  

---

## 📁 Project Structure

```
├── app.py                # Streamlit main app
├── best.pt               # YOLOv11 trained model (BISINDO)
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 🧪 Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/alfhisa/bisindo-yolo-streamlit.git
cd bisindo-yolo-streamlit
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add model file

Place your `best.pt` file (YOLOv11 trained weights) in the root directory.

### 4. Run Streamlit app

```bash
streamlit run app.py
```

---

## 🧠 Model

Trained using YOLOv11 on a custom BISINDO sign dataset.  
Supports **A-Z** hand gesture detection.

> Want to train your own? Use [Ultralytics](https://docs.ultralytics.com) and label your dataset in YOLO format.

---

## 📆 Deployment

This app is fully compatible with [Streamlit Cloud](https://streamlit.io/cloud).

Just push your repo and model, and it runs in the browser!

---

## 📷 Troubleshooting Webcam

If webcam fails to load:
- Make sure your browser gives camera access
- Use Google Chrome or Firefox (Safari might block WebRTC)
- Ensure no other app is using webcam
- If deployed: Check STUN server and firewall

---

## 🙌 Credits

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv11
- [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) for browser webcam support
- [@alfhisa](https://github.com/alfhisa) for project development

---

## 📄 License

MIT © 2025 Muhammad Alfhi Saputra
