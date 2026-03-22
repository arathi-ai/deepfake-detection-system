# 🔍 Deepfake Detection System

A deep learning web application that detects AI-generated deepfake images and videos using MobileNetV2 architecture.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/arathi-ai/deepfake-detector)
[![GitHub](https://img.shields.io/badge/GitHub-arathi--ai-black)](https://github.com/arathi-ai/deepfake-detection-system)

---

## 🚀 Live Demo

Try it here → [https://huggingface.co/spaces/arathi-ai/deepfake-detector](https://huggingface.co/spaces/arathi-ai/deepfake-detector)

> Test the app live — upload any face image or video to get a real vs deepfake prediction instantly.

---

## 📌 About

With the rapid rise of AI-generated media, detecting manipulated facial content has become a critical challenge. This project addresses that problem by building an end-to-end deepfake detection system capable of analyzing both images and videos in real time.

---

## ✨ Features

- ✅ Real-time image deepfake detection
- ✅ Video deepfake detection with frame-by-frame face analysis
- ✅ Confidence score displayed for every prediction
- ✅ Clean web interface built with Gradio
- ✅ Flask REST API version included

---

## 🔧 How It Works

**Image Detection**
- Uploaded face image is resized to 160x160
- Normalized and passed through fine-tuned MobileNetV2 model
- Returns REAL or FAKE with confidence score

**Video Detection**
- OpenCV samples approximately 40 frames from the video
- Haar Cascade face detector finds faces in each frame
- Each detected face is analyzed independently through the model
- Final verdict based on minimum real-score across all analyzed frames

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | TensorFlow / Keras |
| Model | MobileNetV2 (transfer learning) |
| Computer Vision | OpenCV |
| Web Framework | Flask, Gradio |
| Data Processing | NumPy, Pillow |
| Language | Python |

---

## 📁 Project Structure

```
deepfake-detection-system/
│
├── app.py              # Gradio web app (Hugging Face deployment)
├── flask_app.py        # Original Flask web application
├── requirements.txt    # Dependencies
└── README.md
```

---

## ⚙️ Installation
# Install dependencies
pip install -r requirements.txt

# Run Flask app
python flask_app.py
```

---

## 📊 Model

MobileNetV2 was chosen for its lightweight architecture and strong performance on image classification tasks. The model was fine-tuned on a deepfake dataset with:
- Image resizing to 160x160
- Normalization and data augmentation
- Transfer learning from ImageNet weights

---

## 👩‍💻 Built By

[GitHub](https://github.com/arathi-ai) | [Hugging Face](https://huggingface.co/spaces/arathi-ai/deepfake-detector)
