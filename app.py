import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ------------------ LOAD MODEL ------------------
model = tf.keras.models.load_model("deepfake_mobilenetv2_model.keras")
IMG_SIZE = (160, 160)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ------------------ PREDICTION FUNCTIONS ------------------

def predict_full_image(img):
    try:
        img_resized = img.resize(IMG_SIZE)
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array, verbose=0)[0][0]
        label = "REAL" if pred >= 0.5 else "FAKE"
        confidence = pred if pred >= 0.5 else 1 - pred

        return label, float(confidence)
    except Exception as e:
        return f"Error: {str(e)}", 0.0


def predict_face_frame(face):
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, IMG_SIZE)
    img_array = np.expand_dims(face_resized, axis=0)
    img_array = preprocess_input(img_array)
    pred = model.predict(img_array, verbose=0)[0][0]
    return pred


# ------------------ GRADIO FUNCTIONS ------------------

def image_predict(pil_image):
    if pil_image is None:
        return "Please upload an image.", ""

    label, confidence = predict_full_image(pil_image)
    confidence_pct = f"{confidence * 100:.2f}%"

    if label == "FAKE":
        result = f"DEEPFAKE DETECTED\nConfidence: {confidence_pct}"
    else:
        result = f"REAL IMAGE\nConfidence: {confidence_pct}"

    return result


def video_predict(video_path):
    if video_path is None:
        return "Please upload a video."

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = max(frame_count // 40, 1)

    predictions = []
    frames_analyzed = 0

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        if i % sample_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                pred = predict_face_frame(face)
                predictions.append(pred)
                frames_analyzed += 1

    cap.release()

    if len(predictions) == 0:
        return "No face detected in video. Please upload a video with a visible face."

    min_pred = np.min(predictions)
    avg_pred = np.mean(predictions)
    label = "FAKE" if min_pred < 0.5 else "REAL"

    if label == "FAKE":
        result = (
            f"DEEPFAKE DETECTED\n"
            f"Frames analyzed: {frames_analyzed}\n"
            f"Minimum real score: {min_pred:.4f}\n"
            f"Average real score: {avg_pred:.4f}"
        )
    else:
        result = (
            f"REAL VIDEO\n"
            f"Frames analyzed: {frames_analyzed}\n"
            f"Minimum real score: {min_pred:.4f}\n"
            f"Average real score: {avg_pred:.4f}"
        )

    return result


# ------------------ GRADIO UI ------------------

with gr.Blocks(title="Deepfake Detection System") as demo:

    gr.Markdown("""
    # 🔍 Deepfake Detection System
    **Detect AI-generated fake images and videos using MobileNetV2 deep learning model.**
    """)

    with gr.Tabs():

        with gr.Tab("Image Detection"):
            gr.Markdown("Upload a face image to check if it is real or AI-generated.")
            with gr.Row():
                image_input = gr.Image(type="pil", label="Upload Image")
                image_output = gr.Textbox(label="Result", lines=3)
            image_btn = gr.Button("Detect", variant="primary")
            image_btn.click(fn=image_predict, inputs=image_input, outputs=image_output)

        with gr.Tab("Video Detection"):
            gr.Markdown("Upload a video. The system will sample frames, detect faces, and analyze them.")
            video_input = gr.Video(label="Upload Video")
            video_output = gr.Textbox(label="Result", lines=6)
            video_btn = gr.Button("Detect", variant="primary")
            video_btn.click(fn=video_predict, inputs=video_input, outputs=video_output)

    gr.Markdown("""
    ---
    **How it works:** MobileNetV2 is a lightweight convolutional neural network trained to distinguish 
    real face images from AI-generated deepfakes. For videos, faces are detected per frame using 
    OpenCV Haar cascades and analyzed individually.
    """)

demo.launch()
