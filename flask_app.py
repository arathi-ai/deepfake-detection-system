from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ------------------ LOAD MODEL ------------------
model = tf.keras.models.load_model("deepfake_mobilenetv2_model.keras")
IMG_SIZE = (160, 160)

# Face detector (for video frames)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ------------------ PREDICTION FUNCTIONS ------------------

# Face prediction for video frames
def predict_face(face):
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, IMG_SIZE)
    img_array = np.expand_dims(face_resized, axis=0)
    img_array = preprocess_input(img_array)
    pred = model.predict(img_array, verbose=0)[0][0]
    label = "REAL" if pred >= 0.5 else "FAKE"
    confidence = pred if pred >= 0.5 else 1 - pred
    return pred, label, face_rgb, confidence

# Full-image prediction (Tkinter-style logic)
def predict_full_image(img_path):
    try:
        img = Image.open(img_path).convert("RGB")
        img_resized = img.resize(IMG_SIZE)

        # SAME preprocessing your model was trained with
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array, verbose=0)[0][0]
        label = "REAL" if pred >= 0.5 else "FAKE"
        confidence = pred if pred >= 0.5 else 1 - pred

        return label, confidence
    except Exception as e:
        return f"Error: {str(e)}", 0


# ------------------ ROUTES ------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_image", methods=["POST"])
def predict_image():
    file = request.files["image"]
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # --------- Tkinter-style full-image prediction ---------
    label, confidence = predict_full_image(file_path)

    return render_template("result.html",
                           result=label,
                           score=f"{confidence:.4f}",
                           images=[file.filename])

@app.route("/predict_video", methods=["POST"])
def predict_video():
    file = request.files["video"]
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    cap = cv2.VideoCapture(file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = max(frame_count // 40, 1)  # sample ~40 frames

    predictions = []
    face_frames = []  # store up to 4 suspicious frames

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        if i % sample_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                pred, label, face_rgb, _ = predict_face(face)
                predictions.append(pred)

                if len(face_frames) < 4:
                    face_frames.append(face_rgb)

    cap.release()

    if len(predictions) == 0:
        return render_template("result.html",
                               result="No face detected in video",
                               score="N/A",
                               images=None)

    min_pred = np.min(predictions)
    label = "FAKE" if min_pred < 0.5 else "REAL"

    # Save suspicious frames as images
    frame_filenames = []
    for idx, frame_rgb in enumerate(face_frames):
        pil_img = Image.fromarray(frame_rgb)
        frame_file = f"{os.path.splitext(file.filename)[0]}_frame{idx}.jpg"
        pil_img.save(os.path.join(app.config["UPLOAD_FOLDER"], frame_file))
        frame_filenames.append(frame_file)

    return render_template("result.html",
                           result=label,
                           score=f"{min_pred:.4f}",
                           images=frame_filenames)

# ------------------ RUN ------------------
if __name__ == "__main__":
    app.run(debug=True)
