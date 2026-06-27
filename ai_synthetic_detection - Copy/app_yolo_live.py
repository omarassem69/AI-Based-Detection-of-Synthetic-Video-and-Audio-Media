import cv2
import av
import numpy as np
import streamlit as st
from ultralytics import YOLO
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="YOLO + Deepfake Live Detection",
    layout="wide"
)

st.title("YOLO + Deepfake Detection")

# -----------------------------
# SETTINGS
# -----------------------------
IMG_SIZE = 160
VIDEO_MODEL_PATH = r"models\video_model_balanced.keras"

REAL_TH = 0.55
FAKE_TH = 0.75

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_models():
    yolo = YOLO("yolo11n.pt")
    cnn = load_model(VIDEO_MODEL_PATH)
    return yolo, cnn

yolo_model, deepfake_model = load_models()

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def classify_crop(crop):
    if crop is None or crop.size == 0:
        return "NO CROP", 0.0, (255, 255, 255)

    img = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    pred = float(deepfake_model.predict(img, verbose=0)[0][0])

    if pred >= FAKE_TH:
        return f"FAKE {pred:.2f}", pred, (0, 0, 255)
    elif pred >= REAL_TH:
        return f"SUSPICIOUS {pred:.2f}", pred, (0, 255, 255)
    else:
        return f"REAL {pred:.2f}", pred, (0, 255, 0)


def get_upper_body_crop(frame, x1, y1, x2, y2):
    h = y2 - y1
    upper_y2 = y1 + int(h * 0.45)
    crop = frame[y1:upper_y2, x1:x2]
    return crop


def process_frame(frame):
    results = yolo_model(frame, conf=0.4, classes=[0], verbose=False)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            crop = get_upper_body_crop(frame, x1, y1, x2, y2)
            label, pred, color = classify_crop(crop)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    return frame


# -----------------------------
# LIVE CAMERA PROCESSOR
# -----------------------------
class YOLODeepfakeProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = process_frame(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# -----------------------------
# SINGLE PAGE INTERFACE
# -----------------------------
st.sidebar.title("Detection Mode")

mode = st.sidebar.radio(
    "Choose option",
    ["Home", "Live Camera", "Upload Video"]
)

if mode == "Home":
    st.subheader("AI-Based Detection of Synthetic Video Media")

    st.write("""
    This system supports near real-time video analysis using YOLO for person localization
    and a CNN model for fake/real classification.
    """)

    st.info("Choose Live Camera or Upload Video from the sidebar.")

elif mode == "Live Camera":
    st.subheader("Live Camera Detection")

    st.write("Click START below. Your browser will ask for camera permission.")

    webrtc_streamer(
        key="yolo-deepfake-live",
        video_processor_factory=YOLODeepfakeProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )

elif mode == "Upload Video":
    st.subheader("Upload Video Detection")

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:
        temp_path = "temp_uploaded_video.mp4"

        with open(temp_path, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(temp_path)

        frame_placeholder = st.empty()
        stop_button = st.button("Stop Video")

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame = process_frame(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame_placeholder.image(frame, channels="RGB")

            if stop_button:
                break

        cap.release()