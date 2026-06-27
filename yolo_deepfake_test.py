import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# -----------------------------
# LOAD MODELS
# -----------------------------
yolo_model = YOLO("yolo11n.pt")

deepfake_model = load_model(
    r"models\video_model_balanced.keras"
)

IMG_SIZE = 160

# -----------------------------
# VIDEO PATH
# -----------------------------
video_path = r"C:\Users\omara\OneDrive\Pictures\Camera Roll\WIN_20260506_18_08_56_Pro.mp4"

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # -----------------------------
    # YOLO DETECTION
    # -----------------------------
    results = yolo_model(frame, conf=0.4, classes=[0])

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # crop person
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            # preprocess for CNN
            img = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            img = img.astype("float32") / 255.0
            img = np.expand_dims(img, axis=0)

            # prediction
            pred = deepfake_model.predict(img, verbose=0)[0][0]

            # label
            if pred >= 0.75:
                label = f"FAKE {pred:.2f}"
                color = (0, 0, 255)

            elif pred >= 0.55:
                label = f"SUSPICIOUS {pred:.2f}"
                color = (0, 255, 255)

            else:
                label = f"REAL {pred:.2f}"
                color = (0, 255, 0)

            # draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    cv2.imshow("YOLO + Deepfake Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()