from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # small and fast model

model.predict(source=0, show=True, conf=0.4)