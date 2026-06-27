from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.predict(
    source=r"C:\Users\omara\OneDrive\Pictures\Camera Roll\WIN_20260506_18_08_56_Pro.mp4",
    show=True,
    save=True,
    conf=0.4,
    classes=[0]   # only person
)