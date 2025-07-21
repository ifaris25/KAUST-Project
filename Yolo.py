from ultralytics import YOLO

# Load a pretrained detection model
model = YOLO("yolo11n.pt")  # or any variant like yolo11s.pt, yolo11m.pt, etc.

# Run inference on an image
results = model("IMG_3029.jpeg")

# Display predictions
results[0].show()