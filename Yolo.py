from ultralytics import YOLO
from main import *
# Load a pretrained detection model
model = YOLO("yolo11n.pt")  # or any variant like yolo11s.pt, yolo11m.pt, etc.

# Run inference on an image


# # Display predictions
# results[0].show()

def detect_objects_yolo(frame):
    """
    Run YOLO object detection on a frame.
    Returns:
        - List of detected class names (for captioning)
        - Annotated frame with bounding boxes drawn
    """
    results = model(frame,verbose=False)[0]
    annotated_frame = frame.copy()

    if results.boxes is None:
        return [], annotated_frame

    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    confs = results.boxes.conf.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy()
    names = results.names

    for cls_id, conf, box in zip(class_ids, confs, boxes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names[cls_id]} {conf:.2f}"

        # Draw box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Draw label
        cv2.putText(annotated_frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    class_names = list({names[c] for c in class_ids})
    return class_names, annotated_frame