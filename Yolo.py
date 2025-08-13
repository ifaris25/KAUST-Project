from collections import Counter
import cv2
from ultralytics import YOLO

# Load a pretrained detection model once
# You can swap to "yolo11s.pt" / "yolo11m.pt" if you need better accuracy
model = YOLO("yolo11n.pt")


def detect_objects_yolo(frame, conf: float = 0.25, iou: float = 0.5):
    """
    Run YOLO object detection on a frame.

    Returns:
        raw_names: list[str]     -> class names for each detection (duplicates kept)
        counts: Counter          -> counts per class name
        annotated_frame: ndarray -> original frame with boxes/labels drawn
    """
    results = model(frame, verbose=False, conf=conf, iou=iou)[0]
    annotated_frame = frame.copy()

    # Handle no detections
    if results.boxes is None or len(results.boxes) == 0:
        return [], Counter(), annotated_frame

    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    confs = results.boxes.conf.cpu().numpy()
    boxes = results.boxes.xyxy.cpu().numpy()
    names = results.names  # list/dict from ultralytics

    # Draw detections
    for cls_id, c, box in zip(class_ids, confs, boxes):
        x1, y1, x2, y2 = map(int, box)
        label = f"{names[int(cls_id)]} {float(c):.2f}"

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated_frame,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    raw_names = [names[int(c)] for c in class_ids]
    counts = Counter(raw_names)

    return raw_names, counts, annotated_frame
