import cv2
from PIL import Image
from datetime import datetime
import json
import os
from main import *
from Yolo import detect_objects_yolo
from LLMs import useCohere
from collections import Counter

def find_working_camera():
    for index in range(5):
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print(f"✅ Camera found at index {index}")
                return index
    raise IOError("❌ No working camera found. Try plugging in a different webcam.")

def live_caption_camera(every_n_frames=10, batch_size=4):
    cam_index = find_working_camera()
    cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        raise IOError("Cannot open camera")

    frame_idx = 0
    imgs, meta_info = [], []
    captions_this_minute = {}
    last_minute = datetime.now().replace(second=0, microsecond=0)
    preds = []

    os.makedirs("summaries", exist_ok=True)
    print("Live captioning started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        detected_objects, annotated = detect_objects_yolo(frame)

        if frame_idx % every_n_frames == 0 and detected_objects:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgs.append(pil_img)
            obj_counts = Counter(detected_objects)
            summary_str = ", ".join([f"{cls}: {count}" for cls, count in obj_counts.items()])
            meta_info.append(summary_str)

        if len(imgs) == batch_size:
            try:
                preds = predict_captions(imgs, extra_info=meta_info)    
                for i, pred in enumerate(preds):
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    if timestamp not in captions_this_minute:
                        captions_this_minute[timestamp] = []
                    captions_this_minute[timestamp].append(pred)
            except Exception as e:
                print("⚠️ Captioning error:", e)
                preds = []
            imgs, meta_info = [], []

        if preds:
            cv2.putText(annotated, preds[-1], (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow("🔴 Live Camera Captioning (YOLO boxes)", annotated)

        current_minute = datetime.now().replace(second=0, microsecond=0)
        if current_minute > last_minute and captions_this_minute:
            try:
                summary = useCohere(captions_this_minute)
                log_entry = {
                    "time": last_minute.strftime("%Y-%m-%d %H:%M"),
                    "summary": summary
                }
                with open("summaries/live_summaries.json", "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
                
                captions_this_minute = {}
                last_minute = current_minute
            except Exception as e:
                print("⚠️ Summarization error:", e)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Quitting...")
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_caption_camera()