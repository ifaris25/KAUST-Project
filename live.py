import os
import json
from collections import Counter
from datetime import datetime

import cv2
from PIL import Image

from Captioning import predict_captions
from Yolo import detect_objects_yolo
from LLMs import useCohere


def find_working_camera(max_indexes: int = 5) -> int:
    """Try common camera indices and return the first that works (cross‑platform)."""
    for index in range(max_indexes):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print(f"✅ Camera found at index {index}")
                return index
    raise IOError("❌ No working camera found. Try plugging in a different webcam.")


def live_caption_camera(every_n_frames: int = 10, batch_size: int = 4):
    cam_index = find_working_camera()
    cap = cv2.VideoCapture(cam_index)
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

        # YOLO detections (FIX: updated to new return signature)
        raw_names, counts, annotated = detect_objects_yolo(frame)

        if frame_idx % every_n_frames == 0 and counts:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgs.append(pil_img)

            # Build a stable "person: 3, car: 1" string
            summary_str = ", ".join(f"{cls}: {cnt}" for cls, cnt in sorted(counts.items()))
            meta_info.append(summary_str)

        if len(imgs) >= batch_size:
            try:
                preds = predict_captions(imgs, extra_info=meta_info)
                # Aggregate captions by timestamp (per-minute)
                timestamp = datetime.now().strftime("%H:%M:%S")
                captions_this_minute.setdefault(timestamp, []).extend(preds)
            except Exception as e:
                print("⚠️ Captioning error:", e)
                preds = []
            imgs.clear()
            meta_info.clear()

        if preds:
            # Draw the most recent caption on the frame
            last_text = preds[-1]
            cv2.putText(
                annotated,
                last_text[:80] + ("…" if len(last_text) > 80 else ""),
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("🔴 Live Camera Captioning (YOLO boxes)", annotated)

        # Roll up minute summary -> Cohere
        current_minute = datetime.now().replace(second=0, microsecond=0)
        if current_minute > last_minute and captions_this_minute:
            try:
                summary = useCohere(captions_this_minute)
                log_entry = {"time": last_minute.strftime("%Y-%m-%d %H:%M"), "summary": summary}
                with open("summaries/live_summaries.json", "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                captions_this_minute = {}
                last_minute = current_minute
            except Exception as e:
                print("⚠️ Summarization error:", e)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("👋 Quitting...")
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    live_caption_camera()
