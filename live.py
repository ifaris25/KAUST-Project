import cv2
from PIL import Image
from main import *

def find_working_camera():
    """Try multiple camera indices to find one that works."""
    for index in range(5):
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print(f"✅ Camera found at index {index}")
                return index
    raise IOError("❌ No working camera found. Try plugging in a different webcam.")

def live_caption_camera(every_n_frames=60, batch_size=4):
    cam_index = find_working_camera()
    cap = cv2.VideoCapture(cam_index, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        raise IOError("❌ Cannot open camera")

    frame_idx = 0
    imgs, idxs = [], []
    last_caption = ""

    print("📷 Live captioning started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame.")
            break

        # Capture every N frames
        if frame_idx % every_n_frames == 0:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgs.append(pil_img)
            idxs.append(frame_idx)

        # Predict captions in batches
        if len(imgs) == batch_size:
            try:
                preds = predict_captions(imgs)
                last_caption = preds[-1] if preds else ""
            except Exception as e:
                print("⚠️ Captioning error:", e)
            imgs, idxs = [], []

        # Display video with the latest caption
        if last_caption:
            cv2.putText(frame, last_caption, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("🔴 Live Camera Captioning", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Quitting...")
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_caption_camera()