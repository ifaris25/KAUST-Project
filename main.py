import cv2
from PIL import Image

from Captioning import predict_captions


def caption_video(video_path: str, every_n_frames: int = 30, batch_size: int = 8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    captions = {}
    imgs, idxs = [], []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n_frames == 0:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgs.append(pil_img)
            idxs.append(frame_idx)

        if len(imgs) == batch_size:
            preds = predict_captions(imgs)
            captions.update(dict(zip(idxs, preds)))
            imgs, idxs = [], []

        frame_idx += 1

    if imgs:
        preds = predict_captions(imgs)
        captions.update(dict(zip(idxs, preds)))

    cap.release()
    return captions


def visualize_captions(video_path, captions_dict):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    current_frame = 0
    all_frames = sorted(captions_dict.keys())
    if not all_frames:
        cap.release()
        return

    while True:
        ret, frame = cap.read()
        if not ret or current_frame > all_frames[-1]:
            break

        if current_frame in captions_dict:
            caption = captions_dict[current_frame]
            cv2.putText(
                frame,
                caption,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(f"Frame {current_frame}", frame)
            key = cv2.waitKey(0)  # Wait until a key is pressed
            if key == ord("q"):
                break
            cv2.destroyWindow(f"Frame {current_frame}")

        current_frame += 1

    cap.release()
    cv2.destroyAllWindows()


def caption_video_by_scenes(video_path, scene_frames, batch_size=8):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    captions = {}
    imgs, idxs = [], []
    frame_idx = 0
    scene_set = set(scene_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in scene_set:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgs.append(pil_img)
            idxs.append(frame_idx)

        if len(imgs) == batch_size:
            preds = predict_captions(imgs)
            captions.update(dict(zip(idxs, preds)))
            imgs, idxs = [], []

        frame_idx += 1

    if imgs:
        preds = predict_captions(imgs)
        captions.update(dict(zip(idxs, preds)))

    cap.release()
    return captions


def detect_scene_changes(video_path, threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    scenes = [0]
    prev_hist = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA)
            if diff > threshold:
                scenes.append(frame_idx)

        prev_hist = hist
        frame_idx += 1

    cap.release()
    return scenes


# if __name__ == "__main__":
#     video_file = "Videos/a.MP4"
#     scenes = detect_scene_changes(video_file, threshold=0.7)
#     print("Detected scenes at frames:", scenes)

#     captions = caption_video_by_scenes(video_file, scenes)
#     final_summary = useCohere(captions)
    
#     print("\n🎯 Final Summary:")
#     print(final_summary)

# # ---------- Example Usage ----------
# if __name__ == "__main__":
#     video_file = "Videos/b.mp4"
#     captions = caption_video(video_file, every_n_frames=30, batch_size=8)
    
#     # for frame, caption in captions.items():
#     #     print(f"Frame {frame}: {caption}")
    
#     final_summary = useCohere(captions)
#     print("\n🎯 Final Summary:")
#     print(final_summary)



