import cv2
from PIL import Image
from Captioning import predict_captions  # Make sure this is your correct caption file
from Yolo import detect_objects_yolo
import streamlit as st
import numpy as np


def find_working_camera():
    for index in range(5):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                return index
    raise IOError("❌ No working camera found. Try plugging in a different webcam.")


def live_caption_streamlit(every_n_frames=15, batch_size=1):
    cam_index = find_working_camera()
    cap = cv2.VideoCapture(cam_index)

    frame_idx = 0
    imgs, meta_info = [], []
    last_caption = ""

    stframe = st.empty()
    caption_placeholder = st.empty()
    st.markdown("**Press 'Stop' in the top right to exit the stream**")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to read frame from camera.")
            break

        detected_objects, annotated = detect_objects_yolo(frame)

        if frame_idx % every_n_frames == 0 and detected_objects:
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgs.append(pil_img)
            meta_info.append(", ".join(detected_objects))

        if len(imgs) == batch_size:
            try:
                preds = predict_captions(imgs, extra_info=meta_info)
                last_caption = preds[-1] if preds else ""
            except Exception as e:
                st.warning(f"⚠️ Captioning error: {e}")
                last_caption = ""
            imgs, meta_info = [], []

        if last_caption:
            caption_placeholder.markdown(f"### 📝 Caption: {last_caption}")

        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_rgb, channels="RGB")

        frame_idx += 1

    cap.release()


def run_streamlit_ui():
    st.set_page_config(page_title="Live Image Captioning", layout="centered")
    st.title("📸 Live Image Captioning with YOLO + BLIP")

    option = st.radio("Choose Input Source:", ("Upload Image", "Live Webcam Video"))

    if option == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.write("Generating caption...")
            captions = predict_captions([image])
            st.success(f"📝 Caption: {captions[0]}")

    elif option == "Live Webcam Video":
        if st.button("Start Live Captioning"):
            live_caption_streamlit()


if __name__ == "__main__":
    run_streamlit_ui()