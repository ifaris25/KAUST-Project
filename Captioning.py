from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

# Load the model and processor
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()  # IMPORTANT: eval mode

# Generation settings (tune as you like)
gen_kwargs = {
    "max_length": 32,
    "do_sample": True,   # set to False for deterministic output
    "top_k": 50,
    "top_p": 0.95,
}


@torch.inference_mode()
def predict_captions(images, extra_info=None):
    """
    Accepts a list of PIL.Image objects, returns a list of generated captions.
    Optionally takes extra_info (e.g., "person: 3, car: 1") to append as contextual hints.
    """
    if not images:
        return []

    # Ensure all images are RGB
    processed_images = [img.convert("RGB") if getattr(img, "mode", None) != "RGB" else img for img in images]

    # Preprocess
    pixel_values = feature_extractor(images=processed_images, return_tensors="pt").pixel_values.to(device)

    # Generate captions
    output_ids = model.generate(pixel_values, **gen_kwargs)
    captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    captions = [c.strip() for c in captions]

    # Optionally append object info
    if extra_info:
        # Normalize and align lengths
        if not isinstance(extra_info, (list, tuple)):
            extra_info = [str(extra_info)] * len(captions)
        if len(extra_info) != len(captions):
            # pad or truncate
            if len(extra_info) < len(captions):
                extra_info = list(extra_info) + [""] * (len(captions) - len(extra_info))
            else:
                extra_info = list(extra_info)[:len(captions)]

        captions = [
            f"{caption} (Detected: {det})" if det else caption
            for caption, det in zip(captions, extra_info)
        ]

    return captions


def predict_from_paths(image_paths):
    """Accepts list of image paths, loads them, and predicts captions."""
    images = [Image.open(path) for path in image_paths]
    return predict_captions(images)


# # ---------- TEST ----------
# if __name__ == "__main__":
#     test_image = "/Users/faris/Desktop/Record screen/a.png"
#     captions = predict_from_paths([test_image])
#     print("------->", captions)