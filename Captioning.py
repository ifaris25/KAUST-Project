from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

# Load the model and processor
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

gen_kwargs = {
    "max_length": 32,
    "do_sample": True,      # or False for greedy
    "top_k": 50,
    "top_p": 0.95,
}

def predict_captions(images):
    """
    Accepts a list of PIL.Image objects, returns a list of generated captions.
    """
    # Ensure all images are RGB
    processed_images = [img.convert("RGB") if img.mode != "RGB" else img for img in images]

    # Preprocess
    pixel_values = feature_extractor(images=processed_images, return_tensors="pt").pixel_values.to(device)

    # Generate captions
    output_ids = model.generate(pixel_values, **gen_kwargs)
    captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [caption.strip() for caption in captions]

def predict_from_paths(image_paths):
    """
    Accepts list of image paths, loads them, and predicts captions.
    """
    images = [Image.open(path) for path in image_paths]
    return predict_captions(images)

# # ---------- TEST ----------
# if __name__ == "__main__":
#     test_image = "/Users/faris/Desktop/Record screen/a.png"
#     captions = predict_from_paths([test_image])
#     print("------->", captions)