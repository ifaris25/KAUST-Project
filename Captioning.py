# from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
# import torch
# from PIL import Image

# # Load the model and processor
# model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
# tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# # Set device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device).eval()  # IMPORTANT: eval mode

# # Generation settings (tune as you like)
# gen_kwargs = {
#     "max_length": 32,
#     "do_sample": True,   # set to False for deterministic output
#     "top_k": 50,
#     "top_p": 0.95,
# }


# @torch.inference_mode()
# def predict_captions(images, extra_info=None):
#     """
#     Accepts a list of PIL.Image objects, returns a list of generated captions.
#     Optionally takes extra_info (e.g., "person: 3, car: 1") to append as contextual hints.
#     """
#     if not images:
#         return []

#     # Ensure all images are RGB
#     processed_images = [img.convert("RGB") if getattr(img, "mode", None) != "RGB" else img for img in images]

#     # Preprocess
#     pixel_values = feature_extractor(images=processed_images, return_tensors="pt").pixel_values.to(device)

#     # Generate captions
#     output_ids = model.generate(pixel_values, **gen_kwargs)
#     captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
#     captions = [c.strip() for c in captions]

#     # Optionally append object info
#     if extra_info:
#         # Normalize and align lengths
#         if not isinstance(extra_info, (list, tuple)):
#             extra_info = [str(extra_info)] * len(captions)
#         if len(extra_info) != len(captions):
#             # pad or truncate
#             if len(extra_info) < len(captions):
#                 extra_info = list(extra_info) + [""] * (len(captions) - len(extra_info))
#             else:
#                 extra_info = list(extra_info)[:len(captions)]

#         captions = [
#             f"{caption} (Detected: {det})" if det else caption
#             for caption, det in zip(captions, extra_info)
#         ]

#     return captions


# def predict_from_paths(image_paths):
#     """Accepts list of image paths, loads them, and predicts captions."""
#     images = [Image.open(path) for path in image_paths]
#     return predict_captions(images)


# # # ---------- TEST ----------
# # if __name__ == "__main__":
# #     test_image = "/Users/faris/Desktop/Record screen/a.png"
# #     captions = predict_from_paths([test_image])
# #     print("------->", captions)

# Captioning.py — Florence‑2 captioning (drop‑in replacement)

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# Model config
_MODEL_ID = "microsoft/Florence-2-large"
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Default task prompt (see model card tasks)
_DEFAULT_PROMPT = "<DETAILED_CAPTION>"

# Generation settings (deterministic by default)
_GEN_KWARGS = dict(
    max_new_tokens=128,   # adjust up to ~256 if you need longer text
    num_beams=3,
    do_sample=False,
)

# Lazy singletons
_model = None
_processor = None


def _lazy_init():
    """Load Florence‑2 once, using the official HF API with remote code."""
    global _model, _processor
    if _model is None:
        _model = AutoModelForCausalLM.from_pretrained(
            _MODEL_ID, torch_dtype=_DTYPE, trust_remote_code=True
        ).to(_DEVICE)
        _processor = AutoProcessor.from_pretrained(
            _MODEL_ID, trust_remote_code=True
        )


@torch.inference_mode()
def predict_captions(images, extra_info=None, prompt=_DEFAULT_PROMPT):
    """
    Accepts a list of PIL.Image objects, returns a list[str] of captions.
    If extra_info is provided (e.g., "person: 3, car: 1"), it appends it as '(Detected: ...)'.

    The 'prompt' defaults to '<CAPTION>' but you can pass '<DETAILED_CAPTION>' if you want.
    """
    if not images:
        return []

    _lazy_init()

    # Normalize inputs
    if not isinstance(images, (list, tuple)):
        images = [images]

    if extra_info is not None and not isinstance(extra_info, (list, tuple)):
        extra_info = [str(extra_info)] * len(images)

    outputs = []
    for idx, img in enumerate(images):
        # Ensure RGB
        if getattr(img, "mode", None) != "RGB":
            img = img.convert("RGB")

        # Preprocess + generate
        inputs = _processor(text=prompt, images=img, return_tensors="pt").to(_DEVICE, _DTYPE)
        gen_ids = _model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            **_GEN_KWARGS,
        )

        # Decode + postprocess per the model card
        gen_text = _processor.batch_decode(gen_ids, skip_special_tokens=False)[0]
        parsed = _processor.post_process_generation(
            gen_text, task=prompt, image_size=(img.width, img.height)
        )
        # For caption tasks, parsed has {'<CAPTION>': '...'} or similar
        cap = parsed.get(prompt, parsed) if isinstance(parsed, dict) else parsed
        if isinstance(cap, (list, tuple)):
            cap = cap[0] if cap else ""
        cap = str(cap).strip()

        # Append detected-object hint if provided
        if extra_info:
            hint = extra_info[idx] if idx < len(extra_info) else ""
            if hint:
                cap = f"{cap} (Detected: {hint})"

        outputs.append(cap)

    return outputs


def predict_from_paths(image_paths, prompt=_DEFAULT_PROMPT):
    """Convenience helper matching your old API."""
    images = [Image.open(p).convert("RGB") for p in image_paths]
    return predict_captions(images, prompt=prompt)
