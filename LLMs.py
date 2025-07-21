
import cohere
from init import my_key  
from transformers import pipeline



co = cohere.Client(my_key) 


def useCohere(captions):
    caption_text = "\n".join([f"Frame {k}: {v}" for k, v in captions.items()])

    # Summarize using a prompt
    response = co.generate(
        model="command-r-plus",
        prompt=f"""You are an AI assistant. Summarize the following video frame captions into one cohesive and concise description of the scene:
    {caption_text}

    Final Summary:""",
        max_tokens=100,
        temperature=0.4
    )
    return response.generations[0].text.strip()




# Load the summarization model once
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_captions(captions_dict):
    ordered = [captions_dict[k] for k in sorted(captions_dict.keys())]
    full_text = " ".join(ordered)
    max_input_length = 1024
    if len(full_text.split()) > max_input_length:
        full_text = " ".join(full_text.split()[:max_input_length])

    summary = summarizer(full_text, max_length=60, min_length=20, do_sample=False)[0]["summary_text"]
    return summary