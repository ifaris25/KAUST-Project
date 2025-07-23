
import cohere
from init import my_key  
from transformers import pipeline



co = cohere.Client(my_key) 
import re
from collections import defaultdict

import re
from collections import defaultdict
from time import sleep

def extract_max_counts_and_cleaned_captions(captions_dict):
    max_counts = defaultdict(int)
    cleaned_captions = []

    for caption_list in captions_dict.values():
        for caption in caption_list:
            # 1. Clean the caption
            cleaned = re.sub(r"\(Detected:.*?\)", "", caption).strip()
            cleaned_captions.append(cleaned)

            # 2. Parse detected classes
            match = re.search(r"Detected:\s*(.*)", caption)
            if match:
                items = match.group(1).split(",")
                for item in items:
                    parts = item.strip().split(":")
                    if len(parts) == 2:
                        cls = parts[0].strip()
                        try:
                            count = int(parts[1].strip(" )"))
                            max_counts[cls] = max(max_counts[cls], count)
                        except ValueError:
                            continue

    return dict(max_counts), cleaned_captions

def useCohere(captions):
    # caption_text = "\n".join([f"Frame {k}: {v}" for k, v in captions.items()])
    counter, cleaned_captions = extract_max_counts_and_cleaned_captions(captions)
    response = co.generate(
        model="command-r-plus",
        prompt=f"""You are a visual understanding assistant. Your task is to summarize what is happening in the following video frames. 
        The most frequently detected objects in the scene are: {', '.join([f"{k} ({v})" for k, v in counter.items()])}.

        Here are descriptions of the individual frames:
        {chr(10).join(['- ' + cap for cap in cleaned_captions])}

        Based on this, provide a short, human-like summary of the overall scene:

        Final Summary:""",
            max_tokens=100,
            temperature=0.4
        )
    return response.generations[0].text.strip()
    




# # Load the summarization model once
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# def summarize_captions(captions_dict):
#     ordered = [captions_dict[k] for k in sorted(captions_dict.keys())]
#     full_text = " ".join(ordered)
#     max_input_length = 1024
#     if len(full_text.split()) > max_input_length:
#         full_text = " ".join(full_text.split()[:max_input_length])

#     summary = summarizer(full_text, max_length=60, min_length=20, do_sample=False)[0]["summary_text"]
#     return summary



