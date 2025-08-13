import os
import re
from collections import defaultdict
from typing import Dict, List, Union

import cohere

# Prefer environment variable; fall back to optional init.py for convenience
_API_ENV = os.getenv("COHERE_API_KEY")
if not _API_ENV:
    try:
        from init import my_key as _API_ENV  # optional legacy fallback
    except Exception:
        _API_ENV = None

if not _API_ENV:
    raise RuntimeError("Cohere API key not found. Set COHERE_API_KEY env var (or provide init.my_key).")

co = cohere.Client(_API_ENV)


def _normalize_value_to_list(v: Union[str, List[str]]) -> List[str]:
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]


def extract_max_counts_and_cleaned_captions(captions_dict: Dict[Union[int, str], Union[str, List[str]]]):
    """
    Accepts a dict where values can be either a string caption or a list of captions.
    Extracts:
      - max counts per object class from trailing "(Detected: ...)" patterns
      - cleaned captions with the "(Detected: ...)" segment removed
    """
    max_counts = defaultdict(int)
    cleaned_captions: List[str] = []

    for value in captions_dict.values():
        for caption in _normalize_value_to_list(value):
            # 1) Clean the caption text
            cleaned = re.sub(r"\(Detected:.*?\)", "", caption).strip()
            if cleaned:
                cleaned_captions.append(cleaned)

            # 2) Parse detected classes (robust)
            m = re.search(r"Detected:\s*(.*)", caption)
            if not m:
                continue

            items = [p.strip() for p in m.group(1).split(",") if p.strip()]
            for item in items:
                # Allow "person: 3" or just "person"
                if ":" in item:
                    cls, cnt = item.split(":", 1)
                    cls = cls.strip()
                    try:
                        num = int(re.sub(r"[^\d]", "", cnt))
                        if num > max_counts[cls]:
                            max_counts[cls] = num
                    except ValueError:
                        # If the count isn't an int, treat as 1
                        if 1 > max_counts[cls]:
                            max_counts[cls] = 1
                else:
                    cls = item.strip()
                    if 1 > max_counts[cls]:
                        max_counts[cls] = 1

    return dict(max_counts), cleaned_captions


def useCohere(captions_dict: Dict[Union[int, str], Union[str, List[str]]]) -> str:
    """
    Summarize a set of frame captions using Cohere. Values may be a string or a list of strings.
    """
    counter, cleaned_captions = extract_max_counts_and_cleaned_captions(captions_dict)

    # Build a compact objects summary string in deterministic order
    if counter:
        objects_summary = ", ".join(f"{k} ({v})" for k, v in sorted(counter.items()))
    else:
        objects_summary = "none clearly dominant"

    bullets = "\n".join(f"- {cap}" for cap in cleaned_captions) if cleaned_captions else "- (no clean captions parsed)"

    response = co.generate(
        model="command-r-plus",
        prompt=(
            "You are a visual understanding assistant. Summarize what is happening in the following video frames.\n"
            f"Most frequently detected objects: {objects_summary}.\n\n"
            "Here are descriptions of individual frames:\n"
            f"{bullets}\n\n"
            "Provide a short, human-like summary of the overall scene.\n\n"
            "Final Summary:"
        ),
        max_tokens=120,
        temperature=0.4,
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



