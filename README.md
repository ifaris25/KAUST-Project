# KAUST-Project

# Real-time Video-to-Text Summarization

This project turns **live or recorded video into concise summaries** using object detection, image captioning, and language models. It was developed at **King Abdullah University of Science and Technology (KAUST)**.

---

## 🚀 Overview
- Detects objects in video frames with **YOLOv11n**  
- Generates grounded captions with **Florence-2-large**  
- Summarizes captions into short narratives with **Cohere Command-R+**  
- Works in two modes:
  - **Live streaming** → rolling one-minute summaries  
  - **Batch processing** → scene-based summaries  

---

## 🛠️ Tech Stack
- **Python 3.11**, PyTorch, OpenCV, Hugging Face, FastAPI  
- Models: YOLOv11n, Florence-2-large, Cohere Command-R+  
- Frontend: single-page web interface  

---

## 📦 Installation
Clone the repo:
```bash
git clone https://github.com/ifaris25/KAUST-Project.git
cd KAUST-Project
````

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the backend:

```bash
python backend/main1.py
```

Once running, the terminal will show a link (usually `http://127.0.0.1:8000`).
Open that link in your browser to access the web interface.

---

## 👥 Team

* Faris Alhammad – Leader & Programmer
* Safwan Nabeel – Programmer & Researcher
* Mishary Aldawood – Product Manager & Designer
* Rayed Saidi – Hardware Engineer
* Abdurahman Osilan – Product Manager

Mentor: **Dr. Salman Khan**
