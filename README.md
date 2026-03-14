# 🛡️AI-based Deepfake Detection and Live Chat Moderation System

Welcome to **Sahaj Labs**, a full-stack solution designed to keep live chat environments safe, authentic, and free from toxic AI-generated content. This project combines a high-performance **DinoV2-based Deepfake Detector** with an NLP-driven **Inflammatory Content Blocker** to moderate "Charcha" (Live Chat) sessions in real-time.



## 🌟 The "Human" Story Behind the Tech
Most deepfake detectors are slow and struggle with real-world compression. For this project, I used a **DinoV2-base backbone** (frozen to keep those rich features intact) and slapped a **6-layer Transformer Encoder** on top. 

I trained this on a massive dataset (tens of GBs) scraped from Telegram, Twitter, and Facebook. The result? A rock-solid **93% accuracy** on the test set. It doesn’t just look at pixels; it looks at the structural "vibe" of the image to spot AI inconsistencies.

---

## 🚀 Core Features
* **Live "Charcha" Chat**: A WebSocket-powered chat room where every message is screened before it goes live.
* **Deepfake Shield**: Automated DinoV2 analysis for every image upload. If the AI thinks it's a fake (Probability < 0.8), it never hits the chat.
* **Toxicity Filter**: Uses `DistilBERT` to catch negative sentiment and inflammatory language before it ruins the vibe.
* **Standalone Detector**: A dedicated page for users to upload and verify images manually.
* **Secure Auth**: Built-in user system with JWT tokens, BCrypt password hashing, and Pydantic validation.

---

## 🧠 The Architecture
### Image AI (DinoV2 + Transformer)
The model uses a Vision Transformer (ViT) approach. By using DinoV2 as a feature extractor and passing the sequence output through a custom Transformer head, the model identifies "AI-signatures" that standard CNNs often miss.


### Text AI
We use a fine-tuned `SST-2` pipeline. If the negative sentiment score crosses **0.7**, the message is flagged as inflammatory and blocked.

---

## 🛠️ Setup Instructions

### 1. Requirements
You'll need Python 3.9+ and a GPU (optional, but recommended for the DinoV2 inference).

```bash
pip install torch torchvision transformers fastapi uvicorn sqlalchemy passlib[bcrypt] python-multipart PyJWT pillow
