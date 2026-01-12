import torch
import torch.nn as nn
from transformers import Dinov2Model, AutoImageProcessor, pipeline
from PIL import Image
import io

# --- 1. TEXT AI SETUP ---
print("Loading Text AI...")
text_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def check_text_safety(text: str):
    """Returns (is_safe: bool, reason: str)"""
    result = text_pipeline(text)[0]
    if result['label'] == 'NEGATIVE' and result['score'] > 0.7:
        return False, f"Text blocked: Negative sentiment ({int(result['score']*100)}%)"
    return True, "Safe"

# --- 2. IMAGE AI SETUP (DinoV2) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "complex_transformer_bestfinal.pth"

class DinoV2TransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Dinov2Model.from_pretrained("facebook/dinov2-base")
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768, nhead=8, dim_feedforward=2048, 
            dropout=0.1, activation="gelu", batch_first=True
        )
        self.transformer_head = nn.TransformerEncoder(encoder_layer, num_layers=6)
        self.classifier = nn.Sequential(nn.Linear(768, 1))

    def forward(self, pixel_values):
        with torch.no_grad():
            outputs = self.backbone(pixel_values)
        sequence_output = outputs.last_hidden_state
        transformed_sequence = self.transformer_head(sequence_output)
        return self.classifier(transformed_sequence[:, 0, :])

print("Loading DinoV2 Image Model...")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
image_model = DinoV2TransformerClassifier()

try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    image_model.load_state_dict(state_dict)
    image_model.to(DEVICE)
    image_model.eval()
    print("DinoV2 Model Loaded Successfully!")
except FileNotFoundError:
    print("WARNING: .pth file not found. Image check will mock success.")

def check_image_realism(image_bytes):
    """Returns (is_real: bool, score: float)"""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)
        
        with torch.no_grad():
            logits = image_model(pixel_values)
            prob = torch.sigmoid(logits).item()
            
        # Threshold Logic ( > 0.8 is REAL)
        is_real = prob > 0.8
        return is_real, prob
    except Exception as e:
        print(f"Image Check Error: {e}")
        return False, 0.0