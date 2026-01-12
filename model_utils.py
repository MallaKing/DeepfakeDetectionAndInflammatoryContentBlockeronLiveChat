import torch
import torch.nn as nn
from transformers import Dinov2Model, AutoImageProcessor
from PIL import Image
import io

# Uses the same Architecture as AI Utils
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
        outputs = self.backbone(pixel_values)
        sequence_output = outputs.last_hidden_state
        transformed_sequence = self.transformer_head(sequence_output)
        return self.classifier(transformed_sequence[:, 0, :])

print("Loading Detector Model (Public)...")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
detector_model = DinoV2TransformerClassifier()

try:
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    detector_model.load_state_dict(state_dict)
    detector_model.to(DEVICE)
    detector_model.eval()
    print("✅ Detector Ready.")
except FileNotFoundError:
    detector_model = None

def get_prediction(image_bytes):
    """Returns dict for Frontend"""
    if detector_model is None:
        return {"label": "REAL", "confidence": "99% (Mock)"}

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(DEVICE)
        
        with torch.no_grad():
            logits = detector_model(pixel_values)
            prob = torch.sigmoid(logits).item()
            
        label = "REAL" if prob > 0.5 else "FAKE"
        conf = f"{prob*100:.2f}%" if label == "REAL" else f"{(1-prob)*100:.2f}%"
        
        return {"label": label, "confidence": conf}
    except Exception as e:
        return {"error": str(e)}