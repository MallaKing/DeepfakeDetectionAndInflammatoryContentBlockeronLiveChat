from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, Depends, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import Optional
import base64
import json

# Import BOTH utils
import models, auth, schemas
import model_utils  # For Detector Page
import ai_utils     # For Charcha Chat
from models import SessionLocal

app = FastAPI()
models.init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()

# --- ROUTES (Separated Pages) ---
@app.get("/")
def home(): 
    return FileResponse("templates/index.html")

@app.get("/detector")
def detector_page(): 
    return FileResponse("templates/detector.html")

@app.get("/auth")
def auth_page(): 
    return FileResponse("templates/auth.html")

@app.get("/charcha")
def charcha_page(): 
    return FileResponse("templates/chat.html")

# --- AUTH LOGIC ---
@app.post("/register")
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    if db.query(models.User).filter((models.User.username == user.username) | (models.User.email == user.email)).first():
        raise HTTPException(status_code=400, detail="User exists")
    hashed_pw = auth.get_password_hash(user.password)
    db.add(models.User(username=user.username, email=user.email, hashed_password=hashed_pw))
    db.commit()
    return {"msg": "Created"}

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.username == form_data.username).first()
    if not user or not auth.verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Invalid credentials")
    return {"access_token": auth.create_access_token({"sub": user.username}), "token_type": "bearer", "username": user.username}

# --- 1. DETECTOR ENDPOINT (Uses model_utils) ---
@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    image_data = await file.read()
    # Call the dedicated Detector Utility
    return model_utils.get_prediction(image_data)

# --- 2. CHAT SEND ENDPOINT (Uses ai_utils) ---
@app.post("/chat-send")
async def chat_send(
    username: str = Form(...),
    message: str = Form(""),
    file: Optional[UploadFile] = File(None)
):
    # A. Check Text
    if message:
        is_safe_text, reason_text = ai_utils.check_text_safety(message)
        if not is_safe_text:
            return {"status": "blocked", "reason": reason_text}

    # B. Check Image (if exists)
    image_b64 = None
    if file:
        img_data = await file.read()
        is_real, score = ai_utils.check_image_realism(img_data)
        
        if not is_real:
            return {"status": "blocked", "reason": f"Deepfake detected ({ (1-score)*100:.1f}% confidence)"}
        
        image_b64 = base64.b64encode(img_data).decode('utf-8')

    return {
        "status": "allowed",
        "username": username,
        "message": message,
        "image": image_b64
    }

# --- WEBSOCKET ---
class ConnectionManager:
    def __init__(self):
        self.active_connections = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(data)
    except WebSocketDisconnect:
        manager.disconnect(websocket)