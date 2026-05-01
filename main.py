from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import torch
import numpy as np
import cv2
import os
import uuid

from retinaface import RetinaFace
from model import DeepfakeMultiClassModel
from preprocess import preprocess_rgb, preprocess_dct

app = FastAPI()

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

labels = ["REAL", "FACESWAP", "REENACTMENT"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= MODEL =================
model = DeepfakeMultiClassModel(num_classes=3)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "weights/19_multiclass_colab.pkl")

checkpoint = torch.load(MODEL_PATH, map_location=device)
state = checkpoint.get("state_dict", checkpoint)

new_state = {}
for k, v in state.items():
    name = k[7:] if k.startswith('module.') else k
    new_state[name] = v

model.load_state_dict(new_state, strict=False)
model.to(device)
model.eval()


# ================= FACE DETECTION =================
def crop_face(frame, scale=1.2):
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = RetinaFace.detect_faces(rgb)

        if not isinstance(faces, dict) or len(faces) == 0:
            return None

        face = max(faces.values(), key=lambda f: f["score"])

        # ❌ FILTER FACE QUÁ YẾU
        if face["score"] < 0.90:
            return None

        x1, y1, x2, y2 = face["facial_area"]

        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        nw, nh = int(w * scale), int(h * scale)

        x1 = max(cx - nw // 2, 0)
        y1 = max(cy - nh // 2, 0)
        x2 = min(cx + nw // 2, frame.shape[1])
        y2 = min(cy + nh // 2, frame.shape[0])

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None

        # ✔ FIX QUAN TRỌNG: BGR → RGB
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        return face_crop

    except:
        return None


# ================= PREDICT =================
def predict_frame(img):
    x_rgb = preprocess_rgb(img).to(device)
    x_dct = preprocess_dct(img).to(device)

    with torch.no_grad():
        out = model(x_rgb, x_dct)
        prob = torch.softmax(out, dim=1)

    return prob


# ================= VIDEO FRAMES =================
def extract_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    idxs = np.linspace(0, total - 1, num_frames).astype(int)

    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


# ================= API =================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename = file.filename.lower()

        # ================= IMAGE =================
        if filename.endswith((".jpg", ".jpeg", ".png")):

            img = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)

            if img is None:
                return {"error": "invalid image"}

            face = crop_face(img)
            if face is None:
                return {"error": "no face detected"}

            prob = predict_frame(face)

            pred = int(torch.argmax(prob, dim=1))
            conf = float(prob[0][pred])

            return {
                "type": "image",
                "label": labels[pred],
                "confidence": conf
            }

        # ================= VIDEO =================
        elif filename.endswith((".mp4", ".avi", ".mov")):

            temp_path = f"temp_{uuid.uuid4().hex}.mp4"

            with open(temp_path, "wb") as f:
                f.write(content)

            frames = extract_frames(temp_path, 10)

            probs = []

            for frame in frames:
                face = crop_face(frame)

                if face is None:
                    continue

                prob = predict_frame(face)
                probs.append(prob.cpu().numpy()[0])

            os.remove(temp_path)

            if len(probs) == 0:
                return {"error": "no face detected"}

            probs = np.array(probs)

            # ✔ FIX: weighted average (QUAN TRỌNG)
            weights = np.max(probs, axis=1)
            weights = weights / (weights.sum() + 1e-6)

            avg = np.sum(probs * weights[:, None], axis=0)

            pred = int(np.argmax(avg))
            conf = float(avg[pred])

            return {
                "type": "video",
                "label": labels[pred],
                "confidence": conf,
                "frames_used": len(probs)
            }

        else:
            return {"error": "unsupported file type"}

    except Exception as e:
        return {"error": str(e)}