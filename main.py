from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import torch
import numpy as np
import cv2
import os
import uuid

from retinaface import RetinaFace
from network.proposed_model import DeepfakeMultiClassModel
from preprocess import preprocess_rgb, preprocess_dct

import base64
from gradcam import GradCAM, apply_colormap_on_image

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

# ================= GRAD-CAM =================
# Hook vào lớp conv4 để tránh lỗi inplace của ReLU(inplace=True) ở layer tiếp theo
target_layer = model.spatial.model.conv4
grad_cam = GradCAM(model, target_layer)


# ================= FACE DETECTION =================
def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    x1, y1, x2, y2 = face
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize and size_bb < minsize:
        size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)
    return x1, y1, size_bb

def crop_face(frame, scale=1.3):
    try:
        height, width = frame.shape[:2]
        
        # Nhận diện trên ảnh gốc
        faces = RetinaFace.detect_faces(frame)

        if not isinstance(faces, dict) or len(faces) == 0:
            return None

        face = max(faces.values(), key=lambda f: f["score"])

        if face["score"] < 0.90:
            return None

        x_, y_, size = get_boundingbox(face["facial_area"], width, height, scale=scale)
        face_crop = frame[y_:y_+size, x_:x_+size]

        if face_crop.size == 0:
            return None

        # BGR → RGB
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        return face_crop

    except Exception as e:
        return None


# ================= PREDICT =================
def predict_frame(img):
    x_rgb = preprocess_rgb(img).to(device)

    with torch.no_grad():
        out = model(x_rgb)
        prob = torch.softmax(out, dim=1)

    return prob

def predict_heatmap(img, target_class=None):
    x_rgb = preprocess_rgb(img).to(device)
    cam_mask, out = grad_cam(x_rgb, class_idx=target_class)
    
    prob = torch.softmax(out, dim=1)
    
    # Overlay heatmap
    overlay = apply_colormap_on_image(img, cam_mask)
    
    # Encode base64
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', overlay_bgr)
    heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return prob, heatmap_b64


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

            prob, heatmap_b64 = predict_heatmap(face)

            pred = int(torch.argmax(prob, dim=1))
            conf = float(prob[0][pred])

            return {
                "type": "image",
                "label": labels[pred],
                "confidence": conf,
                "heatmap": heatmap_b64
            }

        # ================= VIDEO =================
        elif filename.endswith((".mp4", ".avi", ".mov")):

            temp_path = f"temp_{uuid.uuid4().hex}.mp4"

            with open(temp_path, "wb") as f:
                f.write(content)

            frames = extract_frames(temp_path, 10)

            probs = []
            valid_faces = []

            for frame in frames:
                face = crop_face(frame)

                if face is None:
                    continue
                
                valid_faces.append(face)
                prob = predict_frame(face)
                probs.append(prob.cpu().numpy()[0])

            os.remove(temp_path)

            if len(probs) == 0:
                return {"error": "no face detected"}

            probs = np.array(probs)

            # Chuyển về trung bình cộng (Mean) giống code train chuẩn
            avg = np.mean(probs, axis=0)

            pred = int(np.argmax(avg))
            conf = float(avg[pred])
            
            # Lấy heatmap cho frame có độ tin cậy của nhãn dự đoán cao nhất
            best_frame_idx = np.argmax(probs[:, pred])
            best_face = valid_faces[best_frame_idx]
            
            _, heatmap_b64 = predict_heatmap(best_face, target_class=pred)

            return {
                "type": "video",
                "label": labels[pred],
                "confidence": conf,
                "frames_used": len(probs),
                "heatmap": heatmap_b64
            }

        else:
            return {"error": "unsupported file type"}

    except Exception as e:
        return {"error": str(e)}