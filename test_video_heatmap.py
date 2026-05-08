import torch
import cv2
import numpy as np
import os
import sys

from retinaface import RetinaFace
from network.proposed_model import DeepfakeMultiClassModel
from preprocess import preprocess_rgb
from gradcam import GradCAM, apply_colormap_on_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
labels = ["REAL", "FACESWAP", "REENACTMENT"]
model_path = "weights/19_multiclass_colab.pkl"

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    x1, y1, x2, y2 = face
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize and size_bb < minsize: size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)
    return x1, y1, size_bb

def crop_face(frame, scale=1.3):
    height, width = frame.shape[:2]
    faces = RetinaFace.detect_faces(frame)
    if not isinstance(faces, dict) or len(faces) == 0: return None
    face = max(faces.values(), key=lambda f: f["score"])
    if face["score"] < 0.90: return None
    x_, y_, size = get_boundingbox(face["facial_area"], width, height, scale=scale)
    face_crop = frame[y_:y_+size, x_:x_+size]
    if face_crop.size == 0: return None
    return cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

def main(vid_path, out_path):
    print(f"Processing: {vid_path}")
    print(f"Device: {device}")
    
    model = DeepfakeMultiClassModel(num_classes=3)
    checkpoint = torch.load(model_path, map_location=device)
    state = checkpoint.get("state_dict", checkpoint)
    new_state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
    model.load_state_dict(new_state, strict=False)
    model.to(device)
    model.eval()

    grad_cam = GradCAM(model, model.spatial.model.conv4)

    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print("Error: Cannot open video!")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 25.0

    out_writer = None
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Total frames: {total_frames}. Starting loop...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        face = crop_face(frame)
        if face is not None:
            x_rgb = preprocess_rgb(face).to(device)
            
            with torch.no_grad():
                out_logits = model(x_rgb)
                out_prob = torch.softmax(out_logits, dim=1)
                pred = int(torch.argmax(out_prob, dim=1))
                conf = float(out_prob[0][pred])

            cam_mask, _ = grad_cam(x_rgb, class_idx=pred)
            overlay = apply_colormap_on_image(face, cam_mask)
            
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            overlay_bgr = cv2.resize(overlay_bgr, (300, 300))
            
            # Ghi text kết quả
            label = f"{labels[pred]}: {conf*100:.1f}%"
            cv2.putText(overlay_bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            if out_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(out_path, fourcc, fps, (300, 300))
                
            out_writer.write(overlay_bgr)
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f" Processed {frame_count} frames...")
                
            # Demo 50 frames để chạy cho lẹ
            if frame_count >= 50:
                break

    if out_writer:
        out_writer.release()
    cap.release()
    print(f"DONE! Output video saved at: {out_path}")

if __name__ == "__main__":
    vid = "videos/003_000.mp4"
    if len(sys.argv) > 1:
        vid = sys.argv[1]
    main(vid, "output_heatmap.mp4")
