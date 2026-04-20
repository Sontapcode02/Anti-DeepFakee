import os
import argparse
from os.path import join
import cv2
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm
import numpy as np

# Sử dụng RetinaFace
try:
    from retinaface import RetinaFace
except ImportError:
    print("Vui lòng cài đặt RetinaFace: pip install retina-face")
    exit()

from network.proposed_model import DeepfakeMultiClassModel
from dataset.transform import xception_default_data_transforms

# Để sử dụng Grad-CAM: pip install grad-cam
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRAD_CAM = True
except ImportError:
    print("Không tìm thấy pytorch_grad_cam. Sẽ không áp dụng Heatmap. Dùng: pip install grad-cam")
    HAS_GRAD_CAM = False

def detect_faces_retina(image, confidence_threshold=0.9):
    """
    Phát hiện khuôn mặt bằng RetinaFace. Trả về mảng tọa độ các bounding box.
    """
    faces = RetinaFace.detect_faces(image)
    boxes = []
    if type(faces) == dict:
        for key, face in faces.items():
            if face['score'] >= confidence_threshold:
                boxes.append(face['facial_area'])
    return boxes

def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    x1, y1, x2, y2 = face
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def preprocess_image(image, cuda=True):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image_rgb))
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image

def predict_with_model(image, model, grad_cam=None, cuda=True):
    preprocessed_image = preprocess_image(image, cuda)
    
    # Bật tính toán gradient cho Grad-CAM khi cần thiết
    if grad_cam is not None:
        preprocessed_image.requires_grad = True
    
    output = model(preprocessed_image)
    output = nn.Softmax(dim=1)(output)
    
    probs = output[0].detach().cpu().numpy()
    
    # Logic xác suất của bộ đa lớp
    # 0 -> Real, 1 -> Face Swap, 2 -> Reenactment
    fake_prob = probs[1] + probs[2]
    
    heatmap = None
    if fake_prob < 0.5:
        prediction_label = "REAL"
        confidence = probs[0]
        color = (0, 255, 0) # Xanh lá
    else:
        # Nếu được dự đoán là Fake, ta tiến hành kiểm tra thuộc nhánh Artifact nào lớn nhất
        prediction_id = 1 if probs[1] > probs[2] else 2
        confidence = probs[prediction_id]
        prediction_label = "FACE SWAP" if prediction_id == 1 else "REENACTMENT"
        color = (0, 0, 255) # Đỏ
        
        # Sinh bản đồ nhiệt Heatmap Grad-CAM
        if grad_cam is not None:
            # Truyền Category để CAM tập trung vào việc giải thích nhãn vừa phán đoán.
            # Lưu ý trong một số version grad-cam category cần là list: [prediction_id]
            # hoặc phải khởi tạo đối tượng target. Ở đây dùng API cơ bản.
            try:
                from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
                targets = [ClassifierOutputTarget(prediction_id)]
                cam = grad_cam(input_tensor=preprocessed_image, targets=targets)[0, :]
            except:
                # Fallback API cũ
                cam = grad_cam(input_tensor=preprocessed_image, target_category=prediction_id)[0, :]
                
            img_normalized = cv2.cvtColor(cv2.resize(image, (cam.shape[1], cam.shape[0])), cv2.COLOR_BGR2RGB) / 255.0
            heatmap = show_cam_on_image(img_normalized, cam, use_rgb=True)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)

    return prediction_label, confidence, color, heatmap

def multiclass_detect_video(video_path, model_path, output_path, start_frame=0, end_frame=None, cuda=True):
    print('Starting: {}'.format(video_path))
    reader = cv2.VideoCapture(video_path)
    video_fn = os.path.basename(video_path).split('.')[0] + '_multiclass.avi'
    os.makedirs(output_path, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None

    # Load mô hình đa lớp (Multi-class model)
    model = DeepfakeMultiClassModel(image_size=299, num_classes=3)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        print("WARNING: No model weights found. Running randomly.")
    
    if cuda:
        model = model.cuda()
    model.eval()

    # Init Grad-CAM tại block cuối của module Xception (trong luồng Spatial)
    cam_extractor = None
    if HAS_GRAD_CAM:
        try:
            # Với cấu trúc của xception.py, block cuối cùng thường nằm ở thuộc tính conv4 hoặc liên kết tương tự.
            target_layers = [model.spatial.model.conv4] 
            cam_extractor = GradCAM(model=model, target_layers=target_layers)
        except AttributeError:
            print("Could not find last generic layer of Xception for Grad-CAM.")

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    frame_num = 0
    end_frame = end_frame if end_frame else num_frames
    pbar = tqdm(total=end_frame-start_frame)

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break
        frame_num += 1

        if frame_num < start_frame:
            continue
        pbar.update(1)

        height, width = image.shape[:2]

        if writer is None:
            writer = cv2.VideoWriter(join(output_path, video_fn), fourcc, fps, (width, height))

        # Bước 1. Phát hiện khuôn mặt thay vì dlib ta dùng RetinaFace
        boxes = detect_faces_retina(image)
        
        for face_box in boxes:
            x_, y_, size = get_boundingbox(face_box, width, height)
            cropped_face = image[y_:y_+size, x_:x_+size].copy()
            if cropped_face.size == 0 or cropped_face.shape[0] == 0 or cropped_face.shape[1] == 0:
                continue
                
            # Bước 2 & 3: Dự đoán và phân loại đa lớp / sinh ra Feature Map
            label, score, color, heatmap = predict_with_model(cropped_face, model, grad_cam=cam_extractor, cuda=cuda)
            
            # Phủ Heatmap lên khung khuôn mặt gốc để trực quan
            if heatmap is not None:
                heatmap_resized = cv2.resize(heatmap, (size, size))
                image[y_:y_+size, x_:x_+size] = cv2.addWeighted(image[y_:y_+size, x_:x_+size], 0.6, heatmap_resized, 0.4, 0)
            
            # Vẽ Box & Label
            cv2.rectangle(image, (x_, y_), (x_ + size, y_ + size), color, 2)
            cv2.putText(image, f'{label} {score:.2f}', (x_, y_ - 10), font_face, font_scale, color, thickness, 2)

        if frame_num >= end_frame:
            break

        writer.write(image)

    pbar.close()
    if writer is not None:
        writer.release()
        print('Finished! Output saved under {}'.format(output_path))
    else:
        print('Input video file was empty')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--video_path', '-i', type=str, required=True)
    p.add_argument('--model_path', '-mi', type=str, default=None)
    p.add_argument('--output_path', '-o', type=str, default='.')
    p.add_argument('--start_frame', type=int, default=0)
    p.add_argument('--end_frame', type=int, default=None)
    p.add_argument('--cuda', action='store_true')
    args = p.parse_args()

    video_path = args.video_path
    if video_path.endswith('.mp4') or video_path.endswith('.avi'):
        multiclass_detect_video(**vars(args))
    else:
        videos = os.listdir(video_path)
        for video in videos:
            if video.endswith('.mp4') or video.endswith('.avi'):
                args.video_path = join(video_path, video)
                multiclass_detect_video(**vars(args))
