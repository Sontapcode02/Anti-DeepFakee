import os
import argparse
import cv2
import torch
import torch.nn as nn
from PIL import Image as pil_image
import numpy as np

# Thư viện nhận diện khuôn mặt
try:
    from retinaface import RetinaFace
except ImportError:
    print("Vui lòng cài đặt RetinaFace: pip install retina-face")
    exit()

# Nạp model và hàm tiền xử lý chuẩn của mạng Xception
from network.proposed_model import DeepfakeMultiClassModel
from dataset.transform import xception_default_data_transforms

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

def main(args):
    print(f"\n🎬 ĐANG TEST VIDEO: {args.video_path} ...")
    
    # 1. KHỞI TẠO VÀ NẠP MÔ HÌNH CHUẨN XÁC
    model = DeepfakeMultiClassModel(image_size=299, num_classes=3)
    state_dict = torch.load(args.model_path, map_location='cpu')
    
    # Xử lý triệt để tiền tố 'module.' do train bằng DataParallel
    if any(k.startswith('module.') for k in state_dict.keys()):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k[7:]] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)
        
    if args.cuda and torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    print("✅ Nạp trọng số mô hình thành công!")
    
    # 2. ĐỌC VÀ TRÍCH XUẤT KHUNG HÌNH (FRAMES)
    reader = cv2.VideoCapture(args.video_path)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if num_frames == 0:
        print("❌ Lỗi: Không đọc được file video (File hỏng hoặc đường dẫn sai).")
        return

    # Trích xuất rải đều args.num_frames từ video (Ví dụ: 15 frames)
    frame_idxs = np.linspace(0, num_frames - 1, args.num_frames, dtype=int)
    
    frame_probs = []
    frames_processed = 0
    preprocess = xception_default_data_transforms['test']
    
    print(f"⏳ Đang nội soi {args.num_frames} khung hình rải đều khắp video...\n")
    
    for idx in frame_idxs:
        reader.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, image = reader.read()
        if not ret or image is None:
            continue
            
        height, width = image.shape[:2]
        
        # Dùng RetinaFace dò tìm khuôn mặt
        faces = RetinaFace.detect_faces(image)
        if type(faces) != dict or len(faces) == 0:
            continue # Không thấy mặt thì bỏ qua frame này
            
        # Lấy khuôn mặt có điểm tin cậy cao nhất
        face = list(faces.values())[0]
        if face['score'] < 0.9:
            continue
            
        # Cắt mặt
        x_, y_, size = get_boundingbox(face['facial_area'], width, height)
        cropped_face = image[y_:y_+size, x_:x_+size]
        
        if cropped_face.size == 0:
            continue
            
        # ⚠️ BƯỚC CỰC KỲ QUAN TRỌNG: CHUYỂN BGR SANG RGB
        # Nếu thiết kế API mà bỏ quên bước này, kết quả sẽ rớt xuống 33% như đoán bừa
        cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        
        # Tiền xử lý (Resize đúng 299x299 & Normalize y hệt lúc Train)
        pil_img = pil_image.fromarray(cropped_face_rgb)
        input_tensor = preprocess(pil_img).unsqueeze(0)
        
        if args.cuda and torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            
        # Đưa vào mạng dự đoán Softmax
        with torch.no_grad():
            output = model(input_tensor)
            prob = nn.Softmax(dim=1)(output)[0].cpu().numpy()
            frame_probs.append(prob)
            frames_processed += 1
            print(f"  - Frame {idx:03d} dự đoán: [Real: {prob[0]:.3f}, FaceSwap: {prob[1]:.3f}, Reenact: {prob[2]:.3f}]")

    reader.release()
    
    if frames_processed == 0:
        print("❌ Không tìm thấy khuôn mặt rõ nét nào trong video để phân tích.")
        return
        
    # 3. TÍNH TOÁN KẾT QUẢ CHUNG CUỘC BẰNG TRUNG BÌNH CỘNG (AVERAGING)
    avg_probs = np.mean(frame_probs, axis=0)
    
    labels = ["REAL", "FACE SWAP", "REENACTMENT"]
    predicted_idx = np.argmax(avg_probs)
    final_label = labels[predicted_idx]
    final_confidence = avg_probs[predicted_idx]
    
    print("\n=======================================================")
    print(" KẾT QUẢ TỔNG HỢP TOÀN VIDEO")
    print("=======================================================")
    print(f"Số frames hợp lệ đã soi  : {frames_processed} / {args.num_frames}")
    print(f"Xác suất trung bình      : [Real: {avg_probs[0]:.4f}, FaceSwap: {avg_probs[1]:.4f}, Reenact: {avg_probs[2]:.4f}]")
    print(f"🚀 NHÃN DỰ ĐOÁN CUỐI CÙNG : {final_label}")
    print(f"⭐ ĐỘ TIN CẬY (CONFIDENCE): {final_confidence * 100:.2f}%")
    print("=======================================================\n")
    
    if final_confidence < 0.6:
        print("⚠️ Lời khuyên: Độ tin cậy của AI khá thấp. Hãy kiểm tra lại tiền xử lý (Transform) hoặc chất lượng video.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script chuẩn để debug API / test video trên Colab")
    parser.add_argument('--video_path', '-i', type=str, required=True, help="Đường dẫn file video cần soi")
    parser.add_argument('--model_path', '-m', type=str, required=True, help="Đường dẫn file trọng số (.pkl)")
    parser.add_argument('--num_frames', '-n', type=int, default=15, help="Số frame cần rút trích (Mặc định: 15)")
    parser.add_argument('--cuda', action='store_true', help="Bật GPU để chạy nhanh hơn")
    args = parser.parse_args()
    main(args)
