import os
import cv2
import argparse
from tqdm import tqdm
from retinaface import RetinaFace

def detect_faces_retina(image, confidence_threshold=0.9):
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

    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def process_video(video_path, output_dir, frames_per_video=15, img_size=299):
    # Khởi tạo VideoCapture
    reader = cv2.VideoCapture(video_path)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Nếu video ít khung hình hơn yêu cầu, ta chia khoảng để lấy
    if num_frames == 0:
        return 0
        
    step = max(num_frames // frames_per_video, 1)
    
    video_name = os.path.basename(video_path).split('.')[0]
    out_folder = os.path.join(output_dir, video_name)
    os.makedirs(out_folder, exist_ok=True)
    
    frame_num = 0
    saved_count = 0
    
    while reader.isOpened():
        ret, image = reader.read()
        if not ret:
            break
            
        # Chỉ trích xuất ở các mốc step cụ thể
        if frame_num % step == 0:
            height, width = image.shape[:2]
            boxes = detect_faces_retina(image)
            
            # Giả sử chúng ta chỉ lấy khuôn mặt rõ ràng nhất (đầu tiên/lớn nhất) cho Training
            if len(boxes) > 0:
                face_box = boxes[0] 
                x_, y_, size = get_boundingbox(face_box, width, height)
                
                cropped_face = image[y_:y_+size, x_:x_+size]
                if cropped_face.size > 0:
                    cropped_face = cv2.resize(cropped_face, (img_size, img_size))
                    img_path = os.path.join(out_folder, f"{frame_num}.jpg")
                    cv2.imwrite(img_path, cropped_face)
                    saved_count += 1
            
            if saved_count >= frames_per_video:
                break
                
        frame_num += 1
        
    reader.release()
    return saved_count

def generate_txt_list(dataset_info, output_txt, val_split=0.2):
    # Tạo 2 tệp cho Train và Val
    train_txt = output_txt.replace(".txt", "_train.txt")
    val_txt = output_txt.replace(".txt", "_val.txt")
    
    train_lines = []
    val_lines = []
    
    import random
    
    for folder_path, label in dataset_info:
        if not os.path.exists(folder_path):
            continue
            
        videos = os.listdir(folder_path)
        for video_folder_name in videos:
            vf_path = os.path.join(folder_path, video_folder_name)
            if os.path.isdir(vf_path):
                images = os.listdir(vf_path)
                
                # Chia tách Train/Val theo video thay vì theo ảnh để tránh dữ liệu bị rò rỉ (Leaking)
                # Tức là toàn bộ ảnh của Video 1 phải rơi vào Train, hoặc Val hoàn toàn.
                is_val = random.random() < val_split
                
                for img in images:
                    if img.endswith('.jpg') or img.endswith('.png'):
                        # Dùng đường dẫn tương đối thay vì abspath để dễ mang lên Colab
                        rel_path = os.path.join(vf_path, img).replace('\\', '/')
                        line = f"{rel_path} {label}\n"
                        if is_val:
                            val_lines.append(line)
                        else:
                            train_lines.append(line)
                            
    with open(train_txt, 'w') as f:
        f.writelines(train_lines)
        
    with open(val_txt, 'w') as f:
        f.writelines(val_lines)
        
    print(f"Đã tạo danh sách Train ({len(train_lines)} ảnh) và Val ({len(val_lines)} ảnh).")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, required=True, help="Thư mục chứa video thô (ví dụ: ./dataset_raw/original/c23/videos)")
    parser.add_argument('--output_dir', type=str, required=True, help="Thư mục lưu ảnh khuôn mặt cắt ra")
    parser.add_argument('--frames', type=int, default=15, help="Số khung hình cần chiết xuất trên mỗi Video (Để tiết kiệm thời gian)")
    args = parser.parse_args()
    
    videos = [v for v in os.listdir(args.video_dir) if v.endswith('.mp4') or v.endswith('.avi')]
    print(f"Bắt đầu cắt mặt cho {len(videos)} videos. Tiến trình có thể chạy rất lâu...")
    
    for v in tqdm(videos):
        process_video(os.path.join(args.video_dir, v), args.output_dir, frames_per_video=args.frames)
