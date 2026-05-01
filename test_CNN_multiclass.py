import torch
import torch.nn as nn
import argparse
import os
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from tqdm import tqdm

from network.proposed_model import DeepfakeMultiClassModel
from dataset.transform import xception_default_data_transforms
from dataset.mydataset import MyDataset

def main(args):
    test_list = args.test_list
    batch_size = args.batch_size
    model_path = args.model_path

    if not os.path.exists(test_list):
        print(f"❌ Không tìm thấy file danh sách ảnh: {test_list}")
        return
        
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy file model: {model_path}")
        return

    print(f"Đang tải danh sách ảnh từ {test_list} ...")
    test_dataset = MyDataset(txt_path=test_list, transform=xception_default_data_transforms['test'])
    # Sử dụng shuffle=False để đối chiếu đúng thứ tự khi in ma trận
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2)
    
    print(f"Đang nạp mô hình từ {model_path} ...")
    model = DeepfakeMultiClassModel(image_size=299, num_classes=3)
    
    state_dict = torch.load(model_path, map_location='cpu')
    has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
    if has_module_prefix:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    if args.cuda and torch.cuda.is_available():
        model = model.cuda()
        print("⚡ Đang sử dụng GPU (CUDA) để đánh giá...")
    else:
        print("🐌 Đang sử dụng CPU để đánh giá...")
        
    model.eval()

    all_preds = []
    all_labels = []

    print("\nBắt đầu chạy Đánh giá (Evaluation) ...")
    with torch.no_grad():
        for (image, labels) in tqdm(test_loader, desc="Tiến trình"):
            if args.cuda and torch.cuda.is_available():
                image = image.cuda()
                labels = labels.cuda()
                
            outputs = model(image)
            _, preds = torch.max(outputs.data, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Tính toán các chỉ số
    target_names = ['Real (0)', 'FaceSwap (1)', 'Reenactment (2)']
    
    print("\n" + "="*60)
    print(" BÁO CÁO ĐÁNH GIÁ (CLASSIFICATION REPORT)")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))
    
    print("\n" + "="*60)
    print(" MA TRẬN NHẦM LẪN (CONFUSION MATRIX)")
    print("="*60)
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    print("\n* Chú thích Ma trận: Dòng (Hàng ngang) là nhãn THỰC TẾ | Cột (Hàng dọc) là nhãn DỰ ĐOÁN")
    print("-> Đường chéo chính (từ trên trái xuống dưới phải) là số ảnh đoán ĐÚNG.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Mặc định lấy tập Validation để test
    parser.add_argument('--test_list', type=str, default='./data_list/val_data.txt')
    parser.add_argument('--model_path', type=str, default='./weight/19_multiclass_colab.pkl')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cuda', action='store_true', help="Thêm cờ này nếu chạy trên GPU (Colab)")
    args = parser.parse_args()
    main(args)
