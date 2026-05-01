import torch
import torch.nn as nn
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm

from network.proposed_model import DeepfakeMultiClassModel
from dataset.transform import xception_default_data_transforms
from dataset.mydataset import MyDataset

def main(args):
    test_list = args.test_list
    batch_size = args.batch_size
    weights_dir = args.weights_dir
    max_epochs = args.max_epochs
    model_name_base = args.model_name_base

    if not os.path.exists(test_list):
        print(f"❌ Không tìm thấy list data tại: {test_list}")
        return

    print(f"Đang chuẩn bị dữ liệu Validation từ {test_list} ...")
    val_dataset = MyDataset(txt_path=test_list, transform=xception_default_data_transforms['test'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2)
    val_dataset_size = len(val_dataset)
    
    criterion = nn.CrossEntropyLoss()
    
    epochs = []
    val_losses = []
    val_accs = []

    model = DeepfakeMultiClassModel(image_size=299, num_classes=3)
    if args.cuda and torch.cuda.is_available():
        model = model.cuda()
    
    print("\n🚀 Bắt đầu hành trình cỗ máy thời gian: Khôi phục biểu đồ!")
    
    # Lặp qua từng epoch để chấm điểm lại
    for epoch in range(max_epochs + 1):
        model_path = os.path.join(weights_dir, f"{epoch}_{model_name_base}")
        
        if not os.path.exists(model_path):
            print(f"\n⚠️ Dừng khôi phục: Không tìm thấy file {epoch}_{model_name_base}. Đã chấm xong {epoch} epochs.")
            break
            
        print(f"\n--- Đang nội soi Epoch {epoch} ---")
        
        # Nạp trọng số
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

        model.eval()
        
        running_loss = 0.0
        running_corrects = 0.0
        
        with torch.no_grad():
            # Dùng tqdm để hiển thị thanh tiến trình ngắn gọn
            for image, labels in tqdm(val_loader, desc=f"Chấm điểm Epoch {epoch}", leave=False, colour='cyan'):
                if args.cuda and torch.cuda.is_available():
                    image = image.cuda()
                    labels = labels.cuda()
                    
                outputs = model(image)
                _, preds = torch.max(outputs.data, 1)
                
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                running_corrects += torch.sum(preds == labels.data).to(torch.float32)
                
        epoch_loss = running_loss / val_dataset_size
        epoch_acc = running_corrects / val_dataset_size
        epoch_acc_val = epoch_acc.item() if torch.is_tensor(epoch_acc) else epoch_acc
        
        print(f"👉 Kết quả Epoch {epoch} | Val Loss: {epoch_loss:.4f} | Val Acc: {epoch_acc_val*100:.2f}%")
        
        epochs.append(epoch)
        val_losses.append(epoch_loss)
        val_accs.append(epoch_acc_val)

    # ------------------ VẼ BIỂU ĐỒ ------------------
    if len(epochs) > 0:
        plt.style.use('ggplot')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Biểu đồ Accuracy
        ax1.plot(epochs, val_accs, marker='o', linewidth=2, color='#2ca02c', label='Validation Accuracy')
        ax1.set_title('Validation Accuracy over Epochs', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epochs', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_xticks(epochs)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # Biểu đồ Loss
        ax2.plot(epochs, val_losses, marker='s', linewidth=2, color='#d62728', label='Validation Loss')
        ax2.set_title('Validation Loss over Epochs', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epochs', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_xticks(epochs)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        plt.tight_layout()
        
        output_img = os.path.join(weights_dir, 'validation_learning_curve.png')
        plt.savefig(output_img, dpi=300) # Lưu với độ phân giải cao
        print(f"\n✅ Đã vẽ và lưu xong biểu đồ thành công! File ảnh được cất tại: {output_img}")
    else:
        print("\n❌ Không có dữ liệu để vẽ biểu đồ.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_list', type=str, default='./data_list/val_data.txt')
    parser.add_argument('--weights_dir', type=str, required=True, help="Đường dẫn thư mục chứa các file model .pkl")
    parser.add_argument('--model_name_base', type=str, default='multiclass_colab.pkl')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cuda', action='store_true')
    args = parser.parse_args()
    main(args)
