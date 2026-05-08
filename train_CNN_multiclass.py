import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import os
import cv2

from network.proposed_model import DeepfakeMultiClassModel
from dataset.transform import xception_default_data_transforms
from dataset.mydataset import MyDataset

def main(args):
    name = args.name
    continue_train = args.continue_train
    train_list = args.train_list
    val_list = args.val_list
    epoches = args.epoches
    start_epoch = args.start_epoch
    batch_size = args.batch_size
    model_name = args.model_name
    model_path = args.model_path
    output_dir = args.output_dir
    
    output_path = os.path.join(output_dir, name)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
    
    torch.backends.cudnn.benchmark = True
    
    # Đối với multiclass, DataLoader cần trả về đúng label (0: Real, 1: Face Swap, 2: Reenactment)
    # LƯU Ý: Tệp txt danh sách huấn luyện (data_list) phải được lưu label dưới dạng 0, 1, hoặc 2.
    train_dataset = MyDataset(txt_path=train_list, transform=xception_default_data_transforms['train'])
    val_dataset = MyDataset(txt_path=val_list, transform=xception_default_data_transforms['val'])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)
    
    train_dataset_size = len(train_dataset)
    val_dataset_size = len(val_dataset)
    
    # Khởi tạo mô hình Đa Lớp (3 lớp: Real, Face Swap, Reenactment)
    model = DeepfakeMultiClassModel(image_size=299, num_classes=3)
    if continue_train and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded previous weights from", model_path)
    
    model = model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    model = nn.DataParallel(model)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    iteration = 0
    
    for epoch in range(start_epoch, epoches):
        print('Epoch {}/{}'.format(epoch+1, epoches))
        print('-'*10)
        
        model.train()
        train_loss = 0.0
        train_corrects = 0.0
        
        for (image, labels) in train_loader:
            image = image.cuda()
            labels = labels.cuda()
            # Giả định labels được trả về là giá trị từ 0 đến 2
            
            optimizer.zero_grad()
            outputs = model(image)
            _, preds = torch.max(outputs.data, 1)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            iter_loss = loss.data.item()
            train_loss += iter_loss
            iter_corrects = torch.sum(preds == labels.data).to(torch.float32)
            iteration += 1
            
            if not (iteration % 20):
                print('iteration {} train loss: {:.4f} Acc: {:.4f}'.format(iteration, iter_loss / batch_size, iter_corrects / batch_size))
                
        epoch_loss = train_loss / train_dataset_size
        epoch_acc = train_corrects / train_dataset_size
        print('epoch train loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        model.eval()
        val_loss = 0.0
        val_corrects = 0.0
        with torch.no_grad():
            for (image, labels) in val_loader:
                image = image.cuda()
                labels = labels.cuda()
                
                outputs = model(image)
                _, preds = torch.max(outputs.data, 1)
                
                loss = criterion(outputs, labels)
                val_loss += loss.data.item()
                val_corrects += torch.sum(preds == labels.data).to(torch.float32)
                
            epoch_loss = val_loss / val_dataset_size
            epoch_acc = val_corrects / val_dataset_size
            print('epoch val loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
            
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                
        scheduler.step()
        torch.save(model.module.state_dict(), os.path.join(output_path, str(epoch) + '_' + model_name))
        
    print('Best val Acc: {:.4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    torch.save(model.module.state_dict(), os.path.join(output_path, "best_multiclass.pkl"))

if __name__ == '__main__':
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--name', '-n', type=str, default='multiclass_training')
    parse.add_argument('--train_list', '-tl' , type=str, default = './data_list/FaceSwap_c0_train.txt')
    parse.add_argument('--val_list', '-vl' , type=str, default = './data_list/FaceSwap_c0_val.txt')
    parse.add_argument('--batch_size', '-bz', type=int, default=32)
    parse.add_argument('--epoches', '-e', type=int, default=20)
    parse.add_argument('--start_epoch', '-se', type=int, default=0)
    parse.add_argument('--model_name', '-mn', type=str, default='multiclass_model.pkl')
    parse.add_argument('--continue_train', action='store_true')
    parse.add_argument('--model_path', '-mp', type=str, default='')
    parse.add_argument('--output_dir', '-out', type=str, default='./output', help='Thư mục lưu model')
    args = parse.parse_args()
    main(args)
