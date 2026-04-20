import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from network.xception import xception

class DCT2D(nn.Module):
    """
    Biến đổi Cosine rời rạc 2D (DCT-II) cho PyTorch (Differentiable)
    Hỗ trợ tích cực cho feature extraction tần số.
    """
    def __init__(self, size):
        super(DCT2D, self).__init__()
        # Khởi tạo ma trận DCT
        matrix = torch.zeros(size, size)
        for i in range(size):
            for j in range(size):
                if i == 0:
                    matrix[i, j] = 1 / math.sqrt(size)
                else:
                    matrix[i, j] = math.sqrt(2 / size) * math.cos((2 * j + 1) * i * math.pi / (2 * size))
        self.register_buffer('dct_mat', matrix)

    def forward(self, x):
        # x shape: (B, C, H, W)
        # Thực hiện nhân ma trận để tính 2nd DCT
        # X_dct = C @ X @ C^T
        x_dct = torch.einsum('ih, bchw -> bciw', self.dct_mat, x)
        x_dct = torch.einsum('bciw, jw -> bcij', x_dct, self.dct_mat)
        return x_dct

class SpatialBranch(nn.Module):
    def __init__(self):
        super(SpatialBranch, self).__init__()
        self.model = xception(pretrained=False)
        
        if hasattr(self.model, 'last_linear'):
            self.num_ftrs = self.model.last_linear.in_features
            # Loại bỏ lớp cuối trả về features gốc
            self.model.last_linear = nn.Identity()
        elif hasattr(self.model, 'fc'):
            self.num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)

class FrequencyBranch(nn.Module):
    def __init__(self, in_channels=3):
        super(FrequencyBranch, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        return x.view(x.size(0), -1)

class DeepfakeMultiClassModel(nn.Module):
    """
    Quy trình Phân loại Deepfake Đa lớp:
    Nhánh 1: Đặc trưng Không gian (Xception)
    Nhánh 2: Đặc trưng Tần số (DCT)
    Gộp đặc trưng và xuất ra dự đoán nhóm Fake.
    Classes: 0: REAL, 1: FACE SWAP, 2: REENACTMENT
    """
    def __init__(self, image_size=299, num_classes=3):
        super(DeepfakeMultiClassModel, self).__init__()
        
        self.dct = DCT2D(image_size)
        
        self.spatial = SpatialBranch()
        self.freq = FrequencyBranch(in_channels=3)
        
        spatial_dim = self.spatial.num_ftrs # e.g. 2048 for Xception
        freq_dim = 256
        
        fusion_dim = spatial_dim + freq_dim
        
        # Classifier
        self.fc1 = nn.Linear(fusion_dim, 512)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # 1. Trích xuất Không gian
        feat_spatial = self.spatial(x)
        
        # 2. Trích xuất Tần số
        x_freq = self.dct(x)
        feat_freq = self.freq(x_freq)
        
        # 3. Hợp nhất
        if feat_spatial.dim() > 2:
            feat_spatial = feat_spatial.view(feat_spatial.size(0), -1)
        feat_concat = torch.cat([feat_spatial, feat_freq], dim=1)
        
        # 4. Phân loại 
        feat = self.relu(self.fc1(feat_concat))
        feat = self.dropout(feat)
        out = self.classifier(feat)
        
        return out

if __name__ == '__main__':
    model = DeepfakeMultiClassModel(image_size=299, num_classes=3)
    # Test random tensor
    x = torch.randn(2, 3, 299, 299)
    out = model(x)
    print("Features shape output (Batch Size, Classes):", out.shape)
