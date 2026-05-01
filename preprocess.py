import cv2
import numpy as np
import torch


# ================= RGB (XCEPTION STANDARD) =================
def preprocess_rgb(img):
    """
    Input: BGR image từ OpenCV
    Output: tensor [1, 3, 299, 299]
    """

    img = cv2.resize(img, (299, 299))

    # BGR -> RGB (Đã xử lý ở bước crop_face bên main.py nên comment lại để tránh lỗi đảo ngược hệ màu)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # normalize [0,1]
    img = img.astype(np.float32) / 255.0

    # Xception standard normalization
    img = (img - 0.5) / 0.5

    # HWC -> CHW
    img = np.transpose(img, (2, 0, 1))

    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)


# ================= DCT (STABLE VERSION) =================
def preprocess_dct(img):
    """
    Frequency feature cho deepfake detection
    Output: tensor [1, 1, 32, 32]
    """

    img = cv2.resize(img, (299, 299))

    # BGR -> RGB -> GRAY
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # normalize trước khi DCT (QUAN TRỌNG)
    gray = gray.astype(np.float32) / 255.0

    # DCT transform
    dct = cv2.dct(gray)

    # lấy low-frequency (32x32)
    dct = np.abs(dct[:32, :32])

    # Z-score normalization (ổn định hơn min-max)
    mean = np.mean(dct)
    std = np.std(dct)

    dct = (dct - mean) / (std + 1e-6)

    return torch.tensor(dct, dtype=torch.float32).unsqueeze(0).unsqueeze(0)