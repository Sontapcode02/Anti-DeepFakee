import cv2
import numpy as np
import torch


from PIL import Image
from dataset.transform import xception_default_data_transforms

# ================= RGB (XCEPTION STANDARD) =================
def preprocess_rgb(img):
    """
    Input: RGB image (numpy array) từ crop_face
    Output: tensor [1, 3, 299, 299]
    """
    # Chuyển numpy array thành PIL Image
    pil_img = Image.fromarray(img)
    
    # Sử dụng transform chuẩn của Xception y hệt lúc Train
    preprocess = xception_default_data_transforms['test']
    input_tensor = preprocess(pil_img).unsqueeze(0)
    
    return input_tensor


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