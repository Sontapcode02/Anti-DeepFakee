import torch
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Đăng ký hooks để lấy activation và gradient
        target_layer.register_forward_hook(self.save_activation)
        # Sử dụng register_full_backward_hook cho PyTorch >= 1.8
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.clone().detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].clone().detach()

    def __call__(self, x, class_idx=None):
        self.model.eval()
        
        # Cho phép tính đạo hàm với x (input)
        if not x.requires_grad:
            x.requires_grad = True
            
        out = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(out, dim=1).item()

        self.model.zero_grad()
        
        # Lan truyền ngược để lấy gradient
        out[0, class_idx].backward(retain_graph=False)

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]

        # Global average pooling on gradients (Alpha weights)
        weights = np.mean(gradients, axis=(1, 2))

        # Tính toán Heatmap
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Xóa các giá trị âm (tương tự ReLU)
        cam = np.maximum(cam, 0)
        
        # Resize về đúng kích thước ảnh đầu vào (vd: 299x299)
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        
        # Chuẩn hóa về [0, 1]
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max != 0:
            cam = cam / cam_max
            
        return cam, out

def apply_colormap_on_image(org_img, activation, colormap=cv2.COLORMAP_JET):
    """
    Áp dụng heatmap lên ảnh gốc.
    org_img: Ảnh gốc RGB (numpy array)
    activation: Heatmap [0, 1]
    """
    h, w = activation.shape
    org_img = cv2.resize(org_img, (w, h))
    
    # Tạo heatmap màu (OpenCV trả về BGR)
    heatmap = cv2.applyColorMap(np.uint8(255 * activation), colormap)
    
    # Chuyển heatmap sang RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Chồng ảnh (Trộn)
    overlay = np.float32(heatmap) * 0.4 + np.float32(org_img) * 0.6
    overlay = overlay / np.max(overlay)
    
    # Chuyển lại về dạng uint8
    return np.uint8(255 * overlay)
