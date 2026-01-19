import torch.nn as nn
import torchvision.models as models
from .config import Config

def get_model(model_name: str, num_classes: int = 100) -> nn.Module:
    """
    Khởi tạo kiến trúc mô hình.
    
    Args:
        model_name (str): Tên mô hình ('resnet50' hoặc 'mobilenet_v2')
        num_classes (int): Số lớp đầu ra (CIFAR-100 = 100)
    
    Returns:
        nn.Module: Mô hình PyTorch đã được điều chỉnh layer đầu/cuối.
    """
    print(f"[INFO] Initializing model: {model_name.upper()}...")
    
    if model_name == 'resnet50':
        # Load weights mới nhất
        model = models.resnet50(weights='IMAGENET1K_V1')
        # Điều chỉnh conv1 cho ảnh nhỏ (32x32) để tránh mất mát thông tin
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity() # Bỏ MaxPool
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
        
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        # Điều chỉnh stride lớp đầu tiên
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            in_channels=old_conv.in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=1, 
            padding=old_conv.padding,
            bias=False
        )
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    
    else:
        raise ValueError(f"Model {model_name} chưa được hỗ trợ.")