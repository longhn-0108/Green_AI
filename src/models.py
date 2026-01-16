import torchvision.models as models
from torch import nn

def get_model(model_name: str, num_classes: int):
    print(f"[MODEL] Đang khởi tạo: {model_name}")

    if model_name == 'resnet50':
        # 1. Tải model gốc
        model = models.resnet50(weights='IMAGENET1K_V1')
        
        # 2. (QUAN TRỌNG) Sửa lớp đầu tiên để phù hợp với ảnh nhỏ 32x32
        # Gốc: kernel_size=7, stride=2, padding=3 (Làm ảnh nhỏ đi 1 nửa ngay lập tức)
        # Sửa thành: kernel_size=3, stride=1, padding=1 (Giữ nguyên kích thước ảnh)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # 3. (QUAN TRỌNG) Bỏ lớp MaxPool đầu tiên
        # Lớp này cũng làm ảnh nhỏ đi, với CIFAR-32x32 thì không cần thiết
        model.maxpool = nn.Identity()

        # 4. Thay lớp cuối cùng (FC)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model
        
    
    elif model_name == 'mobilenet_v2':
        # --- CẤU HÌNH MOBILENET-V2 (FIX LỖI 13%) ---
        model = models.mobilenet_v2(weights='IMAGENET1K_V1')
        
        # 1. FIX: Thay đổi lớp convolution đầu tiên
        # Mặc định MobileNetV2 giảm size ảnh ngay lớp đầu (stride=2).
        # Với ảnh 32x32, ta cần stride=1 để giữ thông tin.
        # Lớp đầu của MobileNetV2 nằm trong features[0][0]
        old_conv = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            in_channels=old_conv.in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=1, # <--- QUAN TRỌNG: Sửa từ 2 thành 1
            padding=old_conv.padding,
            bias=False
        )
        
        # 2. Thay lớp classifier cuối cùng
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        return model
        
    else:
        raise ValueError(f"Model {model_name} chưa được hỗ trợ.")