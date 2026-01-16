import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Tỷ lệ chia tập Validation (10%)
VAL_SPLIT_SIZE = 0.1 

def get_cifar100_loaders(batch_size: int = 64, augment: bool = True):
    """
    Tải CIFAR-100 với tùy chọn Augmentation.
    
    Args:
        augment (bool): 
            - True: Dùng cho ResNet (Cắt + Lật) để chống Overfitting.
            - False: Dùng cho MobileNet hoặc Quantization (Chỉ Lật) để ổn định.
    """
    
    # 1. Định nghĩa Transform cho tập TRAIN
    if augment:
        # Augmentation MẠNH (Dành cho ResNet)
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4), # Cắt ngẫu nhiên
            transforms.RandomHorizontalFlip(),    # Lật ngang
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408], 
                std=[0.2675, 0.2565, 0.2761]
            )
        ])
        print("[DATA] Mode: Augmentation MẠNH (Crop + Flip)")
    else:
        # Augmentation NHẸ (Dành cho MobileNet / Quantization)
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(), # Chỉ lật ngang (an toàn)
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408], 
                std=[0.2675, 0.2565, 0.2761]
            )
        ])
        print("[DATA] Mode: Augmentation NHẸ (Chỉ Flip)")

    # 2. Định nghĩa Transform cho tập TEST/VALIDATION (Giữ nguyên)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408], 
            std=[0.2675, 0.2565, 0.2761]
        )
    ])

    # 3. Tải dữ liệu
    full_train_dataset = datasets.CIFAR100(
        root='./data', train=True, download=True, transform=train_transform 
    )
    
    test_dataset = datasets.CIFAR100(
        root='./data', train=False, download=True, transform=test_transform
    )

    # 4. Chia tập Train/Val
    val_size = int(len(full_train_dataset) * VAL_SPLIT_SIZE)
    train_size = len(full_train_dataset) - val_size
    
    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])

    # Kiểm tra có GPU không để bật pin_memory
    is_cuda = torch.cuda.is_available()

    # 5. Tạo Loader
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, 
        num_workers=2, pin_memory=is_cuda
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, 
        num_workers=2, pin_memory=is_cuda
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=2, pin_memory=is_cuda
    )
    
    return train_loader, val_loader, test_loader