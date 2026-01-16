import torch
from torch import nn

# Import các hàm chúng ta đã viết
from src.models import get_model
from src.data_loader import get_cifar100_loaders
from src.engine import evaluate

# --- CẤU HÌNH ---
MODEL_NAME = 'resnet50' # <-- Đổi thành 'mobilenet_v2' nếu cần
MODEL_PATH = './results/resnet50_best.pth' # <-- Đường dẫn tới model đã lưu
BATCH_SIZE = 64
# ------------------

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Đang chạy trên: {DEVICE}")

    # 1. Tải model (chỉ cần cấu trúc)
    model = get_model(model_name=MODEL_NAME, num_classes=100)

    # 2. Tải trọng số đã lưu
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(DEVICE)

    # 3. Tải dữ liệu (chỉ cần tập Test)
    _, _, test_loader = get_cifar100_loaders(batch_size=BATCH_SIZE)

    # 4. Định nghĩa loss
    criterion = nn.CrossEntropyLoss()

    # 5. Chạy đánh giá
    print(f"Đang đánh giá model: {MODEL_NAME}...")
    test_loss, test_accuracy = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=DEVICE
    )

    print("\n--- KẾT QUẢ CUỐI CÙNG ---")
    print(f"Model: {MODEL_NAME}")
    print(f"Độ chính xác (Test): {test_accuracy:.2%}")

if __name__ == "__main__":
    main()