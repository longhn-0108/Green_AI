import torch

class Config:
    """
    Class chứa toàn bộ cấu hình (Hyperparameters) của dự án.
    Giúp quản lý tham số tập trung, dễ dàng thay đổi.
    """
    # Đường dẫn
    DATA_ROOT = './data'
    RESULT_DIR = './results'
    LOG_FILE = './results/experiment_log.csv'
    
    # Tham số huấn luyện
    BATCH_SIZE = 64
    LEARNING_RATE = 0.1
    NUM_EPOCHS = 10  # Chạy demo local thì để thấp, chạy thật thì tăng lên 50
    NUM_WORKERS = 2  # Chỉnh về 0 nếu chạy trên Windows bị lỗi
    
    # Hạt giống ngẫu nhiên (Reproducibility)
    SEED = 42
    
    # Thông số Dataset
    IMG_SIZE = 32
    NUM_CLASSES = 100
    
    # Thiết bị (Tự động nhận diện)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    @staticmethod
    def get_device(technique: str):
        """Quantization bắt buộc chạy trên CPU"""
        if technique == 'quantization':
            return 'cpu'
        return Config.DEVICE