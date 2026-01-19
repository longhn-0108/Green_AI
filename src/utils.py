import os
import csv
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.quantization import QuantStub, DeQuantStub
from codecarbon import EmissionsTracker
from .config import Config

def set_seed(seed: int = 42):
    """Thiết lập hạt giống ngẫu nhiên để đảm bảo tính tái lập (Reproducibility)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class ExperimentTracker:
    """Class wrapper cho CodeCarbon để đo năng lượng dễ dàng hơn."""
    def __init__(self, project_name: str):
        self.tracker = EmissionsTracker(
            project_name=project_name,
            output_dir=Config.RESULT_DIR,
            measure_power_secs=15,
            save_to_file=True,
            log_level='error'
        )
        self.start_time = 0

    def start(self):
        self.start_time = time.time()
        self.tracker.start()

    def stop(self):
        end_time = time.time()
        emissions = self.tracker.stop()
        
        # Lấy năng lượng tiêu thụ an toàn
        energy_kwh = self.tracker.final_emissions_data.energy_consumed if self.tracker.final_emissions_data else 0.0
        elapsed_min = (end_time - self.start_time) / 60
        return elapsed_min, energy_kwh

def log_to_csv(data: dict):
    """Ghi kết quả thí nghiệm vào file CSV."""
    file_exists = os.path.isfile(Config.LOG_FILE)
    fieldnames = [
        'timestamp', 'model_name', 'technique', 'pruning_amount', 
        'best_epoch', 'val_accuracy', 'test_accuracy', 
        'total_time_min', 'total_energy_kwh', 'model_size_mb'
    ]
    
    with open(Config.LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)
    print(f"[LOG] Results saved to {Config.LOG_FILE}")

def get_model_size_mb(model_path: str) -> float:
    if os.path.exists(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)
    return 0.0

# --- LOGIC GREEN AI ---

def apply_pruning(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """Áp dụng L1 Unstructured Pruning."""
    print(f"[GREEN AI] Applying Pruning: {amount*100}% sparsity...")
    parameters_to_prune = []
    
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
            
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    
    # Loại bỏ buffer để giảm kích thước model vĩnh viễn
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')
        
    return model

class QuantizedModelWrapper(nn.Module):
    """
    Wrapper class để khắc phục lỗi input type mismatch.
    Chuyển đổi: Input (Float) -> QuantStub -> Model (Int8) -> DeQuantStub -> Output (Float)
    """
    def __init__(self, model_fp32):
        super(QuantizedModelWrapper, self).__init__()
        self.quant = QuantStub()
        self.model_fp32 = model_fp32
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x

def apply_quantization(model: nn.Module, calibration_loader) -> nn.Module:
    """Thực hiện Post-Training Static Quantization."""
    print("[GREEN AI] Starting Quantization process...")
    
    model.to('cpu')
    model.eval()

    # Bọc model
    quantized_model = QuantizedModelWrapper(model)
    
    # Cấu hình backend (fbgemm cho x86, qnnpack cho ARM)
    backend = 'fbgemm'
    quantized_model.qconfig = torch.quantization.get_default_qconfig(backend)
    
    # Prepare
    torch.quantization.prepare(quantized_model, inplace=True)

    # Calibration
    print("[GREEN AI] Calibrating (using 20 batches)...")
    with torch.no_grad():
        for i, (images, _) in enumerate(calibration_loader):
            if i >= 20: break
            quantized_model(images.to('cpu'))

    # Convert
    print("[GREEN AI] Converting model to INT8...")
    torch.quantization.convert(quantized_model, inplace=True)
    
    return quantized_model