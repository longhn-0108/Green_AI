import time
import os
import csv
import torch
import torch.nn.utils.prune as prune
import torch.quantization
from codecarbon import EmissionsTracker

class TrainingTracker:
    """
    Class quản lý việc theo dõi thời gian và điện năng tiêu thụ.
    Sử dụng CodeCarbon để đo lường.
    """
    def __init__(self, output_dir: str, project_name: str):
        self.project_name = project_name
        self.output_dir = output_dir
        self.start_time = None
        self.end_time = None
        
        # log_level='error' giúp tắt các dòng thông báo in liên tục ra màn hình
        self.tracker = EmissionsTracker(
            project_name=project_name,
            output_dir=output_dir,
            measure_power_secs=15,
            log_level='error' 
        )

    def start(self):
        print(f"[TRACKER] Bắt đầu theo dõi năng lượng: {self.project_name}")
        self.start_time = time.time()
        self.tracker.start()

    def stop(self):
        self.end_time = time.time()
        emissions_data = self.tracker.stop()
        
        # Xử lý trường hợp không đo được dữ liệu
        if emissions_data is None:
            emissions_data = 0.0

        elapsed_time_sec = self.end_time - self.start_time
        elapsed_time_min = elapsed_time_sec / 60
        
        # Lấy tổng năng lượng tiêu thụ (kWh)
        energy_kwh = self.tracker.final_emissions_data.energy_consumed if self.tracker.final_emissions_data else 0.0

        print(f"[TRACKER] Hoàn tất. Thời gian: {elapsed_time_min:.2f} phút. Năng lượng: {energy_kwh:.6f} kWh")
        return elapsed_time_min, energy_kwh

def get_model_size_mb(model_path: str) -> float:
    """Tính kích thước file model (đơn vị MB)."""
    if os.path.exists(model_path):
        size_in_bytes = os.path.getsize(model_path)
        return size_in_bytes / (1024 * 1024)
    return 0.0

def log_experiment_to_csv(data: dict, log_file: str = "experiment_log.csv"):
    """
    Ghi kết quả thí nghiệm vào file CSV chung.
    Tự động tạo header nếu file chưa tồn tại.
    """
    file_exists = os.path.isfile(log_file)
    
    # Định nghĩa các cột dữ liệu cần lưu
    fieldnames = [
        'timestamp', 'model_name', 'technique', 'pruning_amount',
        'best_epoch', 'val_accuracy', 'test_accuracy', 
        'total_time_min', 'total_energy_kwh', 'model_size_mb'
    ]
    
    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)
    
    print(f"[LOG] Đã lưu kết quả vào sổ nhật ký: {log_file}")

# --- PHẦN KỸ THUẬT TỐI ƯU ---

def apply_pruning(model, amount=0.3):
    """
    Áp dụng Global Unstructured Pruning.
    Cắt bỏ 'amount' (ví dụ 0.3 = 30%) các trọng số nhỏ nhất trên toàn mạng.
    """
    print(f"[PRUNING] Đang tiến hành tỉa thưa {amount*100}% trọng số...")
    
    # 1. Chọn các lớp Conv2d và Linear để cắt
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    # 2. Thực hiện cắt tỉa (Global L1 Unstructured)
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )

    # 3. Loại bỏ lớp mặt nạ (làm cho việc cắt tỉa thành vĩnh viễn)
    # Giúp giảm kích thước file thực tế khi lưu
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')

    # 4. Tính toán và in ra độ thưa (Sparsity) để kiểm tra
    total_params = 0
    zero_params = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            total_params += module.weight.nelement()
            zero_params += torch.sum(module.weight == 0).item()
    
    if total_params > 0:
        sparsity = 100. * zero_params / total_params
        print(f"[PRUNING] Hoàn tất. Độ thưa thực tế: {sparsity:.2f}%")
    
    return model

def apply_quantization(model, calibration_loader):
    """
    Áp dụng Post-Training Static Quantization (PTQ).
    Chuyển đổi trọng số từ Float32 sang Int8.
    """
    print(f"[QUANTIZATION] Đang chuẩn bị Lượng tử hóa (Static PTQ)...")
    
    # 1. Chuyển model về CPU (Bắt buộc với PyTorch Quantization hiện tại)
    model.to('cpu')
    model.eval()

    # 2. Cấu hình (QConfig)
    # 'fbgemm' là backend tối ưu cho Server/PC (Intel/AMD)
    # Nếu chạy lỗi trên một số máy, có thể thử đổi thành 'default'
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # 3. Prepare (Chèn các observer để quan sát luồng dữ liệu)
    torch.quantization.prepare(model, inplace=True)

    # 4. Calibration (Hiệu chỉnh)
    # Chạy thử một lượng nhỏ dữ liệu để model xác định khoảng giá trị min/max
    print("[QUANTIZATION] Đang hiệu chỉnh (Calibration) - Vui lòng chờ...")
    with torch.no_grad():
        for i, (images, labels) in enumerate(calibration_loader):
            if i >= 20: break # Chỉ cần khoảng 20 batches là đủ để hiệu chỉnh
            images = images.to('cpu')
            model(images) # Chạy forward pass (không cần tính loss)
            
    # 5. Convert (Chuyển đổi thật sự sang Int8)
    print("[QUANTIZATION] Đang chuyển đổi sang INT8...")
    torch.quantization.convert(model, inplace=True)
    
    print("[QUANTIZATION] Hoàn tất chuyển đổi.")
    return model