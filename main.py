import argparse
import torch
import os
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from datetime import datetime

# Import modules
from src.models import get_model
from src.data_loader import get_cifar100_loaders
from src.utils import (
    TrainingTracker, get_model_size_mb, log_experiment_to_csv, 
    apply_pruning, apply_quantization
)
from src.engine import train_one_epoch, evaluate

def main():
    parser = argparse.ArgumentParser(description="Energy-Aware DL")
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'mobilenet_v2'])
    parser.add_argument('--batch_size', type=int, default=64)
    
    # --- SỬA ĐỔI 1: Cố định mặc định 50 Epochs ---
    parser.add_argument('--epochs', type=int, default=50, help="Số epoch cố định (Fixed Budget)")
    
    parser.add_argument('--lr', type=float, default=0.1) 
    # Patience không còn quan trọng nữa vì ta sẽ tắt tính năng dừng sớm
    parser.add_argument('--patience', type=int, default=100) 
    
    parser.add_argument('--technique', type=str, default='baseline', 
                        choices=['baseline', 'pruning', 'quantization'],
                        help='Chế độ chạy: baseline, pruning, quantization')
    
    parser.add_argument('--pruning_amount', type=float, default=0.3, 
                        help='Ty le cat tia')
    
    args = parser.parse_args()

    # --- Cấu hình thiết bị ---
    if args.technique == 'quantization':
        DEVICE = 'cpu'
        print("[INFO] Chuyển sang CPU để chạy Quantization.")
    else:
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
    print(f"--- RUNNING: {args.model} | Mode: {args.technique} | Epochs: {args.epochs} (FIXED) ---")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    # --- LOGIC CHỌN AUGMENTATION ---
    use_aug = False
    if args.model == 'resnet50' and args.technique != 'quantization':
        use_aug = True
    
    print(f"[INFO] Cấu hình Data Loader: Augmentation = {use_aug}")
    
    train_loader, val_loader, test_loader = get_cifar100_loaders(args.batch_size, augment=use_aug)
    
    model = get_model(args.model, num_classes=100)
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()

    # --- XỬ LÝ LOGIC THEO TỪNG KỸ THUẬT ---
    project_name = f"{args.model}_{args.technique}"

    if args.technique == 'quantization':
        # ... (Phần Quantization giữ nguyên như cũ vì nó không train epoch) ...
        # (Để ngắn gọn, mình lược bỏ phần code quantization ở đây vì bạn đã có ở tin nhắn trước
        #  và Quantization không dùng vòng lặp epoch. Bạn giữ nguyên code phần này nhé.)
        
        # Nếu bạn cần copy lại full code Quantization thì bảo mình, 
        # nhưng logic chính cần sửa nằm ở vòng lặp for bên dưới.
        pass 

    elif args.technique == 'pruning':
        baseline_path = f"./results/{args.model}_baseline_best.pth"
        if not os.path.exists(baseline_path): baseline_path = f"./results/{args.model}_best.pth"

        if os.path.exists(baseline_path):
            print(f"[INFO] Tải trọng số từ: {baseline_path}")
            model.load_state_dict(torch.load(baseline_path, map_location=DEVICE))
        
        model = apply_pruning(model, amount=args.pruning_amount)
        print("[INFO] Chế độ Fine-tuning (LR thấp)...")
        optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
        project_name += f"_{args.pruning_amount}"

    else: # Baseline
        print("[INFO] Chế độ Train Baseline...")
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        project_name = f"{args.model}_baseline"

    # --- CHUNG CHO BASELINE VÀ PRUNING ---
    # Điều chỉnh Scheduler cho phù hợp với 50 epochs
    # Giảm LR tại epoch 25 và 40
    scheduler = MultiStepLR(optimizer, milestones=[25, 40], gamma=0.1)
    
    tracker = TrainingTracker(output_dir="./results", project_name=project_name)
    best_model_path = os.path.join("./results", f"{project_name}_best.pth")

    best_val_acc = 0.0
    best_epoch = 0
    # patience_counter = 0  <-- BỎ: Không cần đếm kiên nhẫn nữa

    tracker.start()

    # --- VÒNG LẶP TRAINING CỐ ĐỊNH ---
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        if scheduler: scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader, criterion, DEVICE)
        
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.2%}")

        # Vẫn lưu lại model tốt nhất nếu gặp
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)
            # Không reset patience nữa
        
        # --- SỬA ĐỔI 2: XÓA BỎ LOGIC EARLY STOPPING ---
        # (Đã xóa đoạn check patience_counter >= args.patience)
        # Code sẽ chạy đủ số epoch quy định rồi mới dừng.

    total_time_min, total_energy_kwh = tracker.stop()

    # Đánh giá cuối cùng (trên model tốt nhất đã lưu)
    print(f"[INFO] Đang đánh giá model tốt nhất (Epoch {best_epoch})...")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("[CẢNH BÁO] Không tìm thấy model best, sử dụng model hiện tại.")

    test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
    
    # Ghi log
    log_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': args.model,
        'technique': args.technique,
        'pruning_amount': args.pruning_amount if args.technique == 'pruning' else 0,
        'best_epoch': best_epoch,
        'val_accuracy': round(best_val_acc, 4),
        'test_accuracy': round(test_acc, 4),
        'total_time_min': round(total_time_min, 2),
        'total_energy_kwh': round(total_energy_kwh, 6),
        'model_size_mb': round(get_model_size_mb(best_model_path), 2)
    }
    
    log_experiment_to_csv(log_data)
    print("\n--- KẾT QUẢ FIXED 50 EPOCHS ---")
    print(log_data)

if __name__ == '__main__':
    main()