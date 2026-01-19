import argparse
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from datetime import datetime

from src.config import Config
from src.models import get_model
from src.data_loader import get_cifar100_loaders
from src.engine import train_one_epoch, evaluate
from src.utils import (
    set_seed, ExperimentTracker, log_to_csv, get_model_size_mb,
    apply_pruning, apply_quantization
)

def run_experiment(args):
    # 1. Cố định Random Seed
    set_seed(Config.SEED)
    
    # 2. Setup Device & Directory
    device = Config.get_device(args.technique)
    if not os.path.exists(Config.RESULT_DIR):
        os.makedirs(Config.RESULT_DIR)
        
    print(f"🚀 STARTING EXPERIMENT: {args.model} | {args.technique} | Device: {device}")

    # 3. Load Data
    use_aug = (args.technique != 'quantization')
    train_loader, val_loader, test_loader = get_cifar100_loaders(augment=use_aug)

    # 4. Initialize Model
    model = get_model(args.model, num_classes=Config.NUM_CLASSES)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # --- PIPELINE LOGIC ---

    # === CASE A: QUANTIZATION ===
    if args.technique == 'quantization':
        # Logic tìm file weights tốt nhất để nén
        pruned_file = f"{args.model}_pruning_{args.pruning_amount}_best.pth"
        baseline_file = f"{args.model}_baseline_best.pth"
        
        load_path = os.path.join(Config.RESULT_DIR, pruned_file)
        if not os.path.exists(load_path):
            load_path = os.path.join(Config.RESULT_DIR, baseline_file)
            
        if not os.path.exists(load_path):
            print(f"❌ Error: Weights not found at {load_path}. Please train baseline/pruning first.")
            return

        print(f"📥 Loading weights for quantization: {load_path}")
        model.load_state_dict(torch.load(load_path, map_location='cpu'))
        
        # Nếu load model pruning thì phải tái tạo cấu trúc pruning trước
        if 'pruning' in load_path:
            model = apply_pruning(model, amount=args.pruning_amount)

        # Đo đạc quá trình Quantization
        tracker = ExperimentTracker(f"{args.model}_quantization")
        tracker.start()
        
        quantized_model = apply_quantization(model, train_loader)
        
        print("[INFO] Evaluating Quantized Model...")
        _, test_acc = evaluate(quantized_model, test_loader, criterion, 'cpu')
        
        t_min, e_kwh = tracker.stop()
        
        # Lưu kết quả
        save_path = os.path.join(Config.RESULT_DIR, f"{args.model}_quantized.pth")
        torch.save(quantized_model.state_dict(), save_path)
        
        log_data = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_name': args.model, 'technique': 'quantization',
            'pruning_amount': args.pruning_amount, 'best_epoch': 0,
            'val_accuracy': test_acc, 'test_accuracy': test_acc,
            'total_time_min': t_min, 'total_energy_kwh': e_kwh,
            'model_size_mb': get_model_size_mb(save_path)
        }
        log_to_csv(log_data)
        print(f"✅ DONE. Acc: {test_acc:.4f} | Size: {log_data['model_size_mb']:.2f} MB")
        return

    # === CASE B: PRUNING & BASELINE ===
    
    # Setup Optimizer
    if args.technique == 'pruning':
        # Load Baseline weights
        base_path = os.path.join(Config.RESULT_DIR, f"{args.model}_baseline_best.pth")
        if os.path.exists(base_path):
            print(f"📥 Loading baseline weights: {base_path}")
            model.load_state_dict(torch.load(base_path, map_location=device))
        
        model = apply_pruning(model, amount=args.pruning_amount)
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9) # Fine-tune LR thấp
        experiment_name = f"{args.model}_pruning_{args.pruning_amount}"
    else:
        optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
        experiment_name = f"{args.model}_baseline"

    scheduler = MultiStepLR(optimizer, milestones=[25, 40], gamma=0.1)
    tracker = ExperimentTracker(experiment_name)
    best_acc = 0.0
    save_path = os.path.join(Config.RESULT_DIR, f"{experiment_name}_best.pth")

    # Training Loop
    print(f"[INFO] Training started for {Config.NUM_EPOCHS} epochs...")
    tracker.start()
    
    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.2%}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            
    t_min, e_kwh = tracker.stop()
    
    # Final Test
    print("[INFO] Loading best model for testing...")
    model.load_state_dict(torch.load(save_path))
    _, test_acc = evaluate(model, test_loader, criterion, device)
    
    log_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'model_name': args.model, 'technique': args.technique,
        'pruning_amount': args.pruning_amount if args.technique == 'pruning' else 0,
        'best_epoch': Config.NUM_EPOCHS, 'val_accuracy': best_acc, 'test_accuracy': test_acc,
        'total_time_min': t_min, 'total_energy_kwh': e_kwh,
        'model_size_mb': get_model_size_mb(save_path)
    }
    log_to_csv(log_data)
    print("✅ Experiment Completed Successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Green AI Experiment Runner")
    parser.add_argument('--model', type=str, default='resnet50', choices=['resnet50', 'mobilenet_v2'])
    parser.add_argument('--technique', type=str, default='baseline', choices=['baseline', 'pruning', 'quantization'])
    parser.add_argument('--pruning_amount', type=float, default=0.3, help="Amount of sparsity (0.0 - 1.0)")
    
    args = parser.parse_args()
    run_experiment(args)