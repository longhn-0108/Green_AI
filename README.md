# Project: Energy-Aware Deep Learning
**Giảm tiêu thụ năng lượng trong huấn luyện mô hình học sâu**

Dự án này khám phá các phương pháp nhằm giảm thiểu năng lượng tiêu thụ trong quá trình huấn luyện (training) các mô hình Deep Learning, một phần của nỗ lực "Green AI" (AI Xanh).

Mục tiêu chính là đo lường và phân tích sự **đánh đổi (trade-off)** giữa **Hiệu năng (Accuracy)** và **Năng lượng tiêu thụ (Energy)** khi áp dụng các kỹ thuật tối ưu hóa.

---

## 🎯 Mục tiêu
1.  **Thiết lập Baseline:** Đo lường hiệu năng, thời gian, và năng lượng tiêu thụ của một mô hình CNN (ví dụ: **ResNet-50**) trên bộ dữ liệu chuẩn (ví dụ: **CIFAR-100**).
2.  **Áp dụng Kỹ thuật:** Áp dụng các kỹ thuật tối ưu phổ biến:
    * **Pruning (Tỉa)**
    * **Quantization (Lượng tử hóa)**
3.  **So sánh:** So sánh các kết quả trên với một kiến trúc "nhẹ" (Lightweight Architecture) có sẵn (ví dụ: **MobileNetV2**).
4.  **Phân tích:** Rút ra kết luận về kỹ thuật nào mang lại hiệu quả tiết kiệm năng lượng tốt nhất so với mức sụt giảm hiệu năng.

---

## 🛠️ Cài đặt

1.  Clone kho chứa này về máy:
    ```bash
    git clone [https://github.com/longhn-0108/Green_AI.git](https://github.com/longhn-0108/Green_AI.git)
    cd TEN_REPO_CUA_BAN
    ```

2.  Tạo và kích hoạt môi trường ảo:
    ```bash
    python -m venv venv
    
    # Trên Windows
    .\venv\Scripts\activate

    ```

3.  Cài đặt các thư viện cần thiết:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Cách chạy (Ví dụ cấu trúc)

1.  **Huấn luyện mô hình Baseline (ResNet-50):**
    ```bash
    python train.py --model resnet50 --dataset cifar100 --output_dir ./results/baseline
    ```

2.  **Huấn luyện với Pruning:**
    ```bash
    python train.py --model resnet50 --pruning --dataset cifar100 --output_dir ./results/pruning
    ```

3.  **Huấn luyện với Quantization:**
    ```bash
    python train.py --model resnet50 --quantization --dataset cifar100 --output_dir ./results/quantization
    ```

4.  **Huấn luyện mô hình Lightweight (MobileNetV2):**
    ```bash
    python train.py --model mobilenet_v2 --dataset cifar100 --output_dir ./results/lightweight
    ```

---

## 📊 Kết quả 

Bảng phân tích cuối cùng sẽ so sánh các số liệu quan trọng:

| Mô hình | Kỹ thuật | Accuracy (%) | Năng lượng (kWh) | Thời gian (giờ) | Thiết bị |
| :--- | :--- | :---: | :---: | :---: | :--- |
| ResNet-50 | **Baseline** | 48.95% | 5.723379 kWh | ~44h | CPU Local |
| ResNet-50 | **Pruning 30%** | 70% | 5.707134 kWh | ~43.9h | CPU Local |
| ResNet-50 | **Pruning 50%** | 75.04% | 5.702441 kWh | ~43.8h | CPU Local |
| ResNet-50 | **Pruning 70%** | 74.78% | 5.698115 kWh | ~43.8h | CPU Local |
| ResNet-50 | **Quantization** | (chưa có) | (chưa có) | (chưa có) | CPU Local |
| ResNet-50 | **Baseline** | 74.72% | 0.077634 kWh | ~1.2h | GPU Kaggle |
| ResNet-50 | **Pruning 30%** | 74.89% | 0.077607 kWh | ~1.2h | GPU Kaggle |
| ResNet-50 | **Pruning 50%** | 75.04% | 0.077600 kWh | ~1.2h | GPU Kaggle |
| ResNet-50 | **Pruning 70%** | 74.78% | 0.077867 kWh | ~1.2h | GPU Kaggle |
| ResNet-50 | **Quantization** | (chưa có) | (chưa có) | (chưa có) | GPU Kaggle |
| MobileNetV2 | **Baseline** | 71.55% | 1.447343 kWh | ~11.1h | CPU Local |
| MobileNetV2 | **Baseline** | 71.98% | 0.019623 kWh | ~0.3h | GPU Kaggle |
