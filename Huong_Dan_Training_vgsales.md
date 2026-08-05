# Hướng Dẫn & Đề Xuất Mô Hình Training Dự Đoán Doanh Số (`vgsales.csv`)

## 1. Phân Loại Chi Tiết Mô Hình MACHINE LEARNING (ML) & DEEP LEARNING (DL)

Trong đề xuất này có đầy đủ cả **Machine Learning (ML)** và **Deep Learning (DL)**. Dưới đây là phân loại cụ thể:

### 🅰️ Các Mô Hình MACHINE LEARNING (ML)
1. **XGBoost / LightGBM / CatBoost Regressor**:
   - *Thuộc loại*: Supervised Machine Learning (Gradient Boosted Trees).
   - *Ưu điểm*: Thích hợp nhất cho dữ liệu dạng bảng (Tabular data như CSV). Học rất nhanh và chính xác.
2. **Random Forest Regressor**:
   - *Thuộc loại*: Ensemble Machine Learning (Tập hợp cây quyết định).
   - *Ưu điểm*: Tránh過擬合 (overfitting), dễ giải thích trọng số thuộc tính.
3. **Prophet (Meta)**:
   - *Thuộc loại*: Additive Time-Series ML Model.
   - *Ưu điểm*: Chuyên cho dự báo chuỗi thời gian, tự động tách Trend và Seasonality cho 3 năm tới.

---

### 🅱️ Các Mô Hình DEEP LEARNING (DL)
1. **LSTM (Long Short-Term Memory) / GRU**:
   - *Thuộc loại*: Deep Learning - Recurrent Neural Network (RNN).
   - *Ưu điểm*: Chuyên xử lý chuỗi thời gian (Time-Series) nhờ các cổng nhớ (Forget/Input/Output Gates).
2. **MLP (Multi-Layer Perceptron)**:
   - *Thuộc loại*: Deep Feedforward Neural Network (PyTorch / Keras).
   - *Ưu điểm*: Mạng nơ-ron đa lớp với hàm kích hoạt Non-linear (ReLU/GELU) và Dropout.
3. **1D CNN (Temporal Convolutional Network)**:
   - *Thuộc loại*: Deep Learning - Convolutional Neural Network 1D.
   - *Ưu điểm*: Trích xuất đặc trưng chuỗi thời gian qua các bộ lọc tích chập (kernels).

---

## 2. Bảng So Sánh ML vs DL Cho Dự Án Này

| Tiêu chí | Machine Learning (XGBoost / Prophet) | Deep Learning (LSTM / PyTorch MLP) |
| :--- | :--- | :--- |
| **Độ phù hợp dữ liệu ~16k dòng** | ⭐ Cực kỳ cao, tối ưu cho CSV bảng | ⭐ Tốt nếu chuyển thành dạng chuỗi time-series |
| **Độ phức tạp code** | Đơn giản, thư viện hỗ trợ sẵn | Cần dựng mạng (Architecture), Epochs, Loss |
| **Giá trị làm đồ án / báo cáo** | Đạt chuẩn thực tế doanh nghiệp | Rất ấn tượng khi có so sánh ML vs DL |

---

## 3. Khuyên Dùng Cho Dự Án
👉 **Nên thực hiện kết hợp**:
- Dùng **XGBoost (ML)** làm mô hình baseline chính.
- Triển khai thêm **LSTM (DL)** để dự đoán 3 năm tiếp theo.
- So sánh chỉ số **MAE, RMSE, R²** giữa ML và DL trong báo cáo!

---
*Xem chi tiết kế hoạch triển khai tại file: [implementation_plan.md](file:///C:/Users/Kien/.gemini/antigravity-ide/brain/78f7f779-f15f-49d5-82ec-c86244443950/implementation_plan.md)*
