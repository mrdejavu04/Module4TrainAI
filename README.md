# 🎮 Module4TrainAI - Video Game Sales 3-Year Forecasting

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20Prophet-orange?style=for-the-badge)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-LSTM%20%7C%20PyTorch-red?style=for-the-badge)
![Dashboard](https://img.shields.io/badge/Dashboard-HTML5%20%7C%20Chart.js-blue?style=for-the-badge)

Dự án phân tích dữ liệu bán hàng trò chơi điện tử (**Video Game Sales**) và xây dựng hệ thống dự báo doanh số trong **3 năm tiếp theo** ứng dụng kết hợp các thuật toán **Machine Learning (ML)** và **Deep Learning (DL)**.

---

## 📌 Giới Thiệu Dự Án (Project Overview)

Dự án tập trung xử lý tập dữ liệu `vgsales.csv` (~16,000 dòng dữ liệu về các tựa game trên toàn cầu), thực hiện qua các bước:
1. **Làm sạch & Xử lý dữ liệu (Data Cleaning & Preprocessing)**: Xử lý dữ liệu khuyết thiếu, chuẩn hóa định dạng và chuẩn bị chuỗi thời gian.
2. **Huấn luyện Mô hình Dự báo (3-Year Sales Forecast)**:
   - **Machine Learning**: XGBoost Regressor, Random Forest, Prophet (Meta).
   - **Deep Learning**: LSTM (Long Short-Term Memory), Multi-Layer Perceptron (MLP), 1D CNN.
3. **Đánh Giá & So Sánh**: So sánh hiệu năng giữa ML và DL qua các chỉ số $R^2$, $MAE$, $RMSE$.
4. **Trực quan hóa (Interactive Dashboard)**: Xây dựng Dashboard web trực quan (`dashboard.html`) cho phép theo dõi biểu đồ doanh số và kết quả dự báo.

---

## 👥 Thành Viên Nhóm & Phân Công Nhiệm Vụ (Team Members & Roles)

| STT | Họ và Tên | Vai Trò (Role) | Trách Nhiệm Chi Tiết (Responsibilities) |
| :---: | :--- | :--- | :--- |
| 1 | **Hứa Trung Kiên** | **Team Leader & Lead AI Engineer** | Quản lý tổng thể dự án, định hướng kiến trúc hệ thống, chủ trì xây dựng mô hình huấn luyện & tích hợp sản phẩm. |
| 2 | **Nguyễn Văn Hiệu** | **Data Engineer** | Tiền xử lý dữ liệu (`Data_Cleaning_vgsales.ipynb`), làm sạch dữ liệu khuyết thiếu và trích xuất đặc trưng (Feature Engineering). |
| 3 | **Nguyễn Huỳnh Đăng Nguyên** | **AI/ML Specialist** | Xây dựng, huấn luyện và tinh chỉnh các mô hình Machine Learning (XGBoost, Random Forest, Prophet) & Deep Learning (LSTM, MLP). |
| 4 | **Mai Vũ Tuấn Minh** | **Frontend & Visualization Developer** | Thiết kế và phát triển giao diện Web Dashboard tương tác (`dashboard.html`) để biểu diễn kết quả phân tích & dự báo 3 năm. |
| 5 | **Đinh Văn Hưng** | **Data Analyst & Model Evaluator** | Trực quan hóa dữ liệu khám phá (EDA), đo lường chỉ số đánh giá độ chính xác (MAE, RMSE, $R^2$) và lập báo cáo so sánh ML vs DL. |
| 6 | **Nguyễn Đức Lộc** | **DevOps & Documentation Specialist** | Quản lý Git workflow/GitHub, tự động hóa script sinh Notebook (`generate_nb.py`, `generate_forecast_nb.py`) & hoàn thiện tài liệu kĩ thuật. |

---

## 📂 Cấu Trúc Thư Mục (Repository Structure)

```text
Module4TrainingAI/
│── vgsales.csv                           # Tập dữ liệu gốc bán hàng Video Game
│── vgsales_cleaned.csv                   # Tập dữ liệu đã qua xử lý & làm sạch
│── Data_Cleaning_vgsales.ipynb           # Notebook làm sạch & chuẩn hóa dữ liệu
│── Model_Training_3Year_Forecast.ipynb   # Notebook huấn luyện mô hình ML/DL & dự báo 3 năm
│── generate_nb.py                        # Script tự động tạo Notebook Data Cleaning
│── generate_forecast_nb.py               # Script tự động tạo Notebook Model Training
│── dashboard.html                        # Giao diện Web Dashboard tương tác
│── Huong_Dan_Training_vgsales.md         # Hướng dẫn chi tiết phân loại & so sánh mô hình ML/DL
└── README.md                             # Tài liệu giới thiệu dự án & thông tin nhóm
```

---

## 🚀 Hướng Dẫn Sử Dụng (Getting Started)

### 1. Cài đặt môi trường
Yêu cầu Python version `>= 3.9`. Cài đặt các thư viện cần thiết:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost prophet torch
```

### 2. Thao tác với Notebooks
- Chạy `Data_Cleaning_vgsales.ipynb` để làm sạch dữ liệu từ `vgsales.csv` sang `vgsales_cleaned.csv`.
- Chạy `Model_Training_3Year_Forecast.ipynb` để tiến hành huấn luyện mô hình ML/DL và xuất kết quả dự báo.

### 3. Mở Dashboard Trực Quan
Mở trực tiếp file `dashboard.html` trên trình duyệt web bất kỳ (Chrome, Edge, Firefox) để xem Dashboard theo dõi kết quả.

---

## 📊 Kết Quả & Đánh Giá (Evaluation Metrics)

Các mô hình được đánh giá dựa trên:
- **$R^2$ Score (Coefficient of Determination)**: Đánh giá khả năng giải thích biến thiên của mô hình.
- **MAE (Mean Absolute Error)**: Sai số tuyệt đối trung bình.
- **RMSE (Root Mean Squared Error)**: Căn bậc hai sai số bình phương trung bình.
