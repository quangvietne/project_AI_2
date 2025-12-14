# 📈 Dự báo Giá Chứng khoán VN30 sử dụng mô hình Multi-Feature LSTM

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458)
![Status](https://img.shields.io/badge/Trạng%20thái-Hoàn%20thành-success)

## 📖 Tổng quan
Dự án này xây dựng một giải pháp Deep Learning toàn diện để dự báo xu hướng giá của các cổ phiếu Blue-chip (nhóm VN30) tại Việt Nam. Thay vì chỉ sử dụng dữ liệu giá đóng cửa đơn thuần, hệ thống tích hợp các **Chỉ báo Phân tích Kỹ thuật (RSI, MACD, MA)** vào kiến trúc mạng **Stacked LSTM (Long Short-Term Memory)** để học được các mẫu hình biến động phức tạp của thị trường.

Hệ thống bao gồm một quy trình (pipeline) khép kín: Từ tự động thu thập dữ liệu (CafeF API), tiền xử lý, trích xuất đặc trưng đến huấn luyện mô hình với bộ tham số được tối ưu riêng cho từng mã cổ phiếu.

## 🚀 Tính năng nổi bật

* **Pipeline dữ liệu tự động:**
    * Xây dựng Crawler tùy chỉnh sử dụng thư viện `requests` để tải dữ liệu lịch sử (2020-2024) trực tiếp từ API của **CafeF**.
    * Xử lý thông minh các ngày nghỉ giao dịch và nội suy dữ liệu (interpolation) để đảm bảo tính liên tục của chuỗi thời gian.
* **Feature Engineering (Kỹ thuật đặc trưng) chuyên sâu:**
    * Tích hợp các chỉ báo kỹ thuật quan trọng để cung cấp ngữ cảnh cho mô hình:
        * **Động lượng:** RSI (Relative Strength Index).
        * **Xu hướng:** MACD (Moving Average Convergence Divergence).
        * **Làm mượt:** SMA (Simple Moving Average) & EMA (Exponential Moving Average) chu kỳ 10/50 ngày.
    * Chuẩn hóa dữ liệu đa biến sử dụng `MinMaxScaler`.
* **Kiến trúc Deep Learning:**
    * **Stacked LSTM:** Sử dụng 3 lớp LSTM xếp chồng để trích xuất đặc trưng thời gian ở nhiều cấp độ trừu tượng.
    * **Chống Overfitting:** Tích hợp lớp Dropout để giảm thiểu rủi ro học vẹt trên dữ liệu tài chính nhiều nhiễu.
    * **Tinh chỉnh (Fine-tuning):** Bộ siêu tham số (Hyperparameters) như *Look-back window* (90-120 ngày), *Learning Rate*, và số lượng *Units* được tùy chỉnh riêng biệt cho từng mã (FPT, HPG, VNM,...) để đạt hiệu suất cao nhất.

## 📊 Kết quả thực nghiệm

Mô hình được đánh giá dựa trên sai số **RMSE** (Root Mean Squared Error) và sai số chuẩn hóa **NRMSE** (Normalized RMSE) trên tập kiểm thử (Test set).

| Mã cổ phiếu | RMSE | **NRMSE (%)** | Đánh giá hiệu suất |
| :--- | :---: | :---: | :--- |
| **BID** | 1.16 | **2.46%** | 🌟 Độ chính xác xuất sắc, bám sát xu hướng thực tế. |
| **VNM** | 2.33 | **3.43%** | Dự báo rất ổn định. |
| **REE** | 2.51 | **3.76%** | Tương quan mạnh với giá thực tế. |
| **HPG** | 1.09 | **4.00%** | Xử lý tốt các biến động giá thép. |
| **FPT** | 14.82 | 10.94% | Nắm bắt được xu hướng tăng dài hạn nhưng còn nhiễu ở các đỉnh giá. |

### Trực quan hóa kết quả

**Biểu đồ Dự đoán vs Thực tế (Mã BID)**
*(Bạn hãy chèn ảnh chụp biểu đồ dự đoán của mã BID vào đây)*
![BID Prediction Graph](./images/BID_prediction.png)

**Lịch sử hàm Loss trong quá trình huấn luyện**
*(Bạn hãy chèn ảnh chụp biểu đồ Loss vào đây)*
![Loss Graph](./images/loss_graph.png)

## 🛠️ Công nghệ sử dụng

* **Ngôn ngữ:** Python 3.x
* **Xử lý dữ liệu:** Pandas, NumPy
* **Deep Learning:** TensorFlow (Keras), Scikit-learn
* **Trực quan hóa:** Matplotlib
* **Nguồn dữ liệu:** CafeF API

## ⚙️ Quy trình hoạt động

1.  **Thu thập dữ liệu:** Script tự động tải dữ liệu giao dịch (Open, High, Low, Close, Volume) từ 2020 đến nay.
2.  **Tiền xử lý:**
    * Chuẩn hóa định dạng ngày tháng.
    * Lấp đầy dữ liệu thiếu (Fill missing dates).
    * Tính toán và thêm các cột chỉ báo kỹ thuật (RSI, MACD, SMA, EMA).
3.  **Tạo chuỗi (Sequence Generation):** Dữ liệu được chuyển đổi thành dạng chuỗi 3D (Samples, Time Steps, Features) dựa trên cửa sổ trượt (ví dụ: dùng 90 ngày quá khứ để dự đoán ngày tiếp theo).
4.  **Huấn luyện:** Mô hình LSTM học trên 80% dữ liệu và kiểm định (validate) trên 10%.
5.  **Đánh giá:** Dự đoán trên tập Test (20% còn lại) và so sánh với giá thực tế sau khi inverse-scaling.

## 📂 Cấu trúc dự án