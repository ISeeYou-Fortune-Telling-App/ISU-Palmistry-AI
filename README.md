# 🔮 Fortune On Your Hand: Ứng Dụng Chiêm Tinh AI Bất Biến Theo Góc Nhìn

## 📋 Tóm Tắt

**Phần mềm phát hiện các đường chính trên lòng bàn tay** của chúng tôi được triển khai theo 4 bước chính. Thách thức chính của chúng tôi là đọc các đường chính trên lòng bàn tay **bất kể hướng nhìn** và **điều kiện ánh sáng**:

1. **Chỉnh sửa hình ảnh lòng bàn tay bị nghiêng**
2. **Phát hiện các đường chính trên lòng bàn tay**
3. **Phân loại các đường**
4. **Đo độ dài của từng đường**

<img width="1362" alt="model_architecture" src="https://user-images.githubusercontent.com/81272473/208795260-48ba6c8f-92a1-4b01-9471-6a4703ad0aff.png">

Để chỉnh sửa hình ảnh lòng bàn tay, chúng tôi sử dụng MediaPipe để trích xuất các điểm quan trọng và triển khai biến dạng với các điểm này. Để phát hiện đường chính, chúng tôi xây dựng mô hình học sâu và huấn luyện mô hình với tập dữ liệu hình ảnh lòng bàn tay. Để phân loại đường, chúng tôi sử dụng K-means clustering để phân bổ từng pixel cho đường cụ thể. Để đo độ dài, chúng tôi đặt ngưỡng cho từng đường chính với các mốc được lấy bởi MediaPipe.

## 🚀 Cài Đặt Môi Trường

### 📦 Yêu Cầu Hệ Thống
- **Python 3.7.6+**
- **Windows/Linux/MacOS**

### 🔧 Cài Đặt Virtual Environment

#### Bước 1: Tạo Virtual Environment
```bash
# Từ thư mục gốc của dự án (palmistry/)
python -m venv palm_env
```

#### Bước 2: Kích Hoạt Virtual Environment

**Windows (PowerShell):**
```powershell
# Từ thư mục gốc của dự án
palm_env\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
# Từ thư mục gốc của dự án
palm_env\Scripts\activate.bat
```

**Linux/Mac:**
```bash
# Từ thư mục gốc của dự án
source palm_env/bin/activate
```

#### Bước 3: Cài Đặt Các Thư Viện Cần Thiết
```bash
# Đảm bảo đã kích hoạt virtual environment
pip install -r code/requirements.txt
```

### 📚 Các Thư Viện Chính
- `torch` - PyTorch cho học sâu
- `torchvision` - Thư viện thị giác máy tính
- `scikit-image` - Xử lý hình ảnh
- `opencv-python` - OpenCV cho thị giác máy tính
- `pillow-heif` - Hỗ trợ định dạng HEIF
- `mediapipe` - Phát hiện điểm landmark bàn tay

## 🎯 Chạy Ứng Dụng

### Chuẩn Bị Dữ Liệu Đầu Vào
1. **Chuẩn bị hình ảnh lòng bàn tay** (.jpg hoặc .heic) trong thư mục `./code/input/`
2. Dự án đã cung cấp hình ảnh mẫu: `hand70.jpg`

### Chạy Phân Tích
```bash
# Từ thư mục gốc của dự án (palmistry/)
python code/read_palm.py --input hand70.jpg
```

**Ví dụ đầy đủ:**
```powershell
# Windows PowerShell
PS C:\Users\Windows\Downloads\palmistry> & "palm_env\Scripts\python.exe" code/read_palm.py --input hand70.jpg
```

### Kết Quả
Sau khi chạy, các file kết quả sẽ được lưu trong thư mục `./code/results/`:
- `result.jpg` - Hình ảnh kết quả với các đường được đánh dấu
- `palm_lines.png` - Các đường chính được phát hiện
- `warped_palm.jpg` - Hình ảnh lòng bàn tay đã được chỉnh sửa
- Và các file khác...

## 📊 Ví Dụ Kết Quả

### Lòng Bàn Tay Chuẩn
<img width="1371" alt="standard" src="https://user-images.githubusercontent.com/81272473/208797334-9cf56f18-01b1-46e5-9bab-5a38a696d05f.png">

### Lòng Bàn Tay Bị Nghiêng
<img width="1361" alt="tilted" src="https://user-images.githubusercontent.com/81272473/208797357-fe007daf-0d24-48b0-80af-21d79b64db4a.png">

## 🔍 Chi Tiết Triển Khai Phân Đoạn Đường

**Cập nhật: 22.12.03 21:57**

### Giả Định
- Đường không đi đến biên của hình ảnh (trong trường hợp này, skeletonize của scikit thường không hoạt động. Ngay cả khi skeletonize được, cần sửa đổi thuật toán nhóm một chút)
- Các đường giao nhau tại tối đa một điểm (theo test case. Có thể xử lý với một số triển khai bổ sung)

### Nhóm Đường
- **Giá trị trả về**: danh sách các đường, mỗi đường cũng là danh sách các pixel
```python
# Ví dụ: [ [[1, 2], [2, 3]], [[10, 11], [11, 11]] ]
```

### Giải Thích Triển Khai
1. Đối với tất cả pixel, đếm các giá trị khác 0 trong 8 pixel xung quanh
2. Kết quả đếm: 0: không trên đường, 1: cuối đường, 2: giữa đường, 3: điểm giao nhau
3. Bắt đầu từ pixel cuối đường, khám phá 8 pixel xung quanh, theo pixel có count ≠ 0 và chưa được thăm
4. Tiếp tục cho đến khi đạt pixel có count = 1 hoặc 3
5. Pixel count = 1: tìm thấy một đường, lưu và loại trừ khỏi vòng lặp để tránh khám phá ngược. Pixel count = 3: lưu đường riêng để xử lý sau
6. Kiểm tra các đường kết thúc bằng 3 có thể nối với nhau không: kiểm tra sự khác biệt điểm đầu và cuối để xác định hướng ngược lại cho tất cả tổ hợp và nối thành đường
7. Trả về các đường đã lưu

## ⚠️ Vấn Đề Đã Biết

- **Skeletonize có thể nối các đường không liên quan** (1 trường hợp, một đường có thể dài hơn một chút) → Cần thêm test
- **Xử lý đường bị đứt đoạn không rõ ràng**: Hiện tại bỏ qua và tiếp tục, có thể tính gradient của các đường đã nhóm để xử lý, nhưng có thể nối nhầm đường → Có lẽ nên ẩn các trường hợp này...

## 🤝 Đóng Góp

Chúng tôi hoan nghênh mọi đóng góp! Vui lòng tạo issue hoặc pull request nếu bạn có ý tưởng cải thiện.

## 📄 Giấy Phép

Dự án này được phát triển cho mục đích nghiên cứu và giáo dục.