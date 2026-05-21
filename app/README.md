# YOLOv8 Detection Dashboard

Ứng dụng Streamlit dùng YOLOv8 để nhận diện đối tượng trong ảnh. Người dùng có thể tải ảnh lên, dán ảnh từ clipboard, chụp ảnh bằng camera, chạy detection, xem bounding box, xem bảng kết quả và tải ảnh kết quả về máy.

## Tính năng chính

- Tải ảnh JPG, JPEG, PNG từ máy tính.
- Dán ảnh trực tiếp bằng Ctrl+V.
- Chụp ảnh bằng camera của trình duyệt.
- Chạy YOLOv8 inference với Confidence Threshold và IoU Threshold.
- Hiển thị ảnh gốc, ảnh kết quả, metrics, bảng detection và biểu đồ thống kê.
- Tải ảnh kết quả PNG.
- Tùy chọn lưu ảnh kết quả vào thư mục `outputs/`.

## Cài đặt

Trong thư mục app:

```powershell
python -m pip install -r requirements.txt
```

Nếu máy dùng Python tại đường dẫn cụ thể:

```powershell
C:\Users\quoch\AppData\Local\Programs\Python\Python314\python.exe -m pip install -r requirements.txt
```

## Chạy app

```powershell
python -m streamlit run app.py
```

Hoặc:

```powershell
C:\Users\quoch\AppData\Local\Programs\Python\Python314\python.exe -m streamlit run app.py
```

Sau khi chạy, mở:

```text
http://localhost:8501
```

## Cách sử dụng

1. Chọn nguồn ảnh: `Tải ảnh lên`, `Dán ảnh`, hoặc `Camera`.
2. Cung cấp ảnh hợp lệ.
3. Điều chỉnh `Confidence Threshold`, `IoU Threshold`, và thiết bị chạy trong sidebar nếu cần.
4. Bấm `Run Detection`.
5. Xem ảnh kết quả, bảng detection, thông tin model và thống kê.
6. Bấm `Tải ảnh PNG` nếu muốn tải ảnh kết quả.

## Cấu trúc thư mục

```text
app/
├── app.py
├── requirements.txt
├── README.md
├── app_models/
│   └── YOLOv8s full split/
│       ├── best.pt
│       ├── class_names.json
│       └── inference_config.json
├── src/
│   ├── detector.py
│   ├── image_utils.py
│   ├── config_loader.py
│   └── components/
│       └── paste_image/
│           └── index.html
├── outputs/
└── docs/
    └── usage.md
```

## Đổi model

Tạo một thư mục mới trong `app_models/` có đủ:

```text
best.pt
class_names.json
inference_config.json
```

Sau đó chọn thư mục đó trong sidebar của app.

## Lưu ý demo

- Nếu chọn `cuda` nhưng máy không có CUDA, app sẽ tự chuyển sang CPU và hiển thị cảnh báo.
- Tùy chọn `Tự chạy lại khi đổi threshold` mặc định tắt để tránh chạy model ngoài ý muốn.
- Khi bật tự chạy lại, app không tự lưu thêm file output cho mỗi lần kéo slider.
