# Hướng dẫn sử dụng

## 1. Chuẩn bị model

Copy model YOLOv8 đã train vào:

```text
app_models/yolov8s_subset/best.pt
```

Config mặc định nằm trong:

```text
app_models/yolov8s_subset/class_names.json
app_models/yolov8s_subset/inference_config.json
```

## 2. Chạy app

Trong Visual Studio 2022 Terminal:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## 3. Chạy detection

1. Upload ảnh `.jpg`, `.jpeg` hoặc `.png`.
2. Chọn confidence threshold, IoU threshold và device trong sidebar.
3. Bấm `Run Detection`.
4. Xem ảnh kết quả và bảng detection.
5. Bấm `Download result image` để tải ảnh đã vẽ bounding box.

Sidebar chỉ hiển thị các control phục vụ inference, không hiển thị khối thông tin chi tiết của model.

## 4. Ý nghĩa bảng kết quả

| Cột | Ý nghĩa |
| --- | --- |
| No. | Số thứ tự detection |
| Class | Tên class |
| Confidence | Độ tin cậy |
| Xmin | Tọa độ trái bbox |
| Ymin | Tọa độ trên bbox |
| Xmax | Tọa độ phải bbox |
| Ymax | Tọa độ dưới bbox |

## 5. Đổi model

Tạo một thư mục mới trong `app_models/` có đủ:

```text
best.pt
class_names.json
inference_config.json
```

Sau đó chọn thư mục đó trong sidebar. Nếu chọn `cuda` nhưng máy không có CUDA, app sẽ tự fallback sang CPU.
