# Hướng dẫn sử dụng

## 1. Chuẩn bị model

Đặt model YOLOv8 đã train vào:

```text
app_models/YOLOv8s full split/best.pt
```

Các file cấu hình đi kèm:

```text
app_models/YOLOv8s full split/class_names.json
app_models/YOLOv8s full split/inference_config.json
```

## 2. Chạy ứng dụng

```powershell
python -m streamlit run app.py
```

Nếu cần dùng Python theo đường dẫn cụ thể:

```powershell
C:\Users\quoch\AppData\Local\Programs\Python\Python314\python.exe -m streamlit run app.py
```

Mở trình duyệt tại:

```text
http://localhost:8501
```

## 3. Chạy detection

1. Chọn nguồn ảnh:
   - `Tải ảnh lên`: chọn file JPG, JPEG hoặc PNG.
   - `Dán ảnh`: click vào vùng dán rồi nhấn Ctrl+V.
   - `Camera`: cấp quyền camera và chụp ảnh.
2. Chỉnh `Confidence Threshold` và `IoU Threshold` nếu cần.
3. Bấm `Run Detection`.
4. Xem ảnh kết quả, bảng detection và thống kê.
5. Bấm `Tải ảnh PNG` để tải ảnh đã vẽ bounding box.

## 4. Ý nghĩa bảng kết quả

| Cột | Ý nghĩa |
| --- | --- |
| No. | Số thứ tự detection |
| Class | Tên class |
| Confidence | Độ tin cậy |
| Xmin | Tọa độ trái của bounding box |
| Ymin | Tọa độ trên của bounding box |
| Xmax | Tọa độ phải của bounding box |
| Ymax | Tọa độ dưới của bounding box |

## 5. Các trạng thái thường gặp

- `Chưa có ảnh đầu vào`: hãy tải ảnh, dán ảnh hoặc chụp ảnh trước.
- `Không phát hiện đối tượng nào`: thử giảm Confidence Threshold.
- `Máy hiện không có CUDA khả dụng`: app đã tự chạy bằng CPU.
- `Chỉ hỗ trợ ảnh JPG, JPEG hoặc PNG`: hãy chọn hoặc dán ảnh đúng định dạng.

## 6. Đổi model

Tạo thư mục mới trong `app_models/` có đủ:

```text
best.pt
class_names.json
inference_config.json
```

Sau đó chọn thư mục model mới trong sidebar.
