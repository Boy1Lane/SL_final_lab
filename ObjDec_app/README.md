# Traffic Object Detection Web App

Web app Streamlit dùng YOLOv8 để upload ảnh, chạy object detection, hiển thị ảnh có bounding box, bảng kết quả và tải ảnh kết quả về máy.

## Trạng thái chạy thử

App đã chạy thành công bằng lệnh:

```powershell
python -m streamlit run app.py
```

Terminal hiển thị:

```text
Local URL: http://localhost:8501
Network URL: http://192.168.100.145:8501
```

Nếu terminal dừng ở dòng URL thì app đang chạy. Mở trình duyệt và truy cập:

```text
http://localhost:8501
```

Lưu ý: nếu terminal hiện `Stopping...` thì server đã dừng, cần chạy lại lệnh trên.

## Cấu trúc thư mục

```text
object_detection_web_app/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── app_models/
│   └── yolov8s_subset/
│       ├── best.pt
│       ├── class_names.json
│       └── inference_config.json
├── src/
│   ├── __init__.py
│   ├── detector.py
│   ├── image_utils.py
│   └── config_loader.py
├── uploads/
│   └── .gitkeep
├── outputs/
│   └── .gitkeep
└── docs/
    └── usage.md
```

## Cài môi trường

Khuyến nghị dùng Python 3.10 hoặc 3.11 theo yêu cầu ban đầu. Nếu máy có Python 3.11, tạo virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu gặp lỗi:

```text
No suitable Python runtime found
```

nghĩa là máy chưa cài Python 3.11 cho Python Launcher. Có thể cài Python 3.11, hoặc dùng Python hiện có nếu các dependency đã cài và app chạy được.

Trên máy đã chạy thử, Python mặc định là Python 3.14 và các package đã được cài bằng:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Lỗi activate PowerShell

Nếu chạy:

```powershell
.\.venv\Scripts\Activate.ps1
```

mà gặp lỗi:

```text
running scripts is disabled on this system
```

chạy một lần:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Sau đó mở lại terminal hoặc chạy lại lệnh activate.

Nếu không dùng virtual environment, vẫn có thể chạy app bằng Python hiện tại sau khi đã cài đủ dependency.

## Đặt model

Đặt file model YOLOv8 đã train vào:

```text
app_models/yolov8s_subset/best.pt
```

Nếu thiếu file này, app sẽ hiển thị lỗi thân thiện trên UI và không crash traceback.

File model `.pt` đã được đưa vào `.gitignore`, vì vậy GitHub sẽ không push model lớn. Sau khi clone repo trên máy khác, cần tự copy `best.pt` vào đúng thư mục trên.

## Chạy app

Trong thư mục `object_detection_web_app`, chạy:

```powershell
python -m streamlit run app.py
```

Có thể dùng lệnh ngắn hơn nếu `streamlit` đã nằm trong PATH:

```powershell
streamlit run app.py
```

Khi chạy thành công, terminal sẽ hiển thị URL dạng:

```text
Local URL: http://localhost:8501
Network URL: http://<ip-may-cua-ban>:8501
```

Mở `Local URL` trên trình duyệt để dùng app.

## Cách dùng UI

1. Upload ảnh `.jpg`, `.jpeg` hoặc `.png`.
2. Chọn model folder trong sidebar.
3. Chỉnh `Confidence threshold`, `IoU threshold` và `Device`.
4. Bấm `Run Detection`.
5. Xem ảnh kết quả, bảng detection và tải ảnh bằng `Download result image`.

Nếu chọn `cuda` nhưng máy không có CUDA, app sẽ fallback sang CPU và hiển thị cảnh báo.

## Đổi model subset/full dataset

Khi có model full dataset, tạo thư mục:

```text
app_models/yolov8s_full_dataset/
├── best.pt
├── class_names.json
└── inference_config.json
```

Sau đó chọn `yolov8s_full_dataset` trong sidebar của app. Không cần sửa logic chính trong code.

## Class mặc định

Config mặc định gồm các class:

```text
person, car, bus, bicycle, motorbike
```

App không hard-code class trong logic detection. Class được đọc từ `class_names.json` và thông tin inference được đọc từ `inference_config.json`.

## Mô tả UI mới

Sidebar chỉ còn các control cần thiết: chọn model folder, chỉnh confidence threshold, IoU threshold và device. Khu vực chính có header gọn, vùng upload ảnh, trạng thái chạy, preview ảnh gốc, ảnh kết quả, metric tóm tắt, bảng detection và nút `Download result image`.

App có cấu hình Streamlit mặc định cho dark mode trong `.streamlit/config.toml`. CSS dùng màu theo theme nên không bị lỗi chữ/nền khi đổi dark/light mode.

UI được chỉnh theo hướng Material Design và 10 nguyên tắc Jakob Nielsen:

- Phân cấp thị giác rõ bằng header, section label, spacing và surface 8px radius.
- Sidebar chỉ chứa control inference, tránh nhiễu thông tin.
- Input có label thật và helper text, không ẩn label quan trọng.
- Nút `Run Detection` bị disabled khi thiếu ảnh hoặc config chưa hợp lệ để giảm lỗi thao tác.
- Có trạng thái file/model/kích thước ảnh trước khi chạy.
- Có spinner khi inference, cảnh báo fallback CPU nếu CUDA không khả dụng.
- Empty state và error message chỉ rõ hành động tiếp theo.
- Bảng kết quả và download xuất hiện đúng ngữ cảnh sau khi chạy detection.

## Push lên GitHub

Trước khi push, project nên chỉ gồm source code, config và tài liệu. Các thư mục/file local như `.venv/`, `.vs/`, `__pycache__/`, ảnh trong `outputs/`, ảnh upload và model `.pt` đã được ignore hoặc dọn khỏi project.

Nếu chưa khởi tạo Git:

```powershell
git init
git branch -M main
git add .
git commit -m "Initial object detection web app"
git remote add origin <github-repo-url>
git push -u origin main
```

Kiểm tra file chuẩn bị commit:

```powershell
git status --short
```
