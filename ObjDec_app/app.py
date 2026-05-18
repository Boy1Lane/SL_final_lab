from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config_loader import ConfigError, load_inference_config
from src.detector import ModelNotFoundError, YOLODetector
from src.image_utils import encode_image_for_download, read_uploaded_image, save_result_image


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_ROOT = PROJECT_ROOT / "app_models"
DEFAULT_MODEL_FOLDER = "yolov8s_subset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png"]


st.set_page_config(
    page_title="Traffic Object Detection Web App",
    page_icon="OD",
    layout="wide",
)


def apply_page_styles() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }
        [data-testid="stSidebar"] {
            background: var(--secondary-background-color);
            border-right: 1px solid rgba(128, 128, 128, 0.22);
        }
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            letter-spacing: 0;
        }
        .app-hero {
            border-bottom: 1px solid rgba(128, 128, 128, 0.22);
            padding-bottom: 1rem;
            margin-bottom: 1.25rem;
        }
        .app-title {
            color: var(--text-color);
            font-size: 2rem;
            font-weight: 760;
            line-height: 1.15;
            letter-spacing: 0;
            margin: 0;
        }
        .app-subtitle {
            color: var(--text-color);
            font-size: 0.98rem;
            margin: 0.35rem 0 0;
            opacity: 0.72;
        }
        .section-label {
            color: var(--text-color);
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: 0;
            margin: 0.4rem 0 0.6rem;
        }
        .field-caption {
            color: var(--text-color);
            font-size: 0.84rem;
            line-height: 1.45;
            opacity: 0.68;
            margin: -0.35rem 0 0.65rem;
        }
        .muted-line {
            color: var(--text-color);
            font-size: 0.9rem;
            margin-top: -0.2rem;
            opacity: 0.68;
        }
        .status-surface {
            background: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            min-height: 8.25rem;
        }
        .status-row {
            align-items: center;
            border-bottom: 1px solid rgba(128, 128, 128, 0.16);
            display: flex;
            gap: 0.75rem;
            justify-content: space-between;
            padding: 0.45rem 0;
        }
        .status-row:last-child {
            border-bottom: 0;
        }
        .status-label {
            color: var(--text-color);
            font-size: 0.82rem;
            opacity: 0.68;
        }
        .status-value {
            color: var(--text-color);
            font-size: 0.9rem;
            font-weight: 650;
            max-width: 16rem;
            overflow: hidden;
            text-align: right;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        div[data-testid="stMetric"] {
            background: var(--secondary-background-color);
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 8px;
            padding: 0.7rem 0.85rem;
        }
        div[data-testid="stMetric"] label {
            color: var(--text-color);
            opacity: 0.72;
        }
        .stButton > button,
        .stDownloadButton > button {
            border-radius: 8px;
            min-height: 2.75rem;
            font-weight: 700;
        }
        .stFileUploader {
            border: 1px dashed rgba(128, 128, 128, 0.36);
            border-radius: 8px;
            padding: 0.25rem 0.75rem;
            background: var(--secondary-background-color);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_detector(model_dir: str) -> YOLODetector:
    return YOLODetector(model_dir)


def list_model_folders() -> list[str]:
    if not MODELS_ROOT.exists():
        return [DEFAULT_MODEL_FOLDER]

    folders = [
        path.name
        for path in MODELS_ROOT.iterdir()
        if path.is_dir() and (path / "inference_config.json").exists()
    ]
    if DEFAULT_MODEL_FOLDER not in folders:
        folders.insert(0, DEFAULT_MODEL_FOLDER)
    return folders or [DEFAULT_MODEL_FOLDER]


def render_detection_table(detections: list[dict]) -> None:
    rows = []
    for index, item in enumerate(detections, start=1):
        xmin, ymin, xmax, ymax = item["bbox"]
        rows.append(
            {
                "No.": index,
                "Class": item["class_name"],
                "Confidence": round(float(item["confidence"]), 4),
                "Xmin": xmin,
                "Ymin": ymin,
                "Xmax": xmax,
                "Ymax": ymax,
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Confidence": st.column_config.NumberColumn("Confidence", format="%.4f"),
        },
    )


def format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def render_status_panel(model_folder: str, uploaded_file, image_size: tuple[int, int] | None) -> None:
    file_name = uploaded_file.name if uploaded_file is not None else "Chưa chọn"
    file_size = format_file_size(uploaded_file.size) if uploaded_file is not None else "N/A"
    dimensions = f"{image_size[0]} x {image_size[1]} px" if image_size is not None else "N/A"
    escaped_model_folder = escape(model_folder)
    escaped_file_name = escape(file_name)
    escaped_file_size = escape(file_size)
    escaped_dimensions = escape(dimensions)
    st.markdown(
        f"""
        <div class="status-surface">
            <div class="status-row">
                <span class="status-label">Model</span>
                <span class="status-value" title="{escaped_model_folder}">{escaped_model_folder}</span>
            </div>
            <div class="status-row">
                <span class="status-label">File</span>
                <span class="status-value" title="{escaped_file_name}">{escaped_file_name}</span>
            </div>
            <div class="status-row">
                <span class="status-label">Size</span>
                <span class="status-value">{escaped_file_size}</span>
            </div>
            <div class="status-row">
                <span class="status-label">Image</span>
                <span class="status-value">{escaped_dimensions}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


apply_page_styles()

st.markdown(
    """
    <div class="app-hero">
        <h1 class="app-title">Traffic Object Detection Web App</h1>
        <p class="app-subtitle">Upload ảnh giao thông, chạy YOLOv8 inference và tải ảnh kết quả có bounding box.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model_folders = list_model_folders()
default_index = model_folders.index(DEFAULT_MODEL_FOLDER) if DEFAULT_MODEL_FOLDER in model_folders else 0

with st.sidebar:
    st.header("Detection Settings")
    model_folder = st.selectbox(
        "Model folder",
        model_folders,
        index=default_index,
        help="Chọn thư mục model trong app_models/.",
    )
    model_dir = MODELS_ROOT / model_folder

    config: dict | None = None
    try:
        config = load_inference_config(model_dir)
    except (FileNotFoundError, ConfigError, ValueError) as exc:
        st.error(str(exc))

    default_conf = float(config.get("confidence_threshold", 0.25)) if config else 0.25
    default_iou = float(config.get("iou_threshold", 0.45)) if config else 0.45
    default_conf = min(max(default_conf, 0.05), 0.95)
    default_iou = min(max(default_iou, 0.10), 0.90)
    default_device = str(config.get("device", "auto")) if config else "auto"
    if default_device not in {"auto", "cpu", "cuda"}:
        default_device = "auto"

    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=default_conf,
        step=0.01,
        help="Giá trị cao hơn giúp giảm detection sai nhưng có thể bỏ sót object.",
    )
    iou_threshold = st.slider(
        "IoU threshold",
        min_value=0.10,
        max_value=0.90,
        value=default_iou,
        step=0.01,
        help="Điều chỉnh mức loại bỏ các bounding box trùng nhau.",
    )
    device = st.selectbox(
        "Device",
        ["auto", "cpu", "cuda"],
        index=["auto", "cpu", "cuda"].index(default_device),
        help="Chọn auto để app tự dùng CUDA nếu khả dụng, nếu không sẽ dùng CPU.",
    )

    st.divider()
    st.caption("Kết quả được lưu trong thư mục outputs/ sau mỗi lần chạy.")

image = None
image_size = None
upload_col, status_col = st.columns([1.2, 0.8], gap="large")

with upload_col:
    st.markdown('<div class="section-label">Input image</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="field-caption">Định dạng hỗ trợ: JPG, JPEG, PNG.</p>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Upload image",
        type=ALLOWED_IMAGE_TYPES,
        accept_multiple_files=False,
        help="Chọn một ảnh để chạy object detection.",
    )

if uploaded_file is not None:
    try:
        image = read_uploaded_image(uploaded_file)
        image_size = image.size
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

with status_col:
    st.markdown('<div class="section-label">Run status</div>', unsafe_allow_html=True)
    render_status_panel(model_folder, uploaded_file, image_size)

left_col, right_col = st.columns(2, gap="large")
with left_col:
    st.markdown('<div class="section-label">Original image</div>', unsafe_allow_html=True)
    if image is not None:
        st.image(image, use_container_width=True)
    else:
        st.info("Chưa có ảnh đầu vào.")

with right_col:
    result_placeholder = st.empty()
    with result_placeholder.container():
        st.markdown('<div class="section-label">Detection result</div>', unsafe_allow_html=True)
        st.info("Kết quả sẽ xuất hiện sau khi chạy detection.")

can_run_detection = image is not None and config is not None

run_detection = st.button(
    "Run Detection",
    type="primary",
    use_container_width=True,
    disabled=not can_run_detection,
)

if image is None:
    st.info("Chọn ảnh để bật nút Run Detection.")
    st.stop()

if not config:
    st.error("Config model chưa hợp lệ. Kiểm tra lại class_names.json và inference_config.json.")
    st.stop()

if not run_detection:
    st.stop()

try:
    with st.spinner("Running detection..."):
        detector = get_detector(str(model_dir))
        result = detector.predict(
            image=image,
            conf=confidence_threshold,
            iou=iou_threshold,
            imgsz=int(config["image_size"]),
            device=device,
        )
except ModelNotFoundError as exc:
    st.error(str(exc))
    st.stop()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()
except Exception as exc:
    st.error(f"Không thể chạy detection. Chi tiết: {exc}")
    st.stop()

annotated_image = result["annotated_image"]
detections = result["detections"]
output_path = save_result_image(annotated_image, OUTPUT_DIR)
saved_display_path = Path("outputs") / output_path.name

if device == "cuda" and result.get("device") == "cpu":
    st.warning("CUDA không khả dụng trên máy này. App đã fallback sang CPU.")

summary_cols = st.columns(3)
summary_cols[0].metric("Detected objects", len(detections))
summary_cols[1].metric("Confidence", f"{confidence_threshold:.2f}")
summary_cols[2].metric("IoU", f"{iou_threshold:.2f}")

with result_placeholder.container():
    st.markdown('<div class="section-label">Detection result</div>', unsafe_allow_html=True)
    st.image(annotated_image, use_container_width=True)

st.markdown('<div class="section-label">Detection table</div>', unsafe_allow_html=True)
if detections:
    render_detection_table(detections)
else:
    st.info("Không phát hiện object nào với confidence threshold hiện tại.")

download_col, path_col = st.columns([0.35, 0.65], gap="large")
with download_col:
    st.download_button(
        "Download result image",
        data=encode_image_for_download(annotated_image, image_format="PNG"),
        file_name=output_path.name,
        mime="image/png",
        use_container_width=True,
    )

with path_col:
    st.markdown(f'<p class="muted-line">Saved result image: {saved_display_path}</p>', unsafe_allow_html=True)
