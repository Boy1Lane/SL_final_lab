from __future__ import annotations

import hashlib
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw

from src.config_loader import ConfigError, load_inference_config
from src.detector import ModelNotFoundError, YOLODetector
from src.image_utils import (
    encode_image_for_download,
    read_data_url_image,
    read_image_bytes,
    save_result_image,
)


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_ROOT = PROJECT_ROOT / "app_models"
DEFAULT_MODEL_FOLDER = "YOLOv8s full split"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
PASTE_COMPONENT_DIR = PROJECT_ROOT / "src" / "components" / "paste_image"

paste_image_component = components.declare_component("paste_image", path=str(PASTE_COMPONENT_DIR))


st.set_page_config(
    page_title="Dashboard nhận diện YOLOv8",
    page_icon="CV",
    layout="wide",
)


def apply_page_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;520;650;760;820&display=swap');

        :root {
            --app-bg: #06111f;
            --panel: rgba(15, 23, 42, 0.78);
            --panel-strong: rgba(15, 23, 42, 0.94);
            --line: rgba(148, 163, 184, 0.18);
            --muted: #94a3b8;
            --text: #e5edf7;
            --accent: #38bdf8;
            --accent-strong: #0ea5e9;
            --cyan: #67e8f9;
            --green: #22c55e;
            --yellow: #facc15;
            --red: #ef4444;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(14, 165, 233, 0.18), transparent 36rem),
                linear-gradient(135deg, #06111f 0%, #0f172a 52%, #082f49 100%);
            color: var(--text);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .main .block-container {
            max-width: 1440px;
            padding: 1.5rem 2rem 2.5rem;
        }

        [data-testid="stSidebar"] {
            background: rgba(2, 6, 23, 0.88);
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            letter-spacing: 0;
        }

        [data-testid="stSidebar"] .stSelectbox,
        [data-testid="stSidebar"] .stSlider,
        [data-testid="stSidebar"] .stCheckbox {
            padding-bottom: 0.35rem;
        }

        .app-hero {
            align-items: flex-end;
            border-bottom: 1px solid var(--line);
            display: flex;
            gap: 1rem;
            justify-content: space-between;
            margin-bottom: 1.35rem;
            padding-bottom: 1rem;
        }

        .app-title {
            color: #f8fafc;
            font-size: clamp(1.65rem, 3vw, 2.45rem);
            font-weight: 820;
            letter-spacing: 0;
            line-height: 1.05;
            margin: 0;
        }

        .app-subtitle {
            color: var(--muted);
            font-size: 0.98rem;
            line-height: 1.55;
            margin: 0.45rem 0 0;
            max-width: 58rem;
        }

        .model-pill {
            align-items: center;
            background: rgba(14, 165, 233, 0.12);
            border: 1px solid rgba(125, 211, 252, 0.24);
            border-radius: 8px;
            color: #bae6fd;
            display: inline-flex;
            font-size: 0.82rem;
            font-weight: 650;
            min-height: 2.2rem;
            padding: 0.35rem 0.7rem;
            white-space: nowrap;
        }

        .section-label {
            color: #f8fafc;
            font-size: 1rem;
            font-weight: 760;
            letter-spacing: 0;
            margin: 0.2rem 0 0.7rem;
        }

        .panel {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 18px 60px rgba(0, 0, 0, 0.22);
            padding: 1rem;
        }

        .empty-panel {
            align-items: center;
            border: 1px dashed rgba(125, 211, 252, 0.36);
            border-radius: 8px;
            color: var(--muted);
            display: flex;
            justify-content: center;
            min-height: 25rem;
            padding: 1.2rem;
            text-align: center;
        }

        .empty-panel strong {
            color: #e0f2fe;
            display: block;
            font-size: 1rem;
            margin-bottom: 0.35rem;
        }

        .metric-card {
            background: var(--panel-strong);
            border: 1px solid var(--line);
            border-radius: 8px;
            box-shadow: 0 14px 40px rgba(0, 0, 0, 0.18);
            min-height: 6.6rem;
            padding: 0.95rem;
        }

        .metric-head {
            align-items: center;
            color: var(--muted);
            display: flex;
            font-size: 0.78rem;
            font-weight: 650;
            gap: 0.5rem;
            text-transform: uppercase;
        }

        .metric-icon {
            align-items: center;
            background: rgba(56, 189, 248, 0.13);
            border: 1px solid rgba(125, 211, 252, 0.24);
            border-radius: 8px;
            color: var(--cyan);
            display: inline-flex;
            font-size: 0.7rem;
            font-weight: 820;
            height: 1.7rem;
            justify-content: center;
            width: 1.9rem;
        }

        .metric-value {
            color: #f8fafc;
            font-size: 1.75rem;
            font-weight: 820;
            letter-spacing: 0;
            line-height: 1.1;
            margin-top: 0.7rem;
        }

        .metric-note {
            color: var(--muted);
            font-size: 0.78rem;
            margin-top: 0.35rem;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 8px;
            font-weight: 760;
            min-height: 2.75rem;
        }

        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0284c7 0%, #06b6d4 100%);
            border: 1px solid rgba(125, 211, 252, 0.42);
            box-shadow: 0 12px 32px rgba(14, 165, 233, 0.24);
        }

        .stFileUploader {
            background: rgba(15, 23, 42, 0.54);
            border: 1px dashed rgba(125, 211, 252, 0.34);
            border-radius: 8px;
            padding: 0.45rem 0.75rem;
        }

        div[data-testid="stAlert"] {
            border-radius: 8px;
        }

        @media (max-width: 960px) {
            .app-hero {
                align-items: flex-start;
                flex-direction: column;
            }

        }

        @media (max-width: 640px) {
            .main .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

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


def format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def hash_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_file_input(uploaded_file) -> dict[str, Any] | None:
    if uploaded_file is None:
        return None
    raw = uploaded_file.getvalue()
    return {
        "image": read_image_bytes(BytesIO(raw)),
        "id": hash_bytes(raw),
        "name": uploaded_file.name,
        "size": len(raw),
        "source": "Tải ảnh lên",
    }


def load_pasted_input(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict) or not payload.get("dataUrl"):
        return None
    data_url = str(payload["dataUrl"])
    encoded = data_url.split(",", 1)[-1].encode("utf-8")
    return {
        "image": read_data_url_image(data_url),
        "id": hash_bytes(encoded),
        "name": payload.get("name") or "clipboard-image.png",
        "size": int(len(encoded) * 0.75),
        "source": "Dán từ clipboard",
    }


def render_metric_card(icon: str, label: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-head"><span class="metric-icon">{escape(icon)}</span>{escape(label)}</div>
            <div class="metric-value">{escape(value)}</div>
            <div class="metric-note">{escape(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def confidence_class(value: float) -> str:
    if value > 0.70:
        return "conf-high"
    if value >= 0.40:
        return "conf-mid"
    return "conf-low"


def render_detection_table(detections: list[dict]) -> None:
    if not detections:
        st.info("Không phát hiện đối tượng nào với ngưỡng tin cậy hiện tại.")
        return

    rows = []
    for index, item in enumerate(detections, start=1):
        xmin, ymin, xmax, ymax = item["bbox"]
        confidence = float(item["confidence"])
        rows.append(
            {
                "No.": index,
                "Class": item["class_name"],
                "Confidence": confidence * 100,
                "Xmin": xmin,
                "Ymin": ymin,
                "Xmax": xmax,
                "Ymax": ymax,
            }
        )

    table = pd.DataFrame(rows)
    styled_table = table.style.format({"Confidence": "{:.1f}%"}).map(
        style_confidence_cell,
        subset=["Confidence"],
    )
    st.dataframe(
        styled_table,
        use_container_width=True,
        hide_index=True,
    )


def style_confidence_cell(value: float) -> str:
    if value > 70:
        return "background-color: rgba(34, 197, 94, 0.22); color: #86efac; font-weight: 700;"
    if value >= 40:
        return "background-color: rgba(250, 204, 21, 0.22); color: #fde68a; font-weight: 700;"
    return "background-color: rgba(239, 68, 68, 0.22); color: #fca5a5; font-weight: 700;"


def make_compare_image(original: Image.Image, annotated: Any, reveal_percent: int) -> Image.Image:
    before = original.convert("RGB")
    after = Image.fromarray(np.asarray(annotated)).convert("RGB").resize(before.size)
    split_x = int(before.width * reveal_percent / 100)
    combined = before.copy()
    combined.paste(after.crop((0, 0, split_x, before.height)), (0, 0))
    draw = ImageDraw.Draw(combined)
    draw.line((split_x, 0, split_x, before.height), fill=(103, 232, 249), width=max(3, before.width // 240))
    return combined


def render_detection_statistics(detections: list[dict]) -> None:
    if not detections:
        st.info("Thống kê sẽ hiển thị khi có ít nhất một đối tượng được phát hiện.")
        return

    chart_col, dist_col = st.columns(2, gap="large")
    with chart_col:
        st.markdown('<div class="section-label">Biểu đồ số lượng đối tượng</div>', unsafe_allow_html=True)
        counts = pd.Series([item["class_name"] for item in detections]).value_counts()
        st.bar_chart(counts)
    with dist_col:
        st.markdown('<div class="section-label">Phân bố độ tin cậy</div>', unsafe_allow_html=True)
        bins = pd.cut(
            [float(item["confidence"]) for item in detections],
            bins=[0.0, 0.4, 0.7, 1.0],
            labels=["< 40%", "40-70%", "> 70%"],
            include_lowest=True,
        )
        st.bar_chart(pd.Series(bins).value_counts().sort_index())


def run_inference(
    image: Image.Image,
    model_dir: Path,
    config: dict,
    confidence_threshold: float,
    iou_threshold: float,
    device: str,
    save_output: bool,
) -> dict[str, Any]:
    detector = get_detector(str(model_dir))
    result = detector.predict(
        image=image,
        conf=confidence_threshold,
        iou=iou_threshold,
        imgsz=int(config["image_size"]),
        device=device,
    )
    if save_output:
        result["output_path"] = save_result_image(result["annotated_image"], OUTPUT_DIR)
    return result


apply_page_styles()

st.markdown(
    """
    <div class="app-hero">
        <div>
            <h1 class="app-title">Dashboard nhận diện YOLOv8</h1>
            <p class="app-subtitle">Tải ảnh, dán ảnh từ clipboard hoặc chụp bằng camera, sau đó điều chỉnh ngưỡng nhận diện, xem kết quả và tải ảnh đã vẽ bounding box.</p>
        </div>
        <div class="model-pill">Object Detection Demo</div>
    </div>
    """,
    unsafe_allow_html=True,
)

model_folders = list_model_folders()
default_index = model_folders.index(DEFAULT_MODEL_FOLDER) if DEFAULT_MODEL_FOLDER in model_folders else 0

with st.sidebar:
    st.header("Model")
    model_folder = st.selectbox(
        "Thư mục model",
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
    default_conf = min(max(default_conf, 0.0), 1.0)
    default_iou = min(max(default_iou, 0.0), 1.0)
    default_device = str(config.get("device", "auto")) if config else "auto"
    if default_device not in {"auto", "cpu", "cuda"}:
        default_device = "auto"

    st.divider()
    st.header("Detection")
    confidence_threshold = st.slider(
        "Ngưỡng tin cậy (Confidence Threshold)",
        min_value=0.0,
        max_value=1.0,
        value=default_conf,
        step=0.01,
        help="Các prediction thấp hơn ngưỡng này sẽ bị loại bỏ.",
    )
    iou_threshold = st.slider(
        "Ngưỡng IoU (IoU Threshold)",
        min_value=0.0,
        max_value=1.0,
        value=default_iou,
        step=0.01,
        help="Điều khiển Non-Maximum Suppression để lọc các bounding box chồng lấn.",
    )
    device = st.selectbox(
        "Thiết bị chạy",
        ["auto", "cpu", "cuda"],
        index=["auto", "cpu", "cuda"].index(default_device),
        help="Auto sẽ dùng CUDA nếu có, nếu không sẽ dùng CPU.",
    )

    st.divider()
    st.header("Output")
    save_output = st.checkbox("Lưu kết quả vào outputs/", value=True)
    rerun_on_threshold_change = st.checkbox("Tự chạy lại khi đổi threshold", value=False)
    st.caption("Khi bật tự chạy lại, app sẽ không tự lưu thêm file cho mỗi lần kéo slider.")

input_col, details_col = st.columns([1.35, 0.85], gap="large")
with input_col:
    st.markdown('<div class="section-label">Ảnh đầu vào</div>', unsafe_allow_html=True)
    source_labels = {
        "upload": "Tải ảnh lên",
        "paste": "Dán ảnh",
        "camera": "Camera",
    }
    source_mode = st.radio(
        "Nguồn ảnh",
        ["upload", "paste", "camera"],
        format_func=lambda value: source_labels[value],
        horizontal=True,
        label_visibility="collapsed",
    )
    uploaded_file = None
    pasted_payload = None
    camera_file = None
    if source_mode == "upload":
        uploaded_file = st.file_uploader(
            "Tải ảnh từ máy tính",
            type=ALLOWED_IMAGE_TYPES,
            accept_multiple_files=False,
            help="Định dạng hỗ trợ: JPG, JPEG, PNG.",
        )
    elif source_mode == "paste":
        st.caption("Click vào vùng dán bên dưới, sau đó nhấn Ctrl+V để dán screenshot hoặc ảnh đã copy.")
        pasted_payload = paste_image_component(default=None, key="paste-image")
    else:
        st.caption("Trình duyệt có thể yêu cầu quyền truy cập camera trước khi chụp.")
        camera_file = st.camera_input("Chụp ảnh từ camera")

with details_col:
    st.markdown('<div class="section-label">Điều khiển chạy</div>', unsafe_allow_html=True)
    st.caption("Cung cấp ảnh hợp lệ rồi bấm Run Detection để chạy YOLOv8.")

selected_input = None
try:
    if source_mode == "upload":
        selected_input = load_file_input(uploaded_file)
    elif source_mode == "paste":
        selected_input = load_pasted_input(pasted_payload)
    else:
        selected_input = load_file_input(camera_file)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

image = selected_input["image"] if selected_input else None
image_id = selected_input["id"] if selected_input else None
image_size = image.size if image else None
can_run_detection = image is not None and config is not None

with details_col:
    if selected_input:
        with st.container(border=True):
            st.caption("Nguồn")
            st.write(selected_input["source"])
            st.caption("Tên file")
            st.write(str(selected_input["name"]))
            st.caption("Dung lượng")
            st.write(format_file_size(int(selected_input["size"])))
    else:
        st.info("Chưa có ảnh đầu vào. Hãy chọn Tải ảnh lên, Dán ảnh hoặc Camera.")

    run_detection = st.button("Run Detection", type="primary", use_container_width=True, disabled=not can_run_detection)
    clear_result = st.button(
        "Xóa kết quả",
        use_container_width=True,
        disabled=not st.session_state.get("has_detection", False),
    )
    if not can_run_detection:
        st.caption("Nút Run Detection sẽ bật khi app có ảnh hợp lệ và config model hợp lệ.")

if not config:
    st.error("Config model chưa hợp lệ. Kiểm tra class_names.json và inference_config.json.")
    st.stop()

if clear_result:
    for key in ("last_result", "last_image_id", "last_signature", "has_detection"):
        st.session_state.pop(key, None)
    st.rerun()

settings_signature = (
    image_id,
    model_folder,
    round(confidence_threshold, 4),
    round(iou_threshold, 4),
    device,
)
previous_signature = st.session_state.get("last_signature")
same_image_as_previous = image_id is not None and image_id == st.session_state.get("last_image_id")
threshold_changed_after_detection = (
    rerun_on_threshold_change
    and same_image_as_previous
    and st.session_state.get("has_detection", False)
    and settings_signature != previous_signature
)
should_run_detection = can_run_detection and (run_detection or threshold_changed_after_detection)

if should_run_detection and image is not None:
    is_auto_rerun = threshold_changed_after_detection and not run_detection
    should_save_this_run = save_output and not is_auto_rerun
    try:
        with st.spinner("Đang chạy YOLOv8 detection..."):
            result = run_inference(
                image=image,
                model_dir=model_dir,
                config=config,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
                device=device,
                save_output=should_save_this_run,
            )
    except ModelNotFoundError as exc:
        st.error(str(exc))
        st.stop()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Không thể chạy detection: {exc}")
        st.stop()

    st.session_state["last_result"] = result
    st.session_state["last_image_id"] = image_id
    st.session_state["last_signature"] = settings_signature
    st.session_state["has_detection"] = True
    same_image_as_previous = True
    if is_auto_rerun:
        st.success("Đã cập nhật kết quả theo threshold mới.")
    else:
        st.success("Detection completed successfully!")

result = st.session_state.get("last_result") if same_image_as_previous else None
detections = result.get("detections", []) if result else []
annotated_image = result.get("annotated_image") if result else None

metric_cols = st.columns(4, gap="medium")
with metric_cols[0]:
    render_metric_card("DET", "Đối tượng phát hiện", str(len(detections)), "Sau khi lọc threshold")
with metric_cols[1]:
    render_metric_card("CON", "Confidence Threshold", f"{confidence_threshold:.2f}", format_percent(confidence_threshold))
with metric_cols[2]:
    render_metric_card("IOU", "IoU Threshold", f"{iou_threshold:.2f}", "Lọc box chồng lấn")
with metric_cols[3]:
    inference_note = f"{float(result.get('inference_time_ms', 0)):.0f} ms" if result else "Chưa chạy"
    render_metric_card("TIM", "Thời gian xử lý", inference_note, str(result.get("device", "Đang chờ")).upper() if result else "Đang chờ")

result_col, original_col = st.columns([1.55, 0.95], gap="large")
with result_col:
    st.markdown('<div class="section-label">Kết quả nhận diện</div>', unsafe_allow_html=True)
    if annotated_image is not None:
        st.image(annotated_image, use_container_width=True)
    else:
        st.markdown(
            """
            <div class="empty-panel">
                <div>
                    <strong>Tải ảnh, dán ảnh hoặc chụp ảnh để bắt đầu.</strong>
                    Định dạng hỗ trợ: JPG, JPEG, PNG.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with original_col:
    st.markdown('<div class="section-label">Ảnh gốc</div>', unsafe_allow_html=True)
    if image is not None:
        st.image(image, use_container_width=True)
    else:
        st.markdown(
            """
            <div class="empty-panel">
                <div><strong>Chưa chọn ảnh</strong>Chọn một nguồn ảnh ở phía trên để xem preview.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

if device == "cuda" and result and result.get("device") == "cpu":
    st.warning("Máy hiện không có CUDA khả dụng. App đã tự chuyển sang CPU.")

if result and image_size is not None:
    output_path = result.get("output_path")
    download_col, saved_col = st.columns([0.35, 0.65], gap="large")
    with download_col:
        st.download_button(
            "Tải ảnh PNG",
            data=encode_image_for_download(annotated_image, image_format="PNG"),
            file_name=Path(output_path).name if output_path else "detection_result.png",
            mime="image/png",
            use_container_width=True,
        )
    with saved_col:
        if output_path:
            saved_display_path = Path("outputs") / Path(output_path).name
            st.caption(f"Đã lưu ảnh kết quả: {saved_display_path}")
        else:
            st.caption("Lượt chạy này không tự lưu file output.")

    st.markdown('<div class="section-label">So sánh trước / sau</div>', unsafe_allow_html=True)
    reveal_percent = st.slider("Tỉ lệ hiển thị ảnh đã detect", 0, 100, 50)
    st.image(make_compare_image(image, annotated_image, reveal_percent), use_container_width=True)

st.markdown('<div class="section-label">Bảng kết quả detection</div>', unsafe_allow_html=True)
render_detection_table(detections)

st.markdown('<div class="section-label">Thống kê detection</div>', unsafe_allow_html=True)
render_detection_statistics(detections)
