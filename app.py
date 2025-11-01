# Main Streamlit application. Save as app.py

import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os
import time
import zipfile
import shutil
import yaml
from pathlib import Path
from PIL import Image
from io import BytesIO
from typing import Tuple

st.set_page_config(page_title="Object Detection Studio", layout="wide")

# -------------------- Helpers --------------------
MODEL_CACHE = {}

@st.cache_resource
def load_yolo(model_name: str = "yolov8n.pt"):
    """Load (and cache) a YOLO model from ultralytics."""
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]
    model = YOLO(model_name)
    MODEL_CACHE[model_name] = model
    return model


def draw_boxes(frame: np.ndarray, boxes, scores, class_ids, class_names) -> np.ndarray:
    """Draw bounding boxes and labels onto the image (in-place copy)."""
    img = frame.copy()
    h, w = img.shape[:2]
    for (x1, y1, x2, y2), conf, cid in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        label = f"{class_names[int(cid)]} {conf*100:.1f}%"
        # box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


def parse_results(results, model):
    # results: ultralytics results list
    r = results[0]
    boxes = []
    scores = []
    class_ids = []
    if r.boxes is None or len(r.boxes) == 0:
        return boxes, scores, class_ids
    for box in r.boxes:
        xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else np.array(box.xyxy[0])
        conf = float(box.conf[0]) if hasattr(box.conf[0], 'item') else float(box.conf)
        cls = int(box.cls[0]) if hasattr(box.cls[0], 'item') else int(box.cls)
        boxes.append(xyxy)
        scores.append(conf)
        class_ids.append(cls)
    return boxes, scores, class_ids


# -------------------- UI Layout --------------------
st.title("📷 Object Detection Studio — Starter")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Live / Camera")
    cam_col = st.empty()
    start_cam = st.button("Start Camera")
    stop_cam = st.button("Stop Camera")
    confidence = st.slider("Confidence threshold", 0.05, 0.99, 0.25)
    model_choice = st.selectbox("Model checkpoint", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], index=0)

with col2:
    st.header("Upload / Files")
    img_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"] )
    vid_file = st.file_uploader("Upload a video (mp4, mov)", type=["mp4", "mov", "avi" ] )
    process_image_btn = st.button("Analyze Image")
    process_video_btn = st.button("Analyze Video")

st.markdown("---")

# Training section
st.sidebar.header("Training & Custom Data")
st.sidebar.write("Upload a .zip containing images/ and labels/ in YOLO format (or a prepared dataset)")
train_zip = st.sidebar.file_uploader("Training dataset (.zip)")
train_epochs = st.sidebar.number_input("Epochs", min_value=1, max_value=1000, value=20)
train_btn = st.sidebar.button("Start Training")

# Model load
with st.spinner("Loading model..."):
    model = load_yolo(model_choice)

# Show model classes
st.sidebar.write("Model classes:")
st.sidebar.write(model.names)


# -------------------- Camera Loop (simple) --------------------
if start_cam:
    st.session_state['camera_running'] = True
if 'camera_running' not in st.session_state:
    st.session_state['camera_running'] = False
if stop_cam:
    st.session_state['camera_running'] = False

if st.session_state['camera_running']:
    cam_placeholder = cam_col.empty()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to open webcam. If you're running this in a remote environment, webcam access may be blocked.")
    else:
        fps_time = time.time()
        while st.session_state['camera_running']:
            ret, frame = cap.read()
            if not ret:
                st.warning("Empty frame")
                break
            # Convert BGR->RGB for ultralytics if necessary
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(source=rgb, conf=confidence, imgsz=640, verbose=False)
            boxes, scores, class_ids = parse_results(results, model)
            out = draw_boxes(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), boxes, scores, class_ids, model.names)
            cam_placeholder.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_column_width=True)
            # small sleep to avoid 100% CPU
            if time.time() - fps_time < 0.02:
                time.sleep(0.02)
            fps_time = time.time()
        cap.release()


# -------------------- Image Processing --------------------
if img_file and process_image_btn:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with st.spinner("Analyzing image..."):
        results = model.predict(source=rgb, conf=confidence, imgsz=640, verbose=False)
        boxes, scores, class_ids = parse_results(results, model)
        out = draw_boxes(img, boxes, scores, class_ids, model.names)
    st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="Detection result", use_column_width=True)
    if len(boxes) == 0:
        st.info("No objects detected above the confidence threshold.")
    else:
        df_rows = []
        for cls, conf in zip(class_ids, scores):
            df_rows.append({"label": model.names[int(cls)], "confidence": float(conf)})
        st.table(df_rows)


# -------------------- Video Processing --------------------
def process_and_export_video(input_path: str, output_path: str, model, conf_thresh: float = 0.25):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(source=rgb, conf=conf_thresh, imgsz=640, verbose=False)
        boxes, scores, class_ids = parse_results(results, model)
        out = draw_boxes(frame, boxes, scores, class_ids, model.names)
        writer.write(out)
        frame_idx += 1
    cap.release()
    writer.release()

if vid_file and process_video_btn:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(vid_file.name)[1])
    tfile.write(vid_file.getbuffer())
    tfile.flush()
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    with st.spinner("Processing video (frame-by-frame). This can take time..."):
        process_and_export_video(tfile.name, out_path, model, conf_thresh=confidence)
    st.video(out_path)
    with open(out_path, "rb") as f:
        st.download_button("Download processed video (mp4)", data=f, file_name=f"processed_{Path(vid_file.name).stem}.mp4")
    # cleanup
    try:
        os.remove(tfile.name)
    except Exception:
        pass


# -------------------- Training Flow (simple wrapper) --------------------
if train_btn and train_zip is not None:
    with st.spinner("Preparing dataset and starting training..."):
        tmpdir = tempfile.mkdtemp()
        zip_path = os.path.join(tmpdir, "data.zip")
        with open(zip_path, "wb") as f:
            f.write(train_zip.getbuffer())
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(tmpdir)
        # Expect dataset root contains images/ and labels/ OR a VOC-like structure. Try to find images/ & labels/
        images_dir = None
        labels_dir = None
        for root, dirs, files in os.walk(tmpdir):
            if 'images' in dirs:
                images_dir = os.path.join(root, 'images')
            if 'labels' in dirs:
                labels_dir = os.path.join(root, 'labels')
        if images_dir is None or labels_dir is None:
            st.error("Couldn't find images/ and labels/ folders inside the zip. Make sure your zip contains 'images' and 'labels' folders in YOLO format.")
        else:
            # Create data.yaml
            classes = set()
            # read label files to discover class indices and count
            for lb in Path(labels_dir).glob('*.txt'):
                try:
                    with open(lb, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 1:
                                classes.add(int(parts[0]))
                except Exception:
                    pass
            num_classes = max(classes) + 1 if len(classes) > 0 else 0
            data_yaml = {
                'train': images_dir,
                'val': images_dir,
                'nc': num_classes,
                'names': [str(i) for i in range(num_classes)]
            }
            data_yaml_path = os.path.join(tmpdir, 'data.yaml')
            with open(data_yaml_path, 'w') as f:
                yaml.dump(data_yaml, f)
            st.success(f"Prepared dataset with {num_classes} classes. Starting training for {train_epochs} epochs.")
            # Kick off YOLOv8 training (this runs in-process; for heavy workloads consider using a background worker or remote server)
            try:
                train_model = YOLO('yolov8n.pt')
                train_model.train(data=data_yaml_path, epochs=int(train_epochs), imgsz=640)
                st.success("Training finished. New weights are saved in runs/train/ folder by ultralytics.")
            except Exception as e:
                st.error(f"Training failed: {e}")
        # cleanup tempdir
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


# -------------------- Footer / Tips --------------------
st.markdown("---")
st.write("Tips:")
st.write("- For faster inference use a GPU. Configure Torch and CUDA before running the app.\n- To use this app on mobile, deploy to a server and open the URL in your phone browser or wrap as a PWA. See README section in this file.")


# EOF
