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

st.set_page_config(page_title="Object Detection Studio", layout="wide")

# -------------------- Helpers --------------------
@st.cache_resource
def load_yolo(model_name: str = "yolov8n.pt"):
    """Load (and cache) a YOLO model from ultralytics.
    
    Args:
        model_name: Path to model checkpoint file
        
    Returns:
        YOLO model instance
    """
    return YOLO(model_name)


def draw_boxes(frame: np.ndarray, boxes, scores, class_ids, class_names) -> np.ndarray:
    """Draw bounding boxes and labels onto the image.
    
    Args:
        frame: Input image array (BGR format)
        boxes: List of bounding boxes in xyxy format
        scores: List of confidence scores
        class_ids: List of class IDs
        class_names: Dictionary mapping class IDs to names
        
    Returns:
        Image with drawn bounding boxes
    """
    img = frame.copy()
    for (x1, y1, x2, y2), conf, cid in zip(boxes, scores, class_ids):
        try:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            # Ensure coordinates are within image bounds
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue  # Skip invalid boxes
            
            label = f"{class_names[int(cid)]} {conf*100:.1f}%"
            # box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # label background - draw below box if too close to top
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = y1 - th - 6 if y1 > th + 6 else y2 + th + 6
            cv2.rectangle(img, (x1, label_y - th - 2), (x1 + tw + 4, label_y + 2), (0, 255, 0), -1)
            cv2.putText(img, label, (x1 + 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        except (ValueError, IndexError, KeyError):
            # Skip boxes that can't be drawn
            continue
    return img


def parse_results(results, model):
    """Parse ultralytics results into boxes, scores, and class IDs.
    
    Args:
        results: Ultralytics Results object or list
        model: YOLO model (unused but kept for compatibility)
        
    Returns:
        Tuple of (boxes, scores, class_ids) lists
    """
    if not results or len(results) == 0:
        return [], [], []
    r = results[0]
    boxes = []
    scores = []
    class_ids = []
    if r.boxes is None or len(r.boxes) == 0:
        return boxes, scores, class_ids
    for box in r.boxes:
        try:
            # Handle tensor conversion more robustly
            if hasattr(box.xyxy[0], 'cpu'):
                xyxy = box.xyxy[0].cpu().numpy()
            else:
                xyxy = np.array(box.xyxy[0])
            
            if hasattr(box.conf[0], 'item'):
                conf = float(box.conf[0].item())
            elif hasattr(box.conf[0], 'cpu'):
                conf = float(box.conf[0].cpu().numpy())
            else:
                conf = float(box.conf[0])
            
            if hasattr(box.cls[0], 'item'):
                cls = int(box.cls[0].item())
            elif hasattr(box.cls[0], 'cpu'):
                cls = int(box.cls[0].cpu().numpy())
            else:
                cls = int(box.cls[0])
            
            boxes.append(xyxy)
            scores.append(conf)
            class_ids.append(cls)
        except Exception as e:
            # Skip boxes that fail to parse
            continue
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
# Initialize camera state
if 'camera_running' not in st.session_state:
    st.session_state['camera_running'] = False
if 'camera' not in st.session_state:
    st.session_state['camera'] = None
if 'last_frame_time' not in st.session_state:
    st.session_state['last_frame_time'] = 0

def release_camera():
    """Safely release camera resource."""
    if st.session_state['camera'] is not None:
        try:
            st.session_state['camera'].release()
        except Exception:
            pass  # Ignore errors during cleanup
        finally:
            st.session_state['camera'] = None

if start_cam:
    # Release any existing camera before starting new one
    release_camera()
    st.session_state['camera_running'] = True
    
if stop_cam:
    st.session_state['camera_running'] = False
    release_camera()

# Use Streamlit's camera input for better compatibility
if st.session_state['camera_running']:
    # Note: Streamlit's camera input is better for web deployment
    # For local webcam, we use cv2 but with limited frames
    try:
        # Reuse existing camera if available, otherwise create new one
        cap = st.session_state['camera']
        if cap is None:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Unable to open webcam. If you're running this in a remote environment, webcam access may be blocked.")
                st.session_state['camera_running'] = False
                release_camera()
            else:
                st.session_state['camera'] = cap
        
        # Process frame only if camera is valid
        if cap is not None and cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model.predict(source=rgb, conf=confidence, imgsz=640, verbose=False)
                boxes, scores, class_ids = parse_results(results, model)
                out = draw_boxes(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), boxes, scores, class_ids, model.names)
                cam_col.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_column_width=True)
                if len(boxes) > 0:
                    st.info(f"Detected {len(boxes)} object(s)")
            else:
                st.warning("Could not read frame from camera. Camera may be disconnected.")
                # Check if camera is still valid
                if not cap.isOpened():
                    st.session_state['camera_running'] = False
                    release_camera()
    except cv2.error as e:
        st.error(f"OpenCV camera error: {str(e)}. Please check if another application is using the camera.")
        st.session_state['camera_running'] = False
        release_camera()
    except Exception as e:
        st.error(f"Camera error: {str(e)}")
        st.session_state['camera_running'] = False
        release_camera()
else:
    # Clean up camera when stopped
    release_camera()


# -------------------- Image Processing --------------------
if img_file and process_image_btn:
    # Validate file size (max 50MB)
    max_size = 50 * 1024 * 1024  # 50MB
    img_file.seek(0, 2)  # Seek to end
    file_size = img_file.tell()
    img_file.seek(0)  # Reset to beginning
    
    if file_size > max_size:
        st.error(f"Image file is too large ({file_size / (1024*1024):.1f}MB). Maximum size is 50MB.")
    elif file_size == 0:
        st.error("Image file is empty. Please upload a valid image.")
    else:
        try:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                st.error("Failed to decode image. Please ensure the file is a valid image format (PNG, JPG, JPEG).")
            elif img.size == 0:
                st.error("Decoded image is empty. Please upload a valid image file.")
            else:
                # Check image dimensions (reasonable limits)
                h, w = img.shape[:2]
                if h > 10000 or w > 10000:
                    st.warning(f"Very large image ({w}x{h}). Processing may be slow.")
                
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                with st.spinner("Analyzing image..."):
                    try:
                        results = model.predict(source=rgb, conf=confidence, imgsz=640, verbose=False)
                        boxes, scores, class_ids = parse_results(results, model)
                        out = draw_boxes(img, boxes, scores, class_ids, model.names)
                        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), caption="Detection result", use_column_width=True)
                        if len(boxes) == 0:
                            st.info("No objects detected above the confidence threshold.")
                        else:
                            df_rows = []
                            for cls, conf in zip(class_ids, scores):
                                df_rows.append({"label": model.names[int(cls)], "confidence": f"{float(conf)*100:.1f}%"})
                            st.dataframe(df_rows, use_container_width=True)
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower() or "CUDA" in str(e):
                            st.error("GPU memory error. Try reducing image size or using CPU mode.")
                        else:
                            st.error(f"Model inference error: {str(e)}")
        except MemoryError:
            st.error("Out of memory error. Image is too large. Please use a smaller image.")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")


# -------------------- Video Processing --------------------
def process_and_export_video(input_path: str, output_path: str, model, conf_thresh: float = 0.25, progress_bar=None, status_text=None) -> bool:
    """Process a video file frame-by-frame with object detection.
    
    Args:
        input_path: Path to input video file
        output_path: Path to save processed video
        model: YOLO model instance
        conf_thresh: Confidence threshold for detection
        progress_bar: Optional Streamlit progress bar
        status_text: Optional Streamlit text container for status updates
        
    Returns:
        True if successful, False otherwise
    """
    cap = None
    writer = None
    try:
        if not os.path.exists(input_path):
            if status_text:
                status_text.error("Input video file not found.")
            return False
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            if status_text:
                status_text.error("Could not open video file. Please check the file format.")
            return False
        
        # Try different codecs for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if width == 0 or height == 0:
            if status_text:
                status_text.error("Invalid video dimensions. Could not read video properties.")
            cap.release()
            return False
        
        if total_frames == 0:
            if status_text:
                status_text.warning("Could not determine frame count. Processing may be interrupted.")
        
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            # Try alternative codec
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                if status_text:
                    status_text.error("Could not initialize video writer. Codec not supported.")
                cap.release()
                return False
        
        frame_idx = 0
        error_count = 0
        max_errors = 10  # Allow some frame read errors
        
        while True:
            ret, frame = cap.read()
            if not ret:
                if frame_idx == 0:
                    # No frames read at all
                    if status_text:
                        status_text.error("Could not read any frames from video.")
                    break
                else:
                    # End of video or read error
                    break
            
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model.predict(source=rgb, conf=conf_thresh, imgsz=640, verbose=False)
                boxes, scores, class_ids = parse_results(results, model)
                out = draw_boxes(frame, boxes, scores, class_ids, model.names)
                writer.write(out)
                frame_idx += 1
                error_count = 0  # Reset error count on success
                
                # Update progress if available
                if progress_bar and total_frames > 0:
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))
                
                if status_text and frame_idx % 10 == 0:  # Update every 10 frames
                    status_text.text(f"Processed {frame_idx}/{total_frames if total_frames > 0 else '?'} frames...")
                
            except Exception as frame_error:
                error_count += 1
                if error_count >= max_errors:
                    if status_text:
                        status_text.warning(f"Too many frame processing errors ({error_count}). Stopping.")
                    break
                continue  # Skip this frame and continue
        
        # Verify output file was created and has content
        if frame_idx == 0:
            if status_text:
                status_text.error("No frames were processed.")
            return False
        
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            if status_text:
                status_text.error("Output video file is empty or was not created.")
            return False
        
        return True
        
    except MemoryError:
        if status_text:
            status_text.error("Out of memory. Video is too large or system resources are exhausted.")
        return False
    except Exception as e:
        if status_text:
            status_text.error(f"Video processing error: {str(e)}")
        return False
    finally:
        # Ensure resources are always released
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass

if vid_file and process_video_btn:
    # Validate file size (max 500MB for videos)
    max_video_size = 500 * 1024 * 1024  # 500MB
    vid_file.seek(0, 2)  # Seek to end
    file_size = vid_file.tell()
    vid_file.seek(0)  # Reset to beginning
    
    if file_size > max_video_size:
        st.error(f"Video file is too large ({file_size / (1024*1024):.1f}MB). Maximum size is 500MB.")
    elif file_size == 0:
        st.error("Video file is empty. Please upload a valid video file.")
    else:
        tfile_path = None
        out_path = None
        try:
            # Save uploaded video to temp file using context manager
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(vid_file.name)[1]) as tfile:
                tfile_path = tfile.name
                vid_file.seek(0)  # Reset file pointer
                tfile.write(vid_file.getbuffer())
                tfile.flush()
                # File closed automatically by context manager
            
            # Create output path
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as ofile:
                out_path = ofile.name
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("Processing video (frame-by-frame). This can take time..."):
                status_text.text("Processing video...")
                success = process_and_export_video(tfile_path, out_path, model, conf_thresh=confidence, progress_bar=progress_bar, status_text=status_text)
            
            progress_bar.empty()
            status_text.empty()
            
            if success and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                st.success("Video processed successfully!")
                st.video(out_path)
                try:
                    with open(out_path, "rb") as f:
                        video_data = f.read()
                    st.download_button(
                        "Download processed video (mp4)", 
                        data=video_data, 
                        file_name=f"processed_{Path(vid_file.name).stem}.mp4", 
                        mime="video/mp4"
                    )
                except Exception as e:
                    st.warning(f"Could not prepare download: {str(e)}")
            else:
                st.error("Failed to process video. Please check the video file format and try again.")
        
        except MemoryError:
            st.error("Out of memory error. Video is too large. Please use a smaller video file.")
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
        finally:
            # Ensure cleanup of temp files
            cleanup_errors = []
            if tfile_path and os.path.exists(tfile_path):
                try:
                    os.remove(tfile_path)
                except Exception as e:
                    cleanup_errors.append(f"Input temp file: {str(e)}")
            
            if out_path and os.path.exists(out_path):
                # Only delete output if processing failed or user didn't download
                try:
                    # Keep output file for download, but clean up on next run if not downloaded
                    # For now, we'll keep it and let system temp cleanup handle it
                    pass
                except Exception as e:
                    cleanup_errors.append(f"Output temp file: {str(e)}")
            
            if cleanup_errors:
                st.warning("Some temporary files could not be cleaned up automatically.")


# -------------------- Training Flow (simple wrapper) --------------------
if train_btn and train_zip is not None:
    # Validate zip file size (max 1GB)
    max_zip_size = 1024 * 1024 * 1024  # 1GB
    train_zip.seek(0, 2)
    zip_size = train_zip.tell()
    train_zip.seek(0)
    
    if zip_size > max_zip_size:
        st.error(f"Training dataset is too large ({zip_size / (1024*1024*1024):.1f}GB). Maximum size is 1GB.")
    elif zip_size == 0:
        st.error("Training dataset file is empty. Please upload a valid zip file.")
    elif train_epochs < 1 or train_epochs > 1000:
        st.error("Epochs must be between 1 and 1000.")
    else:
        tmpdir = None
        try:
            with st.spinner("Preparing dataset and starting training..."):
                tmpdir = tempfile.mkdtemp()
                zip_path = os.path.join(tmpdir, "data.zip")
                
                # Write zip file
                with open(zip_path, "wb") as f:
                    f.write(train_zip.getbuffer())
                
                # Validate zip file
                if not zipfile.is_zipfile(zip_path):
                    st.error("Invalid zip file. Please ensure you uploaded a valid zip archive.")
                else:
                    # Extract zip
                    extract_success = True
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as z:
                            # Check for zip bomb
                            total_size = sum(info.file_size for info in z.infolist())
                            if total_size > max_zip_size * 2:  # Allow some expansion
                                st.error("Extracted dataset would be too large. Possible zip bomb detected.")
                                extract_success = False
                            else:
                                z.extractall(tmpdir)
                    except zipfile.BadZipFile:
                        st.error("Corrupted zip file. Please check your zip file and try again.")
                        extract_success = False
                        tmpdir = None  # Prevent cleanup attempt
                    except Exception as e:
                        st.error(f"Error extracting zip file: {str(e)}")
                        extract_success = False
                        tmpdir = None
                    
                    if extract_success and tmpdir:
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
                            # Validate directories have content
                            image_files = list(Path(images_dir).glob('*.jpg')) + list(Path(images_dir).glob('*.png'))
                            label_files = list(Path(labels_dir).glob('*.txt'))
                            
                            if len(image_files) == 0:
                                st.error("No image files found in images/ directory.")
                            elif len(label_files) == 0:
                                st.error("No label files found in labels/ directory.")
                            elif len(image_files) != len(label_files):
                                st.warning(f"Mismatch: {len(image_files)} images but {len(label_files)} labels. Some files may be missing pairs.")
                            
                            # Create data.yaml
                            classes = set()
                            label_errors = []
                            # read label files to discover class indices and count
                            for lb in label_files:
                                try:
                                    with open(lb, 'r') as f:
                                        for line_num, line in enumerate(f, 1):
                                            parts = line.strip().split()
                                            if len(parts) >= 1:
                                                try:
                                                    class_id = int(parts[0])
                                                    if class_id < 0:
                                                        label_errors.append(f"{lb.name}:{line_num} - negative class ID")
                                                    else:
                                                        classes.add(class_id)
                                                except ValueError:
                                                    label_errors.append(f"{lb.name}:{line_num} - invalid class ID: {parts[0]}")
                                except Exception as e:
                                    label_errors.append(f"{lb.name} - read error: {str(e)}")
                            
                            if label_errors and len(label_errors) > 10:
                                st.warning(f"Found {len(label_errors)} label file errors (showing first 10).")
                                for err in label_errors[:10]:
                                    st.text(err)
                            elif label_errors:
                                for err in label_errors:
                                    st.text(err)
                            
                            num_classes = max(classes) + 1 if len(classes) > 0 else 0
                            if num_classes == 0:
                                st.error("No valid classes found in label files. Please check your dataset format.")
                            else:
                                # Split dataset: use 80% for training, 20% for validation
                                # In a real scenario, you'd split the files, but for simplicity we use the same dir
                                # Better approach would be to create train/val splits
                                data_yaml = {
                                    'train': images_dir,
                                    'val': images_dir,  # Note: Ideally should be a separate validation set
                                    'nc': num_classes,
                                    'names': [str(i) for i in range(num_classes)]
                                }
                                data_yaml_path = os.path.join(tmpdir, 'data.yaml')
                                try:
                                    with open(data_yaml_path, 'w') as f:
                                        yaml.dump(data_yaml, f)
                                    
                                    st.success(f"Prepared dataset with {num_classes} classes and {len(image_files)} images. Starting training for {train_epochs} epochs.")
                                    
                                    # Kick off YOLOv8 training (this runs in-process; for heavy workloads consider using a background worker or remote server)
                                    try:
                                        train_model = YOLO('yolov8n.pt')
                                        train_model.train(data=data_yaml_path, epochs=int(train_epochs), imgsz=640)
                                        st.success("Training finished. New weights are saved in runs/train/ folder by ultralytics.")
                                    except RuntimeError as e:
                                        if "out of memory" in str(e).lower() or "CUDA" in str(e):
                                            st.error("GPU memory error during training. Try reducing batch size or using CPU mode.")
                                        else:
                                            st.error(f"Training runtime error: {str(e)}")
                                    except Exception as e:
                                        st.error(f"Training failed: {str(e)}")
                                except IOError as e:
                                    st.error(f"Could not create data.yaml file: {str(e)}")
            
        except MemoryError:
            st.error("Out of memory. Dataset is too large. Please use a smaller dataset.")
        except Exception as e:
            st.error(f"Error during training preparation: {str(e)}")
        finally:
            # cleanup tempdir
            if tmpdir and os.path.exists(tmpdir):
                try:
                    shutil.rmtree(tmpdir)
                except Exception as cleanup_err:
                    st.warning(f"Could not clean up temporary files: {str(cleanup_err)}")


# -------------------- Footer / Tips --------------------
st.markdown("---")
st.write("Tips:")
st.write("- For faster inference use a GPU. Configure Torch and CUDA before running the app.\n- To use this app on mobile, deploy to a server and open the URL in your phone browser or wrap as a PWA. See README section in this file.")


# EOF
