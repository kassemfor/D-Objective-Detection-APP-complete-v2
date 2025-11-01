# 🎯 Object Detection Studio

A complete Streamlit-based web application for real-time object detection using YOLOv8. This app supports live camera detection, image/video processing, and custom model training.

## ✨ Features

- **Live Camera Detection**: Real-time object detection using your webcam
- **Image Processing**: Upload and analyze images with bounding boxes and confidence scores
- **Video Processing**: Process video files frame-by-frame with detection results
- **Custom Training**: Train your own YOLOv8 models using custom datasets
- **Multiple Model Support**: Choose from YOLOv8n, YOLOv8s, or YOLOv8m models
- **Adjustable Confidence Threshold**: Fine-tune detection sensitivity

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- Webcam (for live detection feature)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/kassemfor/D-Objective-Detection-APP-complete-v2.git
   cd D-Objective-Detection-APP-complete-v2
   ```

2. **Create and activate a Python virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

   The app will automatically open in your browser at `http://localhost:8501`

### Model Download

The YOLOv8 model checkpoint (`yolov8n.pt`) will be automatically downloaded by the ultralytics package on first use. No manual download required!

## 📖 Usage

### Live Camera Detection

1. Click **"Start Camera"** to begin real-time detection
2. Adjust the confidence threshold slider to filter detections
3. Click **"Stop Camera"** when finished

### Image Processing

1. Upload an image (PNG, JPG, JPEG)
2. Click **"Analyze Image"**
3. View detected objects with bounding boxes and confidence scores
4. A table showing all detected objects will be displayed

### Video Processing

1. Upload a video file (MP4, MOV, AVI)
2. Click **"Analyze Video"**
3. Wait for processing to complete (this may take time for longer videos)
4. Download the processed video with detection annotations

### Custom Model Training

1. Prepare your dataset in YOLO format:
   - `images/` folder containing JPG/PNG images
   - `labels/` folder containing corresponding .txt label files
   
2. Label format (each line in .txt file):
   ```
   <class_id> <x_center> <y_center> <width> <height>
   ```
   All values should be normalized to 0-1 range.

3. Compress both folders into a single .zip file

4. In the app sidebar:
   - Upload your dataset .zip file
   - Set the number of training epochs
   - Click **"Start Training"**

5. Trained model weights will be saved in the `runs/train/` directory

## 📁 Training Data Format

The application expects YOLOv5/YOLOv8 style datasets:

- **Images**: JPG or PNG files in an `images/` folder
- **Labels**: Text files with the same base name in a `labels/` folder
- **Label Format**: Each line represents one bounding box:
  ```
  <class> <x_center> <y_center> <width> <height>
  ```
  All coordinates are normalized (0-1).

- **data.yaml**: The UI will automatically create a simple `data.yaml` file, or you can provide your own when possible.

## 📱 Mobile Deployment

### Quick Option (Web-based)
Deploy this Streamlit app to a public server (Streamlit Community Cloud, Heroku, AWS, etc.) and access it through your phone's browser or wrap it in a WebView.

### Native App (Advanced)
For native mobile apps, consider:
- **BeeWare/Toga**: Convert Python UI to native apps (requires UI code rework)
- **Kivy**: Cross-platform Python framework for mobile development
- **React Native / Flutter**: Reimplement frontend and call server-side model API

## 💡 Tips & Best Practices

- **GPU Acceleration**: For faster inference, use a GPU-enabled PyTorch installation with CUDA support
- **Confidence Threshold**: Start with 0.25 and adjust based on your use case:
  - Lower values (0.1-0.2): More detections but higher false positive rate
  - Higher values (0.5-0.8): Fewer but more confident detections
- **Model Selection**: 
  - `yolov8n.pt`: Fastest, smallest (best for real-time)
  - `yolov8s.pt`: Balanced speed/accuracy
  - `yolov8m.pt`: More accurate, slower
- **Video Processing**: Large videos take significant time. Consider processing shorter clips or using GPU acceleration.

## 🛠️ Requirements

See `requirements.txt` for complete dependency list. Main dependencies:
- `ultralytics>=8.0.20` - YOLOv8 implementation
- `streamlit` - Web framework
- `opencv-python` - Image/video processing
- `torch>=1.13.0` - Deep learning framework
- `numpy`, `pillow` - Image handling

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/kassemfor/D-Objective-Detection-APP-complete-v2/issues).

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ using YOLOv8 and Streamlit**

