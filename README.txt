# README (quick)
# 1) Create and activate a Python virtualenv (recommended).
# 2) pip install -r requirements.txt
# 3) Download a YOLOv8 checkpoint (the app uses `yolov8n.pt` by default). The ultralytics package will auto-download when needed.
# 4) Run: streamlit run app.py
#
# Training data format:
# - Expect YOLOv5/YOLOv8 style: images/ (jpg/png) and labels/ (same basename .txt) where each txt line is: <class> <x_center> <y_center> <width> <height> (normalized 0-1)
# - Provide a `data.yaml` when possible (or the UI will create a simple one).
#
# Mobile:
# - Quick option: deploy this Streamlit app publicly (Streamlit Community Cloud or any VPS) and open in phone browser or wrap in WebView.
# - Native: consider BeeWare/Toga or Kivy to convert Python UI into native apps (requires rework of UI code). Another route: reimplement the frontend in React Native / Flutter and call a server-side model.
