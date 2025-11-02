# Object Detection Studio - Mobile App for iPhone

A Progressive Web App (PWA) that brings AI-powered object detection to your iPhone. No App Store required - install directly from Safari!

## 🚀 Features

- **AI Object Detection**: Uses TensorFlow.js with COCO-SSD model
- **Camera Integration**: Take photos directly from your iPhone camera
- **Photo Gallery**: Upload images from your photo library
- **Real-time Results**: See detected objects with confidence scores
- **Offline Support**: Works without internet after first load
- **Home Screen Installation**: Install like a native app
- **iOS Optimized**: Designed specifically for iPhone/iPad

## 📱 iPhone Installation Guide

### Method 1: Install to Home Screen (Recommended)

1. **Open Safari** (must use Safari, not Chrome)
2. **Navigate** to the app URL or host the files on a web server
3. **Tap the Share button** 📤 at the bottom of Safari
4. **Scroll down** and tap "Add to Home Screen"
5. **Customize the name** if desired
6. **Tap "Add"** in the top-right corner
7. **Find the app** on your home screen and tap to launch

### Method 2: Direct Installation

1. **Host the files** on a web server
2. **Open the URL** in Safari
3. **Tap "Add"** when the install prompt appears
4. **Or use the Share button → Add to Home Screen**

## 🛠️ Setup Instructions

### For Developers: Hosting the App

1. **Upload all files** to your web server or GitHub Pages
2. **Ensure HTTPS** is enabled (required for PWA features)
3. **Test the installation** using the guide above

### Required Files Structure

```
ObjectDetectionMobile/
├── index.html          # Main app file
├── style.css           # Mobile-optimized styles
├── app.js             # Main application logic
├── manifest.json      # PWA manifest for installation
├── sw.js             # Service worker for offline support
├── icon.svg          # App icon (SVG format)
├── generate-icons.html # Icon generator tool
└── README.md         # This file
```

### Generating App Icons

1. **Open `generate-icons.html`** in a browser
2. **Right-click each canvas** and save as PNG
3. **Save as `icon-192.png`** and `icon-512.png`
4. **Update manifest.json** to reference your icons

## 🎯 How to Use

1. **Launch the app** from your home screen
2. **Grant camera permissions** when prompted
3. **Tap "Open Camera"** to take a photo
4. **Or tap "Choose Photo"** to select from gallery
5. **Wait for AI analysis** (may take 5-10 seconds on first load)
6. **View results** with detected objects and confidence scores
7. **See visual detection boxes** on the image

## 🧠 Supported Objects

The app can detect 80+ object classes including:
- People (person, face)
- Vehicles (car, bicycle, motorcycle, bus, truck, boat)
- Animals (cat, dog, bird, horse, sheep, cow)
- Objects (chair, table, laptop, phone, bottle, cup)
- Indoor/Outdoor items (building, traffic light, stop sign)

## 📱 iPhone Compatibility

- **iOS 13.0+**: Full PWA support
- **iPhone 6s+**: Recommended for best performance
- **Safari**: Required for installation
- **Camera**: Used for photo capture
- **Storage**: ~5-10MB for cached model and resources

## 🔧 Technical Details

- **Frontend**: HTML5, CSS3, JavaScript ES6+
- **AI Model**: TensorFlow.js COCO-SSD
- **Framework**: Progressive Web App (PWA)
- **Offline**: Service Worker caching
- **Icons**: Adaptive icons for all screen sizes
- **Performance**: Optimized for mobile devices

## 🌐 Browser Requirements

- **Safari 13+**: Primary browser (iOS)
- **Chrome/Edge**: Will work but may not support PWA install
- **HTTPS Required**: For camera access and PWA features
- **JavaScript**: Must be enabled

## 🔒 Privacy & Security

- **Local Processing**: AI model runs entirely on your device
- **No Image Upload**: Photos stay on your iPhone
- **Offline First**: Works without internet after initial load
- **Secure**: Uses HTTPS for all connections

## 🚫 Troubleshooting

### Camera Not Working
1. Check camera permissions in Settings → Safari
2. Ensure HTTPS connection
3. Try refreshing the page
4. Restart Safari

### Installation Issues
1. Must use Safari browser
2. Clear Safari cache and try again
3. Check internet connection
4. Ensure you're on iOS 13+

### Model Loading Errors
1. Check internet connection (first load only)
2. Wait for model to fully download (~5-10MB)
3. Refresh page and try again
4. Check Safari has enough storage space

### Performance Issues
1. Close other apps to free memory
2. Use smaller images (<5MB)
3. Wait for model to cache
4. Restart Safari

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure you're using the latest iOS version
3. Try clearing Safari cache and data
4. Use Safari browser specifically

## 🎉 Benefits Over Native App

- **No App Store**: Install immediately without approval
- **Automatic Updates**: Always have the latest version
- **Universal**: Works on any device with a browser
- **Secure**: Runs in sandboxed browser environment
- **Lightweight**: No app download required
- **Cross-Platform**: Same app works on Android too

---

**Ready to detect objects with AI on your iPhone? Follow the installation guide above! 🚀**
