# 📱 How to Install on iPhone - Step by Step Guide

## Prerequisites
1. ✅ Server is running on your computer
2. ✅ Your iPhone is on the same Wi-Fi network as your computer
3. ✅ You have your computer's IP address

## Step 1: Find Your Computer's IP Address

Your computer's local IP addresses are:
- **172.20.10.4** (most likely for iPhone connection)
- 192.168.56.1
- 192.168.120.1  
- 192.168.198.1

To verify which one to use, check your iPhone's Wi-Fi settings - it should be on the same network subnet.

## Step 2: Open Safari on Your iPhone

⚠️ **IMPORTANT**: You MUST use Safari, not Chrome or other browsers.

1. Open the **Safari** app on your iPhone
2. In the address bar, type: `http://172.20.10.4:8000`
   - (Replace `172.20.10.4` with your actual IP if different)
3. Tap **Go**

## Step 3: Install to Home Screen

1. Once the app loads, tap the **Share button** (📤) at the bottom of Safari
2. Scroll down in the share menu
3. Tap **"Add to Home Screen"**
4. Customize the name if you want (or leave it as "Object Detection Studio")
5. Tap **"Add"** in the top-right corner

## Step 4: Launch Your App

1. Find the app icon on your iPhone's home screen
2. Tap it to launch
3. The app will open like a native iPhone app!

## Troubleshooting

### Can't Connect to Server?
- Make sure your iPhone and computer are on the same Wi-Fi network
- Check Windows Firewall - it may be blocking port 8000
- Try the other IP addresses listed above
- Make sure the server is still running on your computer

### Firewall Blocking?
Run this in PowerShell as Administrator to allow connections:
```powershell
New-NetFirewallRule -DisplayName "Object Detection App" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
```

### App Not Installing?
- Make sure you're using Safari (not Chrome)
- Try refreshing the page
- Make sure the page loaded completely before trying to install
- Check that you're on iOS 13.0 or later

### Icons Not Showing?
- The app will work without icons, but they make it look better
- Open `http://172.20.10.4:8000/generate-icons.html` on your computer
- Right-click each icon and save as `icon-192.png` and `icon-512.png` in the ObjectDetectionMobile folder
- Refresh the page on your iPhone

## After Installation

✅ The app will work offline after the first load  
✅ No internet needed once the AI model is cached  
✅ All processing happens on your iPhone - completely private!  
✅ It looks and feels like a native iPhone app  

Enjoy your AI-powered object detection app! 🎉

