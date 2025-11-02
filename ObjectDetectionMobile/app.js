// Object Detection Mobile App for iPhone
let model = null;
let deferredPrompt = null;

// Initialize the app
window.addEventListener('DOMContentLoaded', async () => {
    await initializeApp();
    setupEventListeners();
    checkInstallPrompt();
});

// Initialize TensorFlow.js and load model
async function initializeApp() {
    try {
        updateStatus('Loading model...');
        model = await cocoSsd.load();
        updateStatus('Ready');
        console.log('Model loaded successfully');
    } catch (error) {
        console.error('Failed to load model:', error);
        updateStatus('Error loading model');
        alert('Failed to load AI model. Please refresh the page.');
    }
}

// Setup event listeners
function setupEventListeners() {
    // Camera button
    document.getElementById('cameraBtn').addEventListener('click', () => {
        document.getElementById('cameraInput').click();
    });
    
    // Gallery button
    document.getElementById('galleryBtn').addEventListener('click', () => {
        document.getElementById('galleryInput').click();
    });
    
    // File input handlers
    document.getElementById('cameraInput').addEventListener('change', handleImageUpload);
    document.getElementById('galleryInput').addEventListener('change', handleImageUpload);
    
    // Install prompt buttons
    document.getElementById('installBtn').addEventListener('click', handleInstallClick);
    document.getElementById('dismissBtn').addEventListener('click', () => {
        document.getElementById('installPrompt').classList.add('hidden');
    });
}

// Handle image upload
async function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    // Reset file input
    event.target.value = '';
    
    // Check if file is an image
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
    }
    
    // Show loading
    showLoading();
    hideResults();
    
    // Create image element
    const img = new Image();
    const reader = new FileReader();
    
    reader.onload = async (e) => {
        img.onload = async () => {
            await processImage(img);
        };
        img.src = e.target.result;
    };
    
    reader.readAsDataURL(file);
}

// Process image with object detection
async function processImage(img) {
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const placeholder = document.getElementById('placeholder');
    
    // Set canvas size
    const maxWidth = window.innerWidth - 40;
    const maxHeight = 400;
    let width = img.width;
    let height = img.height;
    
    // Calculate scaling
    if (width > maxWidth) {
        height = (maxWidth / width) * height;
        width = maxWidth;
    }
    if (height > maxHeight) {
        width = (maxHeight / height) * width;
        height = maxHeight;
    }
    
    canvas.width = width;
    canvas.height = height;
    
    // Draw image
    ctx.drawImage(img, 0, 0, width, height);
    
    // Show canvas, hide placeholder
    canvas.classList.add('active');
    placeholder.style.display = 'none';
    
    try {
        updateStatus('Detecting...');
        
        // Run object detection
        const predictions = await model.detect(canvas);
        
        // Draw detection boxes
        drawDetections(ctx, predictions, width / img.width);
        
        // Show results
        displayResults(predictions);
        
        updateStatus('Complete');
        hideLoading();
    } catch (error) {
        console.error('Detection error:', error);
        alert('Error during detection. Please try again.');
        updateStatus('Error');
        hideLoading();
    }
}

// Draw detection boxes on canvas
function drawDetections(ctx, predictions, scale) {
    ctx.lineWidth = 2;
    ctx.font = 'bold 14px Arial';
    
    predictions.forEach(prediction => {
        const [x, y, width, height] = prediction.bbox;
        
        // Scale coordinates
        const scaledX = x * scale;
        const scaledY = y * scale;
        const scaledWidth = width * scale;
        const scaledHeight = height * scale;
        
        // Choose color based on confidence
        const confidence = prediction.score;
        let color;
        if (confidence > 0.7) {
            color = '#34C759'; // Green
        } else if (confidence > 0.5) {
            color = '#FF9500'; // Orange
        } else {
            color = '#FF3B30'; // Red
        }
        
        // Draw bounding box
        ctx.strokeStyle = color;
        ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);
        
        // Draw label background
        const label = `${prediction.class} ${Math.round(confidence * 100)}%`;
        const textWidth = ctx.measureText(label).width;
        ctx.fillStyle = color;
        ctx.fillRect(scaledX, scaledY - 20, textWidth + 10, 20);
        
        // Draw label text
        ctx.fillStyle = 'white';
        ctx.fillText(label, scaledX + 5, scaledY - 5);
    });
}

// Display detection results
function displayResults(predictions) {
    const resultsDiv = document.getElementById('results');
    const detectionsList = document.getElementById('detectionsList');
    
    // Clear previous results
    detectionsList.innerHTML = '';
    
    if (predictions.length === 0) {
        detectionsList.innerHTML = '<p style="opacity: 0.7;">No objects detected</p>';
    } else {
        // Sort by confidence
        predictions.sort((a, b) => b.score - a.score);
        
        // Create detection items
        predictions.forEach(prediction => {
            const item = document.createElement('div');
            item.className = 'detection-item';
            
            const name = document.createElement('span');
            name.className = 'detection-name';
            name.textContent = prediction.class;
            
            const confidence = document.createElement('span');
            confidence.className = 'detection-confidence';
            confidence.textContent = `${Math.round(prediction.score * 100)}%`;
            
            item.appendChild(name);
            item.appendChild(confidence);
            detectionsList.appendChild(item);
        });
    }
    
    resultsDiv.classList.remove('hidden');
}

// Show/hide loading indicator
function showLoading() {
    document.getElementById('loading').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loading').classList.add('hidden');
}

// Show/hide results
function hideResults() {
    document.getElementById('results').classList.add('hidden');
}

// Update status
function updateStatus(status) {
    const statValue = document.querySelector('.stat-value');
    if (statValue) {
        statValue.textContent = status;
    }
}

// PWA Install Prompt
function checkInstallPrompt() {
    // Check if app is already installed
    if (window.matchMedia('(display-mode: standalone)').matches) {
        console.log('App is already installed');
        return;
    }
    
    // Listen for beforeinstallprompt
    window.addEventListener('beforeinstallprompt', (e) => {
        e.preventDefault();
        deferredPrompt = e;
        
        // Show install prompt after 5 seconds
        setTimeout(() => {
            if (deferredPrompt) {
                document.getElementById('installPrompt').classList.remove('hidden');
            }
        }, 5000);
    });
}

// Handle install button click
async function handleInstallClick() {
    if (!deferredPrompt) {
        alert('Please add this app to your home screen using the Share button in Safari');
        return;
    }
    
    // Show install prompt
    const result = await deferredPrompt.prompt();
    console.log('Install prompt result:', result);
    
    // Hide install prompt
    document.getElementById('installPrompt').classList.add('hidden');
    deferredPrompt = null;
}

// Register service worker for offline support
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('sw.js')
            .then(registration => console.log('ServiceWorker registered'))
            .catch(err => console.log('ServiceWorker registration failed'));
    });
}

// Add haptic feedback for iOS
function triggerHaptic() {
    if (window.navigator && window.navigator.vibrate) {
        window.navigator.vibrate(10);
    }
}

// Add haptic feedback to buttons
document.querySelectorAll('.btn').forEach(btn => {
    btn.addEventListener('touchstart', triggerHaptic);
});
