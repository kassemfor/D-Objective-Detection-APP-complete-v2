// Create PNG icons programmatically without external dependencies
// Uses pure Node.js to create valid PNG files

const fs = require('fs');

// Base64 encoded minimal valid PNG (1x1 transparent)
// We'll create a proper sized icon using this as a base
const minimalPNGBase64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';

function createSimpleGradientPNG(size) {
  // Create a simple PNG with gradient using minimal approach
  // This creates a valid PNG that iOS will accept
  // For proper icons, the generate-icons.html method is recommended
  
  // Read the SVG and use it as basis, or create a simple colored PNG
  // Since we can't easily create complex PNGs without libraries,
  // we'll create a simple solid color PNG that works
  
  // This is a minimal approach - creates a purple PNG
  // Actual implementation would use a proper PNG encoder
  console.log(`Creating ${size}x${size} PNG icon...`);
  
  // For now, create using a different method - use HTML canvas via puppeteer if available
  // Or create a very simple PNG
  
  // Create a valid PNG structure - simplified version
  // This requires proper PNG encoding which is complex
  // Instead, let's check if we can use an alternative
  
  return null; // Will be handled differently
}

// Better approach: Use node-canvas if available
let canvas;
try {
  canvas = require('canvas');
  console.log('✓ Canvas library found');
  createCanvasIcons();
} catch (e) {
  console.log('Canvas library not found. Creating icons using alternative method...');
  createBase64Icons();
}

function createCanvasIcons() {
  const { createCanvas } = canvas;
  
  function drawIcon(size) {
    const canv = createCanvas(size, size);
    const ctx = canv.getContext('2d');
    
    // Gradient background
    const gradient = ctx.createLinearGradient(0, 0, size, size);
    gradient.addColorStop(0, '#667eea');
    gradient.addColorStop(1, '#764ba2');
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, size, size);
    
    // Camera body
    const bodyWidth = size * 0.5;
    const bodyHeight = size * 0.35;
    const bodyX = (size - bodyWidth) / 2;
    const bodyY = (size - bodyHeight) / 2 + size * 0.05;
    
    ctx.fillStyle = 'white';
    ctx.fillRect(bodyX, bodyY, bodyWidth, bodyHeight);
    
    // Camera lens
    const lensRadius = size * 0.12;
    const lensCenterX = size / 2;
    const lensCenterY = bodyY + bodyHeight / 2;
    
    ctx.beginPath();
    ctx.arc(lensCenterX, lensCenterY, lensRadius, 0, 2 * Math.PI);
    ctx.fillStyle = '#667eea';
    ctx.fill();
    ctx.strokeStyle = 'white';
    ctx.lineWidth = size * 0.03;
    ctx.stroke();
    
    // Inner lens circle
    ctx.beginPath();
    ctx.arc(lensCenterX, lensCenterY, lensRadius * 0.5, 0, 2 * Math.PI);
    ctx.strokeStyle = 'white';
    ctx.lineWidth = size * 0.02;
    ctx.stroke();
    
    // Flash
    ctx.fillStyle = 'white';
    ctx.fillRect(bodyX + bodyWidth * 0.7, bodyY - size * 0.05, size * 0.06, size * 0.04);
    
    // Text
    ctx.fillStyle = 'white';
    ctx.font = `bold ${size * 0.08}px -apple-system, Arial`;
    ctx.textAlign = 'center';
    ctx.fillText('AI Detect', size / 2, bodyY + bodyHeight + size * 0.12);
    
    return canv.toBuffer('image/png');
  }
  
  const icon192 = drawIcon(192);
  const icon512 = drawIcon(512);
  
  fs.writeFileSync('icon-192.png', icon192);
  fs.writeFileSync('icon-512.png', icon512);
  
  console.log('✓ Created icon-192.png');
  console.log('✓ Created icon-512.png');
  console.log('Icons generated successfully!');
}

function createBase64Icons() {
  // Fallback: Use sharp if available, otherwise guide user
  try {
    const sharp = require('sharp');
    const iconSvg = fs.readFileSync('icon.svg', 'utf-8');
    
    sharp(Buffer.from(iconSvg))
      .resize(192, 192)
      .png()
      .toFile('icon-192.png')
      .then(() => {
        console.log('✓ Created icon-192.png');
        return sharp(Buffer.from(iconSvg))
          .resize(512, 512)
          .png()
          .toFile('icon-512.png');
      })
      .then(() => {
        console.log('✓ Created icon-512.png');
        console.log('Icons generated successfully!');
      });
  } catch (e) {
    console.log('');
    console.log('⚠️  Cannot auto-generate PNG icons without canvas or sharp library.');
    console.log('Please do one of the following:');
    console.log('');
    console.log('Option 1 (Recommended): Install canvas library');
    console.log('  npm install canvas');
    console.log('  Then run: node create-png-icons.js');
    console.log('');
    console.log('Option 2: Install sharp library');
    console.log('  npm install sharp');
    console.log('  Then run: node create-png-icons.js');
    console.log('');
    console.log('Option 3: Manual generation');
    console.log('  1. Make sure server is running');
    console.log('  2. Open http://localhost:8000/generate-icons.html');
    console.log('  3. Right-click 192x192 canvas → Save as icon-192.png');
    console.log('  4. Right-click 512x512 canvas → Save as icon-512.png');
    console.log('');
    process.exit(1);
  }
}

