// Simple script to create placeholder icons using base64 data URIs
// This creates minimal PNG icons that will work for PWA installation

const fs = require('fs');

// Simple 192x192 PNG icon (purple gradient with camera symbol)
// This is a minimal valid PNG - actual icons should be generated using generate-icons.html
// For now, we'll create SVG icons that can be converted

console.log('To create proper icons:');
console.log('1. Open http://localhost:8000/generate-icons.html in your browser');
console.log('2. Right-click the 192x192 canvas and "Save image as" icon-192.png');
console.log('3. Right-click the 512x512 canvas and "Save image as" icon-512.png');
console.log('');
console.log('For now, creating a simple SVG fallback...');

// Create a simple SVG that can be used as apple-touch-icon
const svg192 = `<svg width="192" height="192" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#667eea;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#764ba2;stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="192" height="192" fill="url(#grad)" rx="38"/>
  <rect x="48" y="67" width="96" height="58" fill="white" rx="10"/>
  <circle cx="96" cy="96" r="23" fill="#667eea" stroke="white" stroke-width="6"/>
  <circle cx="96" cy="96" r="12" fill="none" stroke="white" stroke-width="4"/>
  <rect x="125" y="54" width="12" height="8" fill="white" rx="2"/>
  <text x="96" y="150" font-family="Arial" font-size="15" font-weight="bold" fill="white" text-anchor="middle">AI Detect</text>
</svg>`;

fs.writeFileSync('icon.svg', svg192);
console.log('Created icon.svg as fallback');

