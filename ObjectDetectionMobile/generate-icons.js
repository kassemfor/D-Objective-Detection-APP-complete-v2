const fs = require('fs');
const { createCanvas } = require('canvas');

// Note: This requires 'canvas' package. For simpler solution, we'll create SVG fallback
// Or use the HTML generator that already exists

function generateIconSVG(size) {
  return `<?xml version="1.0" encoding="UTF-8"?>
<svg width="${size}" height="${size}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#667eea;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#764ba2;stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="${size}" height="${size}" fill="url(#grad)" rx="${size * 0.2}"/>
  
  <!-- Camera body -->
  <rect x="${size * 0.25}" y="${size * 0.35}" width="${size * 0.5}" height="${size * 0.3}" fill="white" rx="${size * 0.05}"/>
  
  <!-- Camera lens -->
  <circle cx="${size * 0.5}" cy="${size * 0.5}" r="${size * 0.12}" fill="#667eea" stroke="white" stroke-width="${size * 0.03}"/>
  <circle cx="${size * 0.5}" cy="${size * 0.5}" r="${size * 0.06}" fill="none" stroke="white" stroke-width="${size * 0.02}"/>
  
  <!-- Flash -->
  <rect x="${size * 0.65}" y="${size * 0.28}" width="${size * 0.06}" height="${size * 0.04}" fill="white" rx="${size * 0.01}"/>
  
  <!-- Text -->
  <text x="${size * 0.5}" y="${size * 0.78}" font-family="Arial, sans-serif" font-size="${size * 0.08}" font-weight="bold" fill="white" text-anchor="middle">AI Detect</text>
</svg>`;
}

// Since we can't easily generate PNG without canvas package, let's use a simpler approach
// We'll modify the manifest to make icons optional for now, or create a simple data URI icon
console.log('Icons will be generated via the browser using generate-icons.html');

