const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = 8000;
const MIME_TYPES = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpg',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.wav': 'audio/wav',
  '.mp4': 'video/mp4',
  '.woff': 'application/font-woff',
  '.ttf': 'application/font-ttf',
  '.eot': 'application/vnd.ms-fontobject',
  '.otf': 'application/font-otf',
  '.wasm': 'application/wasm'
};

// Define the base directory for serving files (current directory)
const BASE_DIR = __dirname;

const server = http.createServer((req, res) => {
  console.log(`${req.method} ${req.url}`);

  // Parse URL and normalize the path
  // Handle cases where host header might be missing
  let urlPath;
  try {
    const host = req.headers.host || 'localhost';
    urlPath = new URL(req.url, `http://${host}`).pathname;
  } catch (error) {
    // Fallback for malformed URLs - extract path manually
    urlPath = req.url.split('?')[0].split('#')[0];
    if (!urlPath.startsWith('/')) {
      urlPath = '/' + urlPath;
    }
  }
  
  // Handle root path
  let requestedPath = urlPath === '/' ? '/index.html' : urlPath;
  
  // Remove leading slash for path.join and normalize
  if (requestedPath.startsWith('/')) {
    requestedPath = requestedPath.slice(1);
  }
  
  // Resolve the full path and normalize it
  let filePath = path.join(BASE_DIR, requestedPath);
  filePath = path.normalize(filePath);
  
  // Security check: Ensure the resolved path is within BASE_DIR
  // This prevents path traversal attacks (e.g., ../, ..\, %2e%2e%2f, etc.)
  const resolvedBase = path.resolve(BASE_DIR);
  const resolvedPath = path.resolve(filePath);
  
  if (!resolvedPath.startsWith(resolvedBase + path.sep) && resolvedPath !== resolvedBase) {
    console.warn(`Path traversal attempt blocked: ${req.url} -> ${resolvedPath}`);
    res.writeHead(403, { 'Content-Type': 'text/html' });
    res.end('403 Forbidden: Path traversal not allowed', 'utf-8');
    return;
  }

  const extname = String(path.extname(filePath)).toLowerCase();
  const contentType = MIME_TYPES[extname] || 'application/octet-stream';
  
  // Determine if file is binary (should not use UTF-8 encoding)
  const binaryExtensions = ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', '.woff', '.woff2', '.ttf', '.eot', '.otf', '.wasm', '.mp4', '.mp3', '.wav', '.webp'];
  const isBinary = binaryExtensions.includes(extname);

  fs.readFile(filePath, isBinary ? null : 'utf8', (error, content) => {
    if (error) {
      if (error.code === 'ENOENT') {
        res.writeHead(404, { 'Content-Type': 'text/html' });
        res.end('404 Not Found', 'utf-8');
      } else {
        res.writeHead(500);
        res.end(`Server Error: ${error.code}`, 'utf-8');
      }
    } else {
      res.writeHead(200, { 'Content-Type': contentType });
      // Only use encoding for text files, send binary files as-is
      if (isBinary) {
        res.end(content); // No encoding parameter for binary files
      } else {
        res.end(content, 'utf-8'); // UTF-8 encoding for text files
      }
    }
  });
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`Server running at http://localhost:${PORT}/`);
  console.log(`Server accessible on your network!`);
  console.log(`Open on iPhone: http://[YOUR-IP]:${PORT}`);
  console.log('Press Ctrl+C to stop the server');
});

