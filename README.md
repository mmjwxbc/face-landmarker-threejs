# Face Landmarker + Three.js

A browser-only demo that uses the webcam, MediaPipe Face Landmarker, Depth Anything V2 Small, Transformers.js, and Three.js to render real-time face landmarks and a dense AI depth point cloud.

## Features

- Runs directly in the browser.
- Uses the webcam through `navigator.mediaDevices.getUserMedia()`.
- Uses MediaPipe `FaceLandmarker` in `VIDEO` mode for face landmarks.
- Renders 478 face landmarks as Three.js points.
- Renders semantic wireframe paths for face oval, eyes, brows, nose, and lips.
- Includes a 3D face mode with exaggerated depth and scene rotation.
- Includes an AI depth point cloud mode that samples the whole webcam frame into tens of thousands of Three.js particles.
- Uses `onnx-community/depth-anything-v2-small-ONNX` through `@huggingface/transformers` to estimate monocular depth in the browser.
- Falls back to RGB pseudo depth if the depth model is still loading or cannot run in the current browser.
- Includes camera preview, FPS status, and visibility toggles.

## Run

```bash
npm install
npm run dev
```

Open the Local URL printed by Vite, usually:

```text
http://localhost:5173
```

Then allow camera permission.

> Do not open the app from a plain LAN IP such as `http://192.168.x.x:5173` unless you have HTTPS configured. Browser camera access requires a secure context: localhost or HTTPS.

## Modes

### Face landmarks

This mode uses MediaPipe Face Landmarker and renders the detected 478 face points in Three.js. It is accurate for facial structure, but it only covers the face.

### 3D Face

This mode uses the same landmarks, but exaggerates the z value and rotates the Three.js group so the face looks more three-dimensional.

### AI Depth Point Cloud

This mode draws the webcam frame into a low-resolution canvas, runs a browser-side monocular depth estimation model, and maps each sampled pixel into a Three.js particle.

The x/y position comes from the pixel coordinate, color comes from the original RGB frame, and z depth comes from Depth Anything V2 Small. While the model is loading, the demo uses a lightweight RGB/luma pseudo-depth fallback so the scene remains interactive.

Important limitation: monocular depth is still estimated from one RGB camera. It is closer to a Kinect-style point cloud than pure RGB pseudo-depth, but it is not physically measured depth. For physically accurate depth, use a real depth camera such as Kinect / RealSense / LiDAR.

## Troubleshooting

### First AI Depth load is slow

The first click on `AI Depth Point Cloud` downloads and initializes the ONNX depth model. This can take a while depending on your network and device. After the browser cache is warm, it should start faster.

### Browser or GPU is unsupported

The app tries WebGPU first when available, then falls back to WASM, then falls back to RGB pseudo-depth if the model cannot run.

### `Cannot read properties of undefined (reading 'getUserMedia')`

This means the browser did not expose `navigator.mediaDevices.getUserMedia` to the page.

Most common causes:

- You opened a non-secure URL, such as a plain HTTP LAN IP.
- You opened the built files directly instead of using a local dev server.
- You are inside an embedded browser or WebView that blocks camera APIs.
- The browser is too old or camera permission is blocked.

Fix:

```bash
npm run dev
```

Then open:

```text
http://localhost:5173
```

For testing from another device, deploy the app with HTTPS, or use a secure tunnel such as Cloudflare Tunnel, ngrok, or a real HTTPS domain.

## Build

```bash
npm run build
npm run preview
```

## How it works

```text
Webcam video frame
  -> Face mode: MediaPipe FaceLandmarker.detectForVideo(video, timestamp)
  -> 478 normalized 3D face landmarks
  -> Three.js BufferGeometry positions
  -> Points + LineSegments render loop
```

```text
Webcam video frame
  -> AI Depth mode: draw frame to low-resolution canvas
  -> Transformers.js depth-estimation pipeline
  -> Depth Anything V2 Small ONNX model
  -> normalized depth map
  -> sample every pixel into a Three.js point
  -> RGB controls color, depth map controls z
  -> render a dense additive particle cloud
```

The Face Landmarker `z` value is model-estimated relative depth, not a real Kinect-style depth measurement. The AI depth mode uses monocular estimated depth, not sensor-measured depth.

## Main files

- `src/main.js` - camera setup, model setup, landmark smoothing, AI depth estimation, dense point cloud sampling, Three.js rendering.
- `src/style.css` - fullscreen camera and HUD layout.
- `index.html` - app shell.

## Notes

The demo loads the Face Landmarker model from Google Cloud Storage, MediaPipe WASM runtime from jsDelivr, and the Depth Anything V2 Small ONNX model from Hugging Face. For production, host the model assets from your own domain/CDN.
