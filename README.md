# Face Landmarker + Three.js

A browser-only demo that uses the webcam, MediaPipe Face Landmarker, and Three.js to render real-time face landmarks and a dense Kinect-style camera point cloud.

## Features

- Runs directly in the browser.
- Uses the webcam through `navigator.mediaDevices.getUserMedia()`.
- Uses MediaPipe `FaceLandmarker` in `VIDEO` mode for face landmarks.
- Renders 478 face landmarks as Three.js points.
- Renders semantic wireframe paths for face oval, eyes, brows, nose, and lips.
- Includes a 3D face mode with exaggerated depth and scene rotation.
- Includes a dense camera point cloud mode that samples the whole webcam frame into tens of thousands of Three.js particles.
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

### Full Body Point Cloud

This mode samples the whole webcam image into a dense Three.js particle field. It is designed to look closer to the classic Kinect point cloud demo.

Important limitation: a normal webcam does not provide real depth. This mode uses RGB-derived pseudo depth for the visual effect. For physically accurate Kinect-style depth, use a real depth camera such as Kinect / RealSense, or add a browser-compatible monocular depth model.

## Troubleshooting

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
  -> Full Body Point Cloud mode: draw frame to a low-resolution canvas
  -> sample every pixel into a Three.js point
  -> map RGB/luma to color and pseudo z-depth
  -> render a dense additive particle cloud
```

The Face Landmarker `z` value is model-estimated relative depth, not a real Kinect-style depth measurement.

## Main files

- `src/main.js` - camera setup, model setup, landmark smoothing, dense point cloud sampling, Three.js rendering.
- `src/style.css` - fullscreen camera and HUD layout.
- `index.html` - app shell.

## Notes

The demo loads the Face Landmarker model from Google Cloud Storage and the MediaPipe WASM runtime from jsDelivr. For production, download the `.task` model and host it from your own domain/CDN.
