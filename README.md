# Face Landmarker + Three.js

A browser-only demo that uses the webcam, MediaPipe Face Landmarker, and Three.js to render a real-time 3D face point cloud / wireframe.

## Features

- Runs directly in the browser.
- Uses the webcam through `navigator.mediaDevices.getUserMedia()`.
- Uses MediaPipe `FaceLandmarker` in `VIDEO` mode.
- Renders 478 face landmarks as Three.js points.
- Renders semantic wireframe paths for face oval, eyes, brows, nose, and lips.
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
  -> MediaPipe FaceLandmarker.detectForVideo(video, timestamp)
  -> 478 normalized 3D face landmarks
  -> Three.js BufferGeometry positions
  -> Points + LineSegments render loop
```

The `z` value is model-estimated relative depth, not a real Kinect-style depth measurement.

## Main files

- `src/main.js` - camera setup, model setup, landmark smoothing, Three.js rendering.
- `src/style.css` - fullscreen camera and HUD layout.
- `index.html` - app shell.

## Notes

The demo loads the Face Landmarker model from Google Cloud Storage and the MediaPipe WASM runtime from jsDelivr. For production, download the `.task` model and host it from your own domain/CDN.
