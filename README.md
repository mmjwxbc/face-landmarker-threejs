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

Open the local URL printed by Vite, then allow camera permission.

> Camera access usually requires `localhost` or HTTPS.

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
