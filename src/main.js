import "./style.css";
import * as THREE from "three";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const video = document.querySelector("#input-video");
const canvas = document.querySelector("#three-canvas");
const statusEl = document.querySelector("#status");
const toggleVideoBtn = document.querySelector("#toggle-video");
const toggleLinesBtn = document.querySelector("#toggle-lines");
const togglePointsBtn = document.querySelector("#toggle-points");
const toggleSceneBtn = document.querySelector("#toggle-3d");
const toggleCloudBtn = document.querySelector("#toggle-full-body");

const FACE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const LANDMARK_COUNT = 478;

const CLOUD_WIDTH = 220;
const CLOUD_HEIGHT = 124;
const CLOUD_COUNT = CLOUD_WIDTH * CLOUD_HEIGHT;

const WIRE_PATHS = [
  [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10],
  [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 61],
  [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78],
  [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164],
  [98, 97, 2, 326, 327],
  [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173, 33],
  [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
  [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362],
  [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
];

let faceLandmarker;
let lastVideoTime = -1;
let lastCloudVideoTime = -1;
let isPointsVisible = true;
let isLinesVisible = true;
let isThreeSceneMode = false;
let isCloudMode = false;
let smoothedLandmarks = null;
let frameCount = 0;
let lastFpsTime = performance.now();

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(0, 0, 2.35);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(0x000000, 0);

const faceGroup = new THREE.Group();
scene.add(faceGroup);

const pointPositions = new Float32Array(LANDMARK_COUNT * 3);
const pointGeometry = new THREE.BufferGeometry();
pointGeometry.setAttribute("position", new THREE.BufferAttribute(pointPositions, 3));

const pointMaterial = new THREE.PointsMaterial({
  color: 0x7deaff,
  size: 0.012,
  sizeAttenuation: true,
  transparent: true,
  opacity: 0.9,
  depthWrite: false,
  blending: THREE.AdditiveBlending
});

const points = new THREE.Points(pointGeometry, pointMaterial);
faceGroup.add(points);

const linePairs = buildLinePairs(WIRE_PATHS);
const linePositions = new Float32Array(linePairs.length * 2 * 3);
const lineGeometry = new THREE.BufferGeometry();
lineGeometry.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));

const lineMaterial = new THREE.LineBasicMaterial({
  color: 0xffffff,
  transparent: true,
  opacity: 0.82,
  depthWrite: false,
  blending: THREE.AdditiveBlending
});

const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
faceGroup.add(lines);

const cloudGroup = new THREE.Group();
cloudGroup.visible = false;
scene.add(cloudGroup);

const cloudPositions = new Float32Array(CLOUD_COUNT * 3);
const cloudColors = new Float32Array(CLOUD_COUNT * 3);
const cloudNoise = new Float32Array(CLOUD_COUNT);
for (let i = 0; i < CLOUD_COUNT; i++) {
  const s = Math.sin(i * 12.9898) * 43758.5453;
  cloudNoise[i] = s - Math.floor(s);
}

const cloudGeometry = new THREE.BufferGeometry();
cloudGeometry.setAttribute("position", new THREE.BufferAttribute(cloudPositions, 3));
cloudGeometry.setAttribute("color", new THREE.BufferAttribute(cloudColors, 3));

const cloudMaterial = new THREE.PointsMaterial({
  size: 0.012,
  sizeAttenuation: true,
  vertexColors: true,
  transparent: true,
  opacity: 0.96,
  depthWrite: false,
  blending: THREE.AdditiveBlending
});

const cloudPoints = new THREE.Points(cloudGeometry, cloudMaterial);
cloudGroup.add(cloudPoints);

const cloudCanvas = document.createElement("canvas");
cloudCanvas.width = CLOUD_WIDTH;
cloudCanvas.height = CLOUD_HEIGHT;
const cloudCtx = cloudCanvas.getContext("2d", { willReadFrequently: true });

const grid = new THREE.GridHelper(4.8, 22, 0x7deaff, 0x214b68);
grid.rotation.x = Math.PI / 2;
grid.position.z = -1.15;
grid.visible = false;
grid.material.transparent = true;
grid.material.opacity = 0.14;
scene.add(grid);

const scanRing = new THREE.Mesh(
  new THREE.RingGeometry(0.18, 1.55, 128),
  new THREE.MeshBasicMaterial({
    color: 0x7deaff,
    transparent: true,
    opacity: 0,
    side: THREE.DoubleSide,
    depthWrite: false,
    blending: THREE.AdditiveBlending
  })
);
scanRing.position.z = -0.9;
scanRing.visible = false;
scene.add(scanRing);

scene.add(new THREE.AmbientLight(0xffffff, 1.0));

function buildLinePairs(paths) {
  const pairs = [];
  for (const path of paths) {
    for (let i = 0; i < path.length - 1; i++) {
      pairs.push([path[i], path[i + 1]]);
    }
  }
  return pairs;
}

function normalizedToScenePoint(landmark) {
  const aspect = window.innerWidth / window.innerHeight;
  const xyScale = isThreeSceneMode ? 1.9 : 2.15;
  const zScale = isThreeSceneMode ? 5.8 : 1.7;
  const x = (0.5 - landmark.x) * aspect * xyScale;
  const y = -(landmark.y - 0.5) * xyScale;
  const z = -landmark.z * zScale;
  return [x, y, z];
}

function smoothLandmarks(landmarks, alpha = 0.55) {
  if (!smoothedLandmarks || smoothedLandmarks.length !== landmarks.length) {
    smoothedLandmarks = landmarks.map((point) => ({ ...point }));
    return smoothedLandmarks;
  }

  for (let i = 0; i < landmarks.length; i++) {
    smoothedLandmarks[i].x = smoothedLandmarks[i].x * (1 - alpha) + landmarks[i].x * alpha;
    smoothedLandmarks[i].y = smoothedLandmarks[i].y * (1 - alpha) + landmarks[i].y * alpha;
    smoothedLandmarks[i].z = smoothedLandmarks[i].z * (1 - alpha) + landmarks[i].z * alpha;
  }

  return smoothedLandmarks;
}

function updateFaceGeometry(landmarks) {
  const stable = smoothLandmarks(landmarks);

  for (let i = 0; i < LANDMARK_COUNT; i++) {
    const [x, y, z] = normalizedToScenePoint(stable[i]);
    pointPositions[i * 3 + 0] = x;
    pointPositions[i * 3 + 1] = y;
    pointPositions[i * 3 + 2] = z;
  }

  for (let i = 0; i < linePairs.length; i++) {
    const [a, b] = linePairs[i];
    const aOffset = a * 3;
    const bOffset = b * 3;
    const target = i * 6;

    linePositions[target + 0] = pointPositions[aOffset + 0];
    linePositions[target + 1] = pointPositions[aOffset + 1];
    linePositions[target + 2] = pointPositions[aOffset + 2];
    linePositions[target + 3] = pointPositions[bOffset + 0];
    linePositions[target + 4] = pointPositions[bOffset + 1];
    linePositions[target + 5] = pointPositions[bOffset + 2];
  }

  pointGeometry.attributes.position.needsUpdate = true;
  lineGeometry.attributes.position.needsUpdate = true;
}

function updateCameraCloud() {
  cloudCtx.save();
  cloudCtx.scale(-1, 1);
  cloudCtx.drawImage(video, -CLOUD_WIDTH, 0, CLOUD_WIDTH, CLOUD_HEIGHT);
  cloudCtx.restore();

  const pixels = cloudCtx.getImageData(0, 0, CLOUD_WIDTH, CLOUD_HEIGHT).data;
  const aspect = window.innerWidth / window.innerHeight;
  let pointIndex = 0;

  for (let y = 0; y < CLOUD_HEIGHT; y++) {
    const v = y / (CLOUD_HEIGHT - 1);
    for (let x = 0; x < CLOUD_WIDTH; x++) {
      const u = x / (CLOUD_WIDTH - 1);
      const pixelIndex = (y * CLOUD_WIDTH + x) * 4;
      const r = pixels[pixelIndex + 0] / 255;
      const g = pixels[pixelIndex + 1] / 255;
      const b = pixels[pixelIndex + 2] / 255;
      const max = Math.max(r, g, b);
      const min = Math.min(r, g, b);
      const luma = r * 0.2126 + g * 0.7152 + b * 0.0722;
      const sat = max - min;
      const edgeFalloff = 1 - Math.min(1, Math.hypot(u - 0.5, (v - 0.5) * 1.25) * 1.25);
      const shimmer = cloudNoise[pointIndex] - 0.5;
      const visibility = Math.min(1, Math.max(0.08, luma * 1.4 + sat * 0.55));

      cloudPositions[pointIndex * 3 + 0] = (u - 0.5) * aspect * 3.75;
      cloudPositions[pointIndex * 3 + 1] = -(v - 0.5) * 2.18;
      cloudPositions[pointIndex * 3 + 2] = (0.5 - luma) * 1.05 + edgeFalloff * 0.42 + shimmer * 0.08;

      cloudColors[pointIndex * 3 + 0] = 0.02 * visibility + sat * 0.04;
      cloudColors[pointIndex * 3 + 1] = (0.22 + luma * 0.42 + edgeFalloff * 0.22) * visibility;
      cloudColors[pointIndex * 3 + 2] = (0.55 + luma * 0.75 + sat * 0.25) * visibility;

      pointIndex += 1;
    }
  }

  cloudGeometry.attributes.position.needsUpdate = true;
  cloudGeometry.attributes.color.needsUpdate = true;
}

function animateFaceScene(now) {
  if (isThreeSceneMode) {
    renderer.setClearColor(0x030711, 0.94);
    faceGroup.rotation.x = -0.12 + Math.sin(now * 0.00045) * 0.08;
    faceGroup.rotation.y = Math.sin(now * 0.0007) * 0.42;
    faceGroup.rotation.z = Math.sin(now * 0.00035) * 0.035;
    faceGroup.scale.setScalar(1.18);
    camera.position.z += (2.35 - camera.position.z) * 0.08;

    grid.visible = true;
    scanRing.visible = true;
    scanRing.rotation.z = now * 0.00015;
    scanRing.material.opacity = 0.18;
    pointMaterial.size = 0.018;
    pointMaterial.opacity = 1.0;
    lineMaterial.opacity = 0.7;
    return;
  }

  renderer.setClearColor(0x000000, 0);
  faceGroup.rotation.set(0, 0, 0);
  faceGroup.scale.setScalar(1);
  camera.position.z += (2.35 - camera.position.z) * 0.08;
  grid.visible = false;
  scanRing.visible = false;
  scanRing.material.opacity = 0;
  pointMaterial.size = 0.012;
  pointMaterial.opacity = 0.9;
  lineMaterial.opacity = 0.82;
}

function animateCloudScene(now) {
  renderer.setClearColor(0x000000, 1);
  camera.position.z += (3.0 - camera.position.z) * 0.08;
  cloudGroup.rotation.x = -0.08 + Math.sin(now * 0.00025) * 0.05;
  cloudGroup.rotation.y = Math.sin(now * 0.00042) * 0.34;
  cloudGroup.rotation.z = Math.sin(now * 0.00018) * 0.025;
  cloudGroup.scale.setScalar(1.35);
  cloudMaterial.size = 0.01 + Math.sin(now * 0.001) * 0.0015;

  grid.visible = true;
  grid.position.z = -1.35;
  scanRing.visible = true;
  scanRing.scale.setScalar(1.35 + Math.sin(now * 0.0012) * 0.08);
  scanRing.rotation.z = now * 0.00025;
  scanRing.material.opacity = 0.11;
}

function getCameraApiErrorMessage() {
  const origin = window.location.origin;
  const protocol = window.location.protocol;
  const host = window.location.hostname;

  if (!window.isSecureContext) {
    return `Camera API is blocked because this page is not a secure context. Current origin: ${origin}. Open the app with http://localhost:5173 or deploy it with HTTPS.`;
  }

  if (!navigator.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== "function") {
    return `navigator.mediaDevices.getUserMedia is not available in this browser/context. Current protocol: ${protocol}, host: ${host}. Use a modern Chrome, Edge, Firefox, or Safari tab on localhost or HTTPS.`;
  }

  return null;
}

async function startCamera() {
  const cameraApiError = getCameraApiErrorMessage();
  if (cameraApiError) {
    throw new Error(cameraApiError);
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: "user"
    },
    audio: false
  });

  video.srcObject = stream;
  await video.play();
}

async function createFaceLandmarker() {
  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  return FaceLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: FACE_MODEL_URL,
      delegate: "GPU"
    },
    runningMode: "VIDEO",
    numFaces: 1,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
    outputFaceBlendshapes: true,
    outputFacialTransformationMatrixes: true
  });
}

function updateFps(mode) {
  frameCount += 1;
  const now = performance.now();
  const elapsed = now - lastFpsTime;

  if (elapsed > 800) {
    const fps = Math.round((frameCount * 1000) / elapsed);
    statusEl.textContent = `Tracking - ${fps} FPS - ${mode}`;
    frameCount = 0;
    lastFpsTime = now;
  }
}

function renderLoop() {
  const now = performance.now();

  if (isCloudMode) {
    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== lastCloudVideoTime) {
      updateCameraCloud();
      updateFps("dense camera point cloud");
      lastCloudVideoTime = video.currentTime;
    }
    animateCloudScene(now);
  } else {
    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== lastVideoTime) {
      const result = faceLandmarker.detectForVideo(video, now);
      const landmarks = result.faceLandmarks?.[0];

      if (landmarks) {
        updateFaceGeometry(landmarks);
        updateFps(isThreeSceneMode ? "3D face landmarks" : "camera-aligned face landmarks");
      } else {
        statusEl.textContent = "No face detected";
      }

      lastVideoTime = video.currentTime;
    }
    animateFaceScene(now);
  }

  faceGroup.visible = !isCloudMode;
  cloudGroup.visible = isCloudMode;
  points.visible = !isCloudMode && isPointsVisible;
  lines.visible = !isCloudMode && isLinesVisible;

  renderer.render(scene, camera);
  requestAnimationFrame(renderLoop);
}

function resize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

function setThreeSceneMode(enabled) {
  isThreeSceneMode = enabled;
  document.body.classList.toggle("three-scene", isThreeSceneMode);
  toggleSceneBtn.classList.toggle("active", isThreeSceneMode);
  toggleSceneBtn.textContent = isThreeSceneMode ? "Overlay Face" : "3D Face";
}

function setCloudMode(enabled) {
  isCloudMode = enabled;
  if (enabled) {
    setThreeSceneMode(false);
  }
  document.body.classList.toggle("cloud-scene", isCloudMode);
  toggleCloudBtn.classList.toggle("active", isCloudMode);
  toggleCloudBtn.textContent = isCloudMode ? "Face Mode" : "Full Body Point Cloud";
  statusEl.textContent = isCloudMode ? "Dense camera point cloud mode enabled" : "Face landmark mode enabled";
}

function bindControls() {
  toggleVideoBtn.addEventListener("click", () => {
    video.classList.toggle("hidden");
    toggleVideoBtn.textContent = video.classList.contains("hidden") ? "Show Camera" : "Hide Camera";
  });

  toggleLinesBtn.addEventListener("click", () => {
    isLinesVisible = !isLinesVisible;
    toggleLinesBtn.classList.toggle("active", isLinesVisible);
  });

  togglePointsBtn.addEventListener("click", () => {
    isPointsVisible = !isPointsVisible;
    togglePointsBtn.classList.toggle("active", isPointsVisible);
  });

  toggleSceneBtn.addEventListener("click", () => {
    if (isCloudMode) {
      setCloudMode(false);
    }
    setThreeSceneMode(!isThreeSceneMode);
  });

  toggleCloudBtn.addEventListener("click", () => {
    setCloudMode(!isCloudMode);
  });

  toggleLinesBtn.classList.toggle("active", isLinesVisible);
  togglePointsBtn.classList.toggle("active", isPointsVisible);
  window.addEventListener("resize", resize);
}

async function main() {
  try {
    bindControls();
    statusEl.textContent = "Requesting camera...";
    await startCamera();
    statusEl.textContent = "Loading face landmarker model...";
    faceLandmarker = await createFaceLandmarker();
    statusEl.textContent = "Ready";
    renderLoop();
  } catch (error) {
    console.error(error);
    statusEl.textContent = `Error: ${error.message}`;
  }
}

main();
