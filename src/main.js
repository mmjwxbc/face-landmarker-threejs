import "./style.css";
import * as THREE from "three";
import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";

const video = document.querySelector("#input-video");
const canvas = document.querySelector("#three-canvas");
const statusEl = document.querySelector("#status");
const toggleVideoBtn = document.querySelector("#toggle-video");
const toggleLinesBtn = document.querySelector("#toggle-lines");
const togglePointsBtn = document.querySelector("#toggle-points");

const FACE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";

// MediaPipe Face Landmarker currently returns 478 landmarks per detected face.
const LANDMARK_COUNT = 478;

// Lightweight semantic wireframe groups. These are stable landmark paths that make
// the face read clearly without needing the full triangulation list.
const WIRE_PATHS = [
  // Face oval
  [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10],
  // Lips outer and inner
  [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 61],
  [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78],
  // Nose bridge and bottom
  [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 164],
  [98, 97, 2, 326, 327],
  // Left eye and eyebrow
  [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173, 33],
  [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
  // Right eye and eyebrow
  [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362],
  [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
];

let faceLandmarker;
let lastVideoTime = -1;
let isPointsVisible = true;
let isLinesVisible = true;
let smoothedLandmarks = null;
let frameCount = 0;
let lastFpsTime = performance.now();

const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(0, 0, 2.35);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);

const group = new THREE.Group();
scene.add(group);

const pointPositions = new Float32Array(LANDMARK_COUNT * 3);
const pointGeometry = new THREE.BufferGeometry();
pointGeometry.setAttribute("position", new THREE.BufferAttribute(pointPositions, 3));

const pointMaterial = new THREE.PointsMaterial({
  size: 0.012,
  sizeAttenuation: true,
  transparent: true,
  opacity: 0.9,
  depthWrite: false
});

const points = new THREE.Points(pointGeometry, pointMaterial);
group.add(points);

const linePairs = buildLinePairs(WIRE_PATHS);
const linePositions = new Float32Array(linePairs.length * 2 * 3);
const lineGeometry = new THREE.BufferGeometry();
lineGeometry.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));

const lineMaterial = new THREE.LineBasicMaterial({
  transparent: true,
  opacity: 0.82,
  depthWrite: false
});

const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
group.add(lines);

const light = new THREE.AmbientLight(0xffffff, 1.0);
scene.add(light);

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
  // Landmarker x/y are normalized image coordinates. Mirror x so the 3D overlay
  // matches the mirrored front camera preview.
  const aspect = window.innerWidth / window.innerHeight;
  const x = (0.5 - landmark.x) * aspect * 2.15;
  const y = -(landmark.y - 0.5) * 2.15;
  const z = -landmark.z * 1.7;
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

function updateGeometry(landmarks) {
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

async function startCamera() {
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

function updateFps() {
  frameCount += 1;
  const now = performance.now();
  const elapsed = now - lastFpsTime;

  if (elapsed > 800) {
    const fps = Math.round((frameCount * 1000) / elapsed);
    statusEl.textContent = `Tracking face landmarks - ${fps} FPS`;
    frameCount = 0;
    lastFpsTime = now;
  }
}

function renderLoop() {
  if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== lastVideoTime) {
    const result = faceLandmarker.detectForVideo(video, performance.now());
    const landmarks = result.faceLandmarks?.[0];

    if (landmarks) {
      updateGeometry(landmarks);
      updateFps();
    } else {
      statusEl.textContent = "No face detected";
    }

    lastVideoTime = video.currentTime;
  }

  points.visible = isPointsVisible;
  lines.visible = isLinesVisible;
  renderer.render(scene, camera);
  requestAnimationFrame(renderLoop);
}

function resize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

function bindControls() {
  toggleVideoBtn.addEventListener("click", () => {
    video.classList.toggle("hidden");
    toggleVideoBtn.textContent = video.classList.contains("hidden") ? "Show Camera" : "Hide Camera";
  });

  toggleLinesBtn.addEventListener("click", () => {
    isLinesVisible = !isLinesVisible;
  });

  togglePointsBtn.addEventListener("click", () => {
    isPointsVisible = !isPointsVisible;
  });

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
