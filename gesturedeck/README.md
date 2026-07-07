# GestureDeck

GestureDeck is a web-based presentation player controlled by camera hand gestures.

It implements this MVP workflow:

1. Upload a `.ppt`, `.pptx`, `.odp`, or `.pdf` file.
2. The backend converts PowerPoint/OpenDocument presentations to PDF with LibreOffice headless.
3. The frontend renders the PDF with PDF.js.
4. The browser camera recognizes hand gestures with MediaPipe Hand Landmarker.
5. Stable gesture commands control the current slide.

## Gesture mapping

| Gesture | Action |
| --- | --- |
| 1 raised finger | Jump to slide 1 |
| 2 raised fingers | Jump to slide 2 |
| 3 raised fingers | Jump to slide 3 |
| 4 raised fingers | Jump to slide 4 |
| Swipe up | Previous slide |
| Swipe down | Next slide |

The gesture controller uses frame stability and cooldown protection to reduce accidental triggers.

## Project structure

```text
apps/
  api/   FastAPI upload and LibreOffice conversion service
  web/   React + Vite PDF player and MediaPipe gesture controller
```

## Run with Docker Compose

```bash
docker compose up --build
```

Open:

```text
http://localhost:5173
```

Camera access works on `localhost` or HTTPS. A plain HTTP LAN IP may not expose `navigator.mediaDevices.getUserMedia()`.

## Run locally

### Backend

Install LibreOffice first. On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y libreoffice fonts-noto-cjk
```

Then run:

```bash
cd apps/api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd apps/web
npm install
npm run dev
```

## API

```text
POST /api/presentations/upload
GET  /api/presentations/{presentation_id}
GET  /api/presentations/{presentation_id}/file.pdf
```

## MVP limitations

This version converts presentations to static PDF. It is reliable for arbitrary uploads, but it does not preserve PowerPoint animations, transitions, embedded video, embedded audio, or trigger-based interactions.

For a high-fidelity future version, integrate ONLYOFFICE or Collabora Online and let gestures control their presentation iframe or playback API.
