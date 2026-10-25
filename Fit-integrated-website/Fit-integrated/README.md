# Fit

A runnable website combining the Figma Make interface, a private browser scanner adapted from test.py, and a local connection to the original Test2_paper.py scanner.

## Start on Windows

1. Install Node.js LTS and Python 3.11 or 3.12 if needed.
2. Extract the complete project, including the model files in the root folder.
3. Double-click **Start Fit.cmd**. The first run installs dependencies and builds the website.
4. Open **http://localhost:8765** in Chrome or Edge. Keep the command window open.

Alternatively, from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
cd figma-web
npm.cmd install
npm.cmd run build
cd ..
.\.venv\Scripts\python.exe run_fit.py
```

If port 8765 is occupied, run `python run_fit.py --port 8766` using the same Python environment and open localhost:8766.

## Scan your body

**Browser scanner (primary):** Start body scan -> Enable camera -> enter height -> frame full body -> capture front -> continue to side -> capture side -> review -> Reveal. Keep the laptop fixed. Front: arms diagonally down, away from the torso, feet apart. Side: turn 90 degrees with arms down. Capture needs two seconds of stable positioning followed by multiple valid samples. Poor lighting or framing blocks capture rather than substituting sample numbers.

**Original Python scanner (local alternative):** On the scan screen, enter height in the Original Python scanner panel and choose Use Python scanner. The browser camera stops, and Test2_paper.py opens its own camera window on this computer. Follow its front A-pose and hands-on-head full-turn instructions. Its results return automatically to the website after the turn completes. Close/cancel aborts the scan. This requires the local server and Python packages; it does not run on a hosted static website.

No age prediction is performed by the integrated website. The other Python prototypes are preserved as references.

## Compare a garment

After review, add a garment name and paste its actual garment dimensions as CSV in centimeters:

```csv
size,bust,waist,hip
S,95,75,100
M,106,86,110
L,120,100,124
```

These numbers are SYNTHETIC examples for testing, not a real seller's chart. Replace them with verified product dimensions. Omit unavailable columns. Recommended body-measurement charts are not garment dimensions. Product URL is currently a reference only; automatic fetching is not implemented.

Size tabs update the fit zones from garment minus body measurements. The recommendation uses a transparent non-stretch heuristic and never invents missing chart measurements. Click a body-area marker to ask about that zone. Guidance is deterministic and local, not Claude-generated. Stretch, fabric behavior, back gaping and movement are not inferred.

## Where the functions live

- `figma-web/src/App.tsx`: original Figma layout, connected screens and shared state.
- `figma-web/src/useBodyScanner.ts`: browser camera inference and capture state.
- `figma-web/src/bodyMeasurements.ts`: silhouette/front/side geometry adapted from test.py.
- `figma-web/src/PythonScanner.tsx`: local Python scan start/status/cancel UI.
- `backend/scanner_runner.py`: runs the original Test2_paper.py, intercepts its final results, and closes its camera window after completion.
- `run_fit.py`: loopback-only Python server serving the website and scanner endpoints.
- `figma-web/src/fitEngine.ts`: chart validation, fit scoring and zone explanations.

The model and WASM runtime are copied locally during build; no external inference service receives camera images. The original Python scanner files are unchanged. Browser scanning uses test.py's height-calibrated front/side ellipse approach, not Test2_paper.py's full-turn Cauchy method. The Python option runs that original full-turn method directly.

## Privacy and limits

Browser frames stay in the browser. Python-mode frames stay in the local Python process; no frame upload is implemented. Measurements live in UI memory; the local server temporarily holds a Python result until cancellation, replacement, or shutdown. Delete profile clears both stores when the server is available. There is no encrypted persistent profile or login.

Camera estimates are experimental and have not been calibrated against tape measurements. Inseam may be missing; thigh uses an assumed depth ratio. The silhouette is an illustration, not a personalized 3D reconstruction. Test on multiple people and varied lighting before making accuracy claims.

Automatic Amazon/Apify import, Convex persistence, Claude answers, and production deployment are not implemented. Static deployment supports the browser scanner; original Python scanning requires running the local companion on the shopper's computer. A cloud server webcam would belong to the server, not the shopper.

## Development and checks

```powershell
cd figma-web
npm.cmd run dev
npm.cmd run typecheck
npm.cmd test
npm.cmd run build
cd ..
python -m unittest discover -s backend/tests -v
```

Vite serves the frontend on port 8443 during development. Use `run_fit.py` on 8765 for Python integration. Rebuild after editing before testing the integrated Python server.

Checks include geometry/scaling, person-mask components and aspect ratio, chart validation, missing-zone behavior, recommendation changes, backend session/origin checks and Python-result conversion. Webcam accuracy still requires a real-person test.
