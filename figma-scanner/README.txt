Fit: webcam measurements for the existing Figma Make design

Use these files in your Figma Make React/Vite project, not the earlier Next.js replacements.

1. Keep a copy of your current src/App.tsx.
2. Replace src/App.tsx with App.tsx from this folder.
3. Add src/useBodyScanner.ts and src/bodyMeasurements.ts.
4. Append scanner.css to the END of existing src/index.css. Keep the original stylesheet.
5. Add dependency @mediapipe/tasks-vision with exact version 0.10.21. Preserve all other dependencies and Vite settings. With a terminal: npm install @mediapipe/tasks-vision@0.10.21
6. Preview: Start body scan -> Enable camera -> enter actual height -> full-body framing -> Capture front view -> Continue to side scan -> Capture side view -> Review -> Reveal.

Keep the laptop fixed. Use fitted clothing, bare feet if practical, a plain background, and even lighting. Front: arms diagonally down and away from torso; side: turn 90 degrees with arms down. Capture requires two seconds of positioning followed by stable samples (45 front, 30 side, at least three seconds of collection). A failed frame displays guidance; prolonged instability clears samples. Stop camera and Start over are available.

Implemented: camera permission and cleanup, MediaPipe pose plus person segmentation, largest-person mask, framing checks, stability checks, multi-frame medians, front silhouette widths, side silhouette depths, height-based scaling, ellipse circumference math, thigh estimate and optional inseam, real measurements in Review. Measurements remain in React memory and are not saved or sent to a backend.

Adapted from test.py: same front/side measurement approach, height calibration and circumference math. Changes: uses MediaPipe Full instead of Heavy for browser responsiveness, omits iris calibration, uses browser mask processing instead of OpenCV, and adds movement/outlier rejection. The height field is calibration, not manual entry of body measurements. Segmentation is resampled to camera aspect ratio to avoid distorted widths. The thigh assumption is explicitly labeled; all accuracy confidence remains low pending tape-measure validation.

Unchanged demo features: garment import, fit comparison and questions. The review silhouette remains illustrative. No encrypted persistence is claimed. Model/runtime files download from Google and jsDelivr; camera frames are processed locally without upload. No claim of tape-measure accuracy has been validated.

Verification: TypeScript checks passed against installed React and MediaPipe 0.10.21 types. Automated checks passed for ellipse math, height scaling, largest connected person region, aspect ratio, empty masks, missing values, incomplete scan rejection and off-screen framing. Webcam capture and accuracy require testing in the actual browser with a person.
