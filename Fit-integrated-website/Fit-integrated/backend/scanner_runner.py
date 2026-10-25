"""Run the unchanged Test2_paper desktop scanner and return a browser profile.

The process owns the local camera. Frames never travel over HTTP.
"""
import importlib.util
import json
import math
from pathlib import Path
import sys
from datetime import datetime, timezone

LABELS = {'Bust / chest':'bust', 'Waist':'waist', 'Hips':'hip', 'Thigh':'thigh', 'Shoulder width':'shoulder', 'Inseam':'inseam'}

def normalize_rows(rows, height_cm, turn_done):
    measurements = {}
    for label, value, uncertainty, note in rows:
        if label in LABELS and value is not None:
            if not isinstance(value, (int,float)) or not math.isfinite(value) or not 0 < value < 400:
                raise ValueError(f'Invalid measurement: {label}')
            measurements[LABELS[label]] = round(value,1)
    if not all(k in measurements for k in ('bust','waist','hip')):
        raise ValueError('Scanner did not produce bust, waist and hip estimates.')
    return {'heightCm':height_cm,'measurementsCm':measurements,
        'confidence':{k:'low' for k in measurements},
        'source':'python','scannedAt':datetime.now(timezone.utc).isoformat(),
        'notes':['Captured with the original Test2_paper.py scanner on this computer.',
                 'Experimental estimates; accuracy has not been validated against a tape measure.',
                 'Turn scan completed.' if turn_done else 'Turn incomplete: torso circumferences use assumed depth ratios.',
                 'Thigh circumference uses an assumed depth ratio.']}

def main():
    root=Path(__file__).resolve().parents[1]
    output=Path(sys.argv[1]).resolve()
    height=float(sys.argv[2])
    if not 100 <= height <= 250:
        raise ValueError('Height must be 100–250 cm.')
    spec=importlib.util.spec_from_file_location('fit_desktop_scanner',root/'Test2_paper.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.POSE_PATH=str(root/'pose_landmarker_heavy.task')
    captures=0
    finished=False
    def save(rows, info, resolution, tilt, tilt_source):
        nonlocal captures,finished
        captures += 1
        # The first result is front-only; wait for the turn or its explicit skip.
        if captures < 2:
            return
        profile=normalize_rows(rows,height,bool(info['turn_done']))
        output.write_text(json.dumps(profile,allow_nan=False),encoding='utf-8')
        finished=True
    module.save_json=save
    original_wait=module.cv2.waitKey
    def wait(delay):
        return ord('q') if finished else original_wait(delay)
    module.cv2.waitKey=wait
    sys.argv=[str(root/'Test2_paper.py'),'--height-cm',str(height),'--no-voice']
    module.main()
    if not output.exists():
        raise RuntimeError('Scan ended without measurements. Complete the front and turn steps before closing the window.')

if __name__ == '__main__':
    main()
