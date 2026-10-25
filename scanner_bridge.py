"""Run Test2_paper.py and write a body profile for the booth.

The OpenCV window stays on this computer. A status file and a preview frame
are written so the website can show the camera and narrate the current step.
"""
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import sys

LABELS = {
    "Bust / chest": "bust",
    "Waist": "waist",
    "Hips": "hips",
    "Inseam": "inseam",
}


def profile_from_rows(rows, height_cm, turn_done):
    measurements = {}
    for label, value, _uncertainty, _note in rows:
        key = LABELS.get(label)
        if key is None or value is None:
            continue
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value < 400:
            raise ValueError(f"Invalid measurement: {label}")
        measurements[key] = round(value, 1)
    missing = [key for key in ("bust", "waist", "hips") if key not in measurements]
    if missing:
        raise ValueError("Scanner did not produce bust, waist, and hip estimates.")
    return {
        "heightCm": height_cm,
        "measurementsCm": measurements,
        "turnDone": bool(turn_done),
        "source": "python",
        "scannedAt": datetime.now(timezone.utc).isoformat(),
    }


def main():
    root = Path(__file__).resolve().parent
    output = Path(sys.argv[1]).resolve()
    height = float(sys.argv[2])
    status_path = sys.argv[3] if len(sys.argv) > 3 else None
    frame_path = sys.argv[4] if len(sys.argv) > 4 else None
    if not 100 <= height <= 250:
        raise ValueError("Height must be 100–250 cm.")
    spec = importlib.util.spec_from_file_location("fit_desktop_scanner", root / "Test2_paper.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.POSE_PATH = str(root / "pose_landmarker_heavy.task")
    module.STATUS_PATH = status_path
    module.FRAME_PATH = frame_path
    captures = 0
    finished = False

    def save(rows, info, _resolution, _tilt, _tilt_source):
        nonlocal captures, finished
        captures += 1
        if captures < 2:
            return
        output.write_text(
            json.dumps(profile_from_rows(rows, height, bool(info["turn_done"])), allow_nan=False),
            encoding="utf-8",
        )
        finished = True

    module.save_json = save
    original_wait = module.cv2.waitKey

    def wait(delay):
        return ord("q") if finished else original_wait(delay)

    module.cv2.waitKey = wait
    sys.argv = [str(root / "Test2_paper.py"), "--height-cm", str(height), "--no-voice"]
    module.main()
    if not output.exists():
        raise RuntimeError(
            "Scan ended without measurements. Finish the front pose and the turn in the scanner window."
        )


if __name__ == "__main__":
    main()
