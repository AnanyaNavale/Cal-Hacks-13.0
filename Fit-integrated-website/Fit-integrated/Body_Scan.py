"""
body_avatar.py - Guided webcam clothing measurements (inches + cm) with a
fitted avatar overlay, using MediaPipe Pose Landmarker + Face Landmarker.

No height needed: your iris (~11.7 mm across in nearly all adults) is used as
a built-in ruler. The scale is stored as a body measurement (your shoulder
width), so you can move or tilt the camera between steps without recalibrating.

Also estimates your AGE RANGE live while you do the face and body steps (Haar face
detector + age_googlenet.onnx). Once the same range is read 10 times in a row it is
locked in, shown on screen, and saved to measurements.json.

Usage:    python body_avatar_age.py
Options:  --height-in 65 | --height-cm 165   use your height instead of the iris
          --camera 0   --hfov 68   --no-voice
          --age-model PATH   path to age_googlenet.onnx (default: next to this file)
          --blur-face        blur the detected face on screen (display only)

The app walks you through the steps automatically:
  1. Face scale - sit 2-3 ft from the camera and look straight at it.
  2. Body scan  - step back, tilt the screen if needed so head to feet are in
                  view, then hold an A-pose. Capture starts on its own.
  3. Side scan  - turn 90 deg with arms down (optional, press N to skip).

Keys: n skip step   f redo face   c redo body   s redo side   r restart
      a avatar on/off   m mask on/off   q quit
"""
import argparse
import json
import math
import os
import shutil
import ssl
import subprocess
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

POSE_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")
POSE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_heavy.task")
FACE_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task")
FACE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

IRIS_MM = 11.7     # average adult iris diameter
CM_PER_IN = 2.54
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Pose landmark indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 2, 5, 7, 8
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 23, 24, 25, 26, 27, 28
L_FT, R_FT = 31, 32
FULL_BODY = [NOSE, L_SH, R_SH, L_EL, R_EL, L_WR, R_WR,
             L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN]

# Age model (GoogLeNet ONNX) + Haar face detector
HERE = os.path.dirname(os.path.abspath(__file__))
AGE_ONNX = os.path.join(HERE, "age_googlenet.onnx")
HAAR_PATH = os.path.join(HERE, "haarcascade_frontalface_default.xml")
if not os.path.exists(HAAR_PATH):  # fall back to the copy bundled with opencv-python
    HAAR_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
AGE_MEAN = (78.4263377603, 87.7689143744, 114.895847746)
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
AGE_STABLE_FRAMES = 10   # same range this many readings in a row = confirmed
AGE_EVERY_N = 3          # run the age check on every Nth camera frame (saves CPU)

FACE_FRAMES, CALIB_FRAMES, SIDE_FRAMES = 40, 45, 30
STABLE_SEC = 1.0       # camera must be still this long before capturing
HOLD_SEC = 2.0         # hold the pose this long before a scan starts
MOTION_THRESH = 8.0    # background change that counts as camera movement
MAX_MISSES = 15        # bad frames in a row before a scan restarts
TILT_WARN_DEG = 20
MAX_PITCH_DEG = 35

CHEST_F = 0.33  # bust row, as a fraction of the shoulder->hip distance
DEFAULT_DEPTH_RATIO = {"bust": 0.75, "waist": 0.78, "hip": 0.72}
LIMB_DEPTH_RATIO = 0.95
FALLBACK_W = {"upper_arm_w": 10, "forearm_w": 8, "thigh_w": 16,
              "calf_w": 11, "neck_w": 12, "head_w": 15}  # cm, avatar only
NON_LENGTH = {"waist_f", "hip_f", "world_sh_cm", "pitch_deg"}

TITLES = {
    "face": "Step 1 of 3: Face scale - sit 2-3 ft away, look at the camera",
    "frame": "Step 2 of 3: Step back so your whole body is in view",
    "body": "Step 2 of 3: Scanning - hold still",
    "side_wait": "Step 3 of 3: Turn sideways, arms down",
    "side": "Step 3 of 3: Scanning side - hold still",
    "live": "Done! Measurements saved to measurements.json",
}
PROMPTS = {
    "face": "Step 1. Sit about two to three feet from the camera. Look straight "
            "at it, and keep both shoulders in view.",
    "frame": "Step 2. Stand up and step back about eight to ten feet. Tilt the "
             "screen if you need to, so your whole body is in view. Then hold your "
             "arms out in an A shape, with your feet apart.",
    "body": "Scanning. Hold still.",
    "side_wait": "Step 3. Turn sideways, with your arms down at your sides.",
    "side": "Scanning. Hold still.",
    "live": "All done. Your measurements are on screen.",
}
KEYS_HELP = "[n] skip [f] face [c] body [s] side [r] restart [a] avatar [m] mask [q] quit"


# ------------------------------------------------------------------ helpers
def ensure_models():
    for url, path in ((POSE_URL, POSE_PATH), (FACE_URL, FACE_PATH)):
        if not os.path.exists(path):
            print(f"Downloading {os.path.basename(path)}...")
            try:  # python.org Python on macOS often lacks root certificates
                import certifi
                ctx = ssl.create_default_context(cafile=certifi.where())
            except ImportError:
                ctx = ssl.create_default_context()
            try:
                with urllib.request.urlopen(url, context=ctx) as r, open(path, "wb") as f:
                    shutil.copyfileobj(r, f)
            except Exception as e:
                if os.path.exists(path):
                    os.remove(path)
                sys.exit(f"Could not download {url}\n  ({e})\n"
                         f"Download it in your browser and save it as:\n  {path}")


def to_in(cm):
    return cm / CM_PER_IN


def fmt(cm):
    return "--" if cm is None else f"{to_in(cm):.1f} in / {cm:.1f} cm"


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def clen(a, b, vc):
    """Pixel length of a->b with the vertical part corrected for camera tilt."""
    return float(np.hypot(b[0] - a[0], (b[1] - a[1]) * vc))


def mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def largest_blob(mask):
    """Largest connected person region, plus its top and bottom rows."""
    binm = (mask > 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binm, connectivity=8)
    if n <= 1:
        return None, None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    top = int(stats[i, cv2.CC_STAT_TOP])
    return labels == i, (top, top + int(stats[i, cv2.CC_STAT_HEIGHT]) - 1)


def run_width(blob, y, xc):
    """Width (px) of the contiguous body run on row y that contains column xc."""
    h, w = blob.shape
    y, xc = int(y), int(xc)
    if not (0 <= y < h and 0 <= xc < w) or not blob[y, xc]:
        return None
    row = blob[y]
    left_off = np.flatnonzero(~row[:xc][::-1])
    right_off = np.flatnonzero(~row[xc:])
    left = xc - (left_off[0] if left_off.size else xc)
    right = xc + (right_off[0] if right_off.size else w - xc) - 1
    return right - left + 1


def band_width(blob, y, xc, band=2):
    """Median run width over neighbouring rows (reduces mask noise)."""
    ws = [run_width(blob, y + d, xc) for d in range(-band, band + 1)]
    ws = [v for v in ws if v]
    return float(np.median(ws)) if ws else None


def perp_width(blob, p1, p2, t=0.5, max_px=400):
    """Body width perpendicular to segment p1->p2, measured at fraction t."""
    h, w = blob.shape
    p1, p2 = np.array(p1, float), np.array(p2, float)
    d = p2 - p1
    length = np.linalg.norm(d)
    if length < 1:
        return None
    n = np.array([-d[1], d[0]]) / length
    m = p1 + t * d
    mx, my = int(m[0]), int(m[1])
    if not (0 <= mx < w and 0 <= my < h) or not blob[my, mx]:
        return None

    def walk(sign):
        for s in range(1, max_px):
            x, y = (m + sign * s * n).astype(int)
            if x < 0 or y < 0 or x >= w or y >= h or not blob[y, x]:
                return s
        return max_px

    return walk(1) + walk(-1) - 1


def find_crotch(blob, x, y_start, y_stop):
    """Walk up from between the legs to the first body pixel (the crotch)."""
    h, w = blob.shape
    x, y0 = int(x), min(int(y_start), h - 1)
    if not (0 <= x < w) or blob[y0, x]:
        return None
    for y in range(y0, max(int(y_stop), 0), -1):
        if blob[y, x]:
            return y
    return None


def ellipse_perimeter(width, depth):
    """Ramanujan's approximation for an ellipse with given full width/depth."""
    a, b = width / 2, depth / 2
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def landmarks_px(result, w, h):
    lms = result.pose_landmarks[0]
    pts = [(lm.x * w, lm.y * h) for lm in lms]
    vis = [lm.visibility if lm.visibility is not None else 1.0 for lm in lms]
    return pts, vis


def torso_frame(pts):
    sh_m, hip_m = mid(pts[L_SH], pts[R_SH]), mid(pts[L_HIP], pts[R_HIP])
    torso_px = hip_m[1] - sh_m[1]
    row_y = lambda f: sh_m[1] + f * torso_px
    row_x = lambda f: sh_m[0] + min(f, 1.0) * (hip_m[0] - sh_m[0])
    return sh_m, hip_m, torso_px, row_y, row_x


def median_profile(samples):
    out = {}
    for k in samples[0]:
        vals = [s[k] for s in samples if s.get(k) is not None]
        out[k] = float(np.median(vals)) if vals else None
    return out


# ---------------------------------------------------------------------- age
class AgeEstimator:
    """Finds the face in each frame and predicts an age range. Runs alongside the
    pose/face scans; locks in a result once it is stable."""

    def __init__(self, onnx_path, need=AGE_STABLE_FRAMES):
        if not os.path.exists(onnx_path):
            sys.exit(f"Age model not found: {onnx_path}\n"
                     f"Put age_googlenet.onnx next to this script or pass --age-model PATH.")
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        self.haar = cv2.CascadeClassifier(HAAR_PATH)
        self.need = need
        self.n = 0
        self.reset()

    def reset(self):
        self.confirmed = None   # locked-in age range
        self.label = None       # latest raw reading
        self.last, self.streak = None, 0
        self.box, self.missed = None, 0

    def _find_box(self, frame, pts, vis):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.haar.detectMultiScale(gray, 1.2, 5, minSize=(40, 40))
        if len(faces):
            return tuple(int(v) for v in max(faces, key=lambda f: f[2] * f[3]))
        # Fallback for the far-away body scan: build a box from the pose landmarks
        if pts is not None and min(vis[L_EAR], vis[R_EAR], vis[NOSE]) > 0.5:
            d = dist(pts[L_EAR], pts[R_EAR])
            size = int(1.4 * d)
            if size >= 40:
                cx = (pts[L_EAR][0] + pts[R_EAR][0]) / 2
                cy = (pts[L_EAR][1] + pts[R_EAR][1] + 2 * pts[NOSE][1]) / 4
                x, y = int(cx - size / 2), int(cy - size / 2)
                H, W = frame.shape[:2]
                x, y = max(0, x), max(0, y)
                return (x, y, min(size, W - x), min(size, H - y))
        return None

    def update(self, frame, pts, vis, run_net=True):
        """Call once per camera frame. Returns True on the frame the age is locked in."""
        self.n += 1
        if self.n % AGE_EVERY_N:
            return False
        box = self._find_box(frame, pts, vis)
        if box is None:
            self.missed += 1
            if self.missed > 5:
                self.box, self.last, self.streak = None, None, 0
            return False
        self.missed, self.box = 0, box
        if not run_net or self.confirmed:
            return False

        x, y, w, h = box
        pad = 20
        y1, y2 = max(0, y - pad), min(frame.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(frame.shape[1], x + w + pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False
        try:
            blob = cv2.dnn.blobFromImage(crop, 1.0, (227, 227), AGE_MEAN,
                                         swapRB=True, crop=False)
            self.net.setInput(blob)
            age = AGE_BUCKETS[int(self.net.forward().argmax())]
        except cv2.error:
            return False

        self.label = age
        if age == self.last:
            self.streak += 1
        else:
            self.last, self.streak = age, 1
        if self.streak >= self.need:
            self.confirmed = age
            return True
        return False

    def blur(self, img):
        if self.box:
            x, y, w, h = self.box
            roi = img[y:y + h, x:x + w]
            if roi.size:
                img[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (99, 99), 30)

    def draw(self, img):
        h, w = img.shape[:2]
        s = w / 1280
        if self.box and not self.confirmed:
            x, y, bw, bh = self.box
            cv2.rectangle(img, (x, y), (x + bw, y + bh), (50, 50, 255), 1)
        if self.confirmed:
            text, col = f"Age: {self.confirmed}", (0, 255, 0)
        elif self.label:
            text, col = f"Age: {self.label}?  ({self.streak}/{self.need})", (0, 255, 255)
        else:
            text, col = "Age: looking for your face...", (200, 200, 200)
        sc, th = 0.8 * s, max(1, int(2 * s))
        tw = cv2.getTextSize(text, FONT, sc, th)[0][0]
        cv2.putText(img, text, (w - tw - 15, int(34 * s)), FONT, sc, col, th, cv2.LINE_AA)


# ------------------------------------------------- camera motion and tilt
class MotionMonitor:
    """Detects when the camera itself moves (e.g. you tilt the laptop screen)
    by watching the background, ignoring the person."""

    def __init__(self):
        self.prev, self.still_since = None, time.monotonic()

    def update(self, frame, mask):
        small = cv2.GaussianBlur(
            cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
        bg = np.ones(small.shape, bool)
        if mask is not None:
            m = cv2.resize((mask > 0.3).astype(np.uint8), (160, 90),
                           interpolation=cv2.INTER_NEAREST)
            bg = cv2.dilate(m, np.ones((9, 9), np.uint8)) == 0
        now = time.monotonic()
        if self.prev is not None and bg.sum() > 2000:
            if float(np.mean(cv2.absdiff(small, self.prev)[bg])) > MOTION_THRESH:
                self.still_since = now
        self.prev = small
        return now - self.still_since >= STABLE_SEC


def body_pitch(world, vis):
    """Camera tilt (deg), from how much your body's vertical axis leans toward or
    away from the camera in MediaPipe's 3D world landmarks."""
    def m3(a, b):
        return np.array([(world[a].x + world[b].x) / 2, (world[a].y + world[b].y) / 2,
                         (world[a].z + world[b].z) / 2])

    top = m3(L_SH, R_SH)
    bottom = m3(L_AN, R_AN) if min(vis[L_AN], vis[R_AN]) > 0.5 else m3(L_HIP, R_HIP)
    v = top - bottom
    if abs(v[1]) < 0.2:
        return None
    return float(min(math.degrees(math.atan2(abs(v[2]), abs(v[1]))), MAX_PITCH_DEG))


def framing_check(pts, vis, blob, tb, w, h):
    """Is the full body in frame and in an A-pose? Returns (ok, hint)."""
    if blob is None:
        return False, "Step into view"
    head_cut = tb[0] <= 3 or vis[NOSE] < 0.6
    feet_cut = tb[1] >= h - 4 or min(vis[L_AN], vis[R_AN]) < 0.6
    if head_cut and feet_cut:
        return False, "Step back so your whole body fits"
    if head_cut:
        return False, "Head is cut off: tilt the screen back, or step back"
    if feet_cut:
        return False, "Feet are cut off: tilt the screen forward, or step back"
    xs = np.flatnonzero(blob.any(axis=0))
    if xs[0] <= 3 or xs[-1] >= w - 4:
        return False, "Move toward the center of the frame"
    if tb[1] - tb[0] < 0.5 * h:
        return False, "Come a little closer"
    if not all(vis[i] > 0.6 for i in FULL_BODY):
        return False, "Make sure your arms and legs are clearly visible"
    sh_px = dist(pts[L_SH], pts[R_SH])
    if sh_px < 20:
        return False, "Face the camera"
    sh_y = (pts[L_SH][1] + pts[R_SH][1]) / 2
    torso = (pts[L_HIP][1] + pts[R_HIP][1]) / 2 - sh_y
    for wr, hp in ((L_WR, L_HIP), (R_WR, R_HIP)):
        if abs(pts[wr][0] - pts[hp][0]) < 0.6 * sh_px:
            return False, "Hold your arms out about 45 degrees, like an A"
        if pts[wr][1] < sh_y + 0.3 * torso:
            return False, "Lower your arms a bit, to about 45 degrees"
    if abs(pts[L_AN][0] - pts[R_AN][0]) < 0.3 * sh_px:
        return False, "Step your feet apart, about shoulder width"
    return True, ""


def side_check(pts, vis, blob, tb, h):
    if blob is None:
        return False, "Step into view"
    torso = dist(mid(pts[L_SH], pts[R_SH]), mid(pts[L_HIP], pts[R_HIP]))
    if tb[0] <= 3 or tb[1] >= h - 4 or torso < 30 or max(vis[L_HIP], vis[R_HIP]) < 0.5:
        return False, "Keep your whole body in view"
    if dist(pts[L_SH], pts[R_SH]) / torso > 0.35:
        return False, "Turn 90 degrees to your side, arms down"
    return True, ""


# -------------------------------------------------------------------- scale
def iris_sample(face_res, pts, vis, world, w, h, hfov):
    """One estimate of joint-to-joint shoulder width (cm), using the iris as a
    ruler. Iris width is measured horizontally, so screen tilt doesn't affect it."""
    if not face_res.face_landmarks:
        return None, "Look straight at the camera", None
    fl = face_res.face_landmarks[0]
    if len(fl) < 478:
        return None, "Face model returned no iris landmarks", None
    P = lambda i: (fl[i].x * w, fl[i].y * h)
    d1, d2 = dist(P(469), P(471)), dist(P(474), P(476))
    irises = [(P(468), d1), (P(473), d2)]
    if min(d1, d2) < 4:
        return None, "Move closer to the camera", irises
    if min(d1, d2) / max(d1, d2) < 0.85:
        return None, "Turn your face straight toward the camera", irises
    if not all(vis[i] > 0.6 and 0 <= pts[i][0] < w and 0 <= pts[i][1] < h
               for i in (L_SH, R_SH)):
        return None, "Keep both shoulders in view", irises

    iris_px = (d1 + d2) / 2
    f_px = (w / 2) / math.tan(math.radians(hfov / 2))
    z_mm = f_px * IRIS_MM / iris_px
    if z_mm > 1300:
        return None, "Move closer, about 2 to 3 feet away", irises
    if z_mm < 350:
        return None, "Move back a little", irises

    # Shoulders sit a few cm behind the eyes; correct for that depth difference
    dz = (world[L_SH].z + world[R_SH].z) / 2 - (world[L_EYE].z + world[R_EYE].z) / 2
    dz_mm = 1000 * float(np.clip(dz, 0.0, 0.15))
    mm_per_px = IRIS_MM / iris_px * (z_mm + dz_mm) / z_mm
    sh_cm = dist(pts[L_SH], pts[R_SH]) * mm_per_px / 10
    return sh_cm, f"Hold still - about {fmt(z_mm / 10)} away", irises


def pick_scale(height_cm, front, iris_sh_cm):
    """Returns (shoulder joint width in cm, description of where it came from)."""
    if height_cm and front:
        return height_cm / front["stature"], "your height"
    if iris_sh_cm:
        return iris_sh_cm, "iris"
    if front and front.get("world_sh_cm"):
        return front["world_sh_cm"], "body model (rough - press F)"
    return None, None


# -------------------------------------------------------------- measurement
def measure_front(pts, world, blob, top_bot, pitch_deg):
    """Front A-pose sample. Lengths are in 'shoulder units' (joint-to-joint width)."""
    vc = 1.0 / math.cos(math.radians(pitch_deg or 0.0))
    top, bot = top_bot
    sh_px = dist(pts[L_SH], pts[R_SH])
    if bot - top < 200 or sh_px < 30:
        return None
    sh_m, hip_m, torso_px, row_y, row_x = torso_frame(pts)
    if torso_px < 50:
        return None

    chest = band_width(blob, row_y(CHEST_F), row_x(CHEST_F))
    shoulder_outer = band_width(blob, row_y(-0.03), row_x(-0.03))

    waist = None  # natural waist = narrowest point
    for f in np.arange(0.5, 0.9, 0.02):
        v = band_width(blob, row_y(f), row_x(f), band=1)
        if v and (waist is None or v < waist[0]):
            waist = (v, float(f))

    hip = None  # hips = widest point at/below the hip joints
    knee_y = (pts[L_KN][1] + pts[R_KN][1]) / 2
    ankle_y = (pts[L_AN][1] + pts[R_AN][1]) / 2
    f_end = 1 + 0.35 * (knee_y - hip_m[1]) / torso_px
    for f in np.arange(0.85, f_end, 0.02):
        v = band_width(blob, row_y(f), row_x(f), band=1)
        if v and (hip is None or v > hip[0]):
            hip = (v, float(f))

    if not (chest and waist and hip and shoulder_outer):
        return None
    if chest > 1.3 * sh_px or waist[0] > 1.3 * sh_px or hip[0] > 1.6 * sh_px:
        return None  # arms probably touching the torso

    def avg_perp(a1, b1, a2, b2, t, limit):
        vals = [v for v in (perp_width(blob, pts[a1], pts[b1], t),
                            perp_width(blob, pts[a2], pts[b2], t)) if v and v < limit]
        return float(np.mean(vals)) if vals else None

    def avg_len(a1, b1, a2, b2):
        return (clen(pts[a1], pts[b1], vc) + clen(pts[a2], pts[b2], vc)) / 2

    waist_y = row_y(waist[1])
    crotch = find_crotch(blob, hip_m[0], (knee_y + ankle_y) / 2, hip_m[1] - 0.3 * torso_px)
    neck = perp_width(blob, sh_m, pts[NOSE], 0.4)
    head = band_width(blob, pts[NOSE][1], pts[NOSE][0])
    arm = avg_len(L_SH, L_EL, R_SH, R_EL) + avg_len(L_EL, L_WR, R_EL, R_WR)

    # Fallback scale from MediaPipe's 3D world landmarks (metres)
    segs = [(L_SH, R_SH), (L_SH, L_HIP), (R_SH, R_HIP), (L_HIP, L_KN),
            (R_HIP, R_KN), (L_KN, L_AN), (R_KN, R_AN)]
    w3 = sum(math.dist((world[a].x, world[a].y, world[a].z),
                       (world[b].x, world[b].y, world[b].z)) for a, b in segs)
    p2 = sum(clen(pts[a], pts[b], vc) for a, b in segs)
    world_sh_cm = 100 * w3 / p2 * sh_px if p2 > 0 else None

    su = lambda v: v / sh_px if v is not None else None
    return {
        "stature": su((bot - top + 1) * vc),
        "shoulder_outer": su(shoulder_outer),
        "chest_w": su(chest), "waist_w": su(waist[0]), "hip_w": su(hip[0]),
        "waist_f": waist[1], "hip_f": hip[1],
        "torso_len": su(torso_px * vc),
        "shoulder_to_waist": su((waist_y - sh_m[1]) * vc),
        "arm_len": su(arm),
        "inseam": su((bot - crotch) * vc) if crotch else None,
        "outseam": su((bot - waist_y) * vc),
        "upper_arm_w": su(avg_perp(L_SH, L_EL, R_SH, R_EL, 0.45, 0.5 * sh_px)),
        "forearm_w": su(avg_perp(L_EL, L_WR, R_EL, R_WR, 0.25, 0.4 * sh_px)),
        "thigh_w": su(avg_perp(L_HIP, L_KN, R_HIP, R_KN, 0.2, 0.6 * hip[0])),
        "calf_w": su(avg_perp(L_KN, L_AN, R_KN, R_AN, 0.3, 0.45 * sh_px)),
        "neck_w": su(neck) if neck and neck < 0.6 * sh_px else None,
        "head_w": su(head) if head and head < sh_px else None,
        "world_sh_cm": world_sh_cm,
        "pitch_deg": pitch_deg,
    }


def measure_side(pts, blob, top_bot, front, pitch_deg):
    """Side sample. Depths are stored as fractions of your (tilt-corrected) height."""
    vc = 1.0 / math.cos(math.radians(pitch_deg or 0.0))
    top, bot = top_bot
    stature_px = (bot - top + 1) * vc
    if stature_px < 200:
        return None
    _, _, torso_px, row_y, row_x = torso_frame(pts)
    if torso_px < 50:
        return None

    def scan(f0, f1, pick_max):
        best = None
        for f in np.arange(f0, f1, 0.02):
            d = band_width(blob, row_y(f), row_x(f), band=1)
            if d and (best is None or (d > best if pick_max else d < best)):
                best = d
        return best

    bust = scan(0.15, 0.55, True)
    waist = scan(front["waist_f"] - 0.08, front["waist_f"] + 0.08, False)
    hip = scan(front["hip_f"] - 0.1, front["hip_f"] + 0.15, True)
    if not (bust and waist and hip):
        return None
    return {"bust_d": bust / stature_px, "waist_d": waist / stature_px,
            "hip_d": hip / stature_px}


def compute_results(front, side, shoulder_cm):
    r = {}
    for k, v in front.items():
        if k in NON_LENGTH:
            r[k] = v
        else:
            r[k] = v * shoulder_cm if v is not None else None
    r["shoulder_joint"] = shoulder_cm
    if side:
        for k, v in side.items():
            r[k] = v * r["stature"]

    circ = {}
    for name, wk in (("bust", "chest_w"), ("waist", "waist_w"), ("hip", "hip_w")):
        w = r[wk]
        d = r.get(f"{name}_d")
        estimated = d is None
        if estimated:
            d = w * DEFAULT_DEPTH_RATIO[name]
        circ[name] = (ellipse_perimeter(w, d), estimated)
    for name, wk in (("neck", "neck_w"), ("upper_arm", "upper_arm_w"),
                     ("thigh", "thigh_w"), ("calf", "calf_w")):
        w = r.get(wk)
        circ[name] = (ellipse_perimeter(w, w * LIMB_DEPTH_RATIO) if w else None, False)
    r["circ"] = circ
    r["rise"] = (r["outseam"] - r["inseam"]
                 if r.get("inseam") and r.get("outseam") else None)
    return r


def clothing_rows(r):
    c = r["circ"]
    return [
        ("Height (est.)", r["stature"], False),
        ("Bust / chest", *c["bust"]),
        ("Waist", *c["waist"]),
        ("Hips", *c["hip"]),
        ("Shoulder width", r["shoulder_outer"], False),
        ("Arm / sleeve length", r["arm_len"], False),
        ("Inseam", r["inseam"], False),
        ("Outseam (waist-floor)", r["outseam"], False),
        ("Rise", r["rise"], False),
        ("Shoulder to waist", r["shoulder_to_waist"], False),
        ("Neck", *c["neck"]),
        ("Upper arm", *c["upper_arm"]),
        ("Thigh", *c["thigh"]),
        ("Calf", *c["calf"]),
    ]


def print_table(rows, src, tilt):
    print(f"\nMeasurements (scale from {src}, camera tilt ~{tilt or 0:.0f} deg):")
    for label, v, est in rows:
        print(f"  {label:<24}{fmt(v)}{'  (est. - do the side scan)' if est else ''}")


def save_json(rows, src, tilt, age=None):
    data = {"age_range": age,
            "scale_source": src,
            "camera_tilt_deg": round(tilt, 1) if tilt is not None else None,
            "measurements": {
                label: None if v is None else
                {"in": round(to_in(v), 1), "cm": round(v, 1), "estimated_depth": est}
                for label, v, est in rows}}
    with open("measurements.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Saved measurements.json")


# ------------------------------------------------------------------- avatar
def capsule(img, p1, p2, r, color):
    p1 = (int(p1[0]), int(p1[1]))
    p2 = (int(p2[0]), int(p2[1]))
    r = max(2, int(r))
    cv2.line(img, p1, p2, color, 2 * r, cv2.LINE_AA)
    cv2.circle(img, p1, r, color, -1, cv2.LINE_AA)
    cv2.circle(img, p2, r, color, -1, cv2.LINE_AA)


def draw_avatar(frame, pts, vis, r, vc, alpha=0.55):
    sh_m, hip_m = mid(pts[L_SH], pts[R_SH]), mid(pts[L_HIP], pts[R_HIP])
    torso_img = dist(sh_m, hip_m)
    if torso_img < 20 or not r.get("torso_len"):
        return frame
    # Live scale from your torso (tilt-corrected): works at any distance or seated
    cmpp = r["torso_len"] / clen(sh_m, hip_m, vc)
    px = lambda cm: cm / cmpp
    g = lambda k: r.get(k) or FALLBACK_W[k]
    ok = lambda *ids: all(vis[i] > 0.5 for i in ids)

    # Body rotation: shoulders look narrower when you turn
    cos_t = float(np.clip(dist(pts[L_SH], pts[R_SH]) * cmpp / r["shoulder_joint"], 0, 1))
    sin_t = math.sqrt(1 - cos_t ** 2)
    proj = lambda w, d: math.sqrt((w * cos_t) ** 2 + (d * sin_t) ** 2)

    def level(name, wk):
        w = r[wk]
        d = r.get(f"{name}_d") or w * DEFAULT_DEPTH_RATIO[name]
        return proj(w, d)

    overlay = frame.copy()
    skin, body, limb = (180, 200, 255), (255, 160, 60), (230, 140, 50)

    for hp, kn, an, ft in ((L_HIP, L_KN, L_AN, L_FT), (R_HIP, R_KN, R_AN, R_FT)):
        if ok(hp, kn):
            capsule(overlay, pts[hp], pts[kn], px(g("thigh_w")) / 2, limb)
        if ok(kn, an):
            capsule(overlay, pts[kn], pts[an], px(g("calf_w")) / 2, limb)
        if ok(an, ft):
            capsule(overlay, pts[an], pts[ft], px(g("calf_w")) * 0.35, limb)

    u = np.array([hip_m[0] - sh_m[0], hip_m[1] - sh_m[1]]) / torso_img
    n = np.array([-u[1], u[0]])
    bust_d = r.get("bust_d") or r["chest_w"] * DEFAULT_DEPTH_RATIO["bust"]
    sh_outer = proj(r["shoulder_outer"], bust_d)
    levels = sorted([(-0.06, sh_outer * 0.85), (0.04, sh_outer),
                     (CHEST_F, level("bust", "chest_w")),
                     (r["waist_f"], level("waist", "waist_w")),
                     (r["hip_f"], level("hip", "hip_w")),
                     (max(r["hip_f"], 1.0) + 0.08, level("hip", "hip_w") * 0.95)])
    left, right = [], []
    for f, wcm in levels:
        c = np.array(sh_m) + u * f * torso_img
        half = px(wcm) / 2
        left.append(c + n * half)
        right.append(c - n * half)
    cv2.fillPoly(overlay, [np.array(left + right[::-1], dtype=np.int32)], body, cv2.LINE_AA)

    for sh, el, wr in ((L_SH, L_EL, L_WR), (R_SH, R_EL, R_WR)):
        if ok(sh, el):
            capsule(overlay, pts[sh], pts[el], px(g("upper_arm_w")) / 2, limb)
        if ok(el, wr):
            capsule(overlay, pts[el], pts[wr], px(g("forearm_w")) / 2, limb)

    if ok(NOSE, L_EAR, R_EAR):
        ear_m = mid(pts[L_EAR], pts[R_EAR])
        head_w = px(g("head_w"))
        head_h = head_w * 1.3
        capsule(overlay, sh_m, (ear_m[0], ear_m[1] + 0.25 * head_h), px(g("neck_w")) / 2, skin)
        angle = math.degrees(math.atan2(pts[L_EAR][1] - pts[R_EAR][1],
                                        pts[L_EAR][0] - pts[R_EAR][0]))
        center = (int(ear_m[0]), int(ear_m[1] - 0.12 * head_h))
        cv2.ellipse(overlay, center, (int(head_w / 2), int(head_h / 2)),
                    angle, 0, 360, skin, -1, cv2.LINE_AA)

    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


# ----------------------------------------------------------------------- UI
class Voice:
    """Reads prompts aloud with macOS 'say' so you can follow from across the room."""

    def __init__(self, enabled):
        self.cmd = shutil.which("say") if enabled else None
        self.proc, self.last, self.last_t = None, None, 0.0

    def say(self, text, interrupt=True, min_gap=0.0):
        if not self.cmd or not text or text == self.last:
            return
        now = time.monotonic()
        if now - self.last_t < min_gap:
            return
        if self.proc and self.proc.poll() is None:
            if not interrupt:
                return
            self.proc.terminate()
        self.proc = subprocess.Popen([self.cmd, text])
        self.last, self.last_t = text, now

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()


def wrap(text, scale, thick, max_w):
    lines, cur = [], ""
    for word in text.split():
        t = (cur + " " + word).strip()
        if cur and cv2.getTextSize(t, FONT, scale, thick)[0][0] > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = t
    if cur:
        lines.append(cur)
    return lines


def draw_banner(img, title, hint, progress):
    """Large text at the bottom of the screen, readable from a distance."""
    h, w = img.shape[:2]
    s = w / 1280
    ts, tt = 1.0 * s, max(2, int(2.5 * s))
    hs, ht = 0.8 * s, max(1, int(2 * s))
    lines = [(t, ts, tt, (0, 255, 255)) for t in wrap(title, ts, tt, w - 40)]
    if hint:
        lines += [(t, hs, ht, (255, 255, 255)) for t in wrap(hint, hs, ht, w - 40)]
    lh = int(42 * s)
    bar_h = int(14 * s) if progress is not None else 0
    y0 = max(0, h - (lh * len(lines) + bar_h + int(30 * s)))
    roi = img[y0:h, :]
    roi[:] = (roi * 0.3).astype(np.uint8)
    y = y0 + lh
    for text, sc, th, col in lines:
        tw = cv2.getTextSize(text, FONT, sc, th)[0][0]
        cv2.putText(img, text, ((w - tw) // 2, y), FONT, sc, col, th, cv2.LINE_AA)
        y += lh
    if progress is not None:
        x0, x1 = int(w * 0.2), int(w * 0.8)
        top = y - lh + int(14 * s)
        cv2.rectangle(img, (x0, top), (x1, top + bar_h), (90, 90, 90), -1)
        cv2.rectangle(img, (x0, top), (x0 + int((x1 - x0) * min(progress, 1.0)), top + bar_h),
                      (0, 200, 0), -1)


def draw_panel(img, rows, src, tilt):
    s = img.shape[1] / 1280
    lines = [f"Scale from: {src}"]
    lines += [f"{label}: {fmt(v)}{' (est.)' if est else ''}" for label, v, est in rows]
    if tilt is not None:
        lines.append(f"Camera tilt: ~{tilt:.0f} deg")
    lh, sc = int(24 * s), 0.55 * s
    roi = img[0:min(lh * len(lines) + 12, img.shape[0]), 0:min(int(560 * s), img.shape[1])]
    roi[:] = (roi * 0.35).astype(np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, int(22 * s) + lh * i), FONT, sc,
                    (255, 255, 255), 1, cv2.LINE_AA)


def draw_face_guide(img):
    h, w = img.shape[:2]
    cv2.ellipse(img, (w // 2, int(h * 0.38)), (int(w * 0.08), int(h * 0.2)),
                0, 0, 360, (0, 255, 255), 2, cv2.LINE_AA)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Guided webcam clothing measurements + avatar")
    hg = ap.add_mutually_exclusive_group()
    hg.add_argument("--height-in", type=float, help="optional: your height in inches")
    hg.add_argument("--height-cm", type=float, help="optional: your height in cm")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--hfov", type=float, default=68.0,
                    help="camera horizontal field of view in degrees (small correction only)")
    ap.add_argument("--no-voice", action="store_true", help="don't read prompts aloud")
    ap.add_argument("--age-model", default=AGE_ONNX, help="path to age_googlenet.onnx")
    ap.add_argument("--stay", action="store_true",
                    help="keep the live avatar view open after scanning (default: exit when done)")
    ap.add_argument("--blur-face", action="store_true", help="blur the detected face on screen")
    args = ap.parse_args()
    height_cm = args.height_cm or (args.height_in * CM_PER_IN if args.height_in else None)

    ensure_models()
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_PATH),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.6, min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6, output_segmentation_masks=True))
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_PATH),
        running_mode=vision.RunningMode.VIDEO, num_faces=1))

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    voice = Voice(not args.no_voice)
    motion = MotionMonitor()
    age_est = AgeEstimator(args.age_model)

    state = None
    face_s, front_s, side_s = [], [], []
    front = side = iris_sh_cm = None
    results = rows = src = None
    pitch = None
    hold_t, misses, live_t = None, 0, None
    show_avatar, show_mask = True, False
    t0, last_ts = time.perf_counter(), -1

    def go(new):
        nonlocal state, hold_t, misses, live_t
        state, hold_t, misses, live_t = new, None, 0, None
        if new == "face":
            face_s.clear()
        elif new == "body":
            front_s.clear()
        elif new == "side":
            side_s.clear()
        voice.say(PROMPTS[new])

    def refresh():
        nonlocal results, rows, src
        scale, s = pick_scale(height_cm, front, iris_sh_cm)
        if front is None or scale is None:
            return
        results, src = compute_results(front, side, scale), s
        rows = clothing_rows(results)
        print_table(rows, src, front.get("pitch_deg"))
        save_json(rows, src, front.get("pitch_deg"), age_est.confirmed)

    def after_face():
        go("frame" if front is None else "live")

    def after_body():
        go("side_wait" if side is None else "live")

    go("frame" if height_cm else "face")

    while True:
        ok_, frame = cap.read()
        if not ok_:
            print("Camera read failed (check camera permissions).")
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        ts = max(int((time.perf_counter() - t0) * 1000), last_ts + 1)
        last_ts = ts
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = pose.detect_for_video(mp_img, ts)
        now = time.monotonic()
        out = frame.copy()
        hint, progress = "", None

        person = bool(res.pose_landmarks)
        mask = blob = tb = None
        if person:
            pts, vis = landmarks_px(res, w, h)
            world = res.pose_world_landmarks[0]
            if res.segmentation_masks:
                mask = np.squeeze(res.segmentation_masks[0].numpy_view())
                blob, tb = largest_blob(mask)
            if state != "face" and max(vis[L_HIP], vis[R_HIP]) > 0.5:
                p = body_pitch(world, vis)
                if p is not None:
                    pitch = p if pitch is None else 0.85 * pitch + 0.15 * p
        vc = 1.0 / math.cos(math.radians(pitch or 0.0))
        stable = motion.update(frame, mask)

        # ---------------- age (runs during the face and body steps)
        if age_est.confirmed is None or args.blur_face:
            scanning = state in ("face", "frame", "body", "live")
            if age_est.update(frame, pts if person else None, vis if person else None,
                              run_net=scanning):
                print(f"Age range confirmed: {age_est.confirmed}")
                voice.say(f"Age range confirmed: {age_est.confirmed}", interrupt=False)
                if front is not None:
                    refresh()  # re-save measurements.json with the age included
        if args.blur_face:
            age_est.blur(out)

        # ---------------- step 1: face scale
        if state == "face":
            draw_face_guide(out)
            if not stable:
                face_s.clear()
                hint = "Camera moving - hold still"
            elif not person:
                hint = "Sit in front of the camera"
            else:
                fres = face.detect_for_video(mp_img, ts)
                smp, hint, irises = iris_sample(fres, pts, vis, world, w, h, args.hfov)
                for c, d in irises or []:
                    cv2.circle(out, (int(c[0]), int(c[1])), max(1, int(d / 2)),
                               (0, 255, 255), 1, cv2.LINE_AA)
                if smp:
                    face_s.append(smp)
                else:
                    voice.say(hint, interrupt=False, min_gap=4)
            progress = len(face_s) / FACE_FRAMES
            if len(face_s) >= FACE_FRAMES:
                iris_sh_cm = float(np.median(face_s))
                print(f"Scale captured (shoulder joints {fmt(iris_sh_cm)}).")
                refresh()
                after_face()

        # ---------------- step 2: framing + body scan
        elif state in ("frame", "body"):
            good, hint = (framing_check(pts, vis, blob, tb, w, h) if person
                          else (False, "Step into view"))
            if good and not stable:
                good, hint = False, "Camera moving - hold still"
            if state == "frame":
                if good:
                    hold_t = hold_t or now
                    progress = (now - hold_t) / HOLD_SEC
                    hint = "Perfect - hold that pose"
                    voice.say("Perfect, hold that pose.", interrupt=False)
                    if progress >= 1:
                        go("body")
                else:
                    hold_t = None
                    voice.say(hint, interrupt=False, min_gap=4)
            else:
                smp = measure_front(pts, world, blob, tb, pitch) if good else None
                if smp:
                    front_s.append(smp)
                    misses = 0
                    hint = "Hold still"
                else:
                    misses += 1
                    if misses > MAX_MISSES:
                        go("frame")
                progress = len(front_s) / CALIB_FRAMES
                if state == "body" and len(front_s) >= CALIB_FRAMES:
                    front = median_profile(front_s)
                    refresh()
                    after_body()

        # ---------------- step 3: side scan
        elif state in ("side_wait", "side"):
            good, hint = (side_check(pts, vis, blob, tb, h) if person
                          else (False, "Step into view"))
            if good and not stable:
                good, hint = False, "Camera moving - hold still"
            if state == "side_wait":
                if good:
                    hold_t = hold_t or now
                    progress = (now - hold_t) / HOLD_SEC
                    hint = "Good - hold still"
                    if progress >= 1:
                        go("side")
                else:
                    hold_t = None
                    voice.say(hint, interrupt=False, min_gap=4)
                hint += "   (press N to skip)"
            else:
                smp = measure_side(pts, blob, tb, front, pitch) if good else None
                if smp:
                    side_s.append(smp)
                    misses = 0
                    hint = "Hold still"
                else:
                    misses += 1
                    if misses > MAX_MISSES:
                        go("side_wait")
                progress = len(side_s) / SIDE_FRAMES
                if state == "side" and len(side_s) >= SIDE_FRAMES:
                    side = median_profile(side_s)
                    refresh()
                    go("live")

        # ---------------- live avatar
        elif state == "live":
            hint = KEYS_HELP
            live_t = live_t or now
            # Auto-exit: measurements are saved; give the age a moment to lock in
            if (not args.stay and rows and now - live_t >= 3
                    and (age_est.confirmed or now - live_t >= 10)):
                print("\n==============================")
                print("SCAN COMPLETE")
                print(f"Age range: {age_est.confirmed or 'not confirmed'}")
                print("Measurements saved to measurements.json")
                print("==============================\n")
                break
            if pitch is not None and pitch > TILT_WARN_DEG:
                hint = (f"Camera tilted ~{pitch:.0f} deg: raise the laptop toward waist "
                        f"height for best accuracy.  " + hint)

        if show_mask and mask is not None:
            sel = mask > 0.5
            out[sel] = (out[sel] * 0.5 + np.array([0, 120, 0])).astype(np.uint8)
        if show_avatar and results and state == "live" and person:
            out = draw_avatar(out, pts, vis, results, vc)
        if rows and state == "live":
            draw_panel(out, rows, src, front.get("pitch_deg") if front else None)
        age_est.draw(out)
        draw_banner(out, TITLES[state], hint, progress)
        cv2.imshow("Body Avatar", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):
            if state == "face":
                after_face()
            elif state in ("side_wait", "side"):
                go("live")
            elif state in ("frame", "body") and front is not None:
                after_body()
        elif key == ord("f"):
            go("face")
        elif key == ord("c"):
            go("frame")
        elif key == ord("s"):
            if front:
                go("side_wait")
            else:
                print("Do the body scan first.")
        elif key == ord("r"):
            front = side = iris_sh_cm = None
            results = rows = src = None
            age_est.reset()
            go("frame" if height_cm else "face")
        elif key == ord("a"):
            show_avatar = not show_avatar
        elif key == ord("m"):
            show_mask = not show_mask

    voice.close()
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    face.close()


if __name__ == "__main__":
    main()