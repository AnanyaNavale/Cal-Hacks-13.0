"""
age_and_body_scan.py - Stage 1: webcam face scan -> confirmed age range.
Stage 2: webcam body measurements (in + cm) using your height as the scale.
The program exits on its own once both stages have produced their results.
MediaPipe Pose Landmarker (Tasks API) provides landmarks and segmentation.

Usage: python age_and_body_scan.py [--camera 0] [--face-cascade PATH] [--age-model PATH] [--hfov 68] [--tilt-deg N] [--max-width 1920]
                              [--height-cm N | --height-in N] [--no-voice]

Steps:
  0. Start page: type your height in inches or cm, press Enter.
     (--height-cm / --height-in skips this page.)
  1. Front: barefoot or in socks, whole body in view, A-pose: arms ~45 deg out, feet apart.
  2. Turn: hands on your head, turn slowly in place, one full circle.

Start page keys: digits . '  [I] inches  [C] cm  [Tab] switch  [Backspace]  [Enter] start  [Q]/[Esc] quit
Scan keys:       n skip turn  r restart  l tilt lines  a avatar  m mask  d debug  q quit
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

POSE_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
            "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")
POSE_PATH = "pose_landmarker_heavy.task"
CM_PER_IN = 2.54
FONT = cv2.FONT_HERSHEY_SIMPLEX
WINDOW = "Body Measure"

# Pose landmarks
NOSE, L_EAR, R_EAR = 0, 7, 8
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 23, 24, 25, 26, 27, 28
L_HEEL, R_HEEL, L_FT, R_FT = 29, 30, 31, 32
FEET = (L_AN, R_AN, L_HEEL, R_HEEL, L_FT, R_FT)
CORE = [NOSE, L_SH, R_SH, L_HIP, R_HIP, L_KN, R_KN]
FULL_BODY = CORE + [L_EL, R_EL, L_WR, R_WR]

# Height entry
HEIGHT_MIN_CM, HEIGHT_MAX_CM = 90.0, 250.0
HEIGHT_UNC_CM = 1.0               # how accurately people usually know their height

# Capture
FRONT_FRAMES = 45
TURN_MIN_SEC, TURN_MAX_SEC = 6.0, 40.0
YAW_BINS, MIN_PER_BIN, MIN_BINS_FILLED = 12, 2, 9   # 15 deg bins over 180 deg
STABLE_SEC, HOLD_SEC = 1.0, 2.0
MOTION_SHIFT_PX = 0.6             # whole-image shift (at 320 px wide) = camera moved
FRONT_LOST_SEC = 2.0              # pose lost this long during scanning -> back to waiting
TILT_WARN_DEG = 15
VP_EVERY, VP_HISTORY, VP_MIN_LINES, VP_INLIER_DEG, VP_MAX_TILT = 5, 15, 4, 1.5, 40

# Body model
CHEST_F = 0.33
DEPTH_RATIO = {"bust": 0.75, "waist": 0.78, "hip": 0.72}
LIMB_DEPTH_RATIO = 0.95
FALLBACK_W = {"upper_arm_w": 10, "forearm_w": 8, "thigh_w": 16,
              "calf_w": 11, "neck_w": 12, "head_w": 15}
SYS_LEN, SYS_WIDTH, SYS_CIRC, SYS_EST_CIRC = 0.003, 0.01, 0.015, 0.04
LANDMARK_ABS_CM = 1.2             # joint-based lengths (sleeve, inseam, rise)

DEBUG = {}                        # name -> (value, passed), drawn on screen

STEPS = {  # state: (on-screen title, spoken prompt)
    "height": ("Enter your height",
               "Type your height, then press Enter. Press I for inches, or C for centimeters."),
    "front_wait": ("Step 1 of 2: Stand back, arms out in an A shape",
                   "Step 1. Take your shoes off, and stand about nine feet back, so your whole "
                   "body, including your feet, is in view. Hold your arms out in an A shape, "
                   "with your feet apart."),
    "front": ("Step 1 of 2: Scanning - hold still", "Scanning. Hold still."),
    "turn_wait": ("Step 2 of 2: Hands on your head, stay on the same spot",
                  "Step 2. Stay on the same spot, and put both hands on your head."),
    "turn": ("Step 2 of 2: Turn slowly in a full circle", "Now turn slowly in place, one full circle."),
    "live": ("Done! Measurements saved to measurements.json",
             "All done. Your measurements are on screen."),
}
KEYS_HELP = "[n] skip turn [r] restart [l] lines [a] avatar [m] mask [d] debug [q] quit"


# ------------------------------------------------------------------ helpers
def ensure_model():
    if not os.path.exists(POSE_PATH):
        print("Downloading pose model...")
        urllib.request.urlretrieve(POSE_URL, POSE_PATH)


def to_in(cm):
    return cm / CM_PER_IN


def fmt_pm(v, u):
    if v is None:
        return "--"
    if u is None:
        return f"{to_in(v):.1f} in / {v:.1f} cm"
    return f"{to_in(v):.1f} +/- {to_in(u):.1f} in / {v:.1f} +/- {u:.1f} cm"


def fmt_height(v, u=None):
    ft, inch = divmod(to_in(v), 12)
    if u is None:
        return f"{int(ft)}' {inch:.1f}\" / {v:.1f} cm"
    return f"{int(ft)}' {inch:.1f}\" +/- {to_in(u):.1f} in / {v:.1f} +/- {u:.1f} cm"


def dist(a, b):
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def clen(a, b, vc):
    """Pixel length a->b with the vertical part corrected for camera tilt."""
    return float(math.hypot(b[0] - a[0], (b[1] - a[1]) * vc))


def tilt_factor(tilt_deg):
    return 1.0 / math.cos(math.radians(abs(tilt_deg or 0.0)))


def foot_margin(pts):
    return max(8, 0.15 * dist(pts[L_SH], pts[R_SH]))


def robust_stats(a):
    """Median and its noise (neighbouring frames are correlated, so n_eff = n/5)."""
    a = np.asarray(a, float)
    m = float(np.median(a))
    return m, 1.4826 * float(np.median(np.abs(a - m))) / math.sqrt(max(1.0, len(a) / 5))


def landmarks_px(result, w, h):
    lms = result.pose_landmarks[0]
    return ([(lm.x * w, lm.y * h) for lm in lms],
            [lm.visibility if lm.visibility is not None else 1.0 for lm in lms])


def person_blob(mask, pts):
    """Mask component containing the pose landmarks (not just the largest one)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0.5).astype(np.uint8), connectivity=8)
    if n <= 1:
        return None, None
    h, w = labels.shape
    votes = [labels[int(y), int(x)] for x, y in (pts[i] for i in CORE)
             if 0 <= x < w and 0 <= y < h and labels[int(y), int(x)] > 0]
    i = max(set(votes), key=votes.count) if votes else 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    top = int(stats[i, cv2.CC_STAT_TOP])
    return labels == i, (top, top + int(stats[i, cv2.CC_STAT_HEIGHT]) - 1)


def run_width(blob, y, xc):
    h, w = blob.shape
    y, xc = int(y), int(xc)
    if not (0 <= y < h and 0 <= xc < w) or not blob[y, xc]:
        return None
    row = blob[y]
    lo, ro = np.flatnonzero(~row[:xc][::-1]), np.flatnonzero(~row[xc:])
    left = xc - (lo[0] if lo.size else xc)
    right = xc + (ro[0] if ro.size else w - xc) - 1
    return right - left + 1


def band_width(blob, y, xc, band=2):
    ws = [v for v in (run_width(blob, y + d, xc) for d in range(-band, band + 1)) if v]
    return float(np.median(ws)) if ws else None


def perp_width(blob, p1, p2, t=0.5, max_px=400):
    h, w = blob.shape
    p1, p2 = np.array(p1, float), np.array(p2, float)
    d = p2 - p1
    length = np.linalg.norm(d)
    if length < 1:
        return None
    n, m = np.array([-d[1], d[0]]) / length, p1 + t * d
    if not (0 <= int(m[0]) < w and 0 <= int(m[1]) < h) or not blob[int(m[1]), int(m[0])]:
        return None

    def walk(sign):
        for s in range(1, max_px):
            x, y = (m + sign * s * n).astype(int)
            if x < 0 or y < 0 or x >= w or y >= h or not blob[y, x]:
                return s
        return max_px

    return walk(1) + walk(-1) - 1


def find_crotch(blob, x, y_start, y_stop):
    h, w = blob.shape
    x, y0 = int(x), min(int(y_start), h - 1)
    if not (0 <= x < w) or blob[y0, x]:
        return None
    return next((y for y in range(y0, max(int(y_stop), 0), -1) if blob[y, x]), None)


def find_sole(blob, pts, margin, retry=True):
    """Bottom of the foot (zero point). Searches only just below each foot, so floor
    shadows merged into the mask can't pull it down."""
    h, w = blob.shape
    soles = []
    for ids in ((L_AN, L_HEEL, L_FT), (R_AN, R_HEEL, R_FT)):
        xs, ys = [pts[i][0] for i in ids], [pts[i][1] for i in ids]
        x0, x1 = max(0, int(min(xs) - margin)), min(w, int(max(xs) + margin))
        y0, y1 = max(0, int(min(ys) - margin)), min(h, int(max(ys) + 0.6 * margin))
        if x1 > x0 and y1 > y0:
            rows = np.flatnonzero(blob[y0:y1, x0:x1].any(axis=1))
            if rows.size:
                soles.append(y0 + int(rows[-1]))
    if soles:
        return max(soles)
    return find_sole(blob, pts, 2 * margin, False) if retry else None


def find_crown(blob, pts, margin):
    """Top of the head: highest body pixel above the face."""
    xs = [pts[i][0] for i in (L_EAR, R_EAR, NOSE)]
    x0, x1 = max(0, int(min(xs) - margin)), min(blob.shape[1], int(max(xs) + margin))
    if x1 <= x0:
        return None
    rows = np.flatnonzero(blob[:, x0:x1].any(axis=1))
    return int(rows[0]) if rows.size else None


def sole_and_crown(blob, pts, tb):
    m = foot_margin(pts)
    sole, crown = find_sole(blob, pts, m), find_crown(blob, pts, m)
    return (tb[1] if sole is None else sole), (tb[0] if crown is None else crown)


def ellipse_perimeter(width, depth):
    a, b = width / 2, depth / 2
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def torso_frame(pts):
    sh_m, hip_m = mid(pts[L_SH], pts[R_SH]), mid(pts[L_HIP], pts[R_HIP])
    torso_px = hip_m[1] - sh_m[1]
    row_y = lambda f: sh_m[1] + f * torso_px
    row_x = lambda f: sh_m[0] + min(f, 1.0) * (hip_m[0] - sh_m[0])
    return sh_m, hip_m, torso_px, row_y, row_x


# ------------------------------------------------------------ height entry
def parse_height(text, unit):
    """Returns (height_cm or None, error message). Inches accept 69, 69.5, 5'9 or 5 9."""
    t = text.strip().replace('"', "")
    if not t:
        return None, "Type your height first"
    try:
        if unit == "cm":
            cm = float(t)
        elif "'" in t or " " in t:
            ft, _, inch = t.replace("'", " ").partition(" ")
            inch = float(inch.strip() or 0)
            if inch >= 12:
                return None, "The inches part should be less than 12"
            cm = (float(ft) * 12 + inch) * CM_PER_IN
        else:
            cm = float(t) * CM_PER_IN
    except ValueError:
        return None, "That doesn't look like a number"
    if not HEIGHT_MIN_CM <= cm <= HEIGHT_MAX_CM:
        if unit == "cm":
            return None, f"Enter a height between {HEIGHT_MIN_CM:.0f} and {HEIGHT_MAX_CM:.0f} cm"
        return None, (f"Enter a height between {to_in(HEIGHT_MIN_CM):.0f} and "
                      f"{to_in(HEIGHT_MAX_CM):.0f} inches")
    return cm, ""


def edit_height(k, text, unit):
    """Applies one key press on the start page. Returns (text, unit, enter_pressed)."""
    if k < 0:
        return text, unit, False
    ch = chr(k)
    if k in (8, 127):                                  # Backspace (Windows/Linux, macOS)
        return text[:-1], unit, False
    if k in (13, 10):                                  # Enter
        return text, unit, True
    if ch in "iI":
        return (text if unit == "in" else ""), "in", False
    if ch in "cC":
        return (text if unit == "cm" else ""), "cm", False
    if k == 9:                                         # Tab
        return "", ("cm" if unit == "in" else "in"), False
    if len(text) < 8 and (ch.isdigit() or ch == "." or (unit == "in" and ch in "' ")):
        return text + ch, unit, False
    return text, unit, False


def draw_height_page(img, text, unit, preview_cm, error):
    h, w = img.shape[:2]
    s = w / 1280
    img[:] = (img * 0.25).astype(np.uint8)
    th, big = max(1, int(2 * s)), max(2, int(3 * s))

    def center(t, y, sc, col, thick=th):
        tw = cv2.getTextSize(t, FONT, sc, thick)[0][0]
        cv2.putText(img, t, ((w - tw) // 2, int(y)), FONT, sc, col, thick, cv2.LINE_AA)

    center("Body Measure", 0.15 * h, 1.6 * s, (0, 255, 255), big)
    center("How tall are you, without shoes?", 0.25 * h, 1.0 * s, (255, 255, 255))

    # Unit buttons
    bw_, bh_, gap, y0 = int(220 * s), int(60 * s), int(30 * s), int(0.31 * h)
    for i, (u, label) in enumerate((("in", "[I] inches"), ("cm", "[C] cm"))):
        x0 = w // 2 - bw_ - gap // 2 + i * (bw_ + gap)
        on = u == unit
        if on:
            cv2.rectangle(img, (x0, y0), (x0 + bw_, y0 + bh_), (0, 160, 0), -1)
        cv2.rectangle(img, (x0, y0), (x0 + bw_, y0 + bh_),
                      (255, 255, 255) if on else (130, 130, 130), th)
        (tw, tht), _ = cv2.getTextSize(label, FONT, 0.9 * s, th)
        cv2.putText(img, label, (x0 + (bw_ - tw) // 2, y0 + (bh_ + tht) // 2), FONT, 0.9 * s,
                    (255, 255, 255) if on else (170, 170, 170), th, cv2.LINE_AA)

    # Input box
    box_w, box_h = int(460 * s), int(90 * s)
    x0, y0 = (w - box_w) // 2, int(0.45 * h)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (40, 40, 40), -1)
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (255, 255, 255), th)
    cursor = "|" if int(time.monotonic() * 2) % 2 == 0 else " "
    suffix = "" if "'" in text else f" {unit}"
    center(text + cursor + suffix, y0 + box_h * 0.68, 1.5 * s, (255, 255, 255), big)

    y = y0 + box_h + 45 * s
    center("e.g.  69   or   5'9" if unit == "in" else "e.g.  175", y, 0.8 * s, (170, 170, 170))
    y += 50 * s
    if error:
        center(error, y, 0.85 * s, (0, 0, 255))
    elif preview_cm:
        center(f"= {fmt_height(preview_cm)}", y, 0.85 * s, (0, 220, 0))
    center("For the scan, stand barefoot or in socks - your height sets the scale.",
           0.84 * h, 0.7 * s, (0, 255, 255))
    center("[Enter] start   [Backspace] delete   [Tab] switch units   [Q] quit",
           0.92 * h, 0.7 * s, (200, 200, 200))


# ------------------------------------------------------------------ camera
def open_camera(index, max_w):
    """Use the highest resolution the camera actually delivers."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. Check camera permissions.")

    def try_size(cw, ch):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cw)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ch)
        size = None
        for _ in range(3):
            ok, f = cap.read()
            if ok:
                size = (f.shape[1], f.shape[0])
        return size

    sizes = [try_size(cw, ch) for cw, ch in ((3840, 2160), (2560, 1440), (1920, 1080),
                                             (1280, 720), (640, 480)) if cw <= max_w]
    sizes = [s for s in sizes if s]
    if not sizes:
        raise SystemExit("Camera returned no frames. Check camera permissions.")
    best = max(sizes, key=lambda s: s[0] * s[1])
    try_size(*best)
    return cap, best


class MotionMonitor:
    """Camera movement = whole-image background shift. People walking behind the
    subject and exposure changes don't count."""

    def __init__(self):
        self.prev, self.still_since, self.shift = None, time.monotonic(), 0.0
        self.win = cv2.createHanningWindow((320, 180), cv2.CV_32F)

    def update(self, frame, mask):
        g = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = (g - g.mean()) / (g.std() + 1e-6)
        if mask is not None:
            m = cv2.resize((mask > 0.3).astype(np.uint8), (320, 180), interpolation=cv2.INTER_NEAREST)
            g[cv2.dilate(m, np.ones((15, 15), np.uint8)) > 0] = 0
        now = time.monotonic()
        if self.prev is not None:
            (dx, dy), resp = cv2.phaseCorrelate(self.prev, g, self.win)
            self.shift = math.hypot(dx, dy)
            if resp > 0.1 and self.shift > MOTION_SHIFT_PX:
                self.still_since = now
        self.prev = g
        return now - self.still_since >= STABLE_SEC


class VanishingTilt:
    """Camera tilt from the vertical vanishing point of background lines."""

    def __init__(self, hfov, manual=None):
        self.hfov, self.manual = hfov, manual
        self.hist, self.lines, self.n = deque(maxlen=VP_HISTORY), [], 0
        self.rng = np.random.default_rng()

    def reset(self):
        self.hist.clear()
        self.lines = []

    @property
    def tilt(self):
        if self.manual is not None:
            return self.manual
        return float(np.median(self.hist)) if len(self.hist) >= 3 else None

    @property
    def source(self):
        if self.manual is not None:
            return "manual"
        return "background lines" if self.tilt is not None else "no lines found - assuming level"

    def update(self, frame, mask):
        self.n += 1
        if self.manual is None and self.n % VP_EVERY == 0:
            est = self._estimate(frame, mask)
            if est is not None:
                self.hist.append(est)

    @staticmethod
    def _inliers(v, mids, dirs):
        d = v[:2][None, :] - mids * v[2]
        norm = np.linalg.norm(d, axis=1)
        ok = norm > 1e-9
        cos = np.zeros(len(mids))
        cos[ok] = np.abs(np.sum(d[ok] * dirs[ok], axis=1)) / norm[ok]
        return ok & (cos > math.cos(math.radians(VP_INLIER_DEG)))

    def _estimate(self, frame, mask):
        h, w = frame.shape[:2]
        edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0), 50, 150)
        if mask is not None:
            edges[cv2.dilate((mask > 0.3).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0] = 0
        segs = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=60,
                               minLineLength=int(0.12 * h), maxLineGap=6)
        if segs is None:
            return None
        segs = np.asarray(segs, dtype=float).reshape(-1, 4)
        dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
        segs = segs[np.degrees(np.arctan2(np.abs(dx), np.abs(dy))) < 25]
        n = len(segs)
        if n < VP_MIN_LINES:
            return None
        L = np.cross(np.column_stack([segs[:, :2], np.ones(n)]),
                     np.column_stack([segs[:, 2:], np.ones(n)]))
        L /= np.hypot(L[:, 0], L[:, 1])[:, None]
        mids = (segs[:, :2] + segs[:, 2:]) / 2
        dirs = segs[:, 2:] - segs[:, :2]
        dirs /= np.linalg.norm(dirs, axis=1)[:, None]
        best = None
        for _ in range(200):
            i, j = self.rng.choice(n, 2, replace=False)
            v = np.cross(L[i], L[j])
            if not np.allclose(v, 0):
                inl = self._inliers(v, mids, dirs)
                if best is None or inl.sum() > best.sum():
                    best = inl
        if best is None or best.sum() < VP_MIN_LINES or np.ptp(mids[best, 0]) < 0.15 * w:
            return None
        v = np.linalg.svd(L[best])[2][-1]
        f = (w / 2) / math.tan(math.radians(self.hfov / 2))
        D = np.array([v[0] - w / 2 * v[2], v[1] - h / 2 * v[2], f * v[2]])
        tilt = math.degrees(math.asin(min(1.0, abs(D[2]) / np.linalg.norm(D))))
        if tilt > VP_MAX_TILT:
            return None
        if abs(v[2]) > 1e-9 and v[1] / v[2] > h / 2:
            tilt = -tilt
        self.lines = segs[best].astype(int).tolist()
        return tilt


# -------------------------------------------------------------- measurement
def measure_front(pts, blob, tb, vc):
    """Front A-pose sample. Lengths are stored as fractions of height."""
    sole, crown = sole_and_crown(blob, pts, tb)
    raw = sole - crown
    S, sh_px = raw * vc, dist(pts[L_SH], pts[R_SH])
    sh_m, hip_m, torso_px, row_y, row_x = torso_frame(pts)
    if raw < 200 or sh_px < 20 or torso_px < 40:
        return None, "too small - come closer"
    bw = lambda f, band=2: band_width(blob, row_y(f), row_x(f), band)
    chest, shoulder_outer = bw(CHEST_F), bw(-0.03)
    knee_y = (pts[L_KN][1] + pts[R_KN][1]) / 2
    ankle_y = (pts[L_AN][1] + pts[R_AN][1]) / 2
    f_end = 1 + 0.35 * (knee_y - hip_m[1]) / torso_px
    waist = min(((v, float(f)) for f in np.arange(0.5, 0.9, 0.02) if (v := bw(f, 1))), default=None)
    hip = max(((v, float(f)) for f in np.arange(0.85, f_end, 0.02) if (v := bw(f, 1))), default=None)
    if not (chest and waist and hip and shoulder_outer):
        return None, "torso widths not found"
    DEBUG["chest/shoulder"] = (chest / sh_px, chest <= 1.3 * sh_px)
    DEBUG["waist/shoulder"] = (waist[0] / sh_px, waist[0] <= 1.3 * sh_px)
    DEBUG["hip/shoulder"] = (hip[0] / sh_px, hip[0] <= 1.6 * sh_px)
    if chest > 1.3 * sh_px or waist[0] > 1.3 * sh_px or hip[0] > 1.6 * sh_px:
        return None, "arms touching body"

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
    su = lambda v: v / S if v is not None else None
    rel = lambda y: (sole - y) / raw
    return ({
        "shoulder_joint": su(sh_px), "shoulder_outer": su(shoulder_outer),
        "chest_w": su(chest), "waist_w": su(waist[0]), "hip_w": su(hip[0]),
        "torso_len": su(torso_px * vc), "shoulder_to_waist": su((waist_y - sh_m[1]) * vc),
        "arm_len": su(avg_len(L_SH, L_EL, R_SH, R_EL) + avg_len(L_EL, L_WR, R_EL, R_WR)),
        "inseam": su((sole - crotch) * vc) if crotch else None,
        "outseam": su((sole - waist_y) * vc),
        "upper_arm_w": su(avg_perp(L_SH, L_EL, R_SH, R_EL, 0.45, 0.5 * sh_px)),
        "forearm_w": su(avg_perp(L_EL, L_WR, R_EL, R_WR, 0.25, 0.4 * sh_px)),
        "thigh_w": su(avg_perp(L_HIP, L_KN, R_HIP, R_KN, 0.2, 0.6 * hip[0])),
        "calf_w": su(avg_perp(L_KN, L_AN, R_KN, R_AN, 0.3, 0.45 * sh_px)),
        "neck_w": su(neck) if neck and neck < 0.6 * sh_px else None,
        "head_w": su(head) if head and head < sh_px else None,
        # non-length values used later
        "waist_f": waist[1], "hip_f": hip[1],
        "bust_rel": rel(row_y(CHEST_F)), "waist_rel": rel(waist_y), "hip_rel": rel(row_y(hip[1])),
        "stature_raw": float(raw), "stature_px": float(S), "sole_y": float(sole), "sh_px": sh_px,
    }, "ok")


def summarize_front(samples):
    prof, noise = {}, {}
    for k in samples[0]:
        vals = [s[k] for s in samples if s.get(k) is not None]
        prof[k] = None
        if vals:
            prof[k], noise[k] = robust_stats(vals)
    return prof, noise


def body_yaw(world):
    """Body rotation (0-180 deg) from shoulder and hip directions in 3D."""
    v = lambda a, b: np.array([world[a].x - world[b].x, world[a].z - world[b].z])
    s, hp = v(L_SH, R_SH), v(L_HIP, R_HIP)
    if np.dot(s, hp) < 0:
        hp = -hp
    d = s / (np.linalg.norm(s) + 1e-9) + hp / (np.linalg.norm(hp) + 1e-9)
    return math.degrees(math.atan2(d[1], d[0])) % 180.0


def measure_turn(pts, blob, ref):
    """Widths at bust, waist and hip levels while turning, as fractions of height."""
    sole = find_sole(blob, pts, max(8, 0.15 * ref["sh_px"]))
    if sole is None:
        return None, "Keep your feet in view"
    S_raw = ref["stature_raw"]
    if abs(sole - ref["sole_y"]) > 0.015 * S_raw:
        return None, "Turn in place - stay on the same spot"
    xc = (pts[L_HIP][0] + pts[R_HIP][0]) / 2
    out = {}
    for name, half, pick in (("bust", 0.02, max), ("waist", 0.02, min), ("hip", 0.025, max)):
        vals = [v for dr in np.arange(-half, half + 1e-9, 0.005)
                if (v := band_width(blob, sole - (ref[f"{name}_rel"] + dr) * S_raw, xc, band=1))]
        if len(vals) < 3:
            return None, "Hands on your head, elbows out"
        out[name] = pick(vals) / ref["stature_px"]
    return out, "Keep turning slowly"


def fill_circular(vals):
    """Fill missing bins by linear interpolation around the circle."""
    n, known, out = len(vals), [i for i, v in enumerate(vals) if v is not None], list(vals)
    for i in range(n):
        if out[i] is None:
            prev = max((k for k in known if k < i), default=known[-1])
            nxt = min((k for k in known if k > i), default=known[0])
            dp, dn = (i - prev) % n, (nxt - i) % n
            out[i] = vals[prev] + (vals[nxt] - vals[prev]) * dp / (dp + dn) if dp + dn else vals[prev]
    return out


def summarize_turn(bins):
    out = {}
    for name in ("bust", "waist", "hip"):
        med = [float(np.median(b)) if b else None for b in bins[name]]
        if sum(m is not None for m in med) < MIN_BINS_FILLED:
            return None
        out[name] = fill_circular(med)
    out["waist_depth_rel"] = min(out["waist"])  # narrowest view = side view
    return out


def compute_results(height_cm, f_px, front, noise, turn):
    """Everything is scaled from the entered height (floor to top of head)."""
    hcm = height_cm
    Zb = f_px * hcm * 10 / front["stature_px"]          # camera-to-body distance, mm
    scale_rel = HEIGHT_UNC_CM / hcm

    def val(k):
        v = front.get(k)
        return v * hcm if v is not None else None

    def unc(v, sys_rel, k=None, abs_cm=0.0):
        if v is None:
            return None
        nz = noise.get(k, 0.0) * hcm if k else 0.0
        return math.sqrt((v * scale_rel) ** 2 + (v * sys_rel) ** 2 + nz ** 2 + abs_cm ** 2)

    corr = lambda wcm: wcm / math.sqrt(1 + (wcm * 10 / (2 * Zb)) ** 2)  # finite-distance widening

    circ, depths = {}, {}
    for name, wk in (("bust", "chest_w"), ("waist", "waist_w"), ("hip", "hip_w")):
        if turn:
            ws = [corr(r * hcm) for r in turn[name]]
            c = math.pi * float(np.mean(ws))  # Cauchy: perimeter = pi * mean width
            circ[name], depths[name] = (c, unc(c, SYS_CIRC), ""), min(ws)
        else:
            wv = val(wk)
            c = ellipse_perimeter(wv, wv * DEPTH_RATIO[name])
            circ[name] = (c, unc(c, SYS_EST_CIRC, wk), "est. - do the turn")
    for name, wk in (("neck", "neck_w"), ("upper_arm", "upper_arm_w"),
                     ("thigh", "thigh_w"), ("calf", "calf_w")):
        wv = val(wk)
        c = ellipse_perimeter(wv, wv * LIMB_DEPTH_RATIO) if wv else None
        circ[name] = (c, unc(c, SYS_EST_CIRC, wk), "est.")

    inseam, outseam = val("inseam"), val("outseam")
    u_in = unc(inseam, SYS_LEN, "inseam", LANDMARK_ABS_CM)
    u_out = unc(outseam, SYS_LEN, "outseam", 0.6)
    rise = outseam - inseam if inseam and outseam else None
    arm, s2w, sho = val("arm_len"), val("shoulder_to_waist"), val("shoulder_outer")
    rows = [
        ("Height", hcm, None, "entered"),
        ("Bust / chest", *circ["bust"]), ("Waist", *circ["waist"]), ("Hips", *circ["hip"]),
        ("Shoulder width", sho, unc(sho, SYS_WIDTH, "shoulder_outer"), ""),
        ("Arm / sleeve length", arm, unc(arm, SYS_LEN, "arm_len", LANDMARK_ABS_CM), ""),
        ("Inseam", inseam, u_in, ""),
        ("Outseam (waist-floor)", outseam, u_out, ""),
        ("Rise", rise, math.hypot(u_in, u_out) if rise else None, ""),
        ("Shoulder to waist", s2w, unc(s2w, SYS_LEN, "shoulder_to_waist", LANDMARK_ABS_CM), ""),
        ("Neck", *circ["neck"]), ("Upper arm", *circ["upper_arm"]),
        ("Thigh", *circ["thigh"]), ("Calf", *circ["calf"]),
    ]
    av = {k: val(k) for k in ("torso_len", "shoulder_joint", "shoulder_outer", "chest_w",
                              "waist_w", "hip_w", "upper_arm_w", "forearm_w", "thigh_w",
                              "calf_w", "neck_w", "head_w")}
    av.update(waist_f=front["waist_f"], hip_f=front["hip_f"])
    av.update({f"{k}_d": d for k, d in depths.items()})
    info = {"height_cm": hcm, "distance_cm": Zb / 10, "turn_done": bool(turn)}
    return rows, av, info


def tilt_text(tilt, digits=1):
    return f"{tilt:+.{digits}f} deg" if tilt is not None else "--"


def print_table(rows, info, res, tilt, tilt_src):
    print(f"\nMeasurements (camera {res[0]}x{res[1]}, ~{info['distance_cm'] / 100:.1f} m away, "
          f"tilt {tilt_text(tilt)} ({tilt_src})):")
    for label, v, u, note in rows:
        print(f"  {label:<28}{fmt_pm(v, u)}{'  (' + note + ')' if note else ''}")


def save_json(rows, info, res, tilt, tilt_src, age_range=None):
    data = {
        "age_range": age_range,
        "camera_resolution": f"{res[0]}x{res[1]}",
        "height_entered_cm": round(info["height_cm"], 1),
        "distance_m": round(info["distance_cm"] / 100, 2),
        "camera_tilt_deg": round(tilt, 1) if tilt is not None else None,
        "tilt_source": tilt_src,
        "turn_scan_done": info["turn_done"],
        "scale_reference": "entered height = floor to top of head (scanned barefoot)",
        "measurements": {
            label: None if v is None else {
                "in": round(to_in(v), 1), "cm": round(v, 1),
                "plus_minus_in": round(to_in(u), 1) if u else None,
                "plus_minus_cm": round(u, 1) if u else None, "note": note}
            for label, v, u, note in rows},
    }
    with open("measurements.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Saved measurements.json")


# ----------------------------------------------------------------- checks
def feet_in_frame(pts, vis, blob, h):
    """Feet are in frame if the sole is found above the bottom edge."""
    sole = find_sole(blob, pts, foot_margin(pts))
    lowest = max(pts[i][1] for i in FEET)
    lm_ok, sole_ok = lowest < h - 3, sole is not None and sole < h - 3
    DEBUG["feet_lowest_y"] = (f"{lowest:.0f}/{h}", lm_ok)
    DEBUG["sole_y"] = ("none" if sole is None else f"{sole}/{h}", sole_ok)
    return lm_ok and sole_ok


def full_body_check(pts, vis, blob, tb, w, h, need=FULL_BODY, min_vis=0.6):
    if blob is None:
        return False, "Step into view"
    head_cut = tb[0] <= 3 or vis[NOSE] < 0.5
    feet_cut = not feet_in_frame(pts, vis, blob, h)
    if head_cut and feet_cut:
        return False, "Step back so your whole body fits"
    if head_cut:
        return False, "Head is cut off: step back, or tilt the screen back"
    if feet_cut:
        return False, "Feet are cut off: step back, or tilt the screen forward"
    xs = np.flatnonzero(blob.any(axis=0))
    if xs[0] <= 3 or xs[-1] >= w - 4:
        return False, "Move toward the center of the frame"
    if tb[1] - tb[0] < 0.45 * h:
        return False, "Come a little closer"
    if not all(vis[i] > min_vis for i in need):
        return False, "Make sure your arms and legs are clearly visible"
    return True, ""


def framing_check(pts, vis, blob, tb, w, h):
    ok, hint = full_body_check(pts, vis, blob, tb, w, h)
    if not ok:
        return ok, hint
    sh_px = dist(pts[L_SH], pts[R_SH])
    if sh_px < 20:
        return False, "Face the camera"
    for sh, wr, side in ((L_SH, L_WR, "L"), (R_SH, R_WR, "R")):
        ang = math.degrees(math.atan2(abs(pts[wr][0] - pts[sh][0]), pts[wr][1] - pts[sh][1]))
        DEBUG[f"arm_angle_{side}"] = (ang, 30 <= ang <= 65)
        if ang < 30:
            return False, "Raise your arms a little, to about 45 degrees"
        if ang > 65:
            return False, "Lower your arms a bit, to about 45 degrees"
    if abs(pts[L_AN][0] - pts[R_AN][0]) < 0.3 * sh_px:
        return False, "Step your feet apart, about shoulder width"
    return True, ""


def turn_check(pts, vis, blob, tb, h):
    if blob is None:
        return False, "Step into view"
    if tb[0] <= 3:
        return False, "Head is cut off"
    sole = find_sole(blob, pts, foot_margin(pts))
    if sole is None or sole >= h - 3:
        return False, "Feet are cut off"
    if max(vis[L_HIP], vis[R_HIP]) < 0.5:
        return False, "Keep your whole body in view"
    sh_y = min(pts[L_SH][1], pts[R_SH][1])
    up = [pts[i][1] < sh_y for i in (L_WR, R_WR) if vis[i] > 0.3]
    if up and not all(up):
        return False, "Put both hands on your head"
    return True, ""


# --------------------------------------------------------------- drawing
def capsule(img, p1, p2, r, color):
    p1, p2, r = (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), max(2, int(r))
    cv2.line(img, p1, p2, color, 2 * r, cv2.LINE_AA)
    cv2.circle(img, p1, r, color, -1, cv2.LINE_AA)
    cv2.circle(img, p2, r, color, -1, cv2.LINE_AA)


def draw_avatar(frame, pts, vis, r, vc, alpha=0.55):
    sh_m, hip_m = mid(pts[L_SH], pts[R_SH]), mid(pts[L_HIP], pts[R_HIP])
    torso_img = dist(sh_m, hip_m)
    if torso_img < 20 or not r.get("torso_len") or not r.get("shoulder_joint"):
        return frame
    cmpp = r["torso_len"] / clen(sh_m, hip_m, vc)
    px = lambda cm: cm / cmpp
    g = lambda k: r.get(k) or FALLBACK_W[k]
    ok = lambda *ids: all(vis[i] > 0.5 for i in ids)
    cos_t = float(np.clip(dist(pts[L_SH], pts[R_SH]) * cmpp / r["shoulder_joint"], 0, 1))
    sin_t = math.sqrt(1 - cos_t ** 2)
    proj = lambda w, d: math.hypot(w * cos_t, d * sin_t)
    level = lambda name, wk: proj(r[wk], r.get(f"{name}_d") or r[wk] * DEPTH_RATIO[name])

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
    sh_outer = proj(r["shoulder_outer"], r.get("bust_d") or r["chest_w"] * DEPTH_RATIO["bust"])
    levels = sorted([(-0.06, sh_outer * 0.85), (0.04, sh_outer),
                     (CHEST_F, level("bust", "chest_w")),
                     (r["waist_f"], level("waist", "waist_w")),
                     (r["hip_f"], level("hip", "hip_w")),
                     (max(r["hip_f"], 1.0) + 0.08, level("hip", "hip_w") * 0.95)])
    centers = [(np.array(sh_m) + u * f * torso_img, px(wcm) / 2) for f, wcm in levels]
    outline = [c + n * hw for c, hw in centers] + [c - n * hw for c, hw in centers[::-1]]
    cv2.fillPoly(overlay, [np.array(outline, dtype=np.int32)], body, cv2.LINE_AA)

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
        angle = math.degrees(math.atan2(pts[L_EAR][1] - pts[R_EAR][1], pts[L_EAR][0] - pts[R_EAR][0]))
        cv2.ellipse(overlay, (int(ear_m[0]), int(ear_m[1] - 0.12 * head_h)),
                    (int(head_w / 2), int(head_h / 2)), angle, 0, 360, skin, -1, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def draw_height_ruler(img, blob, sole, crown, height_cm=None):
    h, w = img.shape[:2]
    s = w / 1280
    xs = np.flatnonzero(blob.any(axis=0))
    x = int(min(w - 150 * s, xs[-1] + 25)) if xs.size else int(w - 150 * s)
    col, th = (0, 255, 255), max(1, int(2 * s))
    cv2.line(img, (x, crown), (x, sole), col, th, cv2.LINE_AA)
    for y in (crown, sole):
        cv2.line(img, (x - 12, y), (x + 12, y), col, th, cv2.LINE_AA)
    labels = [("0", sole + 6)]
    if height_cm:
        labels += [(f"{to_in(height_cm):.1f} in", crown + 6), (f"{height_cm:.1f} cm", crown + int(30 * s))]
    for text, y in labels:
        cv2.putText(img, text, (x + 16, y), FONT, 0.7 * s, col, th, cv2.LINE_AA)


def wrap(text, scale, thick, max_w):
    lines, cur = [], ""
    for word in text.split():
        t = (cur + " " + word).strip()
        if cur and cv2.getTextSize(t, FONT, scale, thick)[0][0] > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = t
    return lines + [cur] if cur else lines


def draw_banner(img, title, hint, progress):
    """Large text at the bottom, readable from across the room."""
    h, w = img.shape[:2]
    s = w / 1280
    ts, tt, hs, ht = 1.0 * s, max(2, int(2.5 * s)), 0.8 * s, max(1, int(2 * s))
    lines = [(t, ts, tt, (0, 255, 255)) for t in wrap(title, ts, tt, w - 40)]
    if hint:
        lines += [(t, hs, ht, (255, 255, 255)) for t in wrap(hint, hs, ht, w - 40)]
    lh, bar_h = int(42 * s), int(14 * s) if progress is not None else 0
    y0 = max(0, h - (lh * len(lines) + bar_h + int(30 * s)))
    img[y0:h] = (img[y0:h] * 0.3).astype(np.uint8)
    y = y0 + lh
    for text, sc, th, col in lines:
        tw = cv2.getTextSize(text, FONT, sc, th)[0][0]
        cv2.putText(img, text, ((w - tw) // 2, y), FONT, sc, col, th, cv2.LINE_AA)
        y += lh
    if progress is not None:
        x0, x1, top = int(w * 0.2), int(w * 0.8), y - lh + int(14 * s)
        cv2.rectangle(img, (x0, top), (x1, top + bar_h), (90, 90, 90), -1)
        cv2.rectangle(img, (x0, top), (x0 + int((x1 - x0) * min(progress, 1.0)), top + bar_h),
                      (0, 200, 0), -1)


def draw_panel(img, rows, info, res, tilt, tilt_src):
    s = img.shape[1] / 1280
    lines = [f"camera {res[0]}x{res[1]} | ~{info['distance_cm'] / 100:.1f} m | "
             f"tilt {tilt_text(tilt, 0)} ({tilt_src})"]
    for label, v, u, note in rows:
        val = fmt_height(v, u) if label.startswith("Height") and v else fmt_pm(v, u)
        lines.append(f"{label}: {val}{' (' + note + ')' if note else ''}")
    lh, sc = int(22 * s), 0.5 * s
    roi = img[0:min(lh * len(lines) + 12, img.shape[0]), 0:min(int(760 * s), img.shape[1])]
    roi[:] = (roi * 0.35).astype(np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, int(20 * s) + lh * i), FONT, sc, (255, 255, 255), 1, cv2.LINE_AA)


def draw_debug(img):
    """Each check in green (pass) or red (fail)."""
    s = img.shape[1] / 1280
    for i, (k, (v, ok)) in enumerate(DEBUG.items()):
        txt = f"{k}: {v:.3f}" if isinstance(v, (float, np.floating)) else f"{k}: {v}"
        cv2.putText(img, txt, (int(15 * s), int(30 * s) + i * int(26 * s)), FONT, 0.6 * s,
                    (0, 200, 0) if ok else (0, 0, 255), max(1, int(2 * s)), cv2.LINE_AA)


class Voice:
    """Reads prompts aloud with macOS 'say' so the user can follow from across the room."""

    def __init__(self, enabled):
        self.cmd = shutil.which("say") if enabled else None
        self.proc, self.last, self.last_t = None, None, 0.0

    def say(self, text, interrupt=True, min_gap=0.0):
        now = time.monotonic()
        if not self.cmd or not text or text == self.last or now - self.last_t < min_gap:
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


# ------------------------------------------------------------ stage 1: age
FACE_CASCADE_PATH = "/Users/mitushijain/Desktop/Face_scanning code/haarcascade_frontalface_default.xml"
AGE_ONNX_PATH = "/Users/mitushijain/Desktop/Face_scanning code/age_googlenet.onnx"
AGE_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(14-23)', '(25-32)', '(35-43)', '(48-53)', '(60-100)']
REQUIRED_STABLE_FRAMES = 10       # same age bucket this many frames in a row = confirmed


def scan_age(camera, cascade_path, onnx_path):
    """Returns the confirmed age bucket (e.g. '(25-32)'), or None if the user pressed q."""
    for p in (cascade_path, onnx_path):
        if not os.path.exists(p):
            raise SystemExit(f"File not found: {p}\nUse --face-cascade / --age-model to point to it.")
    facedetect = cv2.CascadeClassifier(cascade_path)
    if facedetect.empty():
        raise SystemExit(f"Could not load the face detector: {cascade_path}")
    age_net = cv2.dnn.readNetFromONNX(onnx_path)

    video = cv2.VideoCapture(camera)
    if not video.isOpened():
        raise SystemExit("Could not open the camera. Check camera permissions.")

    print("Webcam active. Scanning face for age confirmation...")
    last_age, stable, confirmed = None, 0, None
    try:
        while True:
            ret, frame = video.read()
            if not ret:
                print("Camera read failed (check camera permissions).")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                last_age, stable = None, 0          # lost the face -> start counting again
            # only the largest face, so two people can't mix up the stability count
            for (x, y, w, h) in sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[:1]:
                pad = 20
                y1, y2 = max(0, y - pad), min(frame.shape[0], y + h + pad)
                x1, x2 = max(0, x - pad), min(frame.shape[1], x + w + pad)
                expanded_face = frame[y1:y2, x1:x2]
                if expanded_face.size == 0:
                    continue

                label = "Analysis Failed"
                try:
                    blob = cv2.dnn.blobFromImage(expanded_face, scalefactor=1.0, size=(227, 227),
                                                 mean=AGE_MEAN_VALUES, swapRB=True, crop=False)
                    age_net.setInput(blob)
                    age = AGE_BUCKETS[int(age_net.forward().argmax())]
                    label = f"Age: {age}"
                    if age == last_age:
                        stable += 1
                    else:
                        last_age, stable = age, 0
                    if stable >= REQUIRED_STABLE_FRAMES:
                        confirmed = age
                except Exception:
                    last_age, stable = None, 0

                if confirmed:
                    break

                # blur the face on screen (after the age was read from the clean image)
                frame[y:y + h, x:x + w] = cv2.GaussianBlur(frame[y:y + h, x:x + w], (99, 99), 30)
                cv2.putText(frame, f"Confirming: {min(stable, REQUIRED_STABLE_FRAMES)}/{REQUIRED_STABLE_FRAMES}",
                            (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

            if confirmed:
                print("\n===============================")
                print("SUCCESSFUL DETECTION!")
                print(f"Final Confirmed Age Range: {confirmed}")
                print("===============================\n")
                break

            cv2.imshow("Age scan", frame)
            if cv2.waitKey(1) == ord("q"):
                break
    finally:
        video.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)        # lets macOS actually close the window
    return confirmed


# ------------------------------------------------------- stage 2: body
def run_body_scan(args, age_range=None):
    """Runs the body scan. Returns True once measurements are computed and saved,
    or False if the user quit first."""
    known_cm = args.height_cm or (args.height_in * CM_PER_IN if args.height_in else None)
    if known_cm and not HEIGHT_MIN_CM <= known_cm <= HEIGHT_MAX_CM:
        raise SystemExit(f"Height must be between {HEIGHT_MIN_CM:.0f} and {HEIGHT_MAX_CM:.0f} cm.")
    height_unit = "cm" if args.height_cm else "in"
    height_text, height_err = "", ""
    start = "front_wait" if known_cm else "height"

    ensure_model()
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_PATH),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.6, min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6, output_segmentation_masks=True))
    cap, res = open_camera(args.camera, args.max_width)
    print(f"Camera delivers {res[0]}x{res[1]}")

    voice, motion = Voice(not args.no_voice), MotionMonitor()
    vp = VanishingTilt(args.hfov, args.tilt_deg)
    state, front_s = None, []
    front = noise = turn = ref = rows = av = info = None
    turn_bins, turn_t0, hold_t = None, None, None
    lost_t, f_px, now = None, 1.0, time.monotonic()
    show = {"avatar": True, "mask": False, "lines": False, "debug": True}
    was_stable, t0, last_ts = True, time.perf_counter(), -1

    def go(new):
        nonlocal state, hold_t, turn_bins, turn_t0
        state, hold_t = new, None
        if new in ("height", "front_wait"):
            front_s.clear()
        elif new == "turn":
            turn_bins = {k: [[] for _ in range(YAW_BINS)] for k in ("bust", "waist", "hip")}
            turn_t0 = time.monotonic()
        voice.say(STEPS[new][1])

    def refresh():
        nonlocal rows, av, info
        if front is not None and known_cm:
            rows, av, info = compute_results(known_cm, f_px, front, noise, turn)
            print_table(rows, info, res, vp.tilt, vp.source)
            save_json(rows, info, res, vp.tilt, vp.source, age_range)

    def hold(good, hint, nxt, ok_msg):
        """Count down while the pose is held, then move to the next state."""
        nonlocal hold_t
        if not good:
            hold_t = None
            voice.say(hint, interrupt=False, min_gap=4)
            return hint, None
        hold_t = hold_t or now
        progress = min(1.0, (now - hold_t) / HOLD_SEC)
        if progress >= 1.0:
            go(nxt)
        return ok_msg, progress

    done = False
    go(start)
    while True:
        if state == "live":      # all measurements computed and saved -> finished
            done = True
            break
        ok_, frame = cap.read()
        if not ok_:
            print("Camera read failed (check camera permissions).")
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        f_px = (w / 2) / math.tan(math.radians(args.hfov / 2))
        now = time.monotonic()

        # ---------------- start page: height entry
        if state == "height":
            preview_cm, _ = parse_height(height_text, height_unit)
            out = frame.copy()
            draw_height_page(out, height_text, height_unit, preview_cm, height_err)
            cv2.imshow(WINDOW, out)
            code = cv2.waitKey(1)
            k = -1 if code == -1 else code & 0xFF
            if k in (27, ord("q"), ord("Q")):
                break
            before = (height_text, height_unit)
            height_text, height_unit, enter = edit_height(k, height_text, height_unit)
            if (height_text, height_unit) != before:
                height_err = ""
            if enter:
                cm, err = parse_height(height_text, height_unit)
                if cm:
                    known_cm = cm
                    print(f"Height: {fmt_height(cm)}")
                    go("front_wait")
                else:
                    height_err = err
            continue

        # ---------------- pose + camera checks for the scan steps
        ts = last_ts = max(int((time.perf_counter() - t0) * 1000), last_ts + 1)
        result = pose.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
        out, hint, progress = frame.copy(), "", None

        mask = blob = tb = pts = vis = world = None
        if result.pose_landmarks:
            pts, vis = landmarks_px(result, w, h)
            world = result.pose_world_landmarks[0]
            if result.segmentation_masks:
                mask = np.squeeze(result.segmentation_masks[0].numpy_view())
                blob, tb = person_blob(mask, pts)
        person = blob is not None

        stable = motion.update(frame, mask)
        if not stable and was_stable:
            vp.reset()
        was_stable = stable
        if stable:
            vp.update(frame, mask)
        tilt = vp.tilt
        vc = tilt_factor(tilt)

        # ---------------- step 1: front A-pose
        if state in ("front_wait", "front"):
            DEBUG.clear()
            good, hint = framing_check(pts, vis, blob, tb, w, h) if person else (False, "Step into view")
            if good and not stable:
                good, hint = False, "Camera moving - hold still"
            DEBUG["framing"] = ("ok" if good else hint[:35], good)
            if state == "front_wait":
                hint, progress = hold(good, hint, "front", "Perfect - hold that pose")
            else:
                if good:
                    lost_t = None
                    smp, why = measure_front(pts, blob, tb, vc)
                    DEBUG["measure"] = (why, smp is not None)
                    if smp:
                        front_s.append(smp)
                        hint = "Hold still"
                    elif why == "arms touching body":
                        hint = "Lift your arms a little away from your body"
                    else:
                        hint = "Hold still"
                else:
                    lost_t = lost_t or now
                    if now - lost_t > FRONT_LOST_SEC:
                        lost_t = None
                        go("front_wait")
                progress = len(front_s) / FRONT_FRAMES
                if state == "front" and len(front_s) >= FRONT_FRAMES:
                    front, noise = summarize_front(front_s)
                    ref = {k: front[k] for k in ("stature_raw", "stature_px", "sole_y", "sh_px",
                                                 "bust_rel", "waist_rel", "hip_rel")}
                    refresh()
                    go("turn_wait")
            if show["debug"]:
                DEBUG["front_frames"] = (f"{len(front_s)}/{FRONT_FRAMES}", True)
                draw_debug(out)

        # ---------------- step 2: turn in place
        elif state in ("turn_wait", "turn"):
            good, hint = turn_check(pts, vis, blob, tb, h) if person else (False, "Step into view")
            if good and not stable:
                good, hint = False, "Camera moving - hold still"
            if state == "turn_wait":
                hint, progress = hold(good, hint, "turn", "Good - get ready to turn")
                hint += "   (press N to skip)"
            else:
                if good:
                    smp, hint = measure_turn(pts, blob, ref)
                    if smp:
                        b = int(body_yaw(world) / (180.0 / YAW_BINS)) % YAW_BINS
                        for k, v in smp.items():
                            turn_bins[k][b].append(v)
                filled = sum(len(x) >= MIN_PER_BIN for x in turn_bins["waist"])
                progress = filled / YAW_BINS
                elapsed = now - turn_t0
                if (filled == YAW_BINS and elapsed >= TURN_MIN_SEC) or elapsed > TURN_MAX_SEC:
                    turn = summarize_turn(turn_bins)
                    if turn is None:
                        print("Turn scan incomplete - circumferences are estimated. "
                              "Press R to restart and turn more slowly.")
                    refresh()
                    go("live")

        # ---------------- results
        elif state == "live":
            hint = KEYS_HELP
            if tilt is not None and abs(tilt) > TILT_WARN_DEG:
                hint = f"Camera tilted {tilt:+.0f} deg: raise the laptop to waist height.  " + hint
            if person and av:
                if show["avatar"]:
                    out = draw_avatar(out, pts, vis, av, vc)
                if tb[0] > 3 and tb[1] < h - 4:
                    sole, crown = sole_and_crown(blob, pts, tb)
                    draw_height_ruler(out, blob, sole, crown, info["height_cm"])

        if show["lines"]:
            for x1, y1, x2, y2 in vp.lines:
                cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2, cv2.LINE_AA)
        if show["mask"] and mask is not None:
            sel = mask > 0.5
            out[sel] = (out[sel] * 0.5 + np.array([0, 120, 0])).astype(np.uint8)
        if rows and state == "live":
            draw_panel(out, rows, info, res, vp.tilt, vp.source)
        draw_banner(out, STEPS[state][0], hint, progress)
        cv2.imshow(WINDOW, out)

        code = cv2.waitKey(1)
        key = "" if code == -1 else chr(code & 0xFF)
        if key == "q":
            break
        elif key == "n" and state in ("turn_wait", "turn"):
            refresh()
            go("live")
        elif key == "r":
            front = noise = turn = ref = rows = av = info = None
            go("height")
        elif key and key in "lamd":
            name = {"l": "lines", "a": "avatar", "m": "mask", "d": "debug"}[key]
            show[name] = not show[name]

    voice.close()
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    pose.close()
    return done


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Webcam body measurements scaled by your height")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--max-width", type=int, default=1920, help="highest capture width to try")
    ap.add_argument("--hfov", type=float, default=68.0, help="camera horizontal field of view")
    ap.add_argument("--tilt-deg", type=float, default=None, help="manual camera tilt")
    ap.add_argument("--height-cm", type=float, default=None, help="your height in cm (skips the start page)")
    ap.add_argument("--height-in", type=float, default=None,
                    help="your height in inches, e.g. 5'5\" = 65 (skips the start page)")
    ap.add_argument("--no-voice", action="store_true")
    ap.add_argument("--face-cascade", default=FACE_CASCADE_PATH, help="haarcascade_frontalface_default.xml")
    ap.add_argument("--age-model", default=AGE_ONNX_PATH, help="age_googlenet.onnx")
    args = ap.parse_args()

    age_range = scan_age(args.camera, args.face_cascade, args.age_model)
    if age_range is None:
        print("Age scan cancelled - exiting.")
        sys.exit(0)

    time.sleep(0.5)          # give the camera a moment to be released before reopening it
    if not run_body_scan(args, age_range):
        print("Body scan cancelled - exiting.")
        sys.exit(0)

    print(f"\nAll information collected. Age range: {age_range}. "
          "Body measurements saved to measurements.json.")
    sys.exit(0)


if __name__ == "__main__":
    main()