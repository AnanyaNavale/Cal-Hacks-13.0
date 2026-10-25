"""
body_measure.py - Webcam body measurements (in + cm) using Letter/A4 paper as the scale.
MediaPipe Pose Landmarker (Tasks API) provides landmarks and segmentation.

Usage: python body_measure.py [--camera 0] [--hfov 68] [--tilt-deg N] [--max-width 1920] [--no-voice]

Steps (voice guided, automatic):
  1. Locate: whole body incl. shoes in view, no paper yet. Step left, step right,
     come back to the middle. The code learns the empty scene behind you.
  2. Paper: hold Letter/A4 paper sideways against your stomach by the top corners,
     eyes on the camera. Only white inside your outline counts as paper; once found,
     the paper's edges are tracked instead of re-checking colour.
  3. Front: drop the paper. A-pose: arms ~45 deg out, feet apart.
  4. Turn: hands on your head, turn slowly in place, one full circle.

Keys: n skip turn  r restart  l tilt lines  a avatar  m mask  d debug  q quit
"""
import argparse
import json
import math
import os
import shutil
import subprocess
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

# Pose landmarks
NOSE, L_EAR, R_EAR = 0, 7, 8
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 23, 24, 25, 26, 27, 28
L_HEEL, R_HEEL, L_FT, R_FT = 29, 30, 31, 32
FEET = (L_AN, R_AN, L_HEEL, R_HEEL, L_FT, R_FT)
CORE = [NOSE, L_SH, R_SH, L_HIP, R_HIP, L_KN, R_KN]      # paper step: hands may be hidden
FULL_BODY = CORE + [L_EL, R_EL, L_WR, R_WR]

# Paper
PAPERS = {"Letter": (279.4, 215.9), "A4": (297.0, 210.0)}  # mm, landscape (W, H)
PAPER_RATIO_TOL = 0.05            # Letter-vs-A4 sanity check; scale uses the horizontal edge
PAPER_SLOPE_TOL = (0.06, 0.10)    # top/bottom and left/right edge slope difference
PAPER_MAX_A, PAPER_MAX_B = 12.0, 30.0   # Lab tint limits: paper passes, skin fails on A
PAPER_SIZE_REL = 0.002
MIN_PAPER_PX = 70
DEFAULT_OFFSET_MM = 100.0         # paper-to-body-center distance if the turn is skipped
PAPER_MIN_FRAMES, PAPER_MAX_FRAMES, STATURE_MIN_FRAMES = 20, 45, 15
PAPER_TARGET_SPREAD = 0.006       # finish early once paper + height are this consistent

# Stricter paper specs (initial detection)
PAPER_W_SHOULDER = (0.40, 1.20)   # paper width / shoulder-joint distance
PAPER_ASPECT = (1.15, 1.60)       # long/short side of the min-area rectangle
PAPER_RECT_FILL = 0.85            # blob area / min-area rectangle area
PAPER_SOLIDITY = 0.92             # blob area / convex hull area
PAPER_MAX_TEXTURE = 0.12          # gray std / mean inside the paper (plain sheet)
PAPER_MIN_BORDER_CONTRAST = 20    # gray levels brighter than the ring around it
PAPER_HAND_DIST = 0.6             # wrist to nearest top corner, x paper width
PAPER_CHECK_HINTS = {
    "paper_on_torso": "Hold the paper in front of your stomach",
    "paper_free": "The paper blends into something white - hold it against your shirt",
    "paper_hands": "Pinch the top two corners of the paper",
    "paper_texture": "Use a plain, blank sheet of paper",
    "paper_aspect": "Hold the paper sideways, flat and taut",
    "paper_enclosed": "Keep the paper inside your outline, edges over your shirt",
}

# Tracking once the paper is found
LOCK_FRAMES = 5                   # consistent detections needed before locking on
TRACK_MAX_MISSES = 6              # bad frames in a row before the lock is dropped
TRACK_MIN_SUPPORT = 0.5           # share of edge samples that must show a clear edge
TRACK_MAX_AREA_CHANGE = 0.12      # vs. size at lock time
TRACK_PARALLEL_DEG = 4.0          # opposite sides must stay parallel

# Capture
FRONT_FRAMES = 45
TURN_MIN_SEC, TURN_MAX_SEC = 6.0, 40.0
YAW_BINS, MIN_PER_BIN, MIN_BINS_FILLED = 12, 2, 9   # 15 deg bins over 180 deg
STABLE_SEC, HOLD_SEC, HINT_DELAY = 1.0, 2.0, 0.6
MOTION_SHIFT_PX = 0.6             # whole-image shift (at 320 px wide) = camera moved
MAX_MISSES = 15
FRONT_LOST_SEC = 2.0              # pose lost this long during scanning -> back to "Step 3"
TILT_WARN_DEG = 15
VP_EVERY, VP_HISTORY, VP_MIN_LINES, VP_INLIER_DEG, VP_MAX_TILT = 5, 15, 4, 1.5, 40

# Step 1: learn the empty scene while the person steps aside
BG_W = 320                  # background model width (pixels)
BG_MIN_SEEN = 3             # frames a pixel must be uncovered before it's trusted
BG_COVER_TARGET = 0.97      # share of the frame learned before moving on
BG_COVER_MIN = 0.85         # accepted after BG_MAX_SEC
BG_MIN_SEC, BG_MAX_SEC = 3.0, 25.0
BG_MIN_MOVES = 10           # frames of real person motion required
BG_DIFF = 28                # colour distance that counts as "not background"
BG_MOTION_DIFF = 20         # frame-to-frame change that counts as motion
LOCATE_TRAVEL = 0.8         # sideways step needed each way, x shoulder width

# Body model
CHEST_F = 0.33
DEPTH_RATIO = {"bust": 0.75, "waist": 0.78, "hip": 0.72}
LIMB_DEPTH_RATIO = 0.95
FALLBACK_W = {"upper_arm_w": 10, "forearm_w": 8, "thigh_w": 16,
              "calf_w": 11, "neck_w": 12, "head_w": 15}
SYS_LEN, SYS_WIDTH, SYS_CIRC, SYS_EST_CIRC = 0.003, 0.01, 0.015, 0.04
LANDMARK_ABS_CM = 1.2             # joint-based lengths (sleeve, inseam, rise)

DEBUG = {}                        # name -> (value, passed), drawn on screen
CAND_VIEW = {}                    # paper candidate mask, drawn on screen

STEPS = {  # state: (on-screen title, spoken prompt)
    "locate": ("Step 1 of 4: Step left, then right, then back to the middle",
               "Step 1. Stand about nine feet back, so your whole body, including your "
               "shoes, is in view. Don't hold the paper yet. Take a big step to your left, "
               "then a big step to your right, then come back to the middle."),
    "paper": ("Step 2 of 4: Hold the paper flat against your stomach",
              "Step 2. Stay in the middle and pick up the paper. Hold it sideways against "
              "your stomach. Pinch the top two corners and pull it taut. Keep your eyes "
              "on the camera."),
    "front_wait": ("Step 3 of 4: Drop the paper, arms out in an A shape",
                   "Step 3. Drop the paper. Hold your arms out in an A shape, with your feet apart."),
    "front": ("Step 3 of 4: Scanning - hold still", "Scanning. Hold still."),
    "turn_wait": ("Step 4 of 4: Hands on your head, stay on the same spot",
                  "Step 4. Stay on the same spot, and put both hands on your head."),
    "turn": ("Step 4 of 4: Turn slowly in a full circle", "Now turn slowly in place, one full circle."),
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


def fmt_height(v, u):
    ft, inch = divmod(to_in(v), 12)
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
    """Bottom of the shoe (zero point). Searches only just below each foot, so floor
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


# ------------------------------------------------------- background / bounds
class BackgroundModel:
    """Learns the empty scene while the person steps left and right: every pixel
    they uncover is averaged into the background. Frame differencing confirms the
    motion comes from the person, not someone walking by or a light flicker."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = self.cnt = self.prev = self.bg = self.known = None
        self.t0 = self.home = None
        self.span = [0.0, 0.0]          # furthest travel to the person's left / right
        self.moves, self.finished = 0, False

    def _small(self, img, interp=cv2.INTER_AREA):
        h, w = img.shape[:2]
        return cv2.resize(img, (BG_W, max(1, round(BG_W * h / w))), interpolation=interp)

    def _person(self, seg, grow=15):
        """Person mask at model resolution, grown to cover hair, edges and shadow."""
        p = self._small((seg > 0.2).astype(np.uint8), cv2.INTER_NEAREST)
        return cv2.dilate(p, np.ones((grow, grow), np.uint8)) > 0

    @property
    def coverage(self):
        return 0.0 if self.cnt is None else float((self.cnt >= BG_MIN_SEEN).mean())

    def progress(self):
        return min(1.0, self.coverage / BG_COVER_TARGET)

    def update(self, frame, seg, pts, body_ok):
        """Learns from one frame and returns the instruction to show."""
        hip_x = mid(pts[L_HIP], pts[R_HIP])[0]
        sh_px = max(dist(pts[L_SH], pts[R_SH]), 1.0)
        if self.home is None:
            if not body_ok:
                return None                 # start once the whole body is in view
            self.home, self.t0 = hip_x, time.monotonic()

        small = self._small(frame).astype(np.float32)
        person = self._person(seg)
        if self.sum is None:
            self.sum = np.zeros_like(small)
            self.cnt = np.zeros(small.shape[:2], np.float32)
        free = ~person
        self.sum[free] += small[free]
        self.cnt[free] += 1

        # Motion: what changed since the last frame, and is it the person?
        gray = cv2.cvtColor(small.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        if self.prev is not None:
            moving = cv2.absdiff(gray, self.prev) > BG_MOTION_DIFF
            n = int(moving.sum())
            on_person = float((moving & person).sum()) / max(n, 1)
            if n > 0.003 * moving.size and on_person > 0.7:
                self.moves += 1
            DEBUG["motion_on_person"] = (on_person, on_person > 0.7)
        self.prev = gray

        # The view is mirrored, so the person's left is screen-left
        off = (self.home - hip_x) / sh_px
        self.span = [max(self.span[0], off), max(self.span[1], -off)]
        elapsed, cov = time.monotonic() - self.t0, self.coverage
        DEBUG["bg_coverage"] = (cov, cov >= BG_COVER_TARGET)
        DEBUG["travel_left/right"] = (f"{self.span[0]:.1f}/{self.span[1]:.1f}",
                                      min(self.span) >= LOCATE_TRAVEL)
        DEBUG["person_motion"] = (f"{self.moves}/{BG_MIN_MOVES}", self.moves >= BG_MIN_MOVES)

        if self.span[0] < LOCATE_TRAVEL:
            return "Take a big step to your left"
        if self.span[1] < LOCATE_TRAVEL:
            return "Now take a big step to your right"
        if not (cov >= BG_COVER_TARGET or (elapsed > BG_MAX_SEC and cov >= BG_COVER_MIN)):
            return "Step to the side again, until the red area is gone"
        if self.moves < BG_MIN_MOVES:
            return "Step left and right once more"
        if abs(off) > 0.3:
            return "Come back to the middle"
        if elapsed >= BG_MIN_SEC:
            self.bg = self.sum / np.maximum(self.cnt, 1)[..., None]
            self.known = self.cnt >= BG_MIN_SEEN
            self.finished = True
        return "Got it - stay in the middle"

    def unseen(self, w, h):
        """Full-res mask of pixels not learned yet (shown in red)."""
        m = (self.cnt < BG_MIN_SEEN).astype(np.uint8)
        return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0

    def classify(self, frame, seg):
        """Full-res (foreground, known_background) masks against the empty scene."""
        h, w = frame.shape[:2]
        small = self._small(frame).astype(np.float32)
        ref = self.known & ~self._person(seg, 25)
        if ref.sum() > 200:                 # follow webcam auto-exposure changes
            small *= float(np.median(self.bg[ref])) / max(float(np.median(small[ref])), 1.0)
        d = np.abs(small - self.bg).max(axis=2)
        up = lambda m: cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        return up(self.known & (d > BG_DIFF)), up(self.known & (d <= BG_DIFF))


def paper_region(seg, fg, known_bg, pts):
    """Where the paper may be: the person's filled outline (segmentation OR
    background difference, holes filled so a paper the segmenter dropped is
    included), minus learned background outside the torso (arm gaps)."""
    m = ((seg > 0.5) | fg).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    blob, _ = person_blob(m.astype(np.float32), pts)
    if blob is None:
        return None
    cnts = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    region = np.zeros(blob.shape, np.uint8)
    cv2.drawContours(region, cnts, -1, 1, cv2.FILLED)
    # Inside the torso, background can't be used to exclude anything: a white
    # paper can match a white wall behind you. Outside it, it removes arm gaps.
    quad = np.array([pts[i] for i in (L_SH, R_SH, R_HIP, L_HIP)], np.float32)
    c = quad.mean(0)
    quad = c + (quad - c) * np.array([1.3, 1.4], np.float32)
    zone = np.zeros_like(region)
    cv2.fillConvexPoly(zone, quad.astype(np.int32), 1)
    return (region > 0) & ~(known_bg & (zone == 0))


# ------------------------------------------------------------ paper detection
def chroma(bgr):
    """< 1 for near-neutral colours (paper, even under warm light); > 1 for skin."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.maximum(np.abs(lab[..., 1] - 128) / PAPER_MAX_A, np.abs(lab[..., 2] - 128) / PAPER_MAX_B)


def bright_neutral(bgr, kern, blur=False):
    """Mask of bright, neutral (paper-like) pixels with thin stripes removed."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    c = chroma(bgr)
    t = cv2.threshold(cv2.GaussianBlur(g, (5, 5), 0) if blur else g, 0, 255,
                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
    m = cv2.morphologyEx(((g > t) & (c < 1)).astype(np.uint8), cv2.MORPH_OPEN, kern)
    return m, g.astype(np.float32), c, t


def paper_blob_checks(comp, st, gray, pts, vis, g, region_roi):
    """[(name, value, passed)] - every check must pass for a blob to count as paper."""
    x, y, bw, bh, area = (int(v) for v in st)
    H, W = comp.shape
    c8 = comp.astype(np.uint8)
    cnts = cv2.findContours(c8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = max(cnts, key=cv2.contourArea)
    _, (rw, rh), _ = cv2.minAreaRect(cnt)
    rect_fill = area / max(rw * rh, 1.0)
    solidity = area / max(cv2.contourArea(cv2.convexHull(cnt)), 1.0)
    aspect = max(rw, rh) / max(min(rw, rh), 1.0)

    # Plain surface inside, darker ring outside
    k = max(3, int(0.06 * min(bw, bh))) | 1
    inner = cv2.erode(c8, np.ones((k, k), np.uint8)) > 0
    if inner.sum() < 50:
        inner = comp
    ring = (cv2.dilate(c8, np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0) & \
           ~(cv2.dilate(c8, np.ones((k, k), np.uint8)) > 0)
    inside = gray[inner]
    texture = float(inside.std() / (inside.mean() + 1e-6))
    contrast = float(inside.mean() - gray[ring].mean()) if ring.any() else 0.0

    # Position: on the stomach, not merged with the search-box border
    ox, oy = g["off"]
    fx, fy = ox + x + bw / 2, oy + y + bh / 2
    on_torso = (abs(fx - g["cx"]) < 0.35 * g["sh_px"] and
                g["sh_y"] + 0.2 * g["torso"] < fy < g["hip_y"] + 0.3 * g["torso"])
    touches = x <= 1 or y <= 1 or x + bw >= W - 1 or y + bh >= H - 1

    # Surrounded by the person (shirt, hands), not by the room
    grown = cv2.dilate(c8, np.ones((5, 5), np.uint8)) > 0
    enclosed = not (grown & ~region_roi).any()

    # Hands near the top corners (only checked when wrists are clearly visible)
    corners = [(ox + x, oy + y), (ox + x + bw, oy + y)]
    wrists = [pts[i] for i in (L_WR, R_WR) if vis[i] > 0.6]
    hand = max((min(dist(p, c) for c in corners) for p in wrists), default=0.0) / max(bw, 1)

    wsh = bw / g["sh_px"]
    return [
        ("paper_w/shoulder", wsh, PAPER_W_SHOULDER[0] <= wsh <= PAPER_W_SHOULDER[1]),
        ("paper_aspect", aspect, PAPER_ASPECT[0] <= aspect <= PAPER_ASPECT[1]),
        ("paper_rect_fill", rect_fill, rect_fill >= PAPER_RECT_FILL),
        ("paper_solidity", solidity, solidity >= PAPER_SOLIDITY),
        ("paper_texture", texture, texture <= PAPER_MAX_TEXTURE),
        ("paper_contrast", contrast, contrast >= PAPER_MIN_BORDER_CONTRAST),
        ("paper_on_torso", "yes" if on_torso else "no", on_torso),
        ("paper_free", "no" if touches else "yes", not touches),
        ("paper_hands", hand, hand <= PAPER_HAND_DIST),
        ("paper_enclosed", "yes" if enclosed else "no", enclosed),
    ]


def subpix_run(profile, valid, t):
    """Sub-pixel threshold crossings at the start and end of the bright run."""
    idx = np.flatnonzero(valid)
    if idx.size < 3 or idx[0] == 0 or idx[-1] >= len(profile) - 1:
        return None
    i, j = int(idx[0]), int(idx[-1])
    a, b, c, d = (float(profile[k]) for k in (i - 1, i, j, j + 1))
    fs = float(np.clip((t - a) / (b - a), 0, 1)) if b != a else 0.5
    fe = float(np.clip((c - t) / (c - d), 0, 1)) if c != d else 0.5
    return (i - 1) + fs, j + fe


def robust_line(a, b):
    """Fit b = m*a + c, iteratively discarding outliers (e.g. a stray finger)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 8:
        return None
    keep = np.ones(a.size, bool)
    for _ in range(4):
        if keep.sum() < 6:
            return None
        m, c = np.polyfit(a[keep], b[keep], 1)
        r = b - (m * a + c)
        keep = np.abs(r) < max(0.75, 2.5 * (1.4826 * np.median(np.abs(r[keep])) + 1e-6))
    return (float(m), float(c)) if keep.sum() >= 6 else None


def order_corners(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s, d = pts.sum(axis=1), pts[:, 1] - pts[:, 0]
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)  # tl, tr, br, bl


def quad_from_mask(mask):
    """Fallback 4-corner fit of the paper region when edge tracing fails."""
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cnts = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
    if not cnts:
        return None
    hull = cv2.convexHull(max(cnts, key=cv2.contourArea))
    if cv2.contourArea(hull) < 400:
        return None
    peri = cv2.arcLength(hull, True)
    for eps in (0.02, 0.03, 0.04, 0.06):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            return order_corners(approx)
    return order_corners(cv2.boxPoints(cv2.minAreaRect(hull)))


def detect_paper(frame, pts, vis, vc, region):
    """Finds the paper held over the torso, looking only inside the person's
    outline. Returns (detection or None, hint)."""
    h, w = frame.shape[:2]
    sh_px = dist(pts[L_SH], pts[R_SH])
    sh_y, hip_y = mid(pts[L_SH], pts[R_SH])[1], mid(pts[L_HIP], pts[R_HIP])[1]
    torso = hip_y - sh_y
    if sh_px < 20 or torso < 30:
        return None, "Face the camera"
    k = max(3, int(0.05 * sh_px)) | 1
    kern = np.ones((k, k), np.uint8)

    # Coarse search: bright, neutral region inside the person that passes every check
    cx = sum(pts[i][0] for i in (L_SH, R_SH, L_HIP, R_HIP)) / 4
    x0, x1 = int(max(0, cx - 1.6 * sh_px)), int(min(w, cx + 1.6 * sh_px))
    y0, y1 = int(max(0, sh_y - 0.1 * torso)), int(min(h, hip_y + 0.6 * torso))
    if x1 - x0 < 30 or y1 - y0 < 30:
        return None, "Move to the center of the frame"
    roi = frame[y0:y1, x0:x1]
    region_roi = region[y0:y1, x0:x1]
    cand = bright_neutral(roi, kern, blur=True)[0] & region_roi.astype(np.uint8)
    CAND_VIEW["cand"] = (cand, x0, y0)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    if n <= 1:
        return None, "Hold the paper sideways, flat against your stomach"

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    geo = dict(off=(x0, y0), sh_px=sh_px, sh_y=sh_y, hip_y=hip_y, torso=torso, cx=cx)
    best, best_checks = None, None
    for i in 1 + np.argsort(stats[1:, cv2.CC_STAT_AREA])[::-1][:5]:   # 5 largest blobs
        if stats[i, cv2.CC_STAT_AREA] < 0.04 * sh_px ** 2:
            break
        checks = paper_blob_checks(labels == i, stats[i], roi_gray, pts, vis, geo, region_roi)
        if all(ok for *_, ok in checks):
            best, best_checks = i, checks
            break
        if best_checks is None or sum(c[2] for c in checks) > sum(c[2] for c in best_checks):
            best_checks = checks
    for name, v, ok in best_checks or []:
        DEBUG[name] = (v, ok)
    if best is None:
        failed = [name for name, _, ok in (best_checks or []) if not ok]
        hint = next((PAPER_CHECK_HINTS[f] for f in failed if f in PAPER_CHECK_HINTS),
                    "Hold the paper sideways, flat against your stomach")
        return None, hint
    bx, by, bw, bh, _ = stats[best]

    # Fine fit on a padded crop with a local threshold
    pad = max(8, int(0.15 * bw))
    X0, Y0 = max(0, x0 + bx - pad), max(0, y0 + by - pad)
    X1, Y1 = min(w, x0 + bx + bw + pad), min(h, y0 + by + bh + pad)
    m, cg, cs, t2 = bright_neutral(frame[Y0:Y1, X0:X1], kern)
    m = m & region[Y0:Y1, X0:X1].astype(np.uint8)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n2 <= 1:
        return None, "Hold the paper flat against your stomach"
    comp = lab2 == 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(comp)
    if xs.size < 200:
        return None, "Step a little closer to the camera"

    # Rotate so the paper's edges are axis-aligned (landscape or portrait)
    mx, my = float(xs.mean()), float(ys.mean())
    vx, vy = np.linalg.eigh(np.cov(np.vstack([xs - mx, ys - my])))[1][:, 1]
    ang = (math.degrees(math.atan2(vy, vx)) + 45) % 90 - 45
    if abs(ang) > 20:
        return None, "Hold the paper level"
    M = cv2.getRotationMatrix2D((mx, my), ang, 1.0)
    size = (m.shape[1], m.shape[0])
    rg, rs = (cv2.warpAffine(a, M, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
              for a in (cg, cs))
    rc = cv2.warpAffine(comp.astype(np.uint8), M, size, flags=cv2.INTER_NEAREST)
    valid = (rg > t2) & (rs < 1) & (cv2.dilate(rc, np.ones((3, 3), np.uint8)) > 0)
    cols, rows = np.flatnonzero(valid.any(axis=0)), np.flatnonzero(valid.any(axis=1))
    if cols.size < 20 or rows.size < 15:
        return None, "Step a little closer to the camera"
    cx0, cx1, ry0, ry1 = cols[0], cols[-1], rows[0], rows[-1]

    # Precise path: sub-pixel edge lines (outlier rows, e.g. fingers, are discarded)
    top, bot, left, right = [], [], [], []
    for x in range(int(cx0 + 0.1 * (cx1 - cx0)), int(cx0 + 0.9 * (cx1 - cx0)) + 1):
        r = subpix_run(rg[:, x], valid[:, x], t2)
        if r:
            top.append((x, r[0]))
            bot.append((x, r[1]))
    for y in range(int(ry0 + 0.1 * (ry1 - ry0)), int(ry0 + 0.9 * (ry1 - ry0)) + 1):
        r = subpix_run(rg[y, :], valid[y, :], t2)
        if r:
            left.append((y, r[0]))
            right.append((y, r[1]))
    DEBUG["edge_top_bottom"] = (len(top), len(top) >= 8)
    DEBUG["edge_sides"] = (len(left), len(left) >= 8)

    lines = None
    if min(len(top), len(left)) >= 8:
        fit = [robust_line(*zip(*p)) for p in (top, bot, left, right)]
        if None not in fit:
            lt, lb, ll, lr = fit
            dh, dv = abs(lt[0] - lb[0]), abs(ll[0] - lr[0])
            DEBUG["slope_top_bottom"] = (dh, dh <= PAPER_SLOPE_TOL[0])
            DEBUG["slope_left_right"] = (dv, dv <= PAPER_SLOPE_TOL[1])
            if dh <= PAPER_SLOPE_TOL[0] and dv <= PAPER_SLOPE_TOL[1]:
                lines = fit

    def inter(lh, lv):  # lh: y = a*x + b, lv: x = c*y + d
        (a, b), (c, d) = lh, lv
        y = (a * d + b) / (1 - a * c)
        return c * y + d, y

    if lines:
        lt, lb, ll, lr = lines
        xc, yc = (cx0 + cx1) / 2, (ry0 + ry1) / 2
        hgt = ((lb[0] * xc + lb[1]) - (lt[0] * xc + lt[1])) / math.sqrt(1 + ((lt[0] + lb[0]) / 2) ** 2)
        wid = ((lr[0] * yc + lr[1]) - (ll[0] * yc + ll[1])) / math.sqrt(1 + ((ll[0] + lr[0]) / 2) ** 2)
        quad_r = [inter(lt, ll), inter(lt, lr), inter(lb, lr), inter(lb, ll)]
        method, ratio_tol = "edges", PAPER_RATIO_TOL
    else:  # fallback: contour corners (noisier per frame, fine after the median)
        q = quad_from_mask(valid)
        if q is None:
            return None, "Hold the paper flat and taut"
        tl, tr, br, bl = q
        wid, hgt = (dist(tl, tr) + dist(bl, br)) / 2, (dist(tl, bl) + dist(tr, br)) / 2
        quad_r = [tuple(p) for p in q]
        method, ratio_tol = "contour", PAPER_RATIO_TOL + 0.03
    DEBUG["method"] = (method, True)

    hv = hgt * vc
    if not np.isfinite(wid) or not np.isfinite(hv) or wid <= 0 or hv <= 0:
        return None, "Hold the paper flat and taut"
    portrait = hv > wid
    DEBUG["orientation"] = ("portrait" if portrait else "landscape", True)
    if wid < (0.77 * MIN_PAPER_PX if portrait else MIN_PAPER_PX):
        return None, "Step a little closer to the camera"
    ratio = max(wid, hv) / min(wid, hv)
    name = min(PAPERS, key=lambda p: abs(ratio / (PAPERS[p][0] / PAPERS[p][1]) - 1))
    W, H = PAPERS[name]
    err = abs(ratio / (W / H) - 1)
    DEBUG["ratio_error"] = (err, err <= ratio_tol)
    DEBUG["paper"] = (name, True)
    if err > ratio_tol:
        return None, "Hold the paper flat, facing the camera straight on"

    Mi = cv2.invertAffineTransform(M)
    quad = [(Mi[0, 0] * x + Mi[0, 1] * y + Mi[0, 2] + X0,
             Mi[1, 0] * x + Mi[1, 1] * y + Mi[1, 2] + Y0) for x, y in quad_r]
    return {"wid": wid, "hv": hv, "name": name, "quad": quad,
            "horiz_mm": H if portrait else W}, "Hold still"


def summarize_paper(samples, statures, f_px):
    """Combines paper scale and body height measured in separate frames."""
    names = [s["name"] for s in samples]
    k, k_noise = robust_stats([s["horiz_mm"] / s["wid"] for s in samples])
    st, st_noise = robust_stats(statures)
    return {"name": max(set(names), key=names.count), "h0_mm": k * st,
            "spread": math.hypot(k_noise / k, st_noise / st), "z_mm": f_px * k}


# ------------------------------------------------------------ paper tracking
def _edge_samples(grayf, p, q, search, n=28):
    """Search along the outward normal of side p->q for the bright-to-dark step.
    Returns sub-pixel edge points and their gradient strength."""
    p, q = np.asarray(p, np.float32), np.asarray(q, np.float32)
    d = q - p
    L = float(np.hypot(d[0], d[1]))
    if L < 10:
        return None, None
    out = np.array([d[1], -d[0]], np.float32) / L        # outward for tl,tr,br,bl order
    t = np.linspace(0.15, 0.85, n, dtype=np.float32)      # skip the corners (fingers)
    base = p + t[:, None] * d
    offs = np.arange(-search, search + 1, dtype=np.float32)
    coords = base[:, None, :] + offs[None, :, None] * out[None, None, :]
    prof = cv2.remap(grayf, np.ascontiguousarray(coords[..., 0]), np.ascontiguousarray(coords[..., 1]),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    g = prof[:, :-1] - prof[:, 1:]                        # paper bright inside, darker outside
    rows, k = np.arange(n), np.argmax(g, axis=1)
    m = g.shape[1]
    a, b, c = g[rows, np.clip(k - 1, 0, m - 1)], g[rows, k], g[rows, np.clip(k + 1, 0, m - 1)]
    den = a - 2 * b + c
    delta = np.where((k > 0) & (k < m - 1) & (np.abs(den) > 1e-6), 0.5 * (a - c) / den, 0.0)
    pos = offs[0] + k + 0.5 + np.clip(delta, -0.5, 0.5)
    return base + pos[:, None] * out, b


def _fit_side(pe, peak, min_peak):
    """Robust line (homogeneous) through the edge points; drops fingers/outliers."""
    keep = peak >= min_peak
    support = float(keep.mean())
    line = None
    for _ in range(2):
        if keep.sum() < 8:
            return None, support
        vx, vy, x0, y0 = cv2.fitLine(pe[keep].astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        line = np.cross([x0, y0, 1.0], [x0 + vx, y0 + vy, 1.0])
        res = np.abs((pe[:, 0] - x0) * vy - (pe[:, 1] - y0) * vx)
        keep = keep & (res < 1.5)
    return line, support


class PaperTracker:
    """Locks onto the paper after a few consistent colour detections, then follows
    its four edges by gradient (no colour thresholds) until it is lost."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.quad, self.prev, self.misses, self.cands = None, None, 0, []

    @property
    def locked(self):
        return self.quad is not None

    @staticmethod
    def _size(q, vc):
        tl, tr, br, bl = q
        return (dist(tl, tr) + dist(bl, br)) / 2, (clen(tl, bl, vc) + clen(tr, br, vc)) / 2

    def offer(self, det, gray8, vc):
        """Feed a colour detection. Locks on after LOCK_FRAMES consistent ones."""
        q = np.array(det["quad"], np.float32)
        if self.cands:
            last = self.cands[-1]
            if (np.linalg.norm(q.mean(0) - np.array(last["quad"], np.float32).mean(0)) > 0.1 * det["wid"]
                    or abs(det["wid"] / last["wid"] - 1) > 0.05 or det["name"] != last["name"]
                    or (det["hv"] > det["wid"]) != (last["hv"] > last["wid"])):
                self.cands = []
        self.cands.append(det)
        if len(self.cands) >= LOCK_FRAMES:
            self._lock(det, q, gray8, vc)
        return self.locked

    def _lock(self, det, q, gray8, vc):
        grayf = cv2.GaussianBlur(gray8, (3, 3), 0).astype(np.float32)
        peaks = []
        for i in range(4):
            _, pk = _edge_samples(grayf, q[i], q[(i + 1) % 4], 4)
            if pk is None:
                self.cands = []
                return
            peaks.append(float(np.median(pk)))
        if min(peaks) < 8:                     # edges too faint to follow
            self.cands = []
            return
        W, H = PAPERS[det["name"]]
        wid, hv = self._size(q, vc)
        self.quad, self.prev, self.misses = q, gray8, 0
        self.contrast, self.name, self.horiz_mm = min(peaks), det["name"], det["horiz_mm"]
        self.ratio, self.portrait, self.area = W / H, hv > wid, wid * hv
        self.search = max(4, int(0.08 * wid))

    def _predict(self, gray8):
        """Move the last quad with the image motion around the paper (optical flow)."""
        mask = np.zeros(gray8.shape, np.uint8)
        cv2.fillConvexPoly(mask, self.quad.astype(np.int32), 255)
        k = max(9, int(0.15 * dist(self.quad[0], self.quad[1]))) | 1
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
        p0 = cv2.goodFeaturesToTrack(self.prev, 60, 0.01, 5, mask=mask)
        if p0 is None or len(p0) < 6:
            return self.quad
        p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray8, p0, None, winSize=(21, 21), maxLevel=3)
        ok = st.ravel() == 1
        if ok.sum() < 6:
            return self.quad
        M, _ = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0)
        return self.quad if M is None else cv2.transform(self.quad[None], M)[0]

    def _refine(self, grayf, q, search):
        lines, supports = [], []
        for i in range(4):
            pe, pk = _edge_samples(grayf, q[i], q[(i + 1) % 4], search)
            if pe is None:
                return None, 0.0
            line, sup = _fit_side(pe, pk, 0.4 * self.contrast)
            supports.append(sup)
            if line is None:
                return None, min(supports)
            lines.append(line)
        corners = []
        for i in range(4):                     # corner i = side i-1 x side i
            v = np.cross(lines[i - 1], lines[i])
            if abs(v[2]) < 1e-9:
                return None, 0.0
            corners.append(v[:2] / v[2])
        return np.array(corners, np.float32), min(supports)

    def _validate(self, q, pred, support, pts, vc):
        if support < TRACK_MIN_SUPPORT:
            return False, f"weak edges {support:.2f}"
        wid, hv = self._size(q, vc)
        if wid <= 0 or hv <= 0 or (hv > wid) != self.portrait:
            return False, "orientation changed"
        err = abs(max(wid, hv) / min(wid, hv) / self.ratio - 1)
        DEBUG["track_ratio_err"] = (err, err <= PAPER_RATIO_TOL)
        if err > PAPER_RATIO_TOL:
            return False, f"ratio err {err:.3f}"
        if abs(wid * hv / self.area - 1) > TRACK_MAX_AREA_CHANGE:
            return False, "size jumped"
        if np.max(np.linalg.norm(q - pred, axis=1)) > 0.15 * wid:
            return False, "corner jumped"
        tl, tr, br, bl = q
        ang = lambda a, b: math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        if abs(ang(tl, tr) - ang(bl, br)) > TRACK_PARALLEL_DEG or \
                abs(ang(tl, bl) - ang(tr, br)) > TRACK_PARALLEL_DEG:
            return False, "not rectangular"
        sh_px = dist(pts[L_SH], pts[R_SH])
        sh_y, hip_y = mid(pts[L_SH], pts[R_SH])[1], mid(pts[L_HIP], pts[R_HIP])[1]
        tcx = sum(pts[i][0] for i in (L_SH, R_SH, L_HIP, R_HIP)) / 4
        cx, cy = q.mean(0)
        if abs(cx - tcx) > 0.6 * sh_px or not (sh_y < cy < hip_y + 0.5 * (hip_y - sh_y)):
            return False, "left the torso"
        return True, ""

    def update(self, gray8, pts, vc):
        grayf = cv2.GaussianBlur(gray8, (3, 3), 0).astype(np.float32)
        pred = self._predict(gray8)
        self.prev = gray8
        q, support, cur = None, 0.0, pred
        for s in (self.search, max(3, self.search // 2)):   # coarse, then fine
            r, sup = self._refine(grayf, cur, s)
            if r is None:
                break
            cur, q, support = r, r, sup
        DEBUG["track_support"] = (support, support >= TRACK_MIN_SUPPORT)
        ok, why = (q is not None), "edges not found"
        if ok:
            ok, why = self._validate(q, pred, support, pts, vc)
        if not ok:
            self.quad = pred                   # keep following the motion
            self.misses += 1
            DEBUG["track"] = (f"miss {self.misses}: {why}", False)
            if self.misses > TRACK_MAX_MISSES:
                self.reset()
                return None, "Lost the paper - hold it flat against your stomach"
            return None, "Hold the paper flat and still"
        self.quad, self.misses = q, 0
        wid, hv = self._size(q, vc)
        DEBUG["track"] = ("locked", True)
        return {"wid": wid, "hv": hv, "name": self.name, "horiz_mm": self.horiz_mm,
                "quad": [(float(x), float(y)) for x, y in q]}, "Hold still"


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


def compute_results(paper, front, noise, turn):
    Z, H0 = paper["z_mm"], paper["h0_mm"]
    # The paper sits in front of the body's center; scale up to the body's plane
    if turn:
        height_mm = H0 / (1 - 0.5 * turn["waist_depth_rel"] * H0 / Z)
        offset, off_unc = 0.5 * turn["waist_depth_rel"] * height_mm, 10.0
    else:
        offset, off_unc = DEFAULT_OFFSET_MM, 20.0
        height_mm = H0 * (Z + offset) / Z
    Zb, hcm = Z + offset, height_mm / 10
    scale_rel = math.sqrt(paper["spread"] ** 2 + (off_unc / Zb) ** 2 + PAPER_SIZE_REL ** 2)

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
        ("Height (shoe sole to head)", hcm, unc(hcm, SYS_LEN), ""),
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
    info = {"height_cm": hcm, "paper": paper["name"], "distance_cm": Zb / 10,
            "paper_offset_cm": offset / 10, "turn_done": bool(turn)}
    return rows, av, info


def tilt_text(tilt, digits=1):
    return f"{tilt:+.{digits}f} deg" if tilt is not None else "--"


def print_table(rows, info, res, tilt, tilt_src):
    print(f"\nMeasurements ({info['paper']} paper, camera {res[0]}x{res[1]}, "
          f"~{info['distance_cm'] / 100:.1f} m away, tilt {tilt_text(tilt)} ({tilt_src})):")
    for label, v, u, note in rows:
        print(f"  {label:<28}{fmt_pm(v, u)}{'  (' + note + ')' if note else ''}")


def save_json(rows, info, res, tilt, tilt_src):
    data = {
        "camera_resolution": f"{res[0]}x{res[1]}",
        "paper": info["paper"],
        "distance_m": round(info["distance_cm"] / 100, 2),
        "camera_tilt_deg": round(tilt, 1) if tilt is not None else None,
        "tilt_source": tilt_src,
        "turn_scan_done": info["turn_done"],
        "height_reference": "bottom of shoe sole = 0, measured to top of head",
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
    """Feet are in frame if the shoe sole is found above the bottom edge."""
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
        return False, "Shoes are cut off: step back, or tilt the screen forward"
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
        return False, "Shoes are cut off"
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
    lines = [f"{info['paper']} paper | camera {res[0]}x{res[1]} | "
             f"~{info['distance_cm'] / 100:.1f} m | tilt {tilt_text(tilt, 0)} ({tilt_src})"]
    for label, v, u, note in rows:
        val = fmt_height(v, u) if label.startswith("Height") and v else fmt_pm(v, u)
        lines.append(f"{label}: {val}{' (' + note + ')' if note else ''}")
    lh, sc = int(22 * s), 0.5 * s
    roi = img[0:min(lh * len(lines) + 12, img.shape[0]), 0:min(int(760 * s), img.shape[1])]
    roi[:] = (roi * 0.35).astype(np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (10, int(20 * s) + lh * i), FONT, sc, (255, 255, 255), 1, cv2.LINE_AA)


def draw_debug(img):
    """Paper candidate pixels in magenta, plus each check (green pass / red fail)."""
    if "cand" in CAND_VIEW:
        m, x0, y0 = CAND_VIEW.pop("cand")
        sub, sel = img[y0:y0 + m.shape[0], x0:x0 + m.shape[1]], m > 0
        sub[sel] = (sub[sel] * 0.4 + np.array([255, 0, 255]) * 0.6).astype(np.uint8)
        cv2.rectangle(img, (x0, y0), (x0 + m.shape[1], y0 + m.shape[0]), (255, 0, 255), 1)
    s = img.shape[1] / 1280
    for i, (k, (v, ok)) in enumerate(DEBUG.items()):
        txt = f"{k}: {v:.3f}" if isinstance(v, (float, np.floating)) else f"{k}: {v}"
        cv2.putText(img, txt, (int(15 * s), int(30 * s) + i * int(26 * s)), FONT, 0.6 * s,
                    (0, 200, 0) if ok else (0, 0, 255), max(1, int(2 * s)), cv2.LINE_AA)


class HintFilter:
    """Shows a new hint only after it persists, so messages don't flicker."""

    def __init__(self, delay=HINT_DELAY):
        self.delay, self.shown, self.cand, self.t = delay, "", None, 0.0

    def __call__(self, hint, now):
        if hint == self.shown:
            self.cand = None
        else:
            if hint != self.cand:
                self.cand, self.t = hint, now
            if now - self.t >= self.delay or not self.shown:
                self.shown, self.cand = hint, None
        return self.shown


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


# -------------------------------------------------------------- paper step
class PaperStep:
    """Collects paper scale and full-body height in separate frames (so they don't
    both have to pass at once), while making sure the person stays on one spot.
    The paper is found by colour inside the person's outline, then tracked by edges."""

    def __init__(self):
        self.hints = HintFilter()
        self.tracker = PaperTracker()
        self.reset()

    def reset(self):
        self.paper, self.stature, self.anchor = [], [], None
        self.tracker.reset()

    def _moved(self, pts):
        cur = np.array([*mid(pts[L_HIP], pts[R_HIP]), dist(pts[L_SH], pts[R_SH])])
        if self.anchor is not None:
            dx, dy, ds = np.abs(cur - self.anchor)
            sh = self.anchor[2]
            if dx > 0.4 * sh or dy > 0.4 * sh or ds / sh > 0.15:
                self.reset()
                self.anchor = cur
                return True
            self.anchor = 0.9 * self.anchor + 0.1 * cur
        else:
            self.anchor = cur
        return False

    def update(self, frame, out, pts, vis, blob, tb, vc, stable, shift, now, region):
        DEBUG["cam_shift_px"] = (shift, stable)
        if not stable:
            hint = "Camera moving - hold still"
        elif self._moved(pts):
            hint = "Stay on the same spot"
        else:
            h, w = frame.shape[:2]
            body_ok, body_hint = full_body_check(pts, vis, blob, tb, w, h, CORE, 0.5)
            DEBUG["full_body"] = ("ok" if body_ok else body_hint[:30], body_ok)
            if body_ok:
                sole, crown = sole_and_crown(blob, pts, tb)
                self.stature.append((sole - crown) * vc)
                draw_height_ruler(out, blob, sole, crown)

            gray8 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.tracker.locked:
                det, det_hint = self.tracker.update(gray8, pts, vc)
            else:
                det, det_hint = detect_paper(frame, pts, vis, vc, region)
                if det and not self.tracker.offer(det, gray8, vc):
                    cv2.polylines(out, [np.array(det["quad"], np.int32)], True, (0, 255, 255), 2, cv2.LINE_AA)
                    det, det_hint = None, f"Paper found - locking on ({len(self.tracker.cands)}/{LOCK_FRAMES})"
            DEBUG["paper_lock"] = ("locked" if self.tracker.locked else "searching", self.tracker.locked)
            DEBUG["paper_check"] = ("ok" if det else det_hint[:40], bool(det))
            if det:
                self.paper.append(det)
                cv2.polylines(out, [np.array(det["quad"], np.int32)], True, (0, 255, 0), 2, cv2.LINE_AA)
            elif self.tracker.locked:   # following, but this frame failed validation
                cv2.polylines(out, [self.tracker.quad.astype(np.int32)], True, (0, 140, 255), 2, cv2.LINE_AA)

            if not body_ok and len(self.stature) < STATURE_MIN_FRAMES:
                hint = body_hint
            elif not det and len(self.paper) < PAPER_MIN_FRAMES:
                hint = det_hint
            else:
                hint = "Paper found - hold still, eyes on the camera"
        return self.hints(hint, now)

    def progress(self):
        DEBUG["paper_frames"] = (f"{len(self.paper)}/{PAPER_MIN_FRAMES}", True)
        DEBUG["height_frames"] = (f"{len(self.stature)}/{STATURE_MIN_FRAMES}", True)
        return min(len(self.paper) / PAPER_MIN_FRAMES, len(self.stature) / STATURE_MIN_FRAMES, 1.0)

    def result(self, f_px):
        """Summary once there are enough consistent frames, otherwise None."""
        if len(self.paper) < PAPER_MIN_FRAMES or len(self.stature) < STATURE_MIN_FRAMES:
            return None
        s = summarize_paper(self.paper, self.stature, f_px)
        return s if len(self.paper) >= PAPER_MAX_FRAMES or s["spread"] < PAPER_TARGET_SPREAD else None


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Webcam body measurements with a paper reference")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--max-width", type=int, default=1920, help="highest capture width to try")
    ap.add_argument("--hfov", type=float, default=68.0, help="camera horizontal field of view")
    ap.add_argument("--tilt-deg", type=float, default=None, help="manual camera tilt")
    ap.add_argument("--no-voice", action="store_true")
    args = ap.parse_args()

    ensure_model()
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=POSE_PATH),
        running_mode=vision.RunningMode.VIDEO, num_poses=1,
        min_pose_detection_confidence=0.6, min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6, output_segmentation_masks=True))
    cap, res = open_camera(args.camera, args.max_width)
    print(f"Camera delivers {res[0]}x{res[1]}")

    voice, motion = Voice(not args.no_voice), MotionMonitor()
    vp, paper_step = VanishingTilt(args.hfov, args.tilt_deg), PaperStep()
    bg, locate_hints = BackgroundModel(), HintFilter()
    state, front_s = None, []
    paper = front = noise = turn = ref = rows = av = info = None
    turn_bins, turn_t0, hold_t, misses = None, None, None, 0
    lost_t = None
    show = {"avatar": True, "mask": False, "lines": False, "debug": True}
    was_stable, t0, last_ts = True, time.perf_counter(), -1

    def go(new):
        nonlocal state, hold_t, misses, turn_bins, turn_t0
        state, hold_t, misses = new, None, 0
        if new == "locate":
            bg.reset()
            paper_step.reset()
            front_s.clear()
        elif new == "paper":
            paper_step.reset()
        elif new == "turn":
            turn_bins = {k: [[] for _ in range(YAW_BINS)] for k in ("bust", "waist", "hip")}
            turn_t0 = time.monotonic()
        voice.say(STEPS[new][1])

    def refresh():
        nonlocal rows, av, info
        if paper is not None and front is not None:
            rows, av, info = compute_results(paper, front, noise, turn)
            print_table(rows, info, res, vp.tilt, vp.source)
            save_json(rows, info, res, vp.tilt, vp.source)

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

    go("locate")
    while True:
        ok_, frame = cap.read()
        if not ok_:
            print("Camera read failed (check camera permissions).")
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        f_px = (w / 2) / math.tan(math.radians(args.hfov / 2))
        ts = last_ts = max(int((time.perf_counter() - t0) * 1000), last_ts + 1)
        result = pose.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), ts)
        now = time.monotonic()
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

        # ---------------- step 1: locate (learn the empty scene)
        if state == "locate":
            DEBUG.clear()
            if not stable:
                bg.reset()
                raw_hint = "Camera moving - keep the laptop still"
            elif not person:
                raw_hint = "Step into view"
            else:
                body_ok, body_hint = full_body_check(pts, vis, blob, tb, w, h, CORE, 0.5)
                raw_hint = bg.update(frame, mask, pts, body_ok) or body_hint
            hint = locate_hints(raw_hint, now)
            voice.say(hint, interrupt=False, min_gap=2)
            if bg.cnt is not None:
                sel = bg.unseen(w, h)
                out[sel] = (out[sel] * 0.6 + np.array([0, 0, 255]) * 0.4).astype(np.uint8)
            progress = bg.progress()
            if show["debug"]:
                draw_debug(out)
            if bg.finished:
                go("paper")

        # ---------------- step 2: paper (white check inside the person only)
        elif state == "paper":
            DEBUG.clear()
            if not stable:
                go("locate")              # the learned background no longer lines up
                hint = "Camera moved - redo step 1"
            elif person:
                fg, known = bg.classify(frame, mask)
                region = paper_region(mask, fg, known, pts)
                if region is None:
                    hint = "Step into view"
                else:
                    if show["debug"]:
                        cnts = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)[-2]
                        cv2.drawContours(out, cnts, -1, (255, 255, 0), 1, cv2.LINE_AA)
                    hint = paper_step.update(frame, out, pts, vis, blob, tb, vc, stable,
                                             motion.shift, now, region)
                    if hint.startswith(("Step", "Head", "Shoes", "Move", "Come", "Stay", "Lost", "Keep")):
                        voice.say(hint, interrupt=False, min_gap=5)
            else:
                hint = "Step into view"
            if state == "paper":
                progress = paper_step.progress()
                res_paper = paper_step.result(f_px)
                if show["debug"]:
                    draw_debug(out)
                if res_paper:
                    paper = res_paper
                    print(f"Paper: {paper['name']}, ~{paper['z_mm'] / 1000:.2f} m away, "
                          f"spread {100 * paper['spread']:.2f}%")
                    go("front_wait")

        # ---------------- step 3: front A-pose
        elif state in ("front_wait", "front"):
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

        # ---------------- step 4: turn in place
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
        cv2.imshow("Body Measure", out)

        key = chr(cv2.waitKey(1) & 0xFF)
        if key == "q":
            break
        elif key == "n" and state in ("turn_wait", "turn"):
            refresh()
            go("live")
        elif key == "r":
            paper = front = noise = turn = ref = rows = av = info = None
            go("locate")
        elif key in "lamd" and key:
            name = {"l": "lines", "a": "avatar", "m": "mask", "d": "debug"}[key]
            show[name] = not show[name]

    voice.close()
    cap.release()
    cv2.destroyAllWindows()
    pose.close()


if __name__ == "__main__":
    main