"""From a video to a list of per-frame measurements, then to the two moments that matter."""
from dataclasses import dataclass, field

import numpy as np

from .geometry import angle_at, smooth

# Heuristic thresholds in degrees. They are rules of thumb, loosely set from measuring two
# professional reference clips (set-point elbow ~70-95, knee ~110-130; release arm ~150-170,
# legs ~165-172), not validated against coach ratings. Change them to suit your own shooters.
THRESHOLDS = {
    "set_elbow": (70, 100),     # elbow roughly square when the ball is set
    "set_knee_max": 130,        # knees visibly bent at the set point
    "release_arm_min": 150,     # shooting arm close to fully extended at release
    "release_leg_min": 160,     # legs extended through the release
}


@dataclass
class Frame:
    index: int
    time: float
    pose: dict | None = None
    ball: tuple | None = None
    angles: dict = field(default_factory=dict)  # elbow, knee (degrees), NaN when not visible


def measure(pose):
    """Elbow and knee angles from a pose dict (pixel coordinates)."""
    if pose is None:
        return {"elbow": float("nan"), "knee": float("nan")}
    return {
        "elbow": angle_at(pose["shoulder"], pose["elbow"], pose["wrist"]),
        "knee": angle_at(pose["hip"], pose["knee"], pose["ankle"]),
    }


def track_video(path, pose_tracker, ball_detector=None, stride=1, max_frames=None):
    """Run pose (and optionally ball) detection over a video. Returns (frames, info)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Phone videos are often stored sideways with a rotation flag; make sure frames come out upright.
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    width = height = None
    frames, index = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok or (max_frames and len(frames) >= max_frames):
            break
        if width is None:
            height, width = bgr.shape[:2]
        if index % stride == 0:
            pose = pose_tracker.process(bgr)
            ball = ball_detector.detect(bgr) if ball_detector else None
            frames.append(Frame(index, index / fps, pose, ball, measure(pose)))
        index += 1
    cap.release()
    return frames, {"fps": fps / stride, "width": width, "height": height, "stride": stride}


def _series(frames, getter):
    return np.array([getter(f) for f in frames], float)


def find_key_frames(frames, fps):
    """Indices (into `frames`) of the set point and the release, either may be None.

    Release: the last frame the ball is still detected in the shooting hand within a third of a second
    of the wrist's highest point (falls back to the wrist peak itself when the ball is not detected,
    which happens often with motion blur).
    Set point: the deepest knee bend in the 3 seconds leading up to the release.
    """
    wrist_y = smooth(_series(frames, lambda f: f.pose["wrist"][1] if f.pose else np.nan))
    if not np.isfinite(wrist_y).any():
        return None, None
    peak = int(np.nanargmin(wrist_y))  # smaller y = higher in the image

    release = peak
    held = []
    for i in range(max(0, peak - int(0.3 * fps)), min(len(frames), peak + int(0.1 * fps) + 1)):
        f = frames[i]
        if f.pose and f.ball:
            forearm = np.hypot(f.pose["wrist"][0] - f.pose["elbow"][0], f.pose["wrist"][1] - f.pose["elbow"][1])
            distance = np.hypot(f.ball[0] - f.pose["wrist"][0], f.ball[1] - f.pose["wrist"][1])
            if distance <= max(forearm, f.ball[2] * 2):
                held.append(i)
    if held:
        release = held[-1]

    knee = smooth(_series(frames, lambda f: f.angles.get("knee", np.nan)))
    start = max(0, release - int(3.0 * fps))
    window = knee[start: release + 1]
    set_point = None
    if len(window) and np.isfinite(window).any():
        set_point = start + int(np.nanargmin(window))
    return set_point, release


def feedback(set_angles, release_angles, thresholds=THRESHOLDS):
    """Compare the two key moments with the rules. Returns a list of {check, value, ok, message}."""
    checks = []

    def add(name, value, ok, message):
        checks.append({"check": name, "value": None if np.isnan(value) else round(float(value), 1),
                       "ok": bool(ok), "message": "OK" if ok else message})

    lo, hi = thresholds["set_elbow"]
    elbow = set_angles["elbow"]
    add("set point: elbow", elbow, lo <= elbow <= hi,
        "Elbow should be close to a right angle (about 90 degrees) when you set the ball.")
    knee = set_angles["knee"]
    add("set point: knee bend", knee, knee <= thresholds["set_knee_max"],
        "Bend your knees more to load power from the legs.")
    arm = release_angles["arm"]
    add("release: arm extension", arm, arm >= thresholds["release_arm_min"],
        "Extend the shooting arm fully at release.")
    leg = release_angles["leg"]
    add("release: leg extension", leg, leg >= thresholds["release_leg_min"],
        "Straighten your legs through the release.")
    return checks


def release_angles(pose):
    return {"arm": angle_at(pose["shoulder"], pose["elbow"], pose["wrist"]),
            "leg": angle_at(pose["hip"], pose["knee"], pose["ankle"])}
