import numpy as np

from shotform.analysis import Frame, feedback, find_key_frames
from shotform.geometry import angle_at, smooth


def test_right_angle_and_straight_line():
    assert abs(angle_at((0, 1), (0, 0), (1, 0)) - 90) < 1e-6
    assert abs(angle_at((-1, 0), (0, 0), (1, 0)) - 180) < 1e-6


def test_angle_uses_pixels_not_normalized_units():
    # a 45 degree elbow stays 45 degrees regardless of frame aspect ratio when given pixel coordinates
    assert abs(angle_at((100, 100), (0, 0), (100, 0)) - 45) < 1e-6


def test_degenerate_angle_is_nan():
    assert np.isnan(angle_at((0, 0), (0, 0), (1, 1)))


def test_smooth_ignores_gaps():
    out = smooth([1, np.nan, 3, 5, np.nan], window=3)
    assert np.isfinite(out).all()


def _frame(i, wrist_y, knee, ball=None):
    pose = {"shoulder": (0, 0), "elbow": (0, 10), "wrist": (0, wrist_y), "hip": (0, 0), "knee": (0, 10), "ankle": (0, 20)}
    return Frame(i, i / 30, pose, ball, {"elbow": 90.0, "knee": knee})


def test_key_frames_set_is_deepest_knee_bend_before_release():
    knees = [170, 160, 120, 110, 130, 160, 175, 178, 178, 178]
    wrist_y = [300, 300, 290, 280, 250, 200, 120, 100, 100, 110]  # wrist rises, peaks at frame 7-8
    frames = [_frame(i, w, k) for i, (w, k) in enumerate(zip(wrist_y, knees))]
    set_i, release_i = find_key_frames(frames, fps=30)
    assert set_i == 3
    assert release_i in (7, 8, 9)


def test_feedback_flags_bad_form():
    checks = feedback({"elbow": 140, "knee": 170}, {"arm": 120, "leg": 150})
    assert [c["ok"] for c in checks] == [False, False, False, False]
    good = feedback({"elbow": 90, "knee": 120}, {"arm": 175, "leg": 175})
    assert all(c["ok"] for c in good)
