"""Skeleton-only drawings: the body as a stick figure on a plain background.

Nothing from the original video is drawn, so the output shows form without showing the person.
"""
import cv2
import numpy as np

from .pose import mp_pose

BG = (39, 24, 17)          # BGR dark navy
LIMB = (170, 150, 140)
OK = (113, 204, 46)        # green
BAD = (60, 76, 231)        # red
BALL = (0, 140, 255)       # orange
TEXT = (235, 235, 235)

# Connections between body landmarks (indices >= 11 skip the face), plus a plain circle for the head.
BODY = [(a, b) for a, b in mp_pose.POSE_CONNECTIONS if a >= 11 and b >= 11]


def _canvas(width, height, max_height):
    scale = min(1.0, max_height / height)
    w, h = int(width * scale), int(height * scale)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = BG
    return img, scale


def draw_skeleton(pose, ball, size, label="", angles=(), max_height=720, trail=()):
    """One frame. `angles` is a sequence of (joint, text, ok) drawn next to the joint on the shooting side."""
    img, s = _canvas(size[0], size[1], max_height)
    thickness = max(2, int(img.shape[0] / 240))
    if pose is not None:
        pts = pose["all"]
        for a, b in BODY:
            if pts[a][2] > 0.3 and pts[b][2] > 0.3:
                cv2.line(img, (int(pts[a][0] * s), int(pts[a][1] * s)), (int(pts[b][0] * s), int(pts[b][1] * s)), LIMB, thickness)
        ears = [pts[i] for i in (7, 8) if pts[i][2] > 0.3]
        nose = pts[0]
        if nose[2] > 0.3:
            radius = int((np.hypot(ears[0][0] - ears[-1][0], ears[0][1] - ears[-1][1]) / 2 if len(ears) == 2 else 40) * s * 1.1)
            cv2.circle(img, (int(nose[0] * s), int(nose[1] * s)), max(radius, 8), LIMB, thickness)
        for joint, text, ok in angles:
            x, y = pose[joint]
            color = OK if ok else BAD
            cv2.circle(img, (int(x * s), int(y * s)), thickness * 3, color, -1)
            scale_font = img.shape[0] / 1100
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale_font, max(1, thickness - 1))
            tx = int(x * s) + 12
            if tx + tw > img.shape[1] - 6:  # keep the label inside the image
                tx = max(6, int(x * s) - 12 - tw)
            ty = int(y * s) - 8 if tx + tw <= img.shape[1] - 6 else int(y * s) + 28
            cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                        scale_font, color, max(1, thickness - 1), cv2.LINE_AA)
    for i, (x, y) in enumerate(trail):
        cv2.circle(img, (int(x * s), int(y * s)), max(2, thickness), BALL, -1)
    if ball is not None:
        cv2.circle(img, (int(ball[0] * s), int(ball[1] * s)), max(int(ball[2] * s), 6), BALL, thickness)
    if label:
        cv2.putText(img, label, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, img.shape[0] / 800, TEXT, 2, cv2.LINE_AA)
    return img


def render_gif(frames, info, path, key_frames=(), step=2, max_height=480, trail_length=25):
    """Skeleton animation with the ball's path; key moments are labelled."""
    import imageio.v2 as imageio
    labels = {i: name for name, i in key_frames if i is not None}
    images, trail = [], []
    for i in range(0, len(frames), step):
        f = frames[i]
        if f.ball:
            trail.append((f.ball[0], f.ball[1]))
        near = next((name for j, name in labels.items() if abs(j - i) <= step), "")
        img = draw_skeleton(f.pose, f.ball, (info["width"], info["height"]), label=near.upper(),
                            max_height=max_height, trail=trail[-trail_length:])
        images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    imageio.mimsave(path, images, duration=step / max(info["fps"] * step, 1) * step, loop=0)
