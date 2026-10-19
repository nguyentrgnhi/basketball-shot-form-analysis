"""Analyze a basketball shot from a side-on video.

    python analyze_shot.py my_shot.mp4 --side right --out output/

Finds the set point and the release, measures elbow / knee / arm / leg angles at those moments,
prints feedback, and writes skeleton-only images and a GIF (no pixels from your video) plus feedback.json.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from shotform.analysis import feedback, find_key_frames, release_angles, track_video
from shotform.ball import BallDetector
from shotform.pose import PoseTracker
from shotform.render import draw_skeleton, render_gif


def analyze(video, side="right", weights=None, stride=1, use_ball=True, max_frames=None):
    tracker = PoseTracker(side)
    detector = BallDetector(weights) if use_ball else None
    frames, info = track_video(video, tracker, detector, stride=stride, max_frames=max_frames)
    tracker.close()
    set_i, release_i = find_key_frames(frames, info["fps"])
    result = {"video": Path(video).name, "side": side, "frames": len(frames), "set_frame": None, "release_frame": None, "checks": []}
    if set_i is None or release_i is None or not (frames[set_i].pose and frames[release_i].pose):
        return frames, info, (set_i, release_i), result
    set_angles = frames[set_i].angles
    rel = release_angles(frames[release_i].pose)
    result.update(set_frame=frames[set_i].index, release_frame=frames[release_i].index,
                  checks=feedback(set_angles, rel))
    return frames, info, (set_i, release_i), result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video")
    parser.add_argument("--side", choices=["right", "left"], default="right", help="shooting hand")
    parser.add_argument("--weights", default=None, help="ball detector weights (default: models/ball_yolov8n.pt)")
    parser.add_argument("--out", default="output")
    parser.add_argument("--stride", type=int, default=1, help="analyze every Nth frame")
    parser.add_argument("--no-ball", action="store_true", help="skip ball detection (release = wrist peak)")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()

    frames, info, (set_i, release_i), result = analyze(args.video, args.side, args.weights, args.stride, not args.no_ball)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not result["checks"]:
        raise SystemExit("Could not find a clear shot (no body pose detected around the release). "
                         "Use a side-on view with the whole body in frame.")

    size = (info["width"], info["height"])
    for name, i in (("set_point", set_i), ("release", release_i)):
        marks = [(c["check"].split(": ")[1], c["value"], c["ok"]) for c in result["checks"] if c["check"].startswith(name.split("_")[0])]
        joint_of = {"elbow": "elbow", "knee bend": "knee", "arm extension": "elbow", "leg extension": "knee"}
        angles = [(joint_of[m[0]], f"{m[0]}: {m[1]:.0f}", m[2]) for m in marks if m[1] is not None]
        img = draw_skeleton(frames[i].pose, frames[i].ball, size, label=name.replace("_", " "), angles=angles)
        cv2.imwrite(str(out / f"{name}.png"), img)
    if not args.no_gif:
        render_gif(frames, info, str(out / "skeleton.gif"), key_frames=[("set", set_i), ("release", release_i)])
    (out / "feedback.json").write_text(json.dumps(result, indent=2))

    print(f"\nSet point at frame {result['set_frame']}, release at frame {result['release_frame']}\n")
    for c in result["checks"]:
        print(f"  {'✓' if c['ok'] else '✗'} {c['check']:<24} {c['value']}°   {'' if c['ok'] else c['message']}")
    print(f"\nWrote {out}/")


if __name__ == "__main__":
    main()
