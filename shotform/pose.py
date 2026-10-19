import mediapipe as mp

mp_pose = mp.solutions.pose
PoseLandmark = mp_pose.PoseLandmark

JOINTS = ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")


class PoseTracker:
    """MediaPipe pose for one person; returns pixel coordinates of the joints on the shooting side."""

    def __init__(self, side="right"):
        if side not in ("right", "left"):
            raise ValueError("side must be 'right' or 'left'")
        self.side = side
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, smooth_landmarks=True)

    def process(self, frame_bgr):
        import cv2
        h, w = frame_bgr.shape[:2]
        result = self.pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not result.pose_landmarks:
            return None
        lm = result.pose_landmarks.landmark
        out = {}
        for joint in JOINTS:
            p = lm[getattr(PoseLandmark, f"{self.side.upper()}_{joint.upper()}").value]
            out[joint] = (p.x * w, p.y * h)
        # full skeleton (for drawing), pixel coordinates + visibility
        out["all"] = [(p.x * w, p.y * h, p.visibility) for p in lm]
        return out

    def close(self):
        self.pose.close()
