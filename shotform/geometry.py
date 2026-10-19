import numpy as np


def angle_at(a, b, c) -> float:
    """Angle in degrees at vertex b between the segments b->a and b->c. Use pixel coordinates:
    normalized (0-1) coordinates distort angles on non-square frames."""
    ba = np.asarray(a, float) - np.asarray(b, float)
    bc = np.asarray(c, float) - np.asarray(b, float)
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))


def smooth(values, window=5):
    """Moving average that ignores NaNs (gaps in pose or ball tracking)."""
    values = np.asarray(values, float)
    out = np.full_like(values, np.nan)
    half = window // 2
    for i in range(len(values)):
        chunk = values[max(0, i - half): i + half + 1]
        if np.isfinite(chunk).any():
            out[i] = np.nanmean(chunk)
    return out
