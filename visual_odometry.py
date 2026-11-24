"""Ground-plane VO: local right/forward metres and CCW yaw, with quality gates."""
from dataclasses import dataclass
import json
from pathlib import Path
import cv2
import numpy as np
from src.ground_calibration import project, validate_homography


@dataclass
class VOSettings:
    max_corners: int = 500
    min_tracks: int = 16
    min_inlier_ratio: float = 0.6
    max_fb_error_px: float = 1.5
    ransac_threshold_m: float = 0.08
    max_residual_m: float = 0.08
    max_scale_change: float = 0.08
    max_dt_s: float = 0.5
    max_speed_mps: float = 12.0
    max_yaw_rate_rad_s: float = 1.8
    min_blur_variance: float = 15.0
    max_ground_distance_m: float = 30.0


def fit_rigid_motion(old, new, settings):
    """Fit feature old->new, then invert it for camera motion in the old frame."""
    model, mask = cv2.estimateAffinePartial2D(np.float32(old), np.float32(new),
        method=cv2.RANSAC, ransacReprojThreshold=settings.ransac_threshold_m,
        maxIters=1500, confidence=0.995)
    if model is None or mask is None:
        raise ValueError("motion_fit_failed")
    keep = mask.ravel().astype(bool)
    if keep.sum() < settings.min_tracks or keep.mean() < settings.min_inlier_ratio:
        raise ValueError("insufficient_inliers")
    scale = float(np.hypot(model[0, 0], model[1, 0]))
    if abs(scale - 1) > settings.max_scale_change:
        raise ValueError("non_rigid_scale_change")
    a, b = np.asarray(old)[keep], np.asarray(new)[keep]
    ac, bc = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ac).T @ (b - bc))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = bc - rotation @ ac
    residual = float(np.sqrt(np.mean(np.sum((a @ rotation.T + translation - b)**2, axis=1))))
    if residual > settings.max_residual_m:
        raise ValueError("excessive_ground_residual")
    camera_delta = -rotation.T @ translation
    yaw = -float(np.arctan2(rotation[1, 0], rotation[0, 0]))
    return camera_delta, yaw, keep, residual, scale


class GroundPlaneVisualOdometry:
    def __init__(self, calibration_path, max_corners=500, min_tracks=16, settings=None):
        calibration = (calibration_path if isinstance(calibration_path, dict) else
                       json.loads(Path(calibration_path).read_text(encoding="utf-8")))
        self.settings = settings or VOSettings(max_corners=max_corners, min_tracks=min_tracks)
        self.image_to_ground = validate_homography(calibration["image_to_ground"])
        self.width, self.height = int(calibration["image_width"]), int(calibration["image_height"])
        if "roi_polygon_px" not in calibration:
            raise ValueError("Calibration lacks a measured ground ROI. Regenerate it with calibrate_ground_plane.py")
        self.roi = np.asarray(calibration["roi_polygon_px"], dtype=np.int32)
        if self.roi.ndim != 2 or self.roi.shape[1] != 2 or len(self.roi) < 3:
            raise ValueError("invalid calibration ROI")
        if np.any(self.roi < 0) or np.any(self.roi >= [self.width, self.height]):
            raise ValueError("calibration ROI outside image")
        self.mask = np.zeros((self.height, self.width), np.uint8)
        cv2.fillPoly(self.mask, [self.roi], 255)
        if np.count_nonzero(self.mask) < 100:
            raise ValueError("ground ROI is too small")
        self.reset()

    def reset(self):
        self.old_gray, self.points, self.last_timestamp = None, None, None
        self.debug_points = np.empty((0, 2))

    def _seed(self, gray):
        self.old_gray = gray
        self.points = cv2.goodFeaturesToTrack(gray, maxCorners=self.settings.max_corners,
            qualityLevel=0.01, minDistance=8, blockSize=7, mask=self.mask)

    def process_frame(self, frame, timestamp_s, frame_id=None):
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(f"calibration_resolution_mismatch: expected {self.width}x{self.height}, got {frame.shape[1]}x{frame.shape[0]}")
        timestamp_s = float(timestamp_s)
        dt = 0.0 if self.last_timestamp is None else timestamp_s - self.last_timestamp
        packet = dict(timestamp_s=timestamp_s, frame_id=frame_id, delta_time_s=max(0.0, dt),
            source="ground_plane_vo", motion_type="metric_ground_plane_vo", motion_valid=False,
            right_delta_m=0.0, forward_delta_m=0.0, yaw_delta_rad=0.0, confidence=0.0,
            tracked_features=0, inliers=0, inlier_ratio=0.0, tracking_error=None,
            fb_error_px=None, status="initializing")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = float(cv2.Laplacian(gray, cv2.CV_64F)[self.mask > 0].var())
        packet["blur_variance"] = blur
        self.last_timestamp = timestamp_s
        reason = None
        if self.old_gray is None:
            reason = "initializing"
        elif dt <= 0 or dt > self.settings.max_dt_s:
            reason = "invalid_frame_interval"
        elif blur < self.settings.min_blur_variance:
            reason = "blurred_frame"
        elif self.points is None or len(self.points) < self.settings.min_tracks:
            reason = "insufficient_features"
        if reason:
            self._seed(gray)
            packet["status"] = reason
            return packet
        try:
            kwargs = dict(winSize=(21, 21), maxLevel=3,
                          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
            nxt, good1, _ = cv2.calcOpticalFlowPyrLK(self.old_gray, gray, self.points, None, **kwargs)
            if nxt is None:
                raise ValueError("tracking_lost")
            back, good2, _ = cv2.calcOpticalFlowPyrLK(gray, self.old_gray, nxt, None, **kwargs)
            if back is None:
                raise ValueError("backward_tracking_lost")
            fb = np.linalg.norm(back.reshape(-1, 2) - self.points.reshape(-1, 2), axis=1)
            xy = nxt.reshape(-1, 2)
            inside = np.array([cv2.pointPolygonTest(self.roi, tuple(map(float, p)), False) >= 0 for p in xy])
            good = good1.ravel().astype(bool) & good2.ravel().astype(bool) & (fb <= self.settings.max_fb_error_px) & inside
            old, new = self.points.reshape(-1, 2)[good], xy[good]
            packet["tracked_features"] = len(old)
            packet["fb_error_px"] = float(np.median(fb[good])) if good.any() else None
            if len(old) < self.settings.min_tracks:
                raise ValueError("insufficient_consistent_tracks")
            a, b = project(old, self.image_to_ground), project(new, self.image_to_ground)
            near = (np.linalg.norm(a, axis=1) < self.settings.max_ground_distance_m) & (np.linalg.norm(b, axis=1) < self.settings.max_ground_distance_m)
            a, b, new = a[near], b[near], new[near]
            if len(a) < self.settings.min_tracks:
                raise ValueError("insufficient_ground_tracks")
            delta, yaw, keep, residual, scale = fit_rigid_motion(a, b, self.settings)
            if np.linalg.norm(delta) / dt > self.settings.max_speed_mps:
                raise ValueError("implausible_speed")
            if abs(yaw) / dt > self.settings.max_yaw_rate_rad_s:
                raise ValueError("implausible_yaw_rate")
            ratio = float(keep.mean())
            self.debug_points = new[keep]
            packet.update(right_delta_m=float(delta[0]), forward_delta_m=float(delta[1]),
                yaw_delta_rad=yaw, motion_valid=True, status="ok", inliers=int(keep.sum()),
                inlier_ratio=ratio, tracking_error=residual, scale=scale,
                confidence=float(ratio * np.exp(-residual / self.settings.max_residual_m)))
        except (ValueError, cv2.error, np.linalg.LinAlgError) as exc:
            packet["status"] = str(exc)
            self.debug_points = np.empty((0, 2))
        # Redetect inside the calibrated patch each frame to avoid feature starvation.
        self._seed(gray)
        return packet

    def annotate(self, frame, packet):
        image = frame.copy()
        cv2.polylines(image, [self.roi], True, (0, 200, 255), 2)
        for p in self.debug_points:
            cv2.circle(image, tuple(map(int, p)), 3, (60, 255, 60), -1)
        cv2.putText(image, f"VO: {packet['status']} | tracks {packet['tracked_features']}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        return image

    def birdseye(self, frame):
        # Debug viewport: -5..5 metres right, 0..15 metres forward, 40 pixels/metre.
        view = np.array([[40, 0, 200], [0, -40, 600], [0, 0, 1]], dtype=float)
        masked = cv2.bitwise_and(frame, frame, mask=self.mask)
        return cv2.warpPerspective(masked, view @ self.image_to_ground, (400, 600))
