"""Validated ground-plane calibration and projection helpers."""
import cv2
import numpy as np


def project(points, matrix):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    homogeneous = np.column_stack([points, np.ones(len(points))]) @ np.asarray(matrix).T
    if not np.isfinite(homogeneous).all() or np.any(np.abs(homogeneous[:, 2]) < 1e-9):
        raise ValueError("homography crosses the horizon or contains non-finite values")
    return homogeneous[:, :2] / homogeneous[:, 2:]


def validate_homography(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("homography must be a finite 3x3 matrix")
    if np.linalg.cond(matrix) > 1e12:
        raise ValueError("homography is singular or numerically unstable")
    return matrix


def _points(rows):
    pixels = np.asarray([r["pixel_xy"] for r in rows], dtype=float)
    ground = np.asarray([r["ground_right_forward_m"] for r in rows], dtype=float)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or ground.shape != pixels.shape:
        raise ValueError("each correspondence needs two pixel and two ground coordinates")
    if not np.isfinite(pixels).all() or not np.isfinite(ground).all():
        raise ValueError("coordinates must be finite")
    return pixels, ground


def fit_calibration(source, max_residual_m=0.15):
    if not np.isfinite(max_residual_m) or max_residual_m <= 0:
        raise ValueError("max_residual_m must be finite and positive")
    rows = [r for r in source["points"] if not r.get("validation", False)]
    if len(rows) < 4:
        raise ValueError("at least four fitting points are required, plus optional validation points")
    pixels, ground = _points(rows)
    width, height = int(source["image_width"]), int(source["image_height"])
    if width <= 0 or height <= 0 or np.any(pixels < 0) or np.any(pixels >= [width, height]):
        raise ValueError("pixel coordinates must be inside the image")
    for points in (pixels, ground):
        singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
        if singular[0] == 0 or singular[1] / singular[0] < 0.01 or len(np.unique(points, axis=0)) < 4:
            raise ValueError("duplicate, collinear, or nearly collinear calibration points")
    matrix, inliers = cv2.findHomography(pixels, ground, cv2.RANSAC, max_residual_m)
    matrix = validate_homography(matrix)
    errors = np.linalg.norm(project(pixels, matrix) - ground, axis=1)
    if inliers is None or int(inliers.sum()) < 4 or float(errors.max()) > max_residual_m:
        raise ValueError(f"calibration residual exceeds {max_residual_m} m; recheck all markers")
    roi = cv2.convexHull(pixels.astype(np.float32)).reshape(-1, 2)
    denominators = np.column_stack([roi, np.ones(len(roi))]) @ matrix[2]
    if np.min(denominators) * np.max(denominators) <= 0:
        raise ValueError("calibration region crosses the homography horizon")
    project(ground, np.linalg.inv(matrix))
    validation = [r for r in source["points"] if r.get("validation", False)]
    validation_errors = []
    if validation:
        vp, vg = _points(validation)
        validation_errors = np.linalg.norm(project(vp, matrix) - vg, axis=1).tolist()
        if max(validation_errors) > max_residual_m:
            raise ValueError("held-out validation residual is too large; recalibrate")
    return dict(schema_version=2, image_width=width, image_height=height,
                image_to_ground=matrix.tolist(), roi_polygon_px=roi.tolist(),
                coordinate_order="right_m, forward_m", calibration_point_count=len(rows),
                inlier_count=int(inliers.sum()), residuals_m=errors.tolist(),
                rms_error_m=float(np.sqrt(np.mean(errors ** 2))),
                validation_residuals_m=validation_errors,
                source_description=source.get("source_description", "unspecified"),
                note="Fixed camera pose and flat ground required; fitting residual is not motion accuracy.")
