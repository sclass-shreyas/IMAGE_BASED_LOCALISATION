import cv2
import numpy as np
import json
import time
import math

class MotionEstimator:
    def __init__(self, max_corners=100):
        self.feature_params = dict(maxCorners=max_corners, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.old_gray = None
        self.p0 = None

    def process_frame(self, frame, timestamp_s, frame_id=None):
        """Return relative image motion as a dict; dx/dy are explicitly pixels."""
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.old_gray is None:
            self.old_gray = frame_gray.copy()
            self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
            return {"timestamp_s": float(timestamp_s), "frame_id": frame_id,
                    "dx_px": 0.0, "dy_px": 0.0, "d_heading_rad": 0.0,
                    "confidence": 0.0, "motion_type": "relative_image"}

        if self.p0 is not None and len(self.p0) > 0:
            p1, status, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)

            if p1 is None or status is None:
                good_new = np.empty((0, 2), dtype=np.float32)
                good_old = np.empty((0, 2), dtype=np.float32)
            else:
                good_new = p1[status.reshape(-1) == 1]
                good_old = self.p0[status.reshape(-1) == 1]

            if len(good_new) >= 2:
                M, inliers = cv2.estimateAffinePartial2D(good_old, good_new)
                if M is not None:
                    dx = M[0, 2]
                    dy = M[1, 2]
                    d_heading = math.atan2(M[1, 0], M[0, 0])
                    confidence = float(np.sum(inliers) / len(good_new)) if inliers is not None else 0.0
                else:
                    dx, dy, d_heading, confidence = 0.0, 0.0, 0.0, 0.0
            else:
                dx, dy, d_heading, confidence = 0.0, 0.0, 0.0, 0.0

            self.old_gray = frame_gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)
            
            data = {
                "timestamp_s": float(timestamp_s),
                "frame_id": frame_id,
                "dx_px": round(float(dx), 3),
                "dy_px": round(float(dy), 3),
                "d_heading_rad": round(float(d_heading), 4),
                "confidence": round(float(confidence), 2),
                "motion_type": "relative_image"
            }
            return data
            
        else:
            self.old_gray = frame_gray.copy()
            self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
            return {"timestamp_s": float(timestamp_s), "frame_id": frame_id,
                    "dx_px": 0.0, "dy_px": 0.0, "d_heading_rad": 0.0,
                    "confidence": 0.0, "motion_type": "relative_image"}

if __name__ == "__main__":
    estimator = MotionEstimator()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open camera 0. Pass a video path through MotionEstimator in an application.")
    session_start = time.perf_counter()
    frame_id = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_id += 1
        motion = estimator.process_frame(frame, time.perf_counter() - session_start, frame_id)
        print(json.dumps(motion))
        
        cv2.imshow("Testing Motion Estimator", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
