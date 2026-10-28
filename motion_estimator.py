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

    def process_frame(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.old_gray is None:
            self.old_gray = frame_gray.copy()
            self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
            return json.dumps({"timestamp": round(time.time(), 2), "dx": 0.0, "dy": 0.0, "d_heading": 0.0, "confidence": 0.0})

        if self.p0 is not None and len(self.p0) > 0:
            p1, status, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)
            
            good_new = p1[status == 1]
            good_old = self.p0[status == 1]

            if len(good_new) >= 2:
                M, inliers = cv2.estimateAffinePartial2D(good_old, good_new)
                if M is not None:
                    dx = M[0, 2]
                    dy = M[1, 2]
                    d_heading = math.atan2(M[1, 0], M[0, 0])
                    confidence = float(np.sum(inliers) / len(good_new))
                else:
                    dx, dy, d_heading, confidence = 0.0, 0.0, 0.0, 0.0
            else:
                dx, dy, d_heading, confidence = 0.0, 0.0, 0.0, 0.0

            self.old_gray = frame_gray.copy()
            self.p0 = good_new.reshape(-1, 1, 2)
            
            data = {
                "timestamp": round(time.time(), 2),
                "dx": round(float(dx), 3),
                "dy": round(float(dy), 3),
                "d_heading": round(float(d_heading), 4),
                "confidence": round(float(confidence), 2)
            }
            return json.dumps(data)
            
        else:
            self.old_gray = frame_gray.copy()
            self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
            return json.dumps({"timestamp": round(time.time(), 2), "dx": 0.0, "dy": 0.0, "d_heading": 0.0, "confidence": 0.0})

if __name__ == "__main__":
    estimator = MotionEstimator()
    cap = cv2.VideoCapture("test_drive.mp4")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        json_output = estimator.process_frame(frame)
        print(json_output)
        
        cv2.imshow("Testing Motion Estimator", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()