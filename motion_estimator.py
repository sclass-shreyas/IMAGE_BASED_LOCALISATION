import cv2
import numpy as np

# --- THE ENGINE (What Person 4 will import later) ---
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
            return {"dx": 0.0, "dy": 0.0, "confidence": 0.0}

        if self.p0 is not None and len(self.p0) > 0:
            p1, status, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)
        else:
            p1, status = None, None

        dx, dy, confidence = 0.0, 0.0, 0.0

        if p1 is not None and status is not None:
            good_new = p1[status == 1]
            good_old = self.p0[status == 1]

            if len(good_new) > 10:
                # RANSAC isolates the true motion and ignores the hood/sky automatically
                transform, inliers = cv2.estimateAffinePartial2D(good_old, good_new, method=cv2.RANSAC)
                
                if transform is not None:
                    dx = float(transform[0, 2])
                    dy = float(transform[1, 2])
                    
                    # Keep only the valid points for the next frame
                    good_new_inliers = good_new[inliers.ravel() == 1]
                    self.p0 = good_new_inliers.reshape(-1, 1, 2)
                    confidence = float(min(1.0, len(good_new_inliers) / 50.0))
                else:
                    self.p0 = good_new.reshape(-1, 1, 2)
            else:
                self.p0 = None
        else:
            self.p0 = None

        if self.p0 is None or len(self.p0) < 30:
            self.p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **self.feature_params)

        self.old_gray = frame_gray.copy()
        return {"dx": dx, "dy": dy, "confidence": confidence}


# --- THE TEST LOOP (To run it on your video right now) ---
if __name__ == "__main__":
    # 1. Load the video
    cap = cv2.VideoCapture('test_drive.mp4')
    
    # 2. Initialize your new robust estimator
    estimator = MotionEstimator()

    print("Starting robust tracking... Press ESC to stop.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # 3. Pass the raw frame to your class
        motion_data = estimator.process_frame(frame)
        
        # 4. Print the clean output to the terminal
        print(f"Motion: dx={motion_data['dx']:+.2f}, dy={motion_data['dy']:+.2f} | Confidence: {motion_data['confidence']:.2f}")

        # 5. Draw green dots on the tracked points so you can verify it visually
        if estimator.p0 is not None:
            for pt in estimator.p0:
                x, y = pt.ravel()
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

        # Show the video
        cv2.imshow('RANSAC Motion Tracking', frame)
        
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()