import os
# Force software pixel buffers and completely bypass hardware texture lockups
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import argparse
import json
import shutil
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
import cv2

# Global buffers
latest_detections = []
ocr_time_ms = 0.0
is_processing_ocr = False

def open_camera(index, width, height):
    # Force DSHOW with manual compression flags to bypass Iris Xe buffer blocks
    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if camera.isOpened():
        # Force the camera to stream in compressed MJPEG mode (Massively unburdens shared RAM)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return camera
    return None

def crop_for_ocr(frame, roi_mode, roi_box):
    height, width = frame.shape[:2]
    if roi_mode == "full":
        return frame, 0, 0
    if roi_mode == "center":
        crop_width, crop_height = width // 2, height // 2
        x, y = (width - crop_width) // 2, (height - crop_height) // 2
    else:
        x, y, crop_width, crop_height = roi_box
    return frame[y:y + crop_height, x:x + crop_width], x, y

def ocr_worker(image_rgb, psm, min_confidence, offset_x, offset_y, session, width, height):
    global latest_detections, ocr_time_ms, is_processing_ocr
    try:
        import pytesseract
        from pytesseract import Output
        
        started = time.perf_counter()
        data = pytesseract.image_to_data(
            image_rgb,
            lang="eng",
            config=f"--oem 1 --psm {psm}",
            output_type=Output.DICT,
        )
        ocr_time_ms = (time.perf_counter() - started) * 1000
        
        temp_detections = []
        for i, word in enumerate(data["text"]):
            word = word.strip()
            try:
                confidence = float(data["conf"][i])
            except (TypeError, ValueError):
                continue
            if not word or confidence < min_confidence:
                continue
            x, y = int(data["left"][i]), int(data["top"][i])
            w, h = int(data["width"][i]), int(data["height"][i])
            points = [[x + offset_x, y + offset_y],
                      [x + w + offset_x, y + offset_y],
                      [x + w + offset_x, y + h + offset_y],
                      [x + offset_x, y + h + offset_y]]
            temp_detections.append({
                "text": word,
                "confidence": confidence / 100.0,
                "bbox_pixels": points,
            })
            
        latest_detections = temp_detections
        
        session["frames"].append({
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "frame_width": width,
            "frame_height": height,
            "ocr_time_ms": round(ocr_time_ms, 2),
            "detections": latest_detections,
        })
    except Exception as e:
        pass
    finally:
        is_processing_ocr = False

def main():
    global latest_detections, ocr_time_ms, is_processing_ocr
    
    parser = argparse.ArgumentParser(description="Live webcam OCR using Tesseract")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--roi", choices=("center", "full", "select"), default="center")
    parser.add_argument("--min-confidence", type=float, default=40.0)
    parser.add_argument("--psm", type=int, default=6)
    parser.add_argument("--output", default="tesseract_webcam.json")
    parser.add_argument("--tesseract-path", default=None)
    args = parser.parse_args()

    import pytesseract
    executable = args.tesseract_path or shutil.which("tesseract")
    if not executable:
        raise SystemExit("Tesseract executable was not found.")
    pytesseract.pytesseract.tesseract_cmd = executable

    # Initialise camera with hardware-safe settings
    camera = open_camera(args.camera, args.width, args.height)
    if camera is None:
        raise SystemExit(f"Could not open camera index {args.camera}.")

    try:
        ok, first_frame = camera.read()
        if not ok:
            raise SystemExit("Camera failed to return standard frame data.")
        
        roi_box = None
        if args.roi == "select":
            x, y, w, h = cv2.selectROI("Select OCR region, press Enter", first_frame, False, False)
            cv2.destroyWindow("Select OCR region, press Enter")
            if w <= 0 or h <= 0:
                raise SystemExit("No OCR region selected.")
            roi_box = (int(x), int(y), int(w), int(h))

        actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        session = {"created_utc": datetime.now(timezone.utc).isoformat(), "engine": "Tesseract", "frames": []}
        last_ocr_time = 0.0

        while True:
            ok, frame = camera.read()
            if not ok:
                break

            now = time.perf_counter()
            if (now - last_ocr_time >= args.interval) and not is_processing_ocr:
                image, offset_x, offset_y = crop_for_ocr(frame, args.roi, roi_box)
                # Clone the image explicitly into memory to sever ties with the live video channel buffer
                image_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
                
                h, w = frame.shape[:2]
                is_processing_ocr = True
                last_ocr_time = now
                
                t = threading.Thread(
                    target=ocr_worker, 
                    args=(image_rgb, args.psm, args.min_confidence, offset_x, offset_y, session, w, h),
                    daemon=True
                )
                t.start()

            if args.roi == "center":
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), (255, 180, 0), 2)
            elif roi_box:
                x, y, w, h = roi_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 180, 0), 2)

            for det in latest_detections:
                pts = det["bbox_pixels"]
                cv2.rectangle(frame, (pts[0][0], pts[0][1]), (pts[2][0], pts[2][1]), (0, 255, 0), 1)
                cv2.putText(frame, det["text"], (pts[0][0], pts[0][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            cv2.putText(frame, f"OCR Latency: {ocr_time_ms:.1f}ms", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow("Threaded Live OCR", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(session, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
