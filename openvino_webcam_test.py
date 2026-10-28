"""Live English printed-text OCR on Intel GPU using OpenVINO.

Models: Open Model Zoo horizontal-text-detection-0001 and text-recognition-0014.
The models are downloaded into ./models on first run and compiled explicitly
for the selected OpenVINO device (GPU by default; no silent CPU fallback).

Install once in the active venv:
    python -m pip install openvino opencv-python numpy

Run:
    python openvino_webcam_test.py --camera 1 --backend dshow --roi select

Press Q or Esc in the preview to quit. This model pair is intended for printed
English text lines, not isolated handwritten characters.
"""

import argparse
import json
import os
import queue
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


MODEL_ROOT = Path(__file__).resolve().parent / "models"
MODEL_BASE_URL = (
    "https://storage.openvinotoolkit.org/repositories/"
    "open_model_zoo/2023.0/models_bin/1"
)
MODEL_NAMES = ("horizontal-text-detection-0001", "text-recognition-0014")


def download_model(model_name, force=False):
    model_dir = MODEL_ROOT / model_name / "FP16"
    model_dir.mkdir(parents=True, exist_ok=True)
    xml_path = model_dir / f"{model_name}.xml"
    bin_path = model_dir / f"{model_name}.bin"
    paths = (xml_path, bin_path)
    if not force and all(path.is_file() and path.stat().st_size > 0 for path in paths):
        return xml_path

    temporary_paths = []
    try:
        for path in paths:
            temporary = path.with_suffix(path.suffix + ".download")
            temporary_paths.append(temporary)
            url = f"{MODEL_BASE_URL}/{model_name}/FP16/{path.name}"
            print(f"Downloading {path.name} ...", flush=True)
            urllib.request.urlretrieve(url, temporary)
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise RuntimeError(f"Downloaded model file is empty: {path.name}")
        # Replace the cached pair only after both downloads complete.
        for temporary, path in zip(temporary_paths, paths):
            os.replace(temporary, path)
    finally:
        for temporary in temporary_paths:
            if temporary.exists():
                temporary.unlink()
    return xml_path


def open_camera(index, width, height, backend_name):
    backends = {"any": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW,
                "msmf": cv2.CAP_MSMF}
    camera = cv2.VideoCapture(index, backends[backend_name])
    if not camera.isOpened():
        camera.release()
        return None
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camera


class BufferlessCapture:
    """Drain camera frames continuously and retain only the newest one."""
    def __init__(self, camera):
        self.camera = camera
        self.frames = queue.Queue(maxsize=1)
        self.running = True
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        while self.running:
            ok, frame = self.camera.read()
            if not ok:
                time.sleep(0.01)
                continue
            if self.frames.full():
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
            try:
                self.frames.put_nowait(frame)
            except queue.Full:
                pass

    def read(self, timeout=0.5):
        try:
            return self.frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def release(self):
        self.running = False
        self.camera.release()
        self.thread.join(timeout=1.0)


def crop_frame(frame, mode, roi):
    height, width = frame.shape[:2]
    if mode == "full":
        return frame, 0, 0
    if mode == "center":
        x, y = width // 4, height // 4
        w, h = width // 2, height // 2
    else:
        x, y, w, h = roi
    return frame[y:y + h, x:x + w], x, y


def decode_text(logits):
    """CTC greedy decode for the Open Model Zoo English recognizer."""
    alphabet = "#1234567890abcdefghijklmnopqrstuvwxyz"
    sequence = np.asarray(logits).squeeze()
    if sequence.ndim == 1:
        sequence = sequence[None, :]
    # The model output is logits. Convert each timestep to probabilities for
    # a meaningful confidence score while keeping argmax unchanged.
    sequence = sequence - np.max(sequence, axis=-1, keepdims=True)
    sequence = np.exp(sequence)
    sequence /= np.maximum(sequence.sum(axis=-1, keepdims=True), 1e-12)
    indices = sequence.argmax(axis=-1)
    chars = []
    confidences = []
    previous = None
    for timestep, index in enumerate(indices.tolist()):
        if index != 0 and index != previous and index < len(alphabet):
            chars.append(alphabet[index])
            confidences.append(float(sequence[timestep, index]))
        previous = index
    confidence = float(np.mean(confidences)) if confidences else 0.0
    return "".join(chars), confidence


class OpenVINOCR:
    def __init__(self, device):
        import openvino as ov

        self.core = ov.Core()
        if device == "GPU" and "GPU" not in self.core.available_devices:
            raise RuntimeError(
                "OpenVINO does not see an Intel GPU. Install/update the Intel "
                "graphics driver and check OpenVINO GPU support. Available: "
                f"{self.core.available_devices}"
            )
        models = []
        for model_name in MODEL_NAMES:
            model_xml = download_model(model_name)
            try:
                model = self.core.read_model(model_xml)
            except RuntimeError as exc:
                print(f"Cached files for {model_name} are invalid; downloading a fresh XML/BIN pair ...",
                      flush=True)
                model_xml = download_model(model_name, force=True)
                try:
                    model = self.core.read_model(model_xml)
                except RuntimeError as retry_exc:
                    raise RuntimeError(
                        f"Could not load OpenVINO model {model_name} after a fresh download: {retry_exc}"
                    ) from exc
            models.append(model)
        print(f"Compiling OCR models for OpenVINO {device} ...", flush=True)
        self.detector = self.core.compile_model(models[0], device)
        self.recognizer = self.core.compile_model(models[1], device)
        self.det_input = self.detector.input(0)
        self.det_output = self.detector.output("boxes")
        self.rec_input = self.recognizer.input(0)
        self.rec_output = self.recognizer.output(0)
        self.det_height = int(self.det_input.shape[2])
        self.det_width = int(self.det_input.shape[3])
        _, channels, self.rec_height, self.rec_width = map(int, self.rec_input.shape)
        if channels != 1:
            raise RuntimeError(f"Unexpected text recognizer input channels: {channels}")

    def recognize(self, image):
        height, width = image.shape[:2]
        resized = cv2.resize(image, (self.det_width, self.det_height))
        detector_input = resized.transpose(2, 0, 1)[None].astype(np.float32)
        boxes = self.detector([detector_input])[self.det_output]
        boxes = np.asarray(boxes).reshape(-1, 5)
        scale_x, scale_y = width / self.det_width, height / self.det_height
        detections = []

        for box in boxes:
            x1, y1, x2, y2, box_score = map(float, box)
            if box_score < 0.35:
                continue
            x1 = max(0, min(width - 1, int(x1 * scale_x)))
            x2 = max(0, min(width, int(x2 * scale_x)))
            y1 = max(0, min(height - 1, int(y1 * scale_y)))
            y2 = max(0, min(height, int(y2 * scale_y)))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            resized_crop = cv2.resize(gray, (self.rec_width, self.rec_height))
            # This Open Model Zoo recognizer expects grayscale intensity in
            # the 0..255 range, matching the official OpenVINO OCR example.
            rec_input = resized_crop[None, None].astype(np.float32)
            logits = self.recognizer([rec_input])[self.rec_output]
            text, confidence = decode_text(logits)
            if text:
                detections.append({
                    "text": text,
                    "confidence": confidence,
                    "detector_confidence": box_score,
                    "bbox_pixels": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                })
        return detections


def main():
    parser = argparse.ArgumentParser(description="OpenVINO webcam OCR on Intel GPU")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--backend", choices=("dshow", "msmf", "any"), default="dshow")
    parser.add_argument("--device", choices=("GPU", "CPU"), default="GPU",
                        help="Inference device. GPU is explicit and has no fallback.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--roi", choices=("center", "full", "select"), default="center")
    parser.add_argument("--output", default="openvino_webcam.json")
    args = parser.parse_args()

    try:
        ocr = OpenVINOCR(args.device)
    except Exception as exc:
        raise SystemExit(f"Could not initialize OpenVINO OCR: {exc}") from exc

    camera = open_camera(args.camera, args.width, args.height, args.backend)
    if camera is None:
        raise SystemExit(f"Could not open camera {args.camera} via {args.backend.upper()}.")
    capture = BufferlessCapture(camera)
    first_frame = capture.read(timeout=5.0)
    if first_frame is None:
        capture.release()
        raise SystemExit("Camera opened but no frames arrived within five seconds.")

    roi = None
    if args.roi == "select":
        print(
            "ROI selection is showing a still frame. Drag over the text and "
            "press Enter/Space; press C to cancel. The live preview starts after this.",
            flush=True,
        )
        x, y, width, height = cv2.selectROI(
            "Select OCR region, press Enter", first_frame,
            showCrosshair=True, fromCenter=False,
        )
        cv2.destroyWindow("Select OCR region, press Enter")
        if width <= 0 or height <= 0:
            capture.release()
            raise SystemExit("No OCR region selected.")
        roi = (int(x), int(y), int(width), int(height))

    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    session = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "engine": "OpenVINO Open Model Zoo OCR",
        "device": args.device,
        "device_name": str(ocr.core.get_property(args.device, "FULL_DEVICE_NAME")),
        "models": list(MODEL_NAMES),
        "camera_index": args.camera,
        "camera_backend": args.backend,
        "frame_width": actual_width,
        "frame_height": actual_height,
        "roi_mode": args.roi,
        "selected_roi_pixels": list(roi) if roi else None,
        "min_confidence": args.min_confidence,
        "frames": [],
    }

    def save_session():
        temp = output_path.with_suffix(output_path.suffix + ".tmp")
        temp.write_text(json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(output_path)

    result_queue = queue.Queue(maxsize=1)
    inference_running = False
    last_started = -float("inf")
    latest_detections = []
    latest_ms = 0.0
    print(f"OpenVINO device: {session['device_name']}; camera {args.camera} "
          f"via {args.backend.upper()}. Press Q/Esc to quit.", flush=True)

    try:
        while True:
            frame = capture.read(timeout=0.5)
            if frame is None:
                cv2.waitKey(1)
                continue
            now = time.perf_counter()
            try:
                finished = result_queue.get_nowait()
            except queue.Empty:
                finished = None
            if finished is not None:
                detections, latest_ms, error, frame_size, offset = finished
                if error:
                    print(f"OpenVINO inference error: {error}")
                else:
                    offset_x, offset_y = offset
                    latest_detections = []
                    for item in detections:
                        if item["confidence"] < args.min_confidence:
                            continue
                        item["bbox_pixels"] = [
                            [x + offset_x, y + offset_y]
                            for x, y in item["bbox_pixels"]
                        ]
                        latest_detections.append(item)
                    frame_height, frame_width = frame_size
                    session["frames"].append({
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "frame_width": frame_width,
                        "frame_height": frame_height,
                        "ocr_time_ms": round(latest_ms, 2),
                        "detections": latest_detections,
                    })
                    save_session()
                inference_running = False

            if not inference_running and now - last_started >= args.interval:
                crop, offset_x, offset_y = crop_frame(frame, args.roi, roi)
                crop = crop.copy()
                frame_height, frame_width = frame.shape[:2]
                inference_running = True
                last_started = now

                def infer(image_copy, size, offset):
                    started = time.perf_counter()
                    try:
                        result = ocr.recognize(image_copy)
                        elapsed = (time.perf_counter() - started) * 1000
                        result_queue.put((result, elapsed, None, size, offset))
                    except Exception as exc:
                        elapsed = (time.perf_counter() - started) * 1000
                        result_queue.put(([], elapsed, str(exc), size, offset))

                threading.Thread(
                    target=infer,
                    args=(crop, (frame_height, frame_width), (offset_x, offset_y)),
                    daemon=True,
                ).start()

            if args.roi == "center":
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (w // 4, h // 4), (3 * w // 4, 3 * h // 4),
                              (255, 180, 0), 2)
            elif roi:
                x, y, w, h = roi
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 180, 0), 2)
            for item in latest_detections:
                points = np.asarray(item["bbox_pixels"], dtype=np.int32)
                cv2.polylines(frame, [points], True, (0, 220, 0), 2)
                x, y = points[0]
                cv2.putText(frame, f"{item['text']} ({item['confidence']:.2f})",
                            (x, max(22, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 220, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"{args.device}: {latest_ms:.0f} ms",
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (30, 220, 255), 2, cv2.LINE_AA)
            cv2.imshow("OpenVINO Iris Xe OCR (Q/Esc to quit)", frame)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        save_session()
        capture.release()
        cv2.destroyAllWindows()
        print(f"Saved {len(session['frames'])} OCR snapshots to {output_path.resolve()}")


if __name__ == "__main__":
    main()
