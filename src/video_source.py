"""Bounded, interruptible frame acquisition shared by calibration and localization.

Capture runs in a child process: a hung camera driver cannot hang the API.
Live sources retain only the latest frame; files preserve every frame and media time.
"""
from dataclasses import dataclass, asdict
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty, Full
import time
import math
from urllib.parse import urlparse

import cv2


@dataclass
class VideoConfig:
    source_type: str = "webcam"
    url: str = ""
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    backend: str = "dshow"
    timeout_ms: int = 3000
    reconnect_s: float = 2.0

    @classmethod
    def from_env(cls, **overrides):
        values = dict(source_type=os.getenv("VIDEO_SOURCE_TYPE", "webcam"),
                      url=os.getenv("VIDEO_SOURCE_URL", ""),
                      camera_index=int(os.getenv("VIDEO_CAMERA_INDEX", "0")),
                      width=int(os.getenv("VIDEO_WIDTH", "1280")),
                      height=int(os.getenv("VIDEO_HEIGHT", "720")),
                      fps=float(os.getenv("VIDEO_FPS", "30")))
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def validate(self):
        if self.source_type not in {"webcam", "network", "file"}:
            raise ValueError("source_type must be webcam, network, or file")
        if self.source_type == "network" and urlparse(self.url).scheme not in {"http", "https", "rtsp"}:
            raise ValueError("network source needs an http(s) or rtsp video URL")
        if self.source_type == "file" and not Path(self.url).is_file():
            raise ValueError("video file does not exist")
        if self.width < 1 or self.height < 1 or not 0 < self.fps <= 240:
            raise ValueError("invalid requested resolution or FPS")
        if not 0 <= self.camera_index <= 99 or self.backend not in {"dshow", "msmf", "any"}:
            raise ValueError("invalid camera index or backend")
        if not 100 <= self.timeout_ms <= 30000 or not 0.1 <= self.reconnect_s <= 30:
            raise ValueError("invalid timeout or reconnect interval")


def _latest(queue, item):
    try:
        queue.put_nowait(item)
    except Full:
        try:
            queue.get_nowait()
        except Empty:
            pass
        try:
            queue.put_nowait(item)
        except Full:
            pass


def _terminal_status(queue, item, stop):
    # Unlike disposable live updates, EOF/failure must reach the consumer.
    # Queue feeder threads make get_nowait/put_nowait replacement racy.
    while not stop.is_set():
        try:
            queue.put(item, timeout=0.1)
            return
        except Full:
            continue


def _capture_worker(config, frames, statuses, stop):
    cap = None
    sequence, generation = 0, 0
    started = time.monotonic()
    while not stop.is_set():
        try:
            _latest(statuses, {"status": "connecting", "generation": generation})
            if config.source_type == "webcam":
                backend = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY}[config.backend]
                cap = cv2.VideoCapture(config.camera_index, backend)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
                cap.set(cv2.CAP_PROP_FPS, config.fps)
            elif config.source_type == "network":
                cap = cv2.VideoCapture(config.url, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, config.timeout_ms,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, config.timeout_ms])
            else:
                cap = cv2.VideoCapture(config.url)
            if not cap.isOpened():
                raise RuntimeError("Could not open video source; check URL/index and access")
            generation += 1
            file_fps = cap.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(file_fps) or file_fps <= 0:
                file_fps = config.fps
            last_media_time = -1.0
            source_frame = 0
            connected_at = time.monotonic()
            while not stop.is_set():
                ok, frame = cap.read()
                received = time.monotonic()
                if not ok:
                    if config.source_type == "file":
                        _terminal_status(statuses, {"status": "eof", "generation": generation}, stop)
                        return
                    raise RuntimeError("Video source stopped delivering frames")
                sequence += 1
                source_frame += 1
                timestamp = received - started
                basis = "host_receive_monotonic"
                if config.source_type == "file":
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    if timestamp <= last_media_time:
                        timestamp = last_media_time + 1.0 / file_fps
                    last_media_time = timestamp
                    basis = "video_media_time"
                packet = dict(frame=frame, timestamp_s=timestamp, received_monotonic=received,
                              frame_id=sequence, generation=generation, timestamp_basis=basis)
                if config.source_type == "file":
                    while not stop.is_set():
                        try:
                            frames.put(packet, timeout=0.1)
                            break
                        except Full:
                            pass
                else:
                    _latest(frames, packet)
                _latest(statuses, {"status": "connected", "generation": generation,
                                   "width": frame.shape[1], "height": frame.shape[0],
                                   "received_frames": sequence, "source_fps": file_fps,
                                   "received_fps": source_frame/max(received-connected_at, 1e-6),
                                   "timestamp_basis": basis, "sensor_latency_ms": None})
        except Exception as exc:
            _latest(statuses, {"status": "reconnecting", "error": str(exc), "generation": generation})
            if config.source_type == "file":
                _terminal_status(statuses, {"status": "failed", "error": str(exc)}, stop)
                return
        finally:
            if cap is not None:
                cap.release()
        stop.wait(config.reconnect_s)


class VideoSource:
    def __init__(self, config):
        config.validate()
        self.config = config
        ctx = mp.get_context("spawn")
        self.frames, self.statuses = ctx.Queue(1), ctx.Queue(1)
        self.stop_event = ctx.Event()
        self.process = ctx.Process(target=_capture_worker,
                                   args=(config, self.frames, self.statuses, self.stop_event), daemon=True)
        self.latest_status = {"status": "idle"}

    def start(self):
        self.process.start()
        return self

    def read(self, timeout=0.1):
        try:
            return self.frames.get(timeout=timeout)
        except Empty:
            return None

    def status(self):
        while True:
            try:
                update = self.statuses.get_nowait()
                self.latest_status.update(update)
                if update.get("status") == "connected":
                    self.latest_status.pop("error", None)
            except Empty:
                return dict(self.latest_status)

    def close(self):
        self.stop_event.set()
        if self.process.pid:
            self.process.join(timeout=0.5)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1.0)
        self.frames.close()
        self.statuses.close()


def add_video_arguments(parser):
    parser.add_argument("--source-type", choices=["webcam", "network", "file"])
    parser.add_argument("--url")
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--fps", type=float)


def config_from_arguments(args):
    return VideoConfig.from_env(**{k: getattr(args, k, None) for k in
                                  ("source_type", "url", "camera_index", "width", "height", "fps")})
