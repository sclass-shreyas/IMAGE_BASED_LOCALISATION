"""Local MJPEG fixture tests exercise networking, not an Android phone."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import time
import cv2
import numpy as np
from src.video_source import VideoConfig, VideoSource


def test_local_mjpeg_reconnect_and_bounded_queue():
    frame=np.random.default_rng(1).integers(0,255,(120,160,3),dtype=np.uint8)
    _,jpg=cv2.imencode('.jpg',frame)
    payload=jpg.tobytes()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type','multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                for _ in range(12):
                    self.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '+str(len(payload)).encode()+b'\r\n\r\n'+payload+b'\r\n')
                    self.wfile.flush()
                    time.sleep(.03)
            except (BrokenPipeError,ConnectionResetError):
                pass
            self.close_connection=True
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    Thread(target=server.serve_forever,daemon=True).start()
    source=VideoSource(VideoConfig(source_type='network',url=f'http://127.0.0.1:{server.server_port}/video',
                                   reconnect_s=.2,timeout_ms=1000)).start()
    packets=[]
    try:
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            packet=source.read(.2)
            if packet is not None:
                packets.append(packet)
                if packet['generation']>=2:
                    break
                time.sleep(.1)  # Deliberately slower than source: old live frames must drop.
        assert packets and packets[-1]['generation']>=2
        assert any(b['frame_id']-a['frame_id']>1 for a,b in zip(packets,packets[1:]))
        assert np.all(np.diff([p['timestamp_s'] for p in packets])>0)
    finally:
        source.close()
        server.shutdown();server.server_close()
    assert not source.process.is_alive()
