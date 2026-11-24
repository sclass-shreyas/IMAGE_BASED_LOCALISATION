import json
import time
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from particlefilterbackend.engine import app
from particlefilterbackend.motion_api import runtime
from src.video_source import VideoSource, VideoConfig
from src.simulation import RouteSimulator
from particlefilterbackend.road_filter import RoadParticleFilter
from test_motion_localization import small_network, calibration
from src.ground_calibration import fit_calibration


def test_simulation_api_without_camera_or_ocr(monkeypatch):
    runtime.stop()
    monkeypatch.setattr(runtime,'network',small_network())
    with TestClient(app) as client:
        assert client.post('/api/localization/start',json={'mode':'simulation'}).status_code==422
        response=client.post('/api/localization/start',json=dict(mode='simulation',lat=12.,lon=77.,
            destination=[12.,77.002],heading_deg=0,playback_rate=30,simulation_speed_mps=12,
            enable_ocr=False,filter_settings={'particles':100}))
        assert response.status_code==200
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            state=client.get('/api/localization/state').json()
            if state['status'] in ('completed','failed'):
                break
            time.sleep(.05)
        assert state['status']=='completed',state
        assert state['sequence']>20
        assert state['vehicle'] and state['reference'] and state['baseline']
        assert state['enable_ocr'] is False
        assert state['diagnostics']['road_status']=='weighted'
        log=[json.loads(line) for line in open(state['log_path'])]
        assert log[0]['config']['reference_policy'].startswith('reference is evaluation')
        from evaluate_localization import evaluate
        metrics,_=evaluate(state['log_path'])
        assert metrics['paired_reference_samples']>20
        assert metrics['rmse_m']>=0
        assert client.get('/').status_code==200
        assert client.get('/legacy.html').status_code==200
        assert client.post('/api/localization/reset').json()['vehicle'] is None


def test_recorded_source_preserves_frames_and_media_timestamps(tmp_path):
    path=tmp_path/'fixture.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(160,120))
    assert writer.isOpened()
    for i in range(8):
        writer.write(np.full((120,160,3),i*20,np.uint8))
    writer.release()
    source=VideoSource(VideoConfig(source_type='file',url=str(path))).start()
    frames=[]
    try:
        deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            packet=source.read(.2)
            if packet is not None:
                frames.append(packet)
            elif source.status().get('status')=='eof':
                break
    finally:
        source.close()
    assert len(frames)==8
    assert np.all(np.diff([p['timestamp_s'] for p in frames])>0)
    assert frames[-1]['frame'].shape[:2]==(120,160)


def test_route_deviation_is_not_forced_to_planned_route():
    network=small_network()
    # Planned eastward route is never passed to the estimator; actual route turns north.
    actual=network.route((12,77),(12.001,77.001))
    simulator=RouteSimulator(actual,network.frame,speed_mps=4,noise_m=0,yaw_drift_rad_s=0)
    pf=RoadParticleFilter(12,77,0,position_sigma_m=.2,heading_sigma_deg=1,network=network)
    while True:
        sample=simulator.step()
        if sample is None:
            break
        packet,reference=sample
        pf.step(packet)
    state=pf.state()
    assert state['vehicle']['lat']>12.0007
    assert 60<state['vehicle']['heading_deg']<120
    assert state['baseline']['lat']==pytest.approx(reference['lat'],abs=1e-6)


def test_recorded_video_full_pipeline_without_ocr(tmp_path):
    runtime.stop()
    calibration_file=tmp_path/'synthetic_calibration.json'
    calibration_file.write_text(json.dumps(fit_calibration(calibration())))
    path=tmp_path/'synthetic_ground.avi'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'MJPG'),10,(640,480))
    rng=np.random.default_rng(4)
    image=cv2.GaussianBlur(rng.integers(0,256,(480,640,3),dtype=np.uint8),(3,3),0)
    for i in range(12):
        writer.write(cv2.warpAffine(image,np.float32([[1,0,0],[0,1,i*2]]),(640,480)))
    writer.release()
    with TestClient(app) as client:
        response=client.post('/api/localization/start',json=dict(mode='file',lat=12,lon=77,
            enable_osm=False,enable_ocr=False,calibration_path=str(calibration_file),
            video={'source_type':'file','url':str(path)},realtime_file=False))
        assert response.status_code==200
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            state=client.get('/api/localization/state').json()
            if state['status'] in ('completed','failed'):
                break
            time.sleep(.05)
        assert state['status']=='completed',state
        assert state['sequence']==12
        assert state['distance_m']==pytest.approx(.22,abs=.04)
        assert state['stream']['timestamp_basis']=='video_media_time'
        assert state['enable_ocr'] is False


def test_unavailable_stream_does_not_block_api():
    runtime.stop()
    with TestClient(app) as client:
        response=client.post('/api/localization/start',json=dict(mode='preview',enable_osm=False,
            video={'source_type':'network','url':'http://127.0.0.1:1/video','timeout_ms':1000,'reconnect_s':.2}))
        assert response.status_code==200
        for _ in range(5):
            tick=time.monotonic()
            assert client.get('/api/localization/state').status_code==200
            assert time.monotonic()-tick<1.5
            time.sleep(.1)
        assert client.post('/api/localization/stop').status_code==200
        assert runtime.thread is None or not runtime.thread.is_alive()
