"""OCR-independent session worker for live video, file playback and simulation."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from threading import Event, RLock, Thread
import time
import uuid

import cv2
import numpy as np
from src.video_source import VideoSource, VideoConfig
from src.road_network import RoadNetwork, load_graph
from src.osm_matching import config
from src.simulation import RouteSimulator
from visual_odometry import GroundPlaneVisualOdometry, VOSettings
from .road_filter import RoadParticleFilter, FilterSettings


class MotionRuntime:
    def __init__(self):
        self.lock = RLock()
        self.stop_event = Event()
        self.thread = None
        self.source = None
        self.paused = False
        self.network = None
        self.area_center, self.area_radius = config.AREA_CENTER, config.AREA_RADIUS_M
        self.latest = {'status':'idle', 'vehicle':None}
        self.preview, self.bird = None, None
        self.bird_requested_at = 0.0

    def get_network(self):
        with self.lock:
            if self.network is not None:
                return self.network
            center, radius = self.area_center, self.area_radius
        network = RoadNetwork(load_graph(center, radius), center)
        with self.lock:
            self.network = network
        return network

    def start(self, request):
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError('Stop the current session before starting another')
            self.stop_event.clear()
            self.paused = False
            self.preview = self.bird = None
            self.latest = {'status':'starting','mode':request['mode'],'vehicle':None,
                           'session_id':str(uuid.uuid4()),'sequence':0}
            self.thread = Thread(target=self._run,args=(request,),daemon=True)
            self.thread.start()
            return dict(self.latest)

    def stop(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=4)
        with self.lock:
            if self.thread and self.thread.is_alive():
                self.latest['status']='stopping'
            elif self.latest.get('status') != 'failed':
                self.latest['status']='stopped'
            return dict(self.latest)

    def pause(self, paused):
        with self.lock:
            self.paused=bool(paused)
            self.latest['status']='paused' if paused else 'running'
        return self.state()

    def reset(self):
        self.stop()
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError('Worker is still stopping')
            self.latest={'status':'idle','vehicle':None}
            self.preview=self.bird=None
        return self.state()

    def state(self):
        with self.lock:
            return dict(self.latest)

    def _set(self, **fields):
        with self.lock:
            self.latest.update(fields)

    def _jpeg(self, frame, bird=None):
        ok, jpg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,80])
        b_ok,b_jpg=(False,None) if bird is None else cv2.imencode('.jpg',bird)
        with self.lock:
            if ok:
                self.preview=jpg.tobytes()
            if b_ok:
                self.bird=b_jpg.tobytes()

    def _run(self, request):
        log=None
        pool=None
        try:
            # Bound OpenCV's native worker pool on shared laptop CPUs.
            cv2.setNumThreads(max(1, int(os.getenv('VO_OPENCV_THREADS', '2'))))
            mode=request['mode']
            network=self.get_network() if mode!='preview' and (request['enable_osm'] or mode=='simulation') else None
            simulator=None
            lat,lon,heading=request['lat'],request['lon'],request['heading_deg']
            if mode=='simulation':
                end=request.get('destination')
                if not end:
                    raise ValueError('Simulation needs a destination selected on the map')
                planned=network.route((lat,lon),end)
                route=planned
                if request.get('via'):
                    route=np.vstack([network.route((lat,lon),request['via'])[:-1],
                                     network.route(request['via'],end)])
                simulator=RouteSimulator(route,network.frame,request['simulation_speed_mps'],
                    noise_m=request['simulation_noise_m'],yaw_drift_rad_s=request['simulation_yaw_drift'],seed=request['seed'])
                lat,lon=map(float,network.frame.latlon(*route[0]))
                heading=float(np.degrees(simulator.heading)%360)
                self._set(planned_route=network.frame.latlon(planned[:,0],planned[:,1]).tolist(),
                          initialization='simulated route start and tangent', actual_route=network.frame.latlon(route[:,0],route[:,1]).tolist())
            pf=None if mode=='preview' else RoadParticleFilter(lat,lon,heading,
                request['position_sigma_m'],request['heading_sigma_deg'],network,
                request['enable_osm'],FilterSettings(**request['filter_settings']),request['seed'])
            if pf:
                self._set(**pf.state())
            started=time.monotonic()
            sequence=0
            logdir=config.PROJECT_ROOT/'results'/'sessions'
            logdir.mkdir(parents=True,exist_ok=True)
            logpath=logdir/f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{self.latest['session_id'][:8]}.jsonl"
            log=logpath.open('w',encoding='utf-8')
            # Do not put stream passwords into the log.
            metadata={k:v for k,v in request.items() if k!='video'}
            metadata.update(video_source_type=request['video']['source_type'],
                            initial_pose=[lat,lon,heading],schema_version=1,
                            map_area=dict(center=self.area_center,radius_m=self.area_radius),
                            reference_policy='reference is evaluation only; never passed to filter')
            log.write(json.dumps({'type':'metadata','config':metadata})+'\n')
            log.flush()
            self._set(log_path=str(logpath),status='running',enable_osm=request['enable_osm'],
                      enable_ocr=request['enable_ocr'],network_source=network.source if network else None)

            def update(motion, reference=None, processing_ms=0):
                nonlocal sequence
                sequence+=1
                result=pf.step(motion)
                if reference:
                    xy=pf.frame.xy(reference['lat'],reference['lon'])
                    est=pf.frame.xy(result['vehicle']['lat'],result['vehicle']['lon'])
                    base=pf.frame.xy(result['baseline']['lat'],result['baseline']['lon'])
                    result['reference']=reference
                    result['position_error_m']=float(np.linalg.norm(est-xy))
                    result['baseline_error_m']=float(np.linalg.norm(base-xy))
                result.update(timestamp_s=motion['timestamp_s'],sequence=sequence,
                              processing_ms=processing_ms,status='running',
                              processing_fps=sequence/max(time.monotonic()-started,1e-6))
                self._set(**result)
                record={k:v for k,v in result.items() if k!='particles'}
                log.write(json.dumps({'type':'update',**record},allow_nan=False)+'\n')
                if sequence%10==0:
                    log.flush()

            if simulator:
                self._set(stream={'status':'simulation','sensor_latency_ms':None})
                while not self.stop_event.is_set():
                    if self.paused:
                        self.stop_event.wait(.1)
                        continue
                    sample=simulator.step()
                    if sample is None:
                        break
                    update(*sample)
                    self.stop_event.wait(simulator.dt/request['playback_rate'])
            else:
                video=VideoConfig(**request['video'])
                vo=None
                if mode!='preview':
                    vo=GroundPlaneVisualOdometry(request['calibration_path'],settings=VOSettings(**request['vo_settings']))
                self.source=VideoSource(video).start()
                last_frame_id,last_generation=0,None
                dropped,last_preview,last_missing=0,0.,0.
                was_paused=False
                ocr_job=None
                last_ocr=0.
                if request['enable_ocr']:
                    pool=ThreadPoolExecutor(max_workers=1)
                    ocr_model=None
                    def recognize(frame):
                        nonlocal ocr_model
                        if ocr_model is None:
                            from openvino_webcam_test import OpenVINOCR
                            ocr_model=OpenVINOCR(request['ocr_device'])
                        return ocr_model.recognize(frame)
                while not self.stop_event.is_set():
                    if self.paused:
                        was_paused=True
                        self.stop_event.wait(.1)
                        continue
                    if was_paused:
                        if vo:
                            vo.reset()
                        was_paused=False
                    packet=self.source.read(.1)
                    source_status=dict(self.state().get('stream', {}))
                    source_status.update(self.source.status())
                    self._set(stream=source_status)
                    if packet is None:
                        if source_status.get('status') in {'eof','failed'}:
                            if source_status['status']=='failed':
                                raise ValueError(source_status.get('error','Video source failed'))
                            break
                        now=time.monotonic()-started
                        if pf and video.source_type!='file' and now-last_missing>.5:
                            # Arrival time is a common clock for gaps and live frames.
                            if pf.last_timestamp is None or now>pf.last_timestamp:
                                update(dict(timestamp_s=now,delta_time_s=.5,motion_valid=False,
                                            source='stream',status='no_frame',confidence=0.,
                                            right_delta_m=0.,forward_delta_m=0.,yaw_delta_rad=0.))
                            last_missing=now
                        continue
                    tick=time.monotonic()
                    timestamp=packet['timestamp_s'] if video.source_type=='file' else packet['received_monotonic']-started
                    if pf and pf.last_timestamp is not None and timestamp<=pf.last_timestamp:
                        dropped+=1
                        continue
                    dropped+=max(0,packet['frame_id']-last_frame_id-1)
                    last_frame_id=packet['frame_id']
                    frame=packet['frame']
                    source_status.update(width=frame.shape[1],height=frame.shape[0],dropped_frames=dropped,
                        timestamp_basis=packet['timestamp_basis'],
                        receive_to_process_ms=max(0,(tick-packet['received_monotonic'])*1000),sensor_latency_ms=None)
                    self._set(stream=source_status)
                    if vo and packet['generation']!=last_generation:
                        vo.reset()
                    last_generation=packet['generation']
                    if vo:
                        motion=vo.process_frame(frame,timestamp,packet['frame_id'])
                        update(motion,processing_ms=(time.monotonic()-tick)*1000)
                    else:
                        motion=None
                    # Optional OCR is diagnostic only and never gates VO or PF.
                    if pool:
                        if ocr_job and ocr_job.done():
                            try:
                                self._set(ocr_detections=ocr_job.result(),ocr_status='diagnostic_only')
                            except Exception as exc:
                                self._set(ocr_status=str(exc))
                            ocr_job=None
                        if ocr_job is None and tick-last_ocr>1:
                            ocr_job=pool.submit(recognize,frame.copy())
                            last_ocr=tick
                    if tick-last_preview>.5:
                        annotated=vo.annotate(frame,motion) if vo else frame
                        with self.lock:
                            want_bird = tick-self.bird_requested_at < 2.0
                        self._jpeg(annotated,vo.birdseye(frame) if vo and want_bird else None)
                        last_preview=tick
                    if video.source_type=='file' and request['realtime_file'] and motion:
                        self.stop_event.wait(max(0,motion['delta_time_s']/request['playback_rate']-(time.monotonic()-tick)))
            self._set(status='stopped' if self.stop_event.is_set() else 'completed')
        except Exception as exc:
            self._set(status='failed',error=str(exc))
        finally:
            if self.source:
                self.source.close()
                self.source=None
            if pool:
                pool.shutdown(wait=False,cancel_futures=True)
            if log:
                log.close()
