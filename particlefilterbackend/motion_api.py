"""API for the default motion-based demo; legacy OCR APIs remain available."""
from dataclasses import asdict
import os
import time
from threading import Thread
from typing import Literal
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field, ConfigDict, model_validator
from src.video_source import VideoConfig
from src.osm_matching import config
from src.road_network import download_graph
from .motion_runtime import MotionRuntime

router=APIRouter(prefix='/api/localization')
runtime=MotionRuntime()
download_status={'status':'idle'}


class SessionRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    mode: Literal['preview','live','file','simulation']='live'
    video: dict=Field(default_factory=lambda:asdict(VideoConfig.from_env()))
    calibration_path: str=Field(default_factory=lambda:os.getenv('VISUAL_ODOMETRY_CALIBRATION','visual_odometry_calibration.json'))
    lat: float|None=Field(default=None,ge=-85,le=85)
    lon: float|None=Field(default=None,ge=-180,le=180)
    heading_deg: float=Field(default=0,ge=0,lt=360)
    position_sigma_m: float=Field(default=3,ge=0,le=500)
    heading_sigma_deg: float=Field(default=10,ge=0,le=180)
    enable_osm: bool=True
    enable_ocr: bool=Field(default_factory=lambda:os.getenv('ENABLE_OCR','false').lower()=='true')
    ocr_device: Literal['CPU','GPU']='CPU'
    destination: tuple[float,float]|None=None
    via: tuple[float,float]|None=None
    simulation_speed_mps: float=Field(default=2,gt=0,le=12)
    simulation_noise_m: float=Field(default=.01,ge=0,le=1)
    simulation_yaw_drift: float=Field(default=.002,ge=-.2,le=.2)
    playback_rate: float=Field(default=1,gt=0,le=30)
    realtime_file: bool=True
    seed: int=0
    vo_settings: dict=Field(default_factory=dict)
    filter_settings: dict=Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_request(self):
        from visual_odometry import VOSettings
        from .road_filter import FilterSettings
        if self.mode!='preview' and (self.lat is None or self.lon is None):
            raise ValueError('Select an initial position on the map')
        if self.mode=='simulation' and self.destination is None:
            raise ValueError('Select a simulation destination')
        for point in (self.destination,self.via):
            if point is not None and not (-85<=point[0]<=85 and -180<=point[1]<=180):
                raise ValueError('Invalid destination/via coordinates')
        try:
            video=VideoConfig.from_env(**self.video)
            if self.mode=='file':
                video.source_type='file'
            if self.mode!='simulation':
                video.validate()
            self.video=asdict(video)
            vo=VOSettings(**self.vo_settings)
            pf=FilterSettings(**self.filter_settings)
            if not isinstance(pf.particles,int) or not 50<=pf.particles<=5000 or min(pf.road_sigma_m,pf.road_heading_sigma_rad,pf.missing_sigma_m_sqrt_s)<=0:
                raise ValueError('Invalid filter settings')
            import math
            if not all(isinstance(v,(float,int)) and math.isfinite(v) for v in asdict(pf).values()):
                raise ValueError('Filter settings must be finite numbers')
            if pf.motion_noise_fraction<0 or not 0<pf.resample_fraction<=1:
                raise ValueError('Invalid filter noise or resampling threshold')
            if not isinstance(vo.min_tracks,int) or not isinstance(vo.max_corners,int) or not 4<=vo.min_tracks<=vo.max_corners<=3000 or not 0<vo.min_inlier_ratio<=1:
                raise ValueError('Invalid VO feature settings')
            for key,value in asdict(vo).items():
                if not isinstance(value,(int,float)) or not 0<float(value)<1e6:
                    raise ValueError(f'Invalid VO setting: {key}')
        except TypeError as exc:
            raise ValueError(str(exc)) from exc
        return self


@router.get('/config')
def configuration():
    return dict(area_center=runtime.area_center,area_radius_m=runtime.area_radius,
                initial_position=config.INITIAL_POSITION,
                initial_heading_deg=config.INITIAL_HEADING_DEG_FROM_EAST,
                video=asdict(VideoConfig.from_env()),
                calibration_path=os.getenv('VISUAL_ODOMETRY_CALIBRATION','visual_odometry_calibration.json'),
                enable_ocr=os.getenv('ENABLE_OCR','false').lower()=='true')


@router.post('/start')
def start(request:SessionRequest):
    try:
        from .engine import live_runtime
        live_runtime.stop()
        if download_status['status']=='downloading':
            raise ValueError('Wait for the map download to finish before starting a session')
        return runtime.start(request.model_dump())
    except ValueError as exc:
        raise HTTPException(409,str(exc)) from exc


@router.post('/stop')
def stop():
    return runtime.stop()


@router.post('/reset')
def reset():
    try:
        return runtime.reset()
    except ValueError as exc:
        raise HTTPException(409,str(exc)) from exc


@router.post('/pause')
def pause(paused:bool=True):
    return runtime.pause(paused)


@router.get('/state')
def state():
    return runtime.state()


@router.get('/preview.jpg')
def preview(bird:bool=False):
    with runtime.lock:
        if bird:
            runtime.bird_requested_at = time.monotonic()
        data=runtime.bird if bird else runtime.preview
    if data is None:
        raise HTTPException(503,'No frame available')
    return Response(data,media_type='image/jpeg',headers={'Cache-Control':'no-store'})


@router.get('/map')
def road_map():
    try:
        network=runtime.get_network()
        return dict(source=network.source,geojson=network.geojson())
    except (ValueError,OSError) as exc:
        raise HTTPException(409,str(exc)) from exc


class AreaRequest(BaseModel):
    model_config=ConfigDict(allow_inf_nan=False)
    lat:float=Field(ge=-85,le=85)
    lon:float=Field(ge=-180,le=180)
    radius_m:float=Field(default=1000,ge=100,le=5000)


@router.post('/map/download')
def map_download(area:AreaRequest):
    if download_status['status']=='downloading':
        raise HTTPException(409,'Map download already running')
    if runtime.thread and runtime.thread.is_alive():
        raise HTTPException(409,'Stop the session before changing the map area')
    download_status.clear()
    download_status.update(status='downloading')
    def work():
        try:
            path=download_graph((area.lat,area.lon),area.radius_m)
            with runtime.lock:
                runtime.area_center=(area.lat,area.lon)
                runtime.area_radius=area.radius_m
                runtime.network=None
            download_status.update(status='complete',path=str(path))
        except Exception as exc:
            download_status.update(status='failed',error=str(exc))
    Thread(target=work,daemon=True).start()
    return dict(download_status)


@router.get('/map/download')
def map_download_status():
    return dict(download_status)
