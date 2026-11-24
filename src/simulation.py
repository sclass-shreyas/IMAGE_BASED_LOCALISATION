"""Synthetic route and noisy body-frame motion; truth is never a filter input."""
import numpy as np
from particlefilterbackend.road_filter import wrap


class RouteSimulator:
    def __init__(self, points, frame, speed_mps=2.0, dt=0.1, noise_m=0.01,
                 yaw_drift_rad_s=0.002, seed=1):
        points = np.asarray(points, dtype=float)
        keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1)>1e-6]
        self.points = points[keep]
        if len(self.points)<2:
            raise ValueError("simulation requires a nonzero route")
        self.frame, self.speed, self.dt = frame, speed_mps, dt
        self.noise_m, self.yaw_drift = noise_m, yaw_drift_rad_s
        self.rng = np.random.default_rng(seed)
        self.arc = np.r_[0,np.cumsum(np.linalg.norm(np.diff(self.points,axis=0),axis=1))]
        self.distance, self.timestamp = 0.0, 0.0
        first=self.points[1]-self.points[0]
        self.heading=float(np.arctan2(first[1],first[0]))
        self.position=self.points[0].copy()

    def step(self):
        if self.distance>=self.arc[-1]:
            return None
        target=min(self.distance+self.speed*self.dt,self.arc[-1])
        xy=np.array([np.interp(target,self.arc,self.points[:,k]) for k in (0,1)])
        i=min(np.searchsorted(self.arc,target,side='right')-1,len(self.points)-2)
        tangent=self.points[i+1]-self.points[i]
        heading=float(np.arctan2(tangent[1],tangent[0]))
        delta=xy-self.position
        forward=float(delta@[np.cos(self.heading),np.sin(self.heading)])
        right=float(delta@[np.sin(self.heading),-np.cos(self.heading)])
        yaw=float(wrap(heading-self.heading))
        self.timestamp+=self.dt
        packet=dict(timestamp_s=self.timestamp,delta_time_s=self.dt,motion_valid=True,
                    right_delta_m=right+float(self.rng.normal(0,self.noise_m)),
                    forward_delta_m=forward+float(self.rng.normal(0,self.noise_m)),
                    yaw_delta_rad=yaw+self.yaw_drift*self.dt,confidence=.9,
                    source='simulation',status='ok',tracked_features=0,inlier_ratio=None)
        self.distance,self.position,self.heading=target,xy,heading
        ll=self.frame.latlon(*xy)
        reference=dict(lat=float(ll[0]),lon=float(ll[1]),heading_deg=float(np.degrees(heading)%360),
                       distance_m=float(target),source='simulated_ground_truth')
        return packet,reference
