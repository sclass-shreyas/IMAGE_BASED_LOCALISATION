"""Metric particle filter for manually initialized, GNSS-denied motion sessions."""
from dataclasses import dataclass
import numpy as np
from src.road_network import LocalFrame


def wrap(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi


def advance_pose(pose, motion):
    """Apply a local-frame displacement once; successive turns compose naturally."""
    right, forward, yaw = (float(motion[k]) for k in ('right_delta_m', 'forward_delta_m', 'yaw_delta_rad'))
    heading = pose[..., 2].copy()
    pose[..., 0] += forward*np.cos(heading) + right*np.sin(heading)
    pose[..., 1] += forward*np.sin(heading) - right*np.cos(heading)
    pose[..., 2] = wrap(heading + yaw)
    return pose


@dataclass
class FilterSettings:
    particles: int = 600
    road_sigma_m: float = 7.0
    road_heading_sigma_rad: float = 0.65
    missing_sigma_m_sqrt_s: float = 2.0
    motion_noise_fraction: float = 0.08
    resample_fraction: float = 0.5


class RoadParticleFilter:
    def __init__(self, lat, lon, heading_deg, position_sigma_m=3, heading_sigma_deg=10,
                 network=None, enable_osm=True, settings=None, seed=0):
        self.settings = settings or FilterSettings()
        self.network, self.enable_osm = network, enable_osm
        self.frame = network.frame if network is not None else LocalFrame(lat, lon)
        self.rng = np.random.default_rng(seed)
        n = self.settings.particles
        self.particles = np.empty((n, 3))
        self.particles[:, :2] = self.frame.xy(lat, lon) + self.rng.normal(0, position_sigma_m, (n, 2))
        self.particles[:, 2] = wrap(np.radians(heading_deg) + self.rng.normal(0, np.radians(heading_sigma_deg), n))
        self.weights = np.full(n, 1/n)
        self.baseline = np.array([*self.frame.xy(lat, lon), np.radians(heading_deg)])
        self.edge_ids = np.full(n, -1, int)
        self.progress = np.zeros(n)
        self.last_timestamp = None
        self.distance_m = 0.0
        self.total, self.rejected = 0, 0
        self.since_road_m = 0.0
        self.diagnostics = {"status": "initialized", "road_status": "disabled"}
        if enable_osm and network is not None:
            self._weight_roads(0.0)

    @property
    def ess(self):
        return float(1/np.sum(self.weights**2))

    def _weight_roads(self, travelled):
        net, s = self.network, self.settings
        match = net.match(self.particles[:, :2], self.particles[:, 2], s.road_sigma_m, s.road_heading_sigma_rad)
        if np.all(match['distance'] > 5*s.road_sigma_m):
            self.diagnostics.update(road_status="no_plausible_road", min_road_distance_m=float(match['distance'].min()))
            return
        transition = net.transition_penalty(self.edge_ids, match['edge_ids'], self.progress,
                                             match['progress'], travelled)
        logweights = np.log(np.maximum(self.weights, 1e-300)) + match['loglike'] + np.log(transition)
        logweights -= logweights.max()
        weights = np.exp(logweights)
        self.weights = weights / weights.sum()
        self.edge_ids, self.progress = match['edge_ids'], match['progress']
        ess_before = self.ess
        resampled = ess_before < s.resample_fraction*len(weights)
        if resampled:
            positions = (self.rng.random() + np.arange(len(weights))) / len(weights)
            indices = np.minimum(np.searchsorted(np.cumsum(self.weights), positions), len(weights)-1)
            self.particles = self.particles[indices].copy()
            self.edge_ids, self.progress = self.edge_ids[indices], self.progress[indices]
            # Small rejuvenation preserves diversity; no forced centerline snapping.
            self.particles[:, :2] += self.rng.normal(0, 0.15, (len(weights), 2))
            self.particles[:, 2] = wrap(self.particles[:, 2] + self.rng.normal(0, 0.01, len(weights)))
            self.weights.fill(1/len(weights))
        self.diagnostics.update(road_status="weighted", ess_before_resampling=ess_before,
                                resampled=bool(resampled), topology_source=net.source)

    def step(self, motion):
        timestamp = float(motion['timestamp_s'])
        if not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("timestamp must be finite and nonnegative")
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            raise ValueError("motion timestamps must increase strictly; duplicate updates rejected")
        dt = float(motion.get('delta_time_s', 0)) if self.last_timestamp is None else timestamp-self.last_timestamp
        if not np.isfinite(dt) or dt < 0:
            raise ValueError("invalid motion interval")
        valid = bool(motion.get('motion_valid', False))
        numbers = [motion.get(k, 0) for k in ('right_delta_m', 'forward_delta_m', 'yaw_delta_rad', 'confidence')]
        if not np.isfinite(numbers).all():
            raise ValueError("motion values must be finite")
        self.last_timestamp = timestamp
        self.total += 1
        distance = float(np.hypot(numbers[0], numbers[1])) if valid else 0.0
        self.diagnostics['status'] = 'tracking' if valid else 'motion_missing'
        self.diagnostics['resampled'] = False
        if valid:
            advance_pose(self.particles, motion)
            advance_pose(self.baseline, motion)
            quality = np.clip(numbers[3], 0, 1)
            sigma = self.settings.motion_noise_fraction * distance + (0.02 + 0.12*(1-quality))*np.sqrt(dt)
            self.particles[:, :2] += self.rng.normal(0, sigma, (len(self.weights), 2))
            yaw_sigma = (0.01 + 0.04*(1-quality))*np.sqrt(dt) + 0.05*abs(numbers[2])
            self.particles[:, 2] = wrap(self.particles[:, 2] + self.rng.normal(0, yaw_sigma, len(self.weights)))
            self.distance_m += distance
            self.since_road_m += distance
            # Do not multiply the same static map prior at camera frame rate.
            if self.enable_osm and self.network is not None and self.since_road_m >= 0.5:
                self._weight_roads(self.since_road_m)
                self.since_road_m = 0.0
        else:
            self.rejected += 1
            self.particles[:, :2] += self.rng.normal(0, self.settings.missing_sigma_m_sqrt_s*np.sqrt(dt), (len(self.weights), 2))
            self.particles[:, 2] = wrap(self.particles[:, 2] + self.rng.normal(0, 0.12*np.sqrt(dt), len(self.weights)))
            self.diagnostics['road_status'] = 'not_reweighted_without_motion' if self.enable_osm else 'disabled'
        result = self.state()
        measurement_dt = float(motion.get('delta_time_s', dt))
        result['speed_mps'] = distance/measurement_dt if valid and measurement_dt > 0 else None
        result['motion'] = motion
        return result

    def state(self):
        mean = np.average(self.particles[:, :2], weights=self.weights, axis=0)
        delta = self.particles[:, :2] - mean
        covariance = (delta*self.weights[:, None]).T @ delta
        heading = np.arctan2(np.sum(self.weights*np.sin(self.particles[:, 2])),
                             np.sum(self.weights*np.cos(self.particles[:, 2])))
        resultant = np.hypot(np.sum(self.weights*np.sin(self.particles[:, 2])), np.sum(self.weights*np.cos(self.particles[:, 2])))
        heading_sigma = float(np.sqrt(-2*np.log(max(resultant, 1e-12))))
        ll = self.frame.latlon(*mean)
        base_ll = self.frame.latlon(*self.baseline[:2])
        sample = np.linspace(0, len(self.weights)-1, min(300, len(self.weights)), dtype=int)
        positions = self.frame.latlon(self.particles[sample, 0], self.particles[sample, 1])
        hypotheses = []
        for eid in np.unique(self.edge_ids[self.edge_ids >= 0]):
            mass = float(self.weights[self.edge_ids == eid].sum())
            if mass > 0.05:
                hypotheses.append({'edge_id': int(eid), 'probability': mass})
        return dict(vehicle=dict(lat=float(ll[0]), lon=float(ll[1]), heading_deg=float(np.degrees(heading)%360),
                    heading_sigma_deg=float(np.degrees(heading_sigma)), covariance_m2=covariance.tolist(),
                    uncertainty_meters=float(np.sqrt(np.trace(covariance))),
                    radius95_m=float(np.sqrt(max(np.linalg.eigvalsh(covariance))*5.991))),
                    baseline=dict(lat=float(base_ll[0]), lon=float(base_ll[1]), heading_deg=float(np.degrees(self.baseline[2])%360)),
                    particles=positions.tolist(), particle_count=len(self.weights), effective_sample_size=self.ess,
                    distance_m=self.distance_m, rejection_rate=self.rejected/max(self.total, 1),
                    road_hypotheses=sorted(hypotheses, key=lambda v: -v['probability']),
                    diagnostics=dict(self.diagnostics))
