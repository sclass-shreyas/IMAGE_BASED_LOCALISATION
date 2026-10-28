"""Particle filter for geographic cues; image pixels are never treated as metres."""
import numpy as np

from src.pipeline_contract import validate_candidate_response


class ParticleFilterEngine:
    def __init__(self, num_particles=1000, initial_lat=None, initial_lon=None,
                 area_radius_m=None, seed=0, sigma_m=20.0, min_ocr_confidence=0.25):
        from src.osm_matching import config

        initial_lat = config.AREA_CENTER[0] if initial_lat is None else initial_lat
        initial_lon = config.AREA_CENTER[1] if initial_lon is None else initial_lon
        area_radius_m = config.AREA_RADIUS_M if area_radius_m is None else area_radius_m
        self.num_particles = int(num_particles)
        self.rng = np.random.default_rng(seed)
        self.origin_lat, self.origin_lon = float(initial_lat), float(initial_lon)
        self.area_radius_m = float(area_radius_m)
        self.sigma_m = float(sigma_m)
        self.min_ocr_confidence = float(min_ocr_confidence)
        self.particles = np.zeros((self.num_particles, 3), dtype=float)
        self.weights = np.full(self.num_particles, 1.0 / self.num_particles)
        self.last_timestamp_s = None
        self.last_diagnostics = {}
        self._geometry_cache = {}
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Uniform-area prior over the configured test area, independent of cues.
        radius = self.area_radius_m * np.sqrt(self.rng.random(self.num_particles))
        angle = self.rng.uniform(0, 2 * np.pi, self.num_particles)
        east, north = radius * np.cos(angle), radius * np.sin(angle)
        self.particles[:, 0] = self.origin_lat + north / 111_000.0
        self.particles[:, 1] = self.origin_lon + east / (111_000.0 * np.cos(np.radians(self.origin_lat)))
        self.particles[:, 2] = self.rng.uniform(-np.pi, np.pi, self.num_particles)
        self.weights.fill(1.0 / self.num_particles)
        self.last_timestamp_s = None
        self.last_diagnostics = {"status": "reset", "effective_sample_size": float(self.num_particles)}

    def predict(self, motion=None, dt=0.0, route_delta_m=None):
        """Predict using explicit route/simulator metres; ignore visual pixel dx/dy."""
        motion = motion or {}
        dt = max(0.0, float(dt))
        if route_delta_m is not None:
            east_m, north_m = map(float, route_delta_m)
            self.particles[:, 0] += north_m / 111_000.0
            self.particles[:, 1] += east_m / (111_000.0 * np.cos(np.radians(self.particles[:, 0])))
        # Image-derived rotation is also relative camera motion, not a calibrated
        # vehicle yaw measurement; preserve it in the event but don't apply it here.
        self.particles[:, 0] += self.rng.normal(0, 0.25 * np.sqrt(dt) / 111_000.0, self.num_particles)
        self.particles[:, 1] += self.rng.normal(0, 0.25 * np.sqrt(dt) /
                                                (111_000.0 * np.cos(np.radians(self.particles[:, 0]))),
                                                self.num_particles)
        self.particles[:, 2] += self.rng.normal(0, 0.03 * np.sqrt(dt), self.num_particles)

    def _projected_segments(self, candidate):
        """Cache each candidate polyline as line endpoints in the filter's metre plane."""
        candidate_id = candidate.get("candidate_id")
        key = (candidate_id, tuple(candidate.get("bbox", []))) if candidate_id is not None else None
        if key is not None and key in self._geometry_cache:
            return self._geometry_cache[key]
        cosine = np.cos(np.radians(self.origin_lat))
        starts, ends = [], []
        for polyline in candidate.get("segments", []):
            points = [((float(lon) - self.origin_lon) * 111_000.0 * cosine,
                       (float(lat) - self.origin_lat) * 111_000.0) for lat, lon in polyline]
            if len(points) == 1:
                starts.append(points[0]); ends.append(points[0])
            for a, b in zip(points, points[1:]):
                starts.append(a); ends.append(b)
        lat, lon = candidate.get("representative_lat"), candidate.get("representative_lon")
        if lat is not None and lon is not None:
            point = ((float(lon) - self.origin_lon) * 111_000.0 * cosine,
                     (float(lat) - self.origin_lat) * 111_000.0)
            starts.append(point); ends.append(point)
        result = (np.asarray(starts, dtype=float), np.asarray(ends, dtype=float))
        if candidate.get("candidate_id") is not None:
            self._geometry_cache[key] = result
        return result

    def _candidate_distances_m(self, points, candidate):
        """Vectorized point-to-polyline distance, chunked to bound temporary memory."""
        starts, ends = self._projected_segments(candidate)
        if not len(starts):
            return np.full(len(points), np.inf)
        vectors = ends - starts
        lengths_sq = np.einsum("ij,ij->i", vectors, vectors)
        result = np.empty(len(points), dtype=float)
        for offset in range(0, len(points), 128):
            chunk = points[offset:offset + 128]
            delta = chunk[:, None, :] - starts[None, :, :]
            numer = np.einsum("pse,se->ps", delta, vectors)
            fraction = np.clip(numer / np.maximum(lengths_sq[None, :], 1e-12), 0.0, 1.0)
            nearest = starts[None, :, :] + fraction[:, :, None] * vectors[None, :, :]
            result[offset:offset + len(chunk)] = np.linalg.norm(chunk[:, None, :] - nearest, axis=2).min(axis=1)
        return result

    def update(self, observation, candidates_response):
        self._did_resample = False
        observation = observation or {}
        candidates_response = candidates_response or {}
        validate_candidate_response(candidates_response)
        candidates = candidates_response.get("candidates", [])
        confidence = float(observation.get("ocr_confidence", observation.get("confidence", 0.0)) or 0.0)
        if not candidates:
            self.last_diagnostics = {"status": "no_candidates", "effective_sample_size": self.effective_sample_size}
            return self.last_diagnostics
        if confidence < self.min_ocr_confidence:
            self.last_diagnostics = {"status": "rejected_low_confidence", "effective_sample_size": self.effective_sample_size}
            return self.last_diagnostics

        # Ambiguous candidates are alternatives: mixture likelihood takes the best
        # supported explanation for each particle, without adding them as evidence.
        cosine = np.cos(np.radians(self.origin_lat))
        particle_points = np.column_stack(((self.particles[:, 1] - self.origin_lon) * 111_000.0 * cosine,
                                           (self.particles[:, 0] - self.origin_lat) * 111_000.0))
        distances = np.column_stack([self._candidate_distances_m(particle_points, candidate)
                                     for candidate in candidates])
        nearest = np.min(distances, axis=1)
        if not np.any(np.isfinite(nearest)):
            self.last_diagnostics = {"status": "rejected_invalid_geometry", "effective_sample_size": self.effective_sample_size}
            return self.last_diagnostics
        if np.all(nearest > max(6 * self.sigma_m, 250.0)):
            self.last_diagnostics = {"status": "rejected_out_of_range", "effective_sample_size": self.effective_sample_size}
            return self.last_diagnostics
        match_scores = np.asarray([max(0.01, min(1.0, float(c.get("match_score", 1.0))))
                                   for c in candidates])
        log_likelihoods = -(distances ** 2) / (2.0 * self.sigma_m ** 2)
        log_likelihoods += np.log(match_scores)[None, :]
        row_max = np.max(log_likelihoods, axis=1)
        # Alternatives are mixed with log-sum-exp; candidate text score is a prior.
        mixture_loglike = row_max + np.log(np.exp(log_likelihoods - row_max[:, None]).sum(axis=1))
        mixture_loglike -= np.log(len(candidates))
        log_weights = np.log(np.maximum(self.weights, 1e-300)) + confidence * mixture_loglike
        log_weights -= np.max(log_weights)
        updated = np.exp(log_weights)
        total = updated.sum()
        if not np.isfinite(total) or total <= 0:
            self.last_diagnostics = {"status": "rejected_degenerate_likelihood", "effective_sample_size": self.effective_sample_size}
            return self.last_diagnostics
        self.weights = updated / total
        ess = self.effective_sample_size
        if ess < 0.5 * self.num_particles:
            self._resample()
            ess = self.effective_sample_size
        self.last_diagnostics = {"status": "ambiguous_used" if candidates_response.get("ambiguous") else "matched",
                                 "effective_sample_size": ess, "resampled": self._did_resample,
                                 "candidate_count": len(candidates)}
        return self.last_diagnostics

    @property
    def effective_sample_size(self):
        return float(1.0 / np.sum(self.weights ** 2))

    def _resample(self):
        positions = (self.rng.random() + np.arange(self.num_particles)) / self.num_particles
        indices = np.searchsorted(np.cumsum(self.weights), positions)
        self.particles = self.particles[np.minimum(indices, self.num_particles - 1)]
        self.weights.fill(1.0 / self.num_particles)
        self._did_resample = True

    def get_estimated_state(self):
        self._did_resample = False
        est_lat = float(np.average(self.particles[:, 0], weights=self.weights))
        est_lon = float(np.average(self.particles[:, 1], weights=self.weights))
        lat_spread = np.sqrt(np.average((self.particles[:, 0] - est_lat) ** 2, weights=self.weights)) * 111_000.0
        lon_spread = (np.sqrt(np.average((self.particles[:, 1] - est_lon) ** 2, weights=self.weights))
                      * 111_000.0 * np.cos(np.radians(est_lat)))
        uncertainty = float(np.hypot(lat_spread, lon_spread))
        return est_lat, est_lon, max(uncertainty, 2.0)
