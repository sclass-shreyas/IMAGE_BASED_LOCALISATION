import numpy as np
from pyproj import Transformer

class ParticleFilterEngine:
    def __init__(self, num_particles=1500, initial_lat=12.9716, initial_lon=77.5946):
        self.num_particles = num_particles
        
        # UTM Zone 43N for Bengaluru meters
        self.to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32643", always_xy=True)
        self.to_gps = Transformer.from_crs("EPSG:32643", "EPSG:4326", always_xy=True)
        
        init_x, init_y = self.to_utm.transform(initial_lon, initial_lat)
        
        # Internal hidden particle state: [x, y, heading, weight]
        self.particles = np.zeros((num_particles, 4))
        self.particles[:, 0] = init_x + np.random.normal(0, 10, num_particles)
        self.particles[:, 1] = init_y + np.random.normal(0, 10, num_particles)
        self.particles[:, 2] = np.random.uniform(0, 2 * np.pi, num_particles)
        self.particles[:, 3] = 1.0 / num_particles

    def predict(self, dx, dy, d_heading, noise_xy=1.0, noise_heading=0.02):
        for i in range(self.num_particles):
            theta = self.particles[i, 2]
            rot_dx = dx * np.cos(theta) - dy * np.sin(theta)
            rot_dy = dx * np.sin(theta) + dy * np.cos(theta)
            
            self.particles[i, 0] += rot_dx + np.random.normal(0, noise_xy)
            self.particles[i, 1] += rot_dy + np.random.normal(0, noise_xy)
            self.particles[i, 2] += d_heading + np.random.normal(0, noise_heading)
            self.particles[i, 2] = np.mod(self.particles[i, 2], 2 * np.pi)

    def update(self, candidates, sigma=20.0):
        if not candidates:
            return
            
        cand_coords = [self.to_utm.transform(c["lon"], c["lat"]) for c in candidates]
        min_distances = np.full(self.num_particles, np.float64(np.inf))
        
        for cx, cy in cand_coords:
            dists = np.sqrt((self.particles[:, 0] - cx)**2 + (self.particles[:, 1] - cy)**2)
            min_distances = np.minimum(min_distances, dists)

        weights = np.exp(-(min_distances ** 2) / (2 * (sigma ** 2)))
        weights += 1.e-300
        
        self.particles[:, 3] *= weights
        self.particles[:, 3] /= np.sum(self.particles[:, 3])

    def resample(self):
        weights = self.particles[:, 3]
        step = 1.0 / self.num_particles
        u = np.random.uniform(0, step)
        c = weights[0]
        i = 0
        
        resampled_indices = np.zeros(self.num_particles, dtype=int)
        for p in range(self.num_particles):
            while u > c:
                i = (i + 1) % self.num_particles
                c += weights[i]
            resampled_indices[p] = i
            u += step
            
        self.particles = self.particles[resampled_indices]
        self.particles[:, 3] = 1.0 / self.num_particles

    def get_estimate(self):
        mean_x = np.average(self.particles[:, 0], weights=self.particles[:, 3])
        mean_y = np.average(self.particles[:, 1], weights=self.particles[:, 3])
        
        variance_x = np.average((self.particles[:, 0] - mean_x)**2, weights=self.particles[:, 3])
        variance_y = np.average((self.particles[:, 1] - mean_y)**2, weights=self.particles[:, 3])
        uncertainty = np.sqrt(variance_x + variance_y)
        
        lon, lat = self.to_gps.transform(mean_x, mean_y)
        return {"lat": lat, "lon": lon, "uncertainty_meters": round(uncertainty, 2)}