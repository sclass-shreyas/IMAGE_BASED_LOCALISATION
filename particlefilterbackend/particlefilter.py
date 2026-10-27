import numpy as np

class ParticleFilterEngine:
    def __init__(self, num_particles=100, initial_lat=12.9716, initial_lon=77.5946):
        self.num_particles = num_particles
        # Initialize particles around the initial coordinate
        self.particles = np.zeros((num_particles, 2)) # [lat, lon]
        self.particles[:, 0] = initial_lat + np.random.normal(0, 0.0001, num_particles)
        self.particles[:, 1] = initial_lon + np.random.normal(0, 0.0001, num_particles)
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, dx_meters, dy_meters):
        # Convert meter motion roughly to delta lat/lon (approx 111,000 meters per degree)
        d_lat = dy_meters / 111000.0
        d_lon = dx_meters / (111000.0 * np.cos(np.radians(12.9716)))
        
        # Add motion with process noise
        self.particles[:, 0] += d_lat + np.random.normal(0, 0.00002, self.num_particles)
        self.particles[:, 1] += d_lon + np.random.normal(0, 0.00002, self.num_particles)

    def update(self, observation_detected):
        if observation_detected:
            # Resweight particles if an observation/landmark cue is matched
            self.weights *= np.random.uniform(0.8, 1.2, self.num_particles)
            self.weights += 1.e-300
            self.weights /= np.sum(self.weights)
            
            # Resampling step (Low Variance Sampling)
            indices = np.random.choice(self.num_particles, size=self.num_particles, p=self.weights)
            self.particles = self.particles[indices]
            self.weights.fill(1.0 / self.num_particles)

    def get_estimated_state(self):
        est_lat = np.mean(self.particles[:, 0])
        est_lon = np.mean(self.particles[:, 1])
        # Calculate uncertainty as standard deviation spread in meters
        lat_spread = np.std(self.particles[:, 0]) * 111000.0
        lon_spread = np.std(self.particles[:, 1]) * 111000.0 * np.cos(np.radians(est_lat))
        uncertainty = float(np.sqrt(lat_spread**2 + lon_spread**2))
        return float(est_lat), float(est_lon), uncertainty