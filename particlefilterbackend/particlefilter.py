import numpy as np

class ParticleFilterEngine:
    def __init__(self, num_particles=300, initial_lat=12.97831, initial_lon=77.60517):
        self.num_particles = num_particles
        # State representation: [lat, lon, heading_rad]
        self.particles = np.zeros((num_particles, 3))
        
        # Initialize particles clustered around starting coordinates
        self.particles[:, 0] = initial_lat + np.random.normal(0, 0.0001, num_particles)
        self.particles[:, 1] = initial_lon + np.random.normal(0, 0.0001, num_particles)
        self.particles[:, 2] = np.random.uniform(0, 2 * np.pi, num_particles)
        
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, p3_motion):
        """
        Person 3 Integration: Motion Estimation (dx, dy, d_heading)
        """
        dx = p3_motion.get("dx", 0.0)
        dy = p3_motion.get("dy", 0.0)
        d_heading = p3_motion.get("d_heading", 0.0)

        # Convert meters to approximate lat/lon deltas in Bengaluru
        d_lat = dy / 111000.0
        d_lon = dx / (111000.0 * np.cos(np.radians(self.particles[:, 0])))

        # Apply motion with process noise
        self.particles[:, 0] += d_lat + np.random.normal(0, 0.00001, self.num_particles)
        self.particles[:, 1] += d_lon + np.random.normal(0, 0.00001, self.num_particles)
        self.particles[:, 2] += d_heading + np.random.normal(0, 0.03, self.num_particles)

    def update(self, p1_observation, p2_osm_data):
        """
        Person 1 & Person 2 Integration: 
        Weigh particles based on proximity to Person 2's OSM candidate segments and representative nodes.
        """
        if not p1_observation or not p2_osm_data or p2_osm_data.get("ambiguous", True):
            return

        candidates = p2_osm_data.get("candidates", [])
        if not candidates:
            return

        # Extract target coordinates from Person 2's candidates
        target_coords = []
        for cand in candidates:
            # Use representative lat/lon or segment midpoints if available
            target_coords.append([cand.get("representative_lat"), cand.get("representative_lon")])
            for seg in cand.get("segments", []):
                for pt in seg:
                    target_coords.append(pt)

        target_coords = np.array(target_coords)
        if target_coords.size == 0:
            return

        # Compute distance matrix between particles and OSM targets
        lat_diff = (self.particles[:, 0, None] - target_coords[:, 0]) * 111000.0
        lon_diff = (self.particles[:, 1, None] - target_coords[:, 1]) * 111000.0 * np.cos(np.radians(self.particles[:, 0, None]))
        distances = np.sqrt(lat_diff**2 + lon_diff**2)
        
        min_distances = np.min(distances, axis=1)
        
        # Gaussian likelihood weighting (sigma = 12 meters)
        sigma = 12.0
        likelihoods = np.exp(-(min_distances ** 2) / (2 * (sigma ** 2)))
        
        self.weights *= likelihoods
        self.weights += 1.e-300
        self.weights /= np.sum(self.weights)

        self._resample()

    def _resample(self):
        cumulative_sum = np.cumsum(self.weights)
        cumulative_sum[-1] = 1.0
        indices = np.searchsorted(cumulative_sum, np.random.uniform(0, 1, self.num_particles))
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def get_estimated_state(self):
        est_lat = np.average(self.particles[:, 0], weights=self.weights)
        est_lon = np.average(self.particles[:, 1], weights=self.weights)
        
        lat_spread = np.std(self.particles[:, 0]) * 111000.0
        lon_spread = np.std(self.particles[:, 1]) * 111000.0 * np.cos(np.radians(est_lat))
        uncertainty = float(np.sqrt(lat_spread**2 + lon_spread**2))
        
        return float(est_lat), float(est_lon), max(uncertainty, 2.0)