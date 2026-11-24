# Repository audit and upgrade

## State inspected before this implementation

| Component | Finding |
|---|---|
| `particlefilterbackend/engine.py` | Working FastAPI, static frontend, legacy OCR live session and timestamped OCR replay. Location publication waited for a usable OCR match. |
| `particlefilterbackend/particlefilter.py` | Legacy cue-weighting filter, image pixels ignored. Optional metric motion existed, but position calculations used latitude/longitude approximations. No road-network likelihood or topology. |
| `visual_odometry.py` | Experimental homography + LK + similarity fit; full-image detection, no forward/backward check, weak quality gating, no ground ROI. |
| `motion_estimator.py` | Relative pixel optical flow only; retained as an experiment. |
| `calibrate_ground_plane.py` | Batch fit from manually prepared point pairs; no interactive selection, degeneracy or held-out validation. |
| `live_runtime.py` | Webcam-only assumption; initialized OCR unconditionally. Motion waited for OCR sampling and local deltas were simply summed across turns. |
| OSM modules | Useful cached geometries and text matching. Original GeoPackage cleaning discarded OSM node IDs, one-way restrictions and access/topology metadata. |
| `src/build_datastream.py` | Legacy OCR timestamp join, unsuitable for lossless metric-motion replay. |
| Frontend | Working Leaflet, HTTP API polling and draggable/resizable camera panel; no manual start pose, particles, simulation or VO telemetry. |

## Implementation choices

The new default uses a modular motion service rather than rewriting the functioning
legacy OCR matcher. Existing OCR modules and `/api/live/*` and replay endpoints
remain available with `legacy.html`. Default `/api/localization/*` never requires
OCR. Optional OCR diagnostics run asynchronously and do not affect position.

| Files | Responsibility |
|---|---|
| `src/video_source.py` | Shared network/webcam/file acquisition in a stoppable child process, bounded live queue, file media timestamps, reconnect/status reporting. |
| `capture_calibration.py`, `src/ground_calibration.py`, `calibrate_ground_plane.py` | Interactive measured correspondences, validation holdouts, conditioning/residual checks and ground ROI. |
| `visual_odometry.py` | Ground-only feature selection, LK forward/backward checks, RANSAC + rigid refit, metric camera motion and configurable quality gates. |
| `src/road_network.py` | GraphML download/cache, pyproj metric coordinates, directed geometry/topology, local road queries and route generation for simulation only. |
| `particlefilterbackend/road_filter.py` | New metric particle state, manual Gaussian initialization, motion composition, quality-dependent noise, road/connectivity weighting, ESS/resampling and uncertainty. |
| `particlefilterbackend/motion_runtime.py`, `motion_api.py`, `engine.py` | Default motion sessions, source lifecycle, pause/reset, logging, async optional OCR and common API. |
| `src/simulation.py` | Noisy metric packets from selected OSM routes; separately retained truth and deviation route. |
| `evaluate_localization.py`, `replay_motion.py` | Independent reference evaluation and identical-motion OSM-on/off comparisons. |
| `particlefilterfrontend/index.html`, `dashboard.js`, `vendor/leaflet` | Existing layout/map/panel extended with initialization, video modes, particles, quality, reference and road controls. Local Leaflet assets include their license. |
| `tests/test_motion_localization.py`, `tests/test_motion_integration.py` | Synthetic calibration, motion geometry, failure behavior, road direction/connectivity, simulation/API, recorded-video and deviation checks. |

Dependencies add SciPy for indexed road lookup and HTTPX for API tests. The legacy
normalizer test now accepts the existing intentional partial-name key `cubbon`,
while still rejecting fabricated initials. OCR matching behavior is unchanged.

## Remaining practical limitations

- Hardware calibration, Android app connectivity and real scooter accuracy require
  physical tests. Synthetic image tests are explicitly marked fixtures.
- Ground-plane assumptions, possible lens distortion and dynamic foreground objects
  remain relevant. No semantic segmentation, IMU fusion or visual SLAM is included.
- MJPEG sensor timestamps are unknown. Host receive timing and bounded queues cannot
  eliminate camera/network delay or jitter.
- Legacy GeoPackage fallback cannot enforce one-way restrictions; GraphML is the
  preferred input. Full OSM turn-restriction relation parsing is not included.
- Soft road constraints do not ensure recovery from divergence, lane accuracy or
  identification of the correct parallel road. Manual initialization is required.
- The covariance-based uncertainty region is a model summary, not demonstrated
  coverage. Multimodal particle distributions can have a mean between roads.
- GPS is evaluation-only; optional continuous GNSS-assisted estimation is not added.

See README for exact setup/start commands and `docs/testing.md` for hardware
acceptance procedures. The primary demo should be proven first in simulation,
then stationary/short measured camera tests, then a controlled live route.
