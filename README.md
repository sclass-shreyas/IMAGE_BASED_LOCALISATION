# Camera motion + OSM localization prototype

The default dashboard uses calibrated ground-plane visual odometry, a manually
initialized particle filter and cached OSM drivable-road constraints. OCR is off
by default. Android HTTP/MJPEG, webcams, recorded video and simulated motion use
the same motion/filter interface. Accuracy on your phone and scooter has **not**
been measured.

## Setup and first simulation (Windows PowerShell)

Run from the project root. If `.venv` already exists, use it.

```powershell
cd C:\Users\shrey\OneDrive\Desktop\grace
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.road_network --download
.\.venv\Scripts\python.exe -m uvicorn particlefilterbackend.engine:app --host 127.0.0.1 --port 8000
```

The configured Vajrahalli drivable GraphML has already been downloaded in this
workspace; the download command is for another machine/area or refreshing data.
Open **http://127.0.0.1:8000/**. The backend also serves the frontend: no second
server is required. Use HTTP, not a `file://` page.

1. Select **Simulation** as the source.
2. Set map click to **Starting location**, and click a road.
3. Set map click to **Simulation destination**, and click another reachable road.
4. Click **Start localization**. Simulation starts at the nearest route node and
   its first tangent; this is simulated initialization, not measured motion.
5. Blue is the estimate, grey the odometry-only baseline, orange simulated truth,
   purple the planned route, and yellow points the particles.
6. Add a **Simulation deviation waypoint** on a connected side road to make the
   actual simulated route deviate. The filter receives neither route; it considers
   nearby connected roads. Clear the waypoint to restore the direct route.

Stop/reset between experiments. Speed, motion noise, yaw drift and playback rate
are configurable. Simulation tests integration, not camera accuracy.

## Connect the Android phone

Use an app such as IP Webcam that exposes an HTTP/MJPEG endpoint. No custom Android
app is needed. Connect phone and laptop to the same Wi-Fi or hotspot. Select the
rear main camera, landscape orientation, a fixed resolution (start at 1280×720),
fixed zoom/focus and 15–30 FPS. Disable electronic stabilization/cropping if possible.
Start the app's server and copy its displayed IP address and port.

First open that address in the laptop browser. An IP Webcam endpoint commonly
looks like `http://PHONE_IP:8080/video`; use the endpoint your app actually exposes.
The address below is illustrative.

```powershell
Test-NetConnection 192.168.1.100 -Port 8080
$env:VIDEO_SOURCE_TYPE = "network"
$env:VIDEO_SOURCE_URL = "http://192.168.1.100:8080/video"
$env:VIDEO_WIDTH = "1280"
$env:VIDEO_HEIGHT = "720"
$env:VIDEO_FPS = "30"
$env:ENABLE_OCR = "false"
.\.venv\Scripts\python.exe -m uvicorn particlefilterbackend.engine:app --host 127.0.0.1 --port 8000
```

In the dashboard select **Android network stream**, enter the endpoint and click
**Connect preview**. This needs neither calibration nor a starting position.
Check the actual resolution and orientation. Network resolution is set on the
phone; width/height settings only request a local webcam mode. Images are never
silently resized before odometry.

MJPEG usually lacks exposure timestamps. Live packets use laptop receive times
on a monotonic clock. Host queue age is reported; sensor/network latency remains
unknown. Bursty delivery can invalidate speed and tracking estimates.

## Calibrate the actual mounted camera

Follow [the physical calibration guide](docs/visual_odometry.md) in a stationary,
flat, traffic-free area. Stop the dashboard camera before opening the utility.

```powershell
.\.venv\Scripts\python.exe capture_calibration.py --source-type network --url http://192.168.1.100:8080/video --output ground_points.json
.\.venv\Scripts\python.exe calibrate_ground_plane.py ground_points.json visual_odometry_calibration.json
```

**Space** freezes the image. Click a measured marker, type
`right_metres,forward_metres`, then **Enter**. **U** undoes; **R** restarts;
**V** toggles independent validation markers; **S** validates/saves point pairs
and the frozen image; **Q/Esc** exits. Use six to nine fitting points and extra
held-out validation points. Four non-collinear fitting points is the minimum.
The convex hull of fitting markers becomes the ground tracking ROI.

For USB/webcam calibration:

```powershell
.\.venv\Scripts\python.exe capture_calibration.py --source-type webcam --camera-index 1 --width 1280 --height 720
```

Recorded calibration footage is supported with
`--source-type file --url C:\Videos\calibration.mp4`. It must show the measured
layout with the exact experiment camera geometry and image configuration.

## Start live or recorded-video localization

```powershell
$env:VISUAL_ODOMETRY_CALIBRATION = (Resolve-Path .\visual_odometry_calibration.json).Path
$env:ENABLE_OCR = "false"
.\.venv\Scripts\python.exe -m uvicorn particlefilterbackend.engine:app --host 127.0.0.1 --port 8000
```

1. Choose the source, enter its URL/local file path or webcam index, and confirm
   the calibration path. Select **Recorded video** for prerecorded footage.
2. Select your **actual starting location** on the map. Enter camera-forward heading:
   **East=0°, North=90°, West=180°, South=270°**. Convert a compass bearing with
   `heading = (90 - compass_bearing) % 360`.
3. Set position sigma in metres and heading sigma in degrees. These are initial
   one-standard-deviation uncertainties, not accuracy promises. Optionally select
   a directed road with **use its travel direction** checked; verify that heading.
4. Enable OSM constraints and click **Start localization**. Every processed frame
   supplies a motion update; OCR does not control the filter rate.
5. Watch rejections, uncertainty and particles. Pause suspends updates. Resuming
   a live stream resets tracking and increases uncertainty for the unseen interval.
   Stop ends the session; Reset clears state.

Recorded playback preserves frames and uses video media timestamps (FPS fallback
if unavailable). Use the same file, configuration and seed for repeatable runs.
The camera lens, mount, crop and resolution must match its calibration.

## OSM maps and filter behavior

`python -m src.road_network --download` writes an OSMnx `drive` MultiDiGraph into
`data/osm/drive_<area-hash>.graphml`. It preserves available directed edges and
node connectivity. If absent, an old GeoPackage can supply **legacy geometry,
assumed bidirectional**; this fallback is labelled in the dashboard and cannot
recover one-way/access/grade-separation metadata discarded by the original writer.

To change area, stop the session, move the map to the desired center, set a radius
and click **Download drivable graph at map center**. Downloads run in the background.
For persistent defaults, change `AREA_CENTER`, `AREA_RADIUS_M`, and `AREA_NAME` in
`src/osm_matching/config.py`, then restart. `INITIAL_POSITION` and
`INITIAL_HEADING_DEG_FROM_EAST` only prefill the UI; map-click initialization works
without editing configuration.

The new filter stores east/north metres in a local azimuthal equidistant projection
using pyproj. Motion is rotated by each particle's current heading. Distance,
heading and connected-transition likelihoods act as soft constraints without
centerline snapping. Road weighting runs after about 0.5 m of accepted motion,
avoiding repeated static-map updates that manufacture confidence. Missing motion
grows uncertainty and suspends reweighting. No plausible-road support is reported.

Leaflet is bundled locally. Background tiles need internet; cached road geometry,
markers and particles work without tiles. Do not bulk-download public OSM tiles.

## Logging, replay and independent evaluation

Each session writes `results/sessions/<timestamp>_<id>.jsonl`; the dashboard displays
its path. Logs include raw motion, quality, estimates, baseline, covariance,
rejections and timing. Reference fields are added only **after** estimator updates.
Stream URLs/passwords are excluded from metadata.

Replace `SESSION.jsonl` with the log shown in your dashboard:

```powershell
.\.venv\Scripts\python.exe evaluate_localization.py results\sessions\SESSION.jsonl --output results\evaluation.json
.\.venv\Scripts\python.exe replay_motion.py results\sessions\SESSION.jsonl --disable-osm --output results\without_osm.jsonl
.\.venv\Scripts\python.exe evaluate_localization.py results\without_osm.jsonl --output results\without_osm_metrics.json
```

For measured movement add `--measured-distance-m 10` (use the actual distance), or
`--measured-distance-m 0` for the stationary test. Reference CSV columns are
`timestamp_s,lat,lon`, optionally `heading_deg` in the east/CCW convention. Align
reference and session clocks explicitly:

```powershell
.\.venv\Scripts\python.exe evaluate_localization.py results\sessions\SESSION.jsonl --reference-csv reference.csv --reference-offset-s 0 --output results\reference_metrics.json
```

Interpolation is limited by `--max-reference-gap-s` (default 2 seconds). Outputs
include paired errors, mean/RMSE, baseline RMSE, final drift, heading error when
available, distance error, rejection rate and processing time/rate. Without a
reference, position-error metrics are omitted. The dashboard loads reference CSVs
for visualization only; reference locations are never sent to the filter. Phone
GPS is approximate reference data. Continuous GNSS-assisted estimation is not
implemented.

`src/build_datastream.py` remains a legacy OCR-session converter. Its nearest-time
OCR join is unsuitable for VO increment replay; use `replay_motion.py` instead.

## APIs and quality thresholds

See **http://127.0.0.1:8000/docs**. The default API is `/api/localization/` with
`config`, `map`, `start`, `state`, `pause`, `stop`, `reset`, `preview.jpg`, and
background `map/download`. A `start` JSON can include `vo_settings` (fields in
`VOSettings`, `visual_odometry.py`) and `filter_settings` (fields in
`FilterSettings`, `particlefilterbackend/road_filter.py`). Example overrides:
`"vo_settings": {"min_tracks":20,"max_dt_s":0.4}` and
`"filter_settings": {"particles":800,"road_sigma_m":8}`. Tune against measured data.

Optional OCR runs asynchronously as **diagnostics only** in the new pipeline.
The original OCR/fuzzy-matching localizer and APIs remain at **/legacy.html**;
see [legacy instructions](docs/legacy_ocr.md). Only use one camera session at a time.

## Testing, troubleshooting and limitations

- [Staged tests and troubleshooting](docs/testing.md)
- [Repository audit and implementation summary](docs/upgrade_audit.md)
- [Physical calibration instructions](docs/visual_odometry.md)

Fixed-plane odometry assumes stable height, pitch and roll. Scooter leaning,
bumps, braking, slopes, blur, lens distortion and moving objects inside the ROI
can introduce errors that quality gates may not detect. This ROI is not semantic
ground segmentation. Recalibrate after mount, lens, zoom, crop or resolution changes.

OSM cannot always resolve parallel roads or along-road drift. This is manual
initialization, not global localization, and standard centerlines are not lane-level
maps. The graph model does not parse every turn-restriction relation. A particle
mean can lie between roads when hypotheses are ambiguous; the covariance circle
is not experimentally validated accuracy. Lost-motion intervals remain uncertain;
recovery does not reconstruct missing travel or guarantee recovery from divergence.

Do not control the scooter with this prototype. Use a secured mount, a traffic-free
test area and a stationary laptop operator.

Implementation references: [OpenCV capture properties](https://docs.opencv.org/4.5.4/d4/d15/group__videoio__flags__base.html)
and [OSMnx graph/GraphML documentation](https://osmnx.readthedocs.io/en/stable/user-reference.html).
