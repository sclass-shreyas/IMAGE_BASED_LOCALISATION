# Staged testing and troubleshooting

All commands below run from the repository root in Windows PowerShell. The software
test suite uses synthetic image/geometry fixtures and generated video; none of its
results should be reported as real scooter accuracy.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Stage checklist

| Test | Procedure | Record and judge |
|---|---|---|
| 1. Mobile streaming | Open phone server in laptop browser; Connect preview; stop phone server, then restart | Actual resolution/orientation, frames received, reconnect status, API remains responsive. Check network delay separately if exposure timestamps are available. |
| 2. Calibration | Fit 6–9 measured markers, reserve extra points as validation | Fit RMS and held-out residuals in metres; independent measurements agree. Do not use fitting residual as motion accuracy. |
| 3. Stationary | Start session with OSM disabled; leave mounted camera stationary for 30 s | Accepted distance, false endpoint drift, rejection rate, median feature count. Evaluate log with `--measured-distance-m 0`. |
| 4. Straight travel | Move fixed setup 1, 3, 10 and 25 measured metres, flat ground; use separate sessions | Accepted distance error and endpoint error if surveyed reference exists. Stop before changing setup/starting another session. Compare both OSM-on and off after VO is credible. |
| 5. Turn | Controlled measured rotation/turn on flat ground, maintain camera geometry | Sign and magnitude of summed yaw; compare final heading to measured angle. A left turn should increase east/CCW heading. Camera yaw is affected by vehicle lean. |
| 6. Filter | Run software tests and a noise-free simulation before adding noise | Local right/forward axes, 90° turn composition, metric projection, covariance, duplicate-time rejection. |
| 7. OSM | Simulate near road and inject off-road/heading-inconsistent hypotheses in tests | Soft weights, effective sample size and resampling; no forced centerline snapping. Check source says `osm_drive_graph`, not legacy fallback, for one-way testing. |
| 8. Deviation | Select simulation start/destination and a deviation waypoint on a reachable side road | Estimated path can follow the actual connected turn rather than the displayed direct planned route. |
| 9. Failure | Cover camera, blur image, change resolution, disconnect Wi-Fi | Rejection reason and uncertainty growth. Wrong resolution should fail clearly. Stop/pause/reset and API must remain responsive. |
| 10. Integrated | Short slow run in a flat, traffic-free controlled area; stationary laptop operator | Logs, independently measured reference, acceptance rate, processing FPS, endpoint/trajectory error. Report all failures and exclusions. |

For real tests, define acceptable error limits before seeing the results. Treat a
failed short-distance test as a reason to fix calibration/tracking before trying
a longer route. There are no preclaimed accuracy thresholds or measured field
results in this repository.

## Exact evaluation examples

Replace the log path with the path shown by your session:

```powershell
.\.venv\Scripts\python.exe evaluate_localization.py results\sessions\SESSION.jsonl --measured-distance-m 10 --output results\ten_metres.json
.\.venv\Scripts\python.exe replay_motion.py results\sessions\SESSION.jsonl --disable-osm --output results\without_osm.jsonl
.\.venv\Scripts\python.exe evaluate_localization.py results\without_osm.jsonl --reference-csv reference.csv --output results\without_osm_metrics.json
```

GPS reference uses `timestamp_s,lat,lon` and optional `heading_deg`. Establish a
shared time origin, or specify `--reference-offset-s`. Interpolation rejects long
reference gaps. The browser's reference overlay uses a sample within one second;
the offline evaluator is preferred for metrics. GPS can itself be noisy; describe
it as approximate reference, not surveyed truth. Do not feed it into the estimator.

## Troubleshooting

| Symptom | Action |
|---|---|
| Port 8000 already in use | Close the earlier server with Ctrl+C, or use `--port 8001` and open that URL. Inspect the owner with `Get-NetTCPConnection -LocalPort 8000`; do not kill unrelated processes. |
| Phone URL unavailable | Check the app server is started, displayed IP/port, same subnet/hotspot, client isolation and phone sleep. Try `Test-NetConnection PHONE_IP -Port 8080`. Some hotspots block peer traffic. |
| Stream reconnecting | Inspect endpoint/protocol; preview in the laptop browser. OpenCV uses its FFmpeg network backend with open/read timeouts. Backend/API stays running while the source retries. |
| Phone image lags | Reduce resolution/FPS or improve Wi-Fi. The latest-frame queue bounds host backlog but cannot remove phone-side buffering. Host queue age is not end-to-end latency. |
| Camera index wrong | Choose another webcam index. Close other camera apps and calibration capture. For USB on Windows try dshow/msmf/any through the API's video backend setting. |
| Blank/failed calibration window | Validate stream first. It uses actual image dimensions without resizing. Use a camera resolution that fits the monitor. |
| Collinear/unstable homography | Spread markers over a 2D patch and recheck pixel order and metre measurements. Reject near-horizon points. |
| High validation residual | Check common ground origin/axes, lens distortion, marker measurement and whether phone settings changed. Do not copy example coordinates. |
| Calibration lacks ROI | Regenerate it using the updated `calibrate_ground_plane.py` from measured correspondences. |
| Resolution mismatch | Configure the phone to the calibrated dimensions, or recalibrate. The backend will not silently resize. |
| Few tracks / frequent blur | Increase visible textured road area, improve lighting/exposure and mount stability. Adjust thresholds only against measured tests. |
| Non-rigid scale / ground residual failures | Examine bumps, pitch/roll, slope, non-ground objects or bad calibration. Reinitialization cannot repair a violated plane model. |
| No location | Select a start and heading, check calibration, inspect the visible session error. Camera preview alone does not initialize the filter. |
| Estimate moves while camera still | Use OSM-off stationary test; inspect optical-flow jitter and cumulative distance. Missing-motion uncertainty can spread the particles without implying observed movement. |
| Graph unavailable / Overpass 502 | Use the existing cache; retry the explicit download later. Do not continually retry from the frame loop. The map download runs separately from localization. |
| Legacy geometry label | Download GraphML. Old name-matching caches discarded direction/topology metadata; fallback assumes bidirectional endpoint connectivity. |
| No route to destination | Select another connected drivable road or expand/download area; one-way restrictions and disconnected components can prevent a route. |
| No plausible road | Check starting point, heading, calibration, coverage and real off-road movement. The filter reports loss of road support; it does not teleport to a road. |
| Basemap missing or HTTP 403 | Open the HTTP dashboard; allow normal referrer behavior. Cached road geometry works without tiles. Do not bypass tile-provider blocking or bulk-fetch public tiles. |
| OCR error | OCR is optional diagnostic code. Turn it off for the main demo; OpenVINO device/model availability is unrelated to VO. |

The API reference is at `/docs`. Stream errors, quality rejections and session log
paths are visible in the dashboard; complete raw measurements are in JSONL logs.
