# Calibrating the scooter's ground-plane camera

Do this before the first metric camera-motion experiment. Simulation and camera
preview need no calibration. Calibration must use the **same phone, lens, image
orientation, resolution, focus/zoom, crop and rigid mounting** as the experiment.

## Physical layout

1. Park the scooter upright on flat ground in a traffic-free area. Secure the phone
   so it faces forward and slightly down. Keep the mount unchanged. The camera
   needs a useful textured patch of ground, not mostly sky or buildings.
2. Mark the camera center's vertical projection on the ground. This is `(0,0)`.
   Lay a tape along the camera's forward axis. Ground coordinates are **right,
   forward**, in metres: right is positive, left negative, forward positive.
   These are camera axes, not compass east/north.
3. Place six to nine visible markers at several distances and on both sides of the
   forward line. Measure each position from the same origin/axes. Place additional
   markers for independent validation. Avoid a single row, tiny cluster, or nearly
   collinear layout. Cover the ground patch you intend to track.
4. Keep people, wheels and other objects out of this patch. Do not move the scooter
   or camera while capturing the measurements. Record actual measurements; no
   example JSON values are supplied as calibration data.

This method fits a plane mapping directly. It does not implement a full lens
distortion calibration. If held-out points fail, investigate distortion, marker
measurement, mount stability and image crop rather than relaxing thresholds blindly.

## Capture marker correspondences

Close the dashboard camera with **Disconnect / stop** first. Start the Android
camera server and verify its URL in your laptop browser. From the repository root:

```powershell
.\.venv\Scripts\python.exe capture_calibration.py --source-type network --url http://PHONE_IP:8080/video --output ground_points.json
```

For a USB camera:

```powershell
.\.venv\Scripts\python.exe capture_calibration.py --source-type webcam --camera-index 1 --width 1280 --height 720
```

The window uses original image pixels. It does not resize the frame to fit the
screen; use a suitable camera resolution if the window exceeds your display.

| Action | Control |
|---|---|
| Freeze the stationary layout | Space |
| Select a marker | Left click |
| Enter its measured coordinates | Type `right,forward`, then Enter |
| Remove pending/last selection | U |
| Unfreeze and clear all selections | R |
| Toggle independent validation-point entry | V |
| Validate and save | S |
| Quit | Q or Esc |

At least four fitting markers are required; six to nine is recommended. Validation
markers are excluded from fitting. The UI saves `ground_points.json` and
`ground_points.png`. Ground coordinate entries can be negative or decimal values.
Do not click the instruction overlay at the top of the image.

The file contains actual `image_width`, `image_height`, and `points`, each with
`pixel_xy`, `ground_right_forward_m` and an optional `validation` flag. Fit it:

```powershell
.\.venv\Scripts\python.exe calibrate_ground_plane.py ground_points.json visual_odometry_calibration.json
```

The fitter rejects insufficient/degenerate configurations, singular mappings,
invalid image coordinates and excessive residuals. Default maximum residual is
0.15 metres, configurable with `--max-residual-m`. It reports fit RMS and held-out
errors in metres. Four fitting points can fit almost exactly while still having
bad measurements; held-out points matter. Neither fit RMS nor a passing check is
a vehicle-localization accuracy measurement.

The fitting markers' convex hull becomes the ground ROI. Features outside it are
excluded to avoid uncalibrated extrapolation and non-ground objects. Keep the
region large and textured enough for tracking. Validation markers should cover
different locations within this region. Regenerate old calibration files that
lack `roi_polygon_px`.

## Start and inspect odometry

```powershell
$env:VIDEO_SOURCE_TYPE = "network"
$env:VIDEO_SOURCE_URL = "http://PHONE_IP:8080/video"
$env:VISUAL_ODOMETRY_CALIBRATION = (Resolve-Path .\visual_odometry_calibration.json).Path
$env:ENABLE_OCR = "false"
.\.venv\Scripts\python.exe -m uvicorn particlefilterbackend.engine:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/. Select the initial position on the map, set heading and
uncertainty, and start localization. For physical odometry checks first disable
OSM, so road constraints cannot hide movement errors. The yellow outline is the
ground ROI; green points are accepted tracked features. Toggle **Bird's-eye preview**
for a fixed debug viewport of right -5..5 m and forward 0..15 m. This viewport does
not alter the estimator and may clip calibration regions outside those bounds.

Proceed through [stationary, straight and turn tests](testing.md). Resolution
mismatch stops the localization worker with a visible error; the API stays alive.
Other poor measurements are logged as invalid, trigger feature reseeding, and
increase filter uncertainty. They are not interpreted as confirmed zero movement.

## Motion and coordinate conventions

The fitted ground-feature transform maps old camera-ground coordinates into new
ones. The estimator inverts this transform to recover **camera** displacement in
the previous frame: positive `right_delta_m`, positive `forward_delta_m`, and
counter-clockwise `yaw_delta_rad`. It uses forward/backward Lucas–Kanade checks,
RANSAC, a rigid SVD refit, ground residuals, scale-change, blur, frame-interval,
speed and yaw-rate gates. Defaults live in `VOSettings` in `visual_odometry.py` and
can be overridden in the session API's `vo_settings` object.

Each motion packet reaches the filter independently of OCR. The filter rotates
each local translation using its current particle heading before adding it;
translations across turns are not simply summed in camera coordinates.
Geographic heading is counter-clockwise from east: E=0°, N=90°, W=180°, S=270°.
The initial heading is the mounted camera's forward direction. Align that with
the scooter's travel direction for the experiment.

The homography assumes constant camera pose relative to the local ground plane.
Leaning, vibration, suspension movement, slopes and braking can break that model
without triggering every quality check. Stop and reassess when these effects
dominate. This software cannot establish accuracy without measured experiments.
