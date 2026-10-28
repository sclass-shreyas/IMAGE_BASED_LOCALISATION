# Webcam OCR with OpenVINO

Live printed English text OCR from a webcam using OpenVINO and Open Model Zoo.
OpenVINO uses the Intel GPU by default and reports an error if it cannot find
one; select `--device CPU` to run explicitly on the CPU.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```powershell
python openvino_webcam_test.py --camera 1 --backend dshow --roi select
```

Use `--camera 0` or another index to choose a camera. Supported backends are
`dshow`, `msmf`, and `any`. The first run downloads the Open Model Zoo models
into `models/`. Press `Q` or `Esc` in the preview to stop; OCR snapshots are
written to `openvino_webcam.json` by default.

Run `python openvino_webcam_test.py --help` to see all options.

## Live campus localization

The live coordinator captures a
shared session-relative clock, runs OpenVINO OCR, rejects low-confidence/generic or weakly
matched text, queries the configured RVCE OSM data, and submits each observation to
the particle filter. Person 2 searches both named roads and named OSM building
footprints within the RVCE area from `config.py`. It does not use GNSS. The API starts
without loading any sample stream, and refuses streams marked as synthetic. If a
campus sign uses a name different from OSM, add a verified mapping in
`FEATURE_NAME_ALIASES` in `config.py`.

Start the local API and frontend together:

```powershell
.venv\Scripts\python.exe -m uvicorn particlefilterbackend.engine:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` (do not open the HTML as a `file://` URL), choose a
camera index and OpenVINO device, then click **Start camera**. The dashboard shows
an annotated camera preview with detected text boxes. Serving the page over HTTP
also lets the browser send the referrer required by the OpenStreetMap tile server.
`/api/live/start` can also be called directly; it defaults to `camera_index=0`. Check
`/api/live/status` for camera/model errors; `/api/state` returns the latest OCR cue
and estimate. Stop the camera with the UI button or `POST /api/live/stop`. Do not run
the standalone OCR program at the same time, since it would contend for the selected camera.

The filter uses OCR-to-OSM road candidates as absolute map observations. Optical-flow
`dx_px`/`dy_px` are retained as relative image measurements and are never converted
to metres; image rotation is not assumed to be vehicle yaw. This means the current
live build can constrain location when it recognizes a usable mapped road name, but
it does not yet provide reliable continuous movement between such cues. A no-GNSS
trajectory needs calibrated camera-only visual odometry (for example, a calibrated
road-plane mapping or visual SLAM) constrained by the OSM road network. That scale
and motion calibration is a prerequisite for claiming continuous campus tracking.

Campus trials are appropriate as **field validation**, but first use a short,
controlled route and record independent ground truth for scoring (surveyed points,
known landmarks, or GNSS used only by the evaluator and never passed into the
localizer). Compare position error, cue recognition rate, and uncertainty. Expand to
the full campus after confirming OSM coverage, camera stability, and cue association.

`datastream/sampledata.json` is labeled synthetic and is not loaded by default; the
API rejects it for replay. To replay a real OCR/OSM stream, generate one with
`python -m src.build_datastream <real_ocr_session.json> datastream/session.json`, set
`LOCALIZATION_DATASTREAM` to that file, and call `POST /api/replay/start`. Replay
validates component integration, but without calibrated movement it cannot validate
continuous-motion accuracy.
