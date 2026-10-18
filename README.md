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
