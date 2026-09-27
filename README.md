# Penghitung Traffic YOLO

Local vehicle counting for the Jasamarga SS Kopo CCTV livestream (Padalarang–Cileunyi KM 135+600). Detects vehicles with YOLO, counts each tracked vehicle once as it crosses a calibrated virtual line, and reports traffic density for Pagi (06–10), Siang (10–15), and Sore (15–19) via a live overlay, local dashboard, and daily CSV export.

## Requirements

- Windows, Python 3.14
- NVIDIA GPU recommended (CPU fallback works at reduced throughput)

## Setup

```powershell
.\setup.ps1
```

## Run

```powershell
.\run.ps1 --no-overlay
```

Dashboard: `http://127.0.0.1:5000/` during the 06:00–19:00 (Asia/Jakarta) reporting window.

## Calibrate the counting line

```powershell
.\run.ps1 --calibrate
```

Saves one still under `data/calibration/` and marks the line calibrated. Edit the `[line]` endpoints in `config.toml` to reposition it.

## Smoke test (no production writes)

```powershell
.\run.ps1 --smoke-seconds 30 --no-overlay
```

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Notes

- Stream URLs rotate periodically; if a playlist goes permanently empty, re-resolve the camera UUID from the Bina Marga CCTV page and update `config.toml`.
- Only the current Jakarta calendar day of data is retained; download the CSV before midnight.
