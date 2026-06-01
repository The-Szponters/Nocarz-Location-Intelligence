# Launch the Nocarz prediction microservice on http://127.0.0.1:8080
# Usage (from repo root):  .\scripts\run_server.ps1
$env:PYTHONPATH = "src"
python -m uvicorn nocarz.app:app --host 127.0.0.1 --port 8080 --workers 1
