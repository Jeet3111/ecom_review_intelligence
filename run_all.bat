@echo off
REM One-command run (Windows):  run_all.bat [--quick]
cd /d "%~dp0"
if not exist .venv ( python -m venv .venv )
call .venv\Scripts\activate
pip install -q -r requirements.txt
python run_pipeline.py %*
python -m unittest discover -s tests -v
python report\build_report.py
echo Starting web app on http://127.0.0.1:8080  (Ctrl+C to stop)
python webapp\app.py
