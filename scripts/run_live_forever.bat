@echo off
rem Supervision loop for the live signal scanner: relaunches run_live.py if it
rem ever exits (crash, MT5 hiccup, unhandled exception). Installed as the
rem Scheduled Task "AMD_Live_Scanner" by scripts\install_live_task.ps1.
rem All output appends to logs\live_scanner.log (run_live timestamps lines).
cd /d "C:\Users\FT\Documents\FT\Market"
if not exist logs mkdir logs
:loop
echo [%date% %time%] supervisor: starting scanner >> logs\live_scanner.log
"C:\Users\FT\anaconda3\python.exe" scripts\run_live.py --balance 500 --telegram --output live_signals.jsonl >> logs\live_scanner.log 2>&1
echo [%date% %time%] supervisor: scanner exited, restart in 60s >> logs\live_scanner.log
timeout /t 60 /nobreak >nul
goto loop
