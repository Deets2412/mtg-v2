@echo off
REM Daily batch job: refresh MTG v2's today.json + today_analogs.parquet
REM from current market state. Invoked by Windows Task Scheduler
REM ("MTG-v2 Daily Publish") at 07:00 local time -- roughly post US
REM market close (4pm ET ~= 06:00 AEST).
REM
REM Output is appended to data\publish.log relative to repo root.
REM Exit code from python is propagated so Task Scheduler's history
REM pane shows failures clearly.

setlocal

REM cd to repo root (scripts\ -> repo root via ..)
cd /d "%~dp0.."

if not exist "data" mkdir "data"

>>"data\publish.log" echo.
>>"data\publish.log" echo [%date% %time%] === daily_publish.cmd started ===

".venv\Scripts\python.exe" -m src.publish >>"data\publish.log" 2>&1
set RC=%ERRORLEVEL%

>>"data\publish.log" echo [%date% %time%] === exit=%RC% ===

exit /b %RC%
