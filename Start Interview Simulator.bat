@echo off
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo This folder does not have the app installed yet.
    echo Open README.md and follow the Setup steps, then double-click this file again.
    echo.
    pause
    exit /b 1
)

echo Starting the interview simulator. Your browser will open in a moment.
echo Leave this window open while you practice. Close it when you are finished.
echo.

".venv\Scripts\python.exe" -m train_speaking
if errorlevel 1 (
    echo.
    echo The app stopped with an error. The message above explains why.
    echo.
    pause
)
