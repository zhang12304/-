@echo off
chcp 65001 >nul 2>&1
cd /d "%~dp0"

copy "%~1" ".\_temp_input.jpg" >nul 2>&1
if not exist ".\_temp_input.jpg" (
    echo Failed to read file. Please copy the image into this folder first.
    pause
    exit /b
)

python plate_recognition.py ".\_temp_input.jpg"
del ".\_temp_input.jpg" >nul 2>&1
pause
