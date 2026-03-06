@echo off
cd /d "%~dp0"
cd ..
:: call conda activate thz_env

echo Starting 2D THz Calibration GUI...
streamlit run Calibration2DGUI.py

pause
