@echo off
cd /d "%~dp0"
cd ..
:: call conda activate thz_env

echo Starting 2D THz Analysis GUI...
streamlit run 2DTempScanGUI.py

pause
