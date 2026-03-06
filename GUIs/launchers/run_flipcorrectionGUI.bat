@echo off
cd /d "%~dp0"
cd ..
:: call conda activate thz_env

echo Starting flip correction GUI...
streamlit run flipcorrectionGUI.py

pause
