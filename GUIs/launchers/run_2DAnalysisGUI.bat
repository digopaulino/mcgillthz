@echo off
:: Navigate to the folder where this script is located
cd /d "%~dp0"
cd ..

:: =======================================================================
:: USER CONFIGURATION
:: Replace the path below with your exact streamlit executable path.
:: To find it on Windows:
:: 1. Open the "Anaconda Prompt"
:: 2. Activate your environment: conda activate PHYS
:: 3. Run this command: where streamlit
:: =======================================================================
set STREAMLIT_PATH="C:\Users\YourUsername\anaconda3\envs\PHYS\Scripts\streamlit.exe"

:: Run the Streamlit app
echo Starting 2D THz Analysis GUI...
%STREAMLIT_PATH% run 2DAnalysisGUI.py

:: Check if it failed
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo =======================================================================
    echo X ERROR: Streamlit failed to launch or the path was not found.
    echo =======================================================================
    echo Please right-click this .bat file, click "Edit", and update the 
    echo STREAMLIT_PATH variable at the top to match this computer.
    echo.
    echo To find the correct path:
    echo   1. Open the "Anaconda Prompt" from your Windows Start Menu.
    echo   2. Type: conda activate PHYS
    echo   3. Type: where streamlit
    echo =======================================================================
    echo.
    pause
)