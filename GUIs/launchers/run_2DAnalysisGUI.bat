@echo off
:: Navigate to the folder where this script is located
cd /d "%~dp0"
cd ..

:: =======================================================================
:: USER CONFIGURATION
:: 1. Set CONDA_BASE_PATH to the folder where Anaconda/Miniconda is installed.
::    (Common paths: C:\Users\Username\anaconda3 OR C:\Users\Username\miniconda3)
:: 2. Set ENV_NAME to your environment name.
:: =======================================================================
set CONDA_BASE_PATH="C:\Users\Rodrigo\anaconda3"
set ENV_NAME="PHYS"

echo Initializing Conda Environment...
:: This officially hooks Conda into Windows so Math DLLs are found!
call %CONDA_BASE_PATH%\Scripts\activate.bat %CONDA_BASE_PATH%

echo Activating %ENV_NAME%...
call conda activate %ENV_NAME%

echo Starting GUI...
streamlit run 2DAnalysisGUI.py

:: Check if it failed to launch or crashed
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo =======================================================================
    echo X ERROR: Streamlit crashed or the environment wasn't found.
    echo =======================================================================
    echo Please right-click this .bat file, click "Edit", and ensure 
    echo CONDA_BASE_PATH is pointing to your actual Anaconda installation.
    echo.
    pause
    exit /b %ERRORLEVEL%
)

:: Catch-all pause for when the server is stopped normally
echo.
echo Streamlit server stopped gracefully.
pause