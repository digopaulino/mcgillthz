#!/bin/bash

# =======================================================================
# USER CONFIGURATION
# Replace the path below with your exact streamlit executable path. 
# To find it, open a terminal, activate your conda environment: "conda activate ENV_NAME"
# Then run: "which streamlit" and copy the output here.
# =======================================================================
STREAMLIT_PATH="/Users/rodrigo/anaconda3/envs/PHYS/bin/streamlit"

# Navigate to the folder where this .command file is located
cd "$(dirname "$0")"
cd ..

# Run the Streamlit app
echo "Starting 2D THz Calibration GUI..."
"$STREAMLIT_PATH" run Calibration2DGUI.py

# Check if the previous command failed (exit code not equal to 0)
if[ $? -ne 0 ]; then
    echo ""
    echo "======================================================================="
    echo "ERROR: Streamlit failed to launch or the path was not found."
    echo "======================================================================="
    echo "Please replace the path in this script with your streamlit package path."
    echo ""
    echo "To find it:"
    echo "  1. Open a new terminal."
    echo "  2. Activate your conda environment:  conda activate ENV_NAME"
    echo "  3. Copy the output of this command:  which streamlit"
    echo ""
    echo "Right-click this .command file, select 'Open With > TextEdit', and"
    echo "paste that path over the old one."
    echo "======================================================================="
    echo ""
    
    # Prevent the terminal window from closing immediately so you can read this!
    read -p "Press [Enter] to close this window..."
fi