#!/bin/bash

# Navigate to the folder where this .command file is located
cd "$(dirname "$0")"

# conda activate PHYS

# Run the Streamlit app
echo "Starting 2D THz Calibration GUI..."
streamlit run Calibration2DGUI.py
