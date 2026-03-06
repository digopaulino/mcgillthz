#!/bin/bash

# Navigate to the folder where this .command file is located
cd "$(dirname "$0")"
cd ..

# Run the Streamlit app
echo "Starting flip correction GUI..."

# Replace the path below to your streamlit package. 
# To find it, activate you conda environment: "conda activate ENV_NAME"
# Copy the output of the following command: "which streamlit"
/Users/rodrigo/anaconda3/envs/PHYS/bin/streamlit run flipcorrectionGUI.py
