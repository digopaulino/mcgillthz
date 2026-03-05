import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import subprocess
import sys

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="2D THz Calibration", layout="wide")

st.title("2D THz Spectroscopy Calibration")
st.markdown("""
This interface generates the calibration file for the 2DTS setup. 
1. Click **Browse File** to locate your calibration data.
2. Adjust the fitting parameters.
3. Click **Run Calibration** to visualize the fits.
4. If the fits look correct, click **Save Calibration**.
""")

# --- HELPER FUNCTION: ISOLATED FILE BROWSER ---
def select_file():
    """
    Opens a native OS file dialog using an isolated subprocess.
    This prevents macOS 'Tcl_WaitForEvent' threading crashes when using Tkinter inside Streamlit.
    """
    script = """
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.attributes('-topmost', True)
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select Calibration Data",
    filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
)
print(file_path)
"""
    # Run the tiny Tkinter script in a separate Python process
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    return result.stdout.strip()

# --- STATE MANAGEMENT ---
# Initialize persistent variables in session_state
if 'file_path' not in st.session_state:
    st.session_state.file_path = ""
if 'calibration_data' not in st.session_state:
    st.session_state.calibration_data = None

def clear_calibration_state():
    """Clears the stored calibration data if the user changes a parameter."""
    st.session_state.calibration_data = None

# --- SIDEBAR: INPUT & OUTPUT SETTINGS ---
st.sidebar.header("1. Input/Output Files")

# Browse Button
if st.sidebar.button("📁 Browse File"):
    chosen_path = select_file()
    if chosen_path:  # Only update if the user didn't hit 'Cancel'
        st.session_state.file_path = chosen_path
        clear_calibration_state()
        st.rerun() # Refresh the UI immediately to show the path

# Text input showing the chosen path
file_path = st.sidebar.text_input(
    "Absolute Path to Calibration Data (.csv)",
    value=st.session_state.file_path,
    on_change=clear_calibration_state
)
st.session_state.file_path = file_path # Sync state

# Determine the default output path based on the input path
if file_path and os.path.isfile(file_path):
    default_out_dir = os.path.dirname(file_path)
    default_out_file = os.path.join(default_out_dir, "calibration.csv")
else:
    default_out_file = "calibration.csv"

# Output file path input
out_file_path = st.sidebar.text_input(
    "Output Calibration File",
    value=default_out_file,
    help="Where to save the resulting calibration file."
)

# --- SIDEBAR: FITTING PARAMETERS ---
st.sidebar.header("2. Fitting Parameters")

order = st.sidebar.number_input(
    "Polynomial Order (Intensity Calibration)", 
    min_value=1, max_value=10, value=4, step=1,
    on_change=clear_calibration_state
)

peak_height = st.sidebar.number_input(
    "Minimum Peak Height", 
    value=1.0, step=0.1,
    on_change=clear_calibration_state
)

min_time = st.sidebar.number_input(
    "Min Time Index", 
    min_value=0, value=1, step=1,
    on_change=clear_calibration_state
)

max_time = st.sidebar.number_input(
    "Max Time Index", 
    min_value=1, value=18, step=1,
    on_change=clear_calibration_state
)

# --- MAIN WORKFLOW: RUN CALIBRATION ---
if st.button("Run Calibration", type="primary"):
    if not file_path or not os.path.isfile(file_path):
        st.error("Please provide a valid, existing input file path before running.")
    else:
        with st.spinner("Processing data and generating fits..."):
            try:
                # ---------------------------------------------------------
                # BACKEND LOGIC
                # ---------------------------------------------------------
                # 1. Get step size from metadata
                with open(file_path, 'r') as f:
                    line = f.readline().strip()

                fields = line.split(',')
                time_step = float(fields[1].split(':')[1]) / 1000

                # 2. Import data
                AB = pd.read_csv(file_path, header=None, skiprows=1).T

                # Validate max_time
                actual_max = min(max_time, len(AB.columns))
                if actual_max != max_time:
                    st.warning(f"Max time index truncated to {actual_max} to match data columns.")

                valid_times =[]
                AB_peak_pixels = []
                AB_peak_heights =[]

                # Extract peaks
                for idx, col in enumerate(AB.columns.values[min_time:actual_max]):
                    peaks, _ = find_peaks(AB[col], height=peak_height)
                    if len(peaks) == 0:
                        st.warning(f"No peaks found in column '{col}' (time {idx * time_step:.3f} ps). Leaving blank.")
                        continue # Skip this column instead of stopping the whole script
                    
                    first_peak = peaks[0]
                    AB_peak_pixels.append(first_peak)
                    AB_peak_heights.append(AB[col][first_peak])
                    valid_times.append(idx * time_step)

                # Stop if NO peaks were found in ANY column
                if len(AB_peak_pixels) == 0:
                    st.error(f"No peaks were found in ANY of the columns using the height threshold {peak_height}. Please lower the 'Minimum Peak Height' parameter.")
                    st.stop()

                times = np.array(valid_times)

                # Normalize peak heights
                AB_peak_heights = np.array(AB_peak_heights) / np.max(AB_peak_heights)

                # 3. Intensity calibration (Polynomial Fit)
                coeffs = np.polyfit(AB_peak_pixels, AB_peak_heights, order)
                calibration_poly = np.poly1d(coeffs)
                pixels = np.arange(0, len(AB[0]), 1)
                intensity_calibration = calibration_poly(pixels)

                # 4. Time calibration (Linear Fit)
                line_func = lambda x, a, b: a * x + b
                pars, _ = curve_fit(line_func, times, np.array(AB_peak_pixels))
                
                time_pixel_cal = (1/pars[0] + 1/pars[0]) / 2
                calibrated_time = np.linspace(0, len(AB) * time_pixel_cal, len(AB))
                normalized_intensity = intensity_calibration / np.max(intensity_calibration)

                # Store the successful results in session_state
                st.session_state.calibration_data = {
                    'AB_peak_pixels': AB_peak_pixels,
                    'AB_peak_heights': AB_peak_heights,
                    'pixels': pixels,
                    'intensity_calibration': intensity_calibration,
                    'times': times,
                    'pars': pars,
                    'time_pixel_cal': time_pixel_cal,
                    'calibrated_time': calibrated_time,
                    'normalized_intensity': normalized_intensity
                }

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
                st.exception(e)

# --- DISPLAY PLOTS AND SAVE BUTTON ---
# This block executes if data was successfully processed and stored in the session state
if st.session_state.calibration_data is not None:
    data = st.session_state.calibration_data
    
    st.divider()
    st.subheader("Calibration Evaluation")
    
    col1, col2 = st.columns(2)

    # Plot 1: Intensity Calibration
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(data['AB_peak_pixels'], data['AB_peak_heights'], 'ok', label="Extracted Peaks")
    ax1.plot(data['pixels'], data['intensity_calibration'], '--k', label=f"Polynomial Fit (Order {order})")
    ax1.set_xlabel('Peak Pixel')
    ax1.set_ylabel('Normalized Peak E-Field')
    ax1.legend()
    ax1.set_title("Intensity Calibration")
    col1.pyplot(fig1)
    plt.close(fig1)

    # Plot 2: Time Calibration
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    line_func = lambda x, a, b: a * x + b
    ax2.plot(data['times'], line_func(data['times'], *data['pars']), '-k', label="Linear Fit")
    ax2.plot(data['times'], data['AB_peak_pixels'], 'ok', label="Extracted Peaks")
    
    text_str = f"Calibration factor:\n{data['time_pixel_cal']:.5f} ps/px"
    ax2.text(0.1, 0.8, text_str, transform=ax2.transAxes, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    
    ax2.set_xlabel('Time (ps)')
    ax2.set_ylabel('Peak Pixel')
    ax2.legend()
    ax2.set_title("Time Calibration")
    col2.pyplot(fig2)
    plt.close(fig2)

    st.info("Please review the plots above. If the fit represents the data accurately, save the calibration file.")
    
    # ---------------------------------------------------------
    # FILE EXPORT WORKFLOW
    # ---------------------------------------------------------
    if st.button("💾 Save Calibration", type="primary"):
        try:
            header_text = '1st row is calibrated time axis (in ps), 2nd is the pixel-dependence of the peak electric field.'
            
            # Save using data from session state
            np.savetxt(out_file_path, 
                       [data['calibrated_time'], data['normalized_intensity']], 
                       header=header_text)
            
            st.success(f"Calibration file successfully saved to:\n`{out_file_path}`")
            st.balloons()
            
        except Exception as e:
            st.error(f"Failed to save file: {e}")