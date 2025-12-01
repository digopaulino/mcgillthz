# Generate calibration file for 2DTS setup

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Inputs here
folder = '/Users/rodrigo/Library/CloudStorage/OneDrive-McGillUniversity/1. Research/4. PbSnTe/2DTHz/Reflection/PbTe-Nov6'
file_name = folder+'/Calibration_AB.csv'

calibration_file = folder+'/calibration.csv'

order = 3      #Polynomial order to fit data. Change this to make the fit more reasonable
peak_height = 1
min_time = 0
max_time = -2

# Getting step size from metadata
with open(file_name, 'r') as f:
    line = f.readline().strip()

fields = line.split(',')    # Split by commas and extract values
time_step = float(fields[1].split(':')[1]) / 1000


# Importing data
AB = pd.read_csv(file_name, header=None, skiprows=1).T

times = np.arange(0, len(AB.columns[min_time:max_time])*time_step, time_step)

AB_peak_pixels = []
AB_peak_heights = []


for col in AB.columns.values[min_time:max_time]:
    AB_peak_pixels.append( find_peaks(AB[col], height=peak_height)[0][0]   )
    AB_peak_heights.append( AB[col][AB_peak_pixels[-1]]  )

AB_peak_heights = AB_peak_heights / np.max(AB_peak_heights)


####### Intensity calibration



coeffs = np.polyfit(AB_peak_pixels, AB_peak_heights, order)
calibration_poly = np.poly1d(coeffs)
pixels = np.arange(0, len(AB[0]), 1)
intensity_calibration = calibration_poly(pixels)


fig1, ax1 = plt.subplots()
ax1.plot(AB_peak_pixels, AB_peak_heights, 'ok')
ax1.plot(pixels, intensity_calibration, '--k')
ax1.set_xlabel('Peak pixel')
ax1.set_ylabel('Peak E field')


# Time calibration

fig2, ax2 = plt.subplots()



line = lambda x, a, b: a*x + b
pars, _ = curve_fit(line, times, np.array(AB_peak_pixels))
ax2.plot(times, line(times, *pars), '-k')
ax2.plot(times, AB_peak_pixels, 'ok')

time_pixel_cal = (1/pars[0] + 1/pars[0])/2
calibrated_time = np.linspace(0, len(AB)*time_pixel_cal , len(AB))
ax2.text(0.3, 0.2, f'Calibration factor: {time_pixel_cal:.5f} ps/px', transform=ax2.transAxes)

ax2.set_xlabel('Time (ps)')
ax2.set_ylabel('Peak pixel')

# Exporting data into calibration_file
header_text = '1st row is calibrated time axis (in ps), 2nd is the pixel-dependence of the peak electric field.'
np.savetxt(calibration_file, [calibrated_time, intensity_calibration/np.max(intensity_calibration)], header=header_text)

print(f'Calibration file saved to {calibration_file} .')

plt.show()