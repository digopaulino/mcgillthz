import os
import numpy as np   # type: ignore

from .fft_utils import *
from .misc import *

def normalize_data(data, normalize):
    """
    Normalizes the data based on the specified normalization parameter.

    Parameters:
    data (ndarray): 2D array with time data in data[0] and corresponding values in data[1].
    normalize (float or int): Normalization option. 1 leaves data as is, 0 normalizes to max, float divides by number.

    Returns:
    ndarray: Normalized data array. data[0] is time, data[1] are the normalized field values
    """
    if normalize == 1:
        norm = 1
    elif normalize == 0:
        norm = data[1][np.argmax(np.abs(data[1]))]
    else:
        norm = normalize

    data[1] = data[1] / norm

    return data

def normalize_data_and_fft(data, normalize, fft_original, window='hann', min_time=-np.inf, max_time=np.inf, pad_power2=1):
    """
    Normalizes the data and performs FFT on the normalized data.

    Parameters:
    data (ndarray): 2D array with time data in data[0] and field values in data[1].
    normalize (float or int): Normalization option. 1 leaves data as is, 0 normalizes to max, float divides by number.
    fft_original (ndarray): Original FFT data array.
    window (str, optional): Window function for FFT. Different options include hann, boxcar, blackman, hamming, bohman, blackmanharris, etc.
                    See documentation of scipy.signal.windows.get_window function for extensive list.
    min_time (float, optional): Minimum time for FFT.
    max_time (float, optional): Maximum time for FFT.
    pad_power2 (int, optional): Power of 2 to pad the data length for FFT.

    Returns:
    tuple: Normalized data array and FFT array. 
                data: 2d array of time and field values.
                FFT: np.array with 5 columns - Frequencies, FFT Amp of normalized data, FFT Phase of normalized data, 
                    FFT Amp error divided by norm, FFT Phase error divided by norm
    """
    if normalize == 1:
        norm = 1
    elif normalize == 0:
        norm = data[1][np.argmax(np.abs(data[1]))]
    else:
        norm = normalize

    data[1] = data[1] / norm

    fft = do_fft(data, window=window, min_time=min_time, max_time=max_time, pad_power2=pad_power2)

    return data, np.array([fft[0], fft[1], fft[2], fft_original[3] / norm, fft_original[4] / norm])

def back_sub(data, max_t_bg=0.1):
    """
    Subtracts the background from the data based on the average before the peak.

    Parameters:
    data (np.ndarray): 2D array with time data in data[0] and field values in data[1].
    max_t_bg (float): Time before the peak to average for background subtraction.

    Returns:
    ndarray: 2D array with time data, background-subtracted values, and background values.
    """
    mask = (data[0] < max_t_bg)
    bg = np.mean(data[1][mask])

    return np.array([data[0], data[1] - bg, bg * np.ones(len(data[0]))])

def pad_td_right(data, n_points):
    """
    Pads the time-domain data with zeros on the right side.

    Parameters:
    data (ndarray): 2D array where the first row is time values and the second row is field values.
    n_points (int): Number of zeros to add to the right of the time-domain data.

    Returns:
    ndarray: 2D array with padded time values and field values.
    """
    dt = np.mean(np.diff(data[0]))
    tf = np.max(data[0])

    # new_time = np.append(np.linspace(-n_points*dt+ t0, t0-dt, n_points), data[0] )
    # new_field = np.append(np.zeros(n_points), data[1])
    new_time  = np.append(data[0], np.linspace(tf+dt, tf+n_points*dt, n_points) )
    new_field = np.append(data[1], np.zeros(n_points))

    return np.array([new_time, new_field])



def import_file(file, normalize=1, window='hann', start_pos=0, pad_power2=1, max_t_bg=0.1, max_time=np.inf, pad_td=0):
    """
    Imports and processes a single time-domain data file.

    Parameters:
    file (str): Path to the data file.
    normalize (float or int, optional): Normalization option. Default is 1.
                              1 leaves data as is, 0 normalizes to max, float divides by specified number.
    window (str, optional): Window function to use for FFT. Examples include "hann", "hamming", "nuttall".
                  See scipy.signal.windows documentation for details. Default is "hann".
    start_pos (float, optional): Initial delay stage position when the scan was started (in mm). Default is 0.
    pad_power2 (int, optional): Power of 2 for zero-padding in FFT, improving frequency resolution. 
                    Defaul is 1, where it pads until next power of 2.
    max_t_bg (float, optional): Maximum time (in ps) for background subtraction. All signal before this time will be considered background.
    max_time (float, optional): Maximum time (in ps) to consider in the time-domain data. Can be used to window data.
    pad_td (int): Number of zeros to add to the right of the time-domain data, useful when reference and sample data have different time ranges.
                    The padding is applied after windowing with "max_time".
    
    Returns:
    tuple: Processed time-domain data and its FFT as numpy arrays.
    """
    raw_data = np.genfromtxt(file).transpose()

    data = back_sub(raw_data, max_t_bg)

    data = normalize_data(data, normalize)

    time = data[0][data[0] < max_time]
    field = data[1][data[0] < max_time]

    start_time = 2 * np.abs(start_pos) * 1e-3 / c / 1e-12  # in ps

    delayed = np.array([time + start_time, field])

    data = pad_td_right(delayed, pad_td)

    fft = do_fft(delayed, window=window, pad_power2=pad_power2)

    return delayed, fft

def import_files(prefix, time, n_averages=1, posfix='.d24', normalize=1, window='hann', max_t_bg=0.1, start_pos=0, pad_power2=1, pad_td=0):
    """
    Imports and processes multiple time-domain data files, averages them, and provides error estimates.

    Parameters:
    prefix (str): Prefix for the data files.
    time (float): Time associated with the first file.
    n_averages (int): Number of files to average.
    posfix (str): File extension, e.g., '.d24'.
    normalize (float or int, optional): Normalization option. Default is 1.
                              1 leaves data as is, 0 normalizes to max, float divides by specified number.
    window (str, optional): Window function to use for FFT. Examples include "hann", "hamming", "nuttall".
                  See scipy.signal.windows documentation for details. Default is "hann".
    start_pos (float, optional): Initial delay stage position when the scan was started (in mm). Default is 0.
    pad_power2 (int, optional): Power of 2 for zero-padding in FFT, improving frequency resolution. 
                    Defaul is 1, where it pads until next power of 2.
    max_t_bg (float, optional): Maximum time (in ps) for background subtraction. All signal before this time will be considered background.
    pad_td (int): Number of zeros to add to the right of the time-domain data, useful when reference and sample data have different time ranges.
                    The padding is applied after windowing with "max_time".
    Returns:
    tuple: Averaged processed time-domain data and its FFT as numpy arrays.
    """
    raw_list = []
    i = 0
    while len(raw_list) < n_averages:
        file = f"{prefix}{int(time) + i}{posfix}"
        if os.path.exists(file):
            raw_list.append(np.genfromtxt(file).transpose())
            print(f'Time of files: {int(time) + i}')
        i += 1

    data_list = [back_sub(raw_data, max_t_bg) for raw_data in raw_list]

    data_list = [normalize_data(data, normalize) for data in data_list]

    start_time = 2 * np.abs(start_pos) * 1e-3 / c / 1e-12  # in ps

    delayed_list = [np.array([d[0] + start_time, d[1]]) for d in data_list]

    data_list = [ pad_td_right(d, pad_td) for d in delayed_list ]


    fft_list = [do_fft(d, window=window, pad_power2=pad_power2) for d in data_list]

    avg_data = avg_err_files(delayed_list)
    avg_fft = avg_err_fft_files(fft_list)

    return avg_data, avg_fft

def import_average_file(file, return_props=False, normalize=1, window='hann', start_pos=0, pad_power2=1, max_t_bg=0.1, max_time=np.inf, pad_td=0):
    """
    Imports and processes data from an averaged time-domain data file with options for normalization, padding, and background subtraction.

    Parameters:
    file (str): Path to the averaged data file.
    return_props (bool, optional): If True, returns properties from the file header such as lock-in unit, start position, 
                         number of data points, time resolution, and normalization used. Default is False.
    normalize (float or int, optional): Normalization option. Default is 1.
                              1 leaves data as is, 0 normalizes to max, float divides by specified number.
    window (str, optional): Window function to use for FFT. Examples include "hann", "hamming", "nuttall".
                  See scipy.signal.windows documentation for details. Default is "hann".
    start_pos (float, optional): Initial delay stage position when the scan was started (in mm). Default is 0.
    pad_power2 (int, optional): Power of 2 for zero-padding in FFT, improving frequency resolution. 
                    Defaul is 1, where it pads until next power of 2.
    max_t_bg (float, optional): Maximum time (in ps) for background subtraction. All signal before this time will be considered background.
    max_time (float, optional): Maximum time (in ps) to consider in the time-domain data. Can be used to window data.
    pad_td (int): Number of zeros to add to the right of the time-domain data, useful when reference and sample data have different time ranges.
                    The padding is applied after windowing with "max_time".

    Returns:
    tuple: If return_props is False, returns (data, fft).
           If return_props is True, returns (data, fft, properties).
           - data: 2D array with time and field values.
           - fft: 2D array with 5 columns: Frequencies, FFT Amplitude, FFT Phase, FFT Amplitude error, and FFT Phase error.
           - properties: Dictionary containing metadata from the file header.
    """
    data = np.genfromtxt(file, skip_header=1).transpose()
    data[0] = np.abs(data[0])
    data = back_sub(data, max_t_bg)
    data = normalize_data(data, normalize)

    time = data[0][data[0] < max_time]
    field = data[1][data[0] < max_time]

    start_time = 2 * np.abs(start_pos) * 1e-3 / c / 1e-12  # in ps

    data = np.array([time + start_time, field])
    data = pad_td_right(data, pad_td)

    fft = do_fft(data, window=window, pad_power2=pad_power2)

    if return_props:
        properties = {}
        with open(file) as open_file:
            first_line = open_file.readline()

        properties['Unit'] = first_line.split('Lock-In 1: ')[1].split('/')[0]
        properties['Start Position'] = first_line.split('THz Start: ')[1].split(', ')[0]
        properties['# of data points'] = first_line.split('mm, ')[1].split(', ')[0]
        properties['Time resolution'] = first_line.split('points, ')[1].split(' - ')[0]
        properties['Norm'] = normalize

        return data, fft, properties
    else:
        return data, fft

def average_files(data_list):
    """
    Averages multiple data arrays.

    Parameters:
    data_list (list of ndarray): List of data arrays to be averaged.

    Returns:
    ndarray: 2D array with averaged time and field values.
    """
    sum_array = np.zeros(len(data_list[0][1]))
    for d in data_list:
        sum_array += d[1]

    return np.array([data_list[0][0], sum_array / len(data_list)])

def average_fft_files(data_list):
    """
    Averages multiple FFT data arrays.

    Parameters:
    data_list (list of ndarray): List of FFT data arrays to be averaged.

    Returns:
    ndarray: 2D array with averaged time, amplitude, and phase.
    """
    sum_array_amp = np.zeros(len(data_list[0][1]))
    sum_array_phase = np.zeros(len(data_list[0][2]))
    for d in data_list:
        sum_array_amp += d[1]
        sum_array_phase += d[2]

    return np.array([data_list[0][0], sum_array_amp / len(data_list), sum_array_phase / len(data_list)])

def avg_err_files(data_list):
    """
    Calculates the average and standard error of multiple data arrays.

    Parameters:
    data_list (list of ndarray): List of data arrays to be averaged.

    Returns:
    ndarray: 2D array with time, average field values, and standard error.
    """
    N = len(data_list)
    average = average_files(data_list)

    sum_array = np.zeros(len(data_list[0][1]))
    for d in data_list:
        sum_array += (d[1] - average[1]) ** 2

    return np.array([data_list[0][0], average[1], np.sqrt(sum_array / (N * (N - 1)))])

def avg_err_fft_files(data_list):
    """
    Calculates the average and standard error of multiple FFT data arrays.

    Parameters:
    data_list (list of ndarray): List of FFT data arrays to be averaged.

    Returns:
    ndarray: 2D array with time, average FFT amplitude, average FFT phase, FFT amplitude error, and FFT phase error.
    """
    N = len(data_list)
    average = average_fft_files(data_list)

    sum_array_amp = np.zeros(len(data_list[0][1]))
    sum_array_phase = np.zeros(len(data_list[0][2]))
    for d in data_list:
        sum_array_amp += (d[1] - average[1]) ** 2
        sum_array_phase += (d[2] - average[2]) ** 2

    return np.array([data_list[0][0], average[1], average[2], np.sqrt(sum_array_amp / (N * (N - 1))), np.sqrt(sum_array_phase / (N * (N - 1)))])
