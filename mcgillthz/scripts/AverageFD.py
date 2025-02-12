""" 
To run this code, copy it in the folder with all data files. Change the values below accordingly.
NOTE: calculate_std not implemented
"""
max_t_bg = 0.1          # Time range at which a background offset will be subtracted
posfix = '.d25'         # Posfix of the data files. Usually .d25, .d24, etc...
multiply_ch1 = 1/100    # I was reading in 0.1 V scale
multiply_ch2 = 1/100
ref_sign = '-'          # Signal used to obtain E_ref from E1 and E2. If '-', then E_ref = E1 - E2
fft_window  = 'hann'    # Window function to apply to the data.
power_of_2 = 10         # Power of 2 the data will be padded before FFT 

import os
import numpy as np                              # type: ignore
from scipy.fft import rfft, rfftfreq, irfft     # type: ignore
from scipy.signal.windows import get_window     # type: ignore
import time
from datetime import datetime

def import_files():
    filedir = os.getcwd()               # Location of .dat files
    filenames = get_average_list(filedir)  # List of average files to open

    open_files(filedir, filenames)


def import_files_to_average(prefix, time, n_averages):
    raw_list = []
    i = 0
    while len(raw_list) < n_averages:
        file = f"{prefix}{int(time) + i:{0}{4}}{posfix}"
        if os.path.exists(file):
            raw_list.append(np.genfromtxt(file).transpose())
        i += 1

        if i > 100:
            raise ValueError(f'File not found. Last tried was {file}')
    
    return raw_list


def get_average_list(filedir):
    """
    Returns a list with the name of the files in the specified directory
    that match the pattern '*Average.*'.
    """
    dirfiles = [f for f in os.listdir(filedir) if 'Average.' in f and os.path.isfile(os.path.join(filedir, f))]
    dirfiles.sort(key=lambda f: datetime.strptime(''.join(f.split(' ')[1:3]) ,'%b%d%H%M'))                          # Sort by date in the file name

    return dirfiles

def get_pump_time(filename):
    with open(filename) as open_file:
        first_line = open_file.readline()
    
    pump_time = first_line.split('Pump Probe Time: ')[1].split(' ps')[0]
    return float(pump_time)

def get_n_averages(filename):
    with open(filename) as open_file:
        first_line = open_file.readline()
    
    avgs = first_line.split('Averages: ')[1].split(' -')[0]
    return int(avgs)

def back_sub(data):
    """
    Subtracts the background from the data based on the average before the peak.

    Parameters:
    data (np.ndarray): A 2D array where:
                       - data[0] contains the time values.
                       - data[1] contains the field values.
                       - data[2] contains additional field values at another chopping frequency.
    max_t_bg (float): Maximum time (in the same units as data[0]) to consider for background subtraction.
                      All field values within this time range are averaged to calculate the background.

    Returns:
    np.ndarray: A 2D array where:
                - The first row is the time values (unchanged).
                - The second row is the field values with the background subtracted.
                - The third row contains the original field values in data[2].
    """
    mask = (data[0] < max_t_bg)
    bg = np.mean(data[1][mask])
    if np.abs(bg) > 0:  # Checks if bg is a number
        return np.array([data[0], data[1] - bg, data[2]])
    else:
        return np.array([data[0], data[1], data[2]])

def pad_to_power2(E, power):
    N0 = len(E)
    N_pad = int((2**power - N0) / 2)

    pad_data = np.append(np.zeros(N_pad), np.append(E, np.zeros(N_pad)))

    return pad_data

def center_and_pad(data):
    global power_of_2

    peak_ind = np.argmax(data[1])
    N_right = len(data[1]) - peak_ind
    N_pad = N_right - peak_ind

    # Pads to the left or right to make the window centered on the peak
    if N_pad > 0:
        new_E1 = np.append(np.zeros(N_pad), data[1])
        new_E2 = np.append(np.zeros(N_pad), data[2])   
    else:
        new_E1 = np.append(data[1], np.zeros(np.abs(N_pad)))
        new_E2 = np.append(data[2], np.zeros(np.abs(N_pad)))

    w = get_window(fft_window, len(new_E1), fftbins=False)
    
    if 2**power_of_2 < len(new_E1):
        power_of_2 = int(np.log2(len(new_E1))) + 1
    
    # Pads
    E1 = pad_to_power2(new_E1, power_of_2)
    E2 = pad_to_power2(new_E2, power_of_2)
    w = pad_to_power2(w, power_of_2)

    return np.array([E1, E2, w])
    


def fourier_transform(time, E1, E2, window=None):
        """
        Calculates the Fourier transform of time-domain data.

        Parameters:
        data (np.ndarray): A 2D array where:
                       - data[0] contains the time values.
                       - data[1] contains the field values.
                       - data[2] contains additional field values at another chopping frequency.
        Returns:
        np.ndarray: A 2D array where:
                - The first row is the frequency in THz.
                - The second row is the complex Fourier transform of data[1].
                - The third row is the complex Fourier transform of data[2].
        """
        if window is None:
            window = np.ones(len(E1))

        N = len(E1)
        dt = abs(time[1] - time[0])

        fft_freq = rfftfreq(N, dt)
        fft_a = rfft(E1 * window)
        fft_b = rfft(E2 * window)
        
        return np.array([fft_freq, fft_a, fft_b])




def averagefd_returntd(raw_list, ref_sign=ref_sign):
    """
    Imports, processes, and averages multiple time-domain data files in the Fourier domain.

    This function reads multiple time-domain data files with a specified prefix and time, processes them by 
    performing background subtraction, calculating the Fourier transform, averaging the FT, and returns the inverse FT. 

    Parameters:
    prefix (str): Prefix for the file names (e.g. "UBB Jan16 "). The files are expected to be sequentially numbered. 
    time (float): Starting time associated with the first file.
    n_averages (int): Number of files to average. The function stops once this many files have been successfully processed.
    posfix (str): File extension (e.g., '.d25') for the data files.
    max_t_bg (float, optional): Maximum time (in ps) for background subtraction. Any signal before this time is treated as background.
    ref_sign (str): Determines what signal is used to transform from E1 and E2 to E_pump and E_ref. Values are:
                    '-': E_ref = E1 - E2, E_pump = E1 + E2
                    '+': E_ref = E1 + E2, E_pump = E1 - E2
                    '0' or '': E_ref = E1, E_pump = E2
    multiply_ch1 (float): Constant factor to multiply data in channel 1.
    multiply_ch2 (float): Constant factor to multiply data in channel 2.

    Returns:
        E_a: Time domain value of the electric field.
        E_b: Time domain value of the additional electric field.
    """

    # Substracts background
    no_bg_list = [back_sub(raw_data) for raw_data in raw_list]

    # Calculates E_ref and E_pump from E1 and E2. If set to '0', only multiplies by the multiplicative factor.
    if ref_sign == '-':
        data_list = [np.array([d[0], d[1]*multiply_ch1-d[2]*multiply_ch2, d[1]*multiply_ch1+d[2]*multiply_ch2])   for d in no_bg_list  ]
    elif ref_sign == '+':
        data_list = [np.array([d[0], d[1]*multiply_ch1+d[2]*multiply_ch2, d[1]*multiply_ch1-d[2]*multiply_ch2])   for d in no_bg_list  ]
    elif ref_sign == '0' or ref_sign == '':
        data_list = [np.array([d[0], d[1]*multiply_ch1, d[2]*multiply_ch2])   for d in no_bg_list  ]
    else:
        raise ValueError("Invalid ref_sign. Use '-', '+' or '0'.")

    fft_list = np.array([ fourier_transform(d[0], d[1], d[2]) for d in data_list])     #Calculates the Fourier transform of 

    fft_a_avg = np.mean(fft_list[:,1], axis=0)
    fft_b_avg = np.mean(fft_list[:,2], axis=0)

    avg_a = irfft(fft_a_avg)
    avg_b = irfft(fft_b_avg)

    return data_list[0][0], avg_a, avg_b



def averagefd_returnfd(raw_list):
    """
    Imports, processes, and averages multiple time-domain data files in the Fourier domain.

    This function reads multiple time-domain data files with a specified prefix and time, processes them by 
    performing background subtraction, calculating the Fourier transform, averaging the FT, and returns the inverse FT. 

    Parameters:
    prefix (str): Prefix for the file names (e.g. "UBB Jan16 "). The files are expected to be sequentially numbered. 
    time (float): Starting time associated with the first file.
    n_averages (int): Number of files to average. The function stops once this many files have been successfully processed.
    posfix (str): File extension (e.g., '.d25') for the data files.
    max_t_bg (float, optional): Maximum time (in ps) for background subtraction. Any signal before this time is treated as background.
    ref_sign (str): Determines what signal is used to transform from E1 and E2 to E_pump and E_ref. Values are:
                    '-': E_ref = E1 - E2, E_pump = E1 + E2
                    '+': E_ref = E1 + E2, E_pump = E1 - E2
                    '0' or '': E_ref = E1, E_pump = E2
    multiply_ch1 (float): Constant factor to multiply data in channel 1.
    multiply_ch2 (float): Constant factor to multiply data in channel 2.

    Returns:
        E_a: Time domain value of the electric field.
        E_b: Time domain value of the additional electric field.
    """

    # Substracts background
    no_bg_list = [back_sub(raw_data) for raw_data in raw_list]

    # Calculates E_ref and E_pump from E1 and E2. If set to '0', only multiplies by the multiplicative factor.
    if ref_sign == '-':
        data_list = [np.array([d[0], d[1]*multiply_ch1-d[2]*multiply_ch2, d[1]*multiply_ch1+d[2]*multiply_ch2])   for d in no_bg_list  ]
    elif ref_sign == '+':
        data_list = [np.array([d[0], d[1]*multiply_ch1+d[2]*multiply_ch2, d[1]*multiply_ch1-d[2]*multiply_ch2])   for d in no_bg_list  ]
    elif ref_sign == '0' or ref_sign == '':
        data_list = [np.array([d[0], d[1]*multiply_ch1, d[2]*multiply_ch2])   for d in no_bg_list  ]
    else:
        raise ValueError("Invalid ref_sign. Use '-', '+' or '0'.")
    
    cnt_pad_list = [center_and_pad(d) for d in data_list]
    fft_list = np.array([ fourier_transform(data_list[0][0], d[0], d[1], window=d[2]) for d in cnt_pad_list])

    fft_freq = fft_list[0][0]                       # Calculate amplitude and angle before doing the statistics
    fft_a_avg = np.mean(fft_list[:,1], axis=0)
    fft_b_avg = np.mean(fft_list[:,2], axis=0)

    fft_a_std = np.std(fft_list[:,1], axis=0, ddof=1) / np.sqrt(len(fft_list))      # Standard error, instead of standard deviation
    fft_b_std = np.std(fft_list[:,2], axis=0, ddof=1) / np.sqrt(len(fft_list))

    return fft_freq, fft_a_avg, fft_b_avg, fft_a_std, fft_b_std

def open_files(filedir, filenames):
    """
    Opens files and creates E1 and E2 2D matrix and Pumpindex .dat files.
    """
    start_time = time.time()
    n_avgs = get_n_averages(os.path.join(filedir, filenames[0]))

    strip_name = filenames[0].split(' ')
    data = import_files_to_average(f'{strip_name[0]} {strip_name[1]} ', strip_name[2], n_avgs)
    time_axis, _, _ = averagefd_returntd(data)
    freq_axis, _, _, _, _ = averagefd_returnfd(data)
    
    m_td = len(time_axis)  # Number of rows
    m_fd = len(freq_axis)
    n = len(filenames)  # Number of columns/files

    pump_times = np.zeros(len(filenames)) 

    # Initialize time domain matrices: E1, E2 and ERef, EPump 
    E1_2D = np.zeros((m_td, 1 + n))
    E2_2D = np.zeros((m_td, 1 + n))
    ERef_2D = np.zeros((m_td, 1 + n))
    EPump_2D = np.zeros((m_td, 1 + n))
    
    # Initialize Fourier domain matrices: ERef, EPump, ERef_std and EPump_std
    ERef_fd_2D = np.zeros((m_fd, 1 + n), dtype=complex)
    EPump_fd_2D = np.zeros((m_fd, 1 + n), dtype=complex)
    ERef_std_fd_2D = np.zeros((m_fd, 1 + n), dtype=complex)
    EPump_std_fd_2D = np.zeros((m_fd, 1 + n), dtype=complex)

    # Copy time axis
    E1_2D[:, 0] = time_axis
    E2_2D[:, 0] = time_axis
    ERef_2D[:, 0] = time_axis
    EPump_2D[:, 0] = time_axis  
    
    # Copy frequency axis
    ERef_fd_2D[:,0] = freq_axis
    EPump_fd_2D[:,0] = freq_axis
    ERef_std_fd_2D[:,0] = freq_axis
    EPump_std_fd_2D[:,0] = freq_axis

    # Write Pumpindex.dat
    np.savetxt('Pumpindex.dat', np.arange(1, n + 1), fmt='%d', newline='\n')


    # Read remaining files and populate E1 and E2 matrices
    for i in range(0, n):
        pump_times[i] = get_pump_time(os.path.join(filedir, filenames[i]))
        
        print(f'Importing delay {pump_times[i]} ps, at {filenames[i]}')
        strip_name = filenames[i].split(' ')
        data = import_files_to_average(f'{strip_name[0]} {strip_name[1]} ', strip_name[2], n_avgs)

        _, E1_2D[:, i + 1], E2_2D[:, i + 1] = averagefd_returntd(data, ref_sign='0')
        _, ERef_2D[:, i + 1], EPump_2D[:, i + 1] = averagefd_returntd(data, ref_sign='-')
        _, ERef_fd_2D[:, i+1], EPump_fd_2D[:,i+1], ERef_std_fd_2D[:,i+1], EPump_std_fd_2D[:,i+1] =  averagefd_returnfd(data)


    # Write PumpTimes.dat if not already existing
    if not os.path.exists('PumpTimes.dat'):
        np.savetxt('PumpTimes.dat', pump_times, fmt='%.2f', newline='\n')
    
    # Write files
    np.savetxt('E1.dat', E1_2D, delimiter='\t', newline='\n', fmt='%-.7f')
    np.savetxt('E2.dat', E2_2D, delimiter='\t', newline='\n', fmt='%-.7f')
    np.savetxt('ERef.dat', ERef_2D, delimiter='\t', newline='\n', fmt='%-.7f')
    np.savetxt('EPump.dat', EPump_2D, delimiter='\t', newline='\n', fmt='%-.7f')
    np.savetxt('ERef_fd.dat', ERef_fd_2D, delimiter='\t', newline='\n', fmt='%-.7f')
    np.savetxt('EPump_fd.dat', EPump_fd_2D, delimiter='\t', newline='\n', fmt='%-.7f')
    np.savetxt('ERef_std_fd.dat', ERef_std_fd_2D, delimiter='\t', newline='\n', fmt='%-.7f')
    np.savetxt('EPump_std_fd.dat', EPump_std_fd_2D, delimiter='\t', newline='\n', fmt='%-.7f')
    

    print(f"Execution time: {time.time() - start_time:.2f} seconds")



if __name__ == "__main__":
    import_files()

    
