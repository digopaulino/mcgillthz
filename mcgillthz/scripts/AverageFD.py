""" 
To run this code, copy it in the folder with all data files. Change the values below accordingly.
NOTE: calculate_std not implemented
"""
max_t_bg = 0.1          # Time range at which a background offset will be subtracted
posfix = '.d25'         # Posfix of the data files. Usually .d25, .d24, etc...
multiply_ch1 = 1/100    # I was reading in 0.1 V scale
multiply_ch2 = 1/100
ref_sign = '-'          # Signal used to obtain E_ref from E1 and E2. If '-', then E_ref = E1 - E2
calculate_std = True



import os
import numpy as np                          # type: ignore
from scipy.fft import rfft, rfftfreq, irfft # type: ignore
import time
from datetime import datetime

def import_files():
    filedir = os.getcwd()               # Location of .dat files
    filenames = get_average_list(filedir)  # List of average files to open

    open_files(filedir, filenames)


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

def back_sub(data, max_t_bg=0.1):
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


def fourier_transform(data):
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
        N = len(data[0])
        dt = abs(data[0][1] - data[0][0])

        fft_freq = rfftfreq(N, dt)
        fft_a = rfft(data[1])
        fft_b = rfft(data[2])
        
        return np.array([fft_freq, fft_a, fft_b])


def inverse_fourier_transform(data):
        """
        Calculates the inverse Fourier transform of time-domain data.

        Parameters:
        data (np.ndarray): A 2D array where:
                       - data[0] contains the frequency values.
                       - data[1] contains the Fourier transform of the field values.
                       - data[2] contains the Fourier transform of additional field values at another chopping frequency.
        Returns:
        E_a: Time domain value of the electric field.
        E_b: Time domain value of the additional electric field.
        """
        E_a = irfft(data[1])
        E_b = irfft(data[2])
        
        return E_a, E_b


def import_n_average_fd(prefix, time, n_averages, posfix, max_t_bg=0.1, ref_sign='-', multiply_ch1=1, multiply_ch2=1):
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

    raw_list = []
    i = 0
    while len(raw_list) < n_averages:
        file = f"{prefix}{int(time) + i:{0}{4}}{posfix}"
        if os.path.exists(file):
            raw_list.append(np.genfromtxt(file).transpose())
            # print(f'File found: {file}')
        i += 1

        if i > 100:
            raise ValueError(f'File not found. Last tried was {file}')
    
    # Substracts background
    no_bg_list = [back_sub(raw_data, max_t_bg) for raw_data in raw_list]

    # Calculates E_ref and E_pump from E1 and E2. If set to '0', only multiplies by the multiplicative factor.
    if ref_sign == '-':
        data_list = [np.array([d[0], d[1]*multiply_ch1-d[2]*multiply_ch2, d[1]*multiply_ch1+d[2]*multiply_ch2])   for d in no_bg_list  ]
    elif ref_sign == '+':
        data_list = [np.array([d[0], d[1]*multiply_ch1+d[2]*multiply_ch2, d[1]*multiply_ch1-d[2]*multiply_ch2])   for d in no_bg_list  ]
    elif ref_sign == '0' or ref_sign == '':
        data_list = [np.array([d[0], d[1]*multiply_ch1, d[2]*multiply_ch2])   for d in no_bg_list  ]
    else:
        raise ValueError("Invalid ref_sign. Use '-', '+' or '0'.")

    fft_list = np.array([ fourier_transform(d) for d in data_list])     #Calculates the Fourier transform of 

    fft_freq = fft_list[0][0]
    fft_a_avg = np.mean(fft_list[:,1], axis=0)
    fft_b_avg = np.mean(fft_list[:,2], axis=0)

    avg_a, avg_b = inverse_fourier_transform([fft_freq, fft_a_avg, fft_b_avg])

    if calculate_std:
        fft_a_max = np.mean(fft_list[:,1], axis=0) + np.std(fft_list[:,1], axis=0)
        fft_a_min = np.mean(fft_list[:,1], axis=0) - np.std(fft_list[:,1], axis=0)

        fft_b_max = np.mean(fft_list[:,2], axis=0) + np.std(fft_list[:,2], axis=0)
        fft_b_min = np.mean(fft_list[:,2], axis=0) - np.std(fft_list[:,2], axis=0)

    return data_list[0][0], avg_a, avg_b


def open_files(filedir, filenames):
    """
    Opens files and creates E1 and E2 2D matrix and Pumpindex .dat files.
    """
    start_time = time.time()
    n_avgs = get_n_averages(os.path.join(filedir, filenames[0]))

    strip_name = filenames[0].split(' ')
    time_axis, _, _ = import_n_average_fd(f'{strip_name[0]} {strip_name[1]} ', strip_name[2], n_avgs, posfix, max_t_bg=max_t_bg, 
                                    ref_sign='-', multiply_ch1=multiply_ch1, multiply_ch2=multiply_ch2)
    
    m = len(time_axis)  # Number of rows
    n = len(filenames)  # Number of columns/files

    pump_times = np.zeros(len(filenames)) 

    # Initialize E1, E2 and ERef, EPump matrices
    E1_2D = np.zeros((m, 1 + n))
    E2_2D = np.zeros((m, 1 + n))
    ERef_2D = np.zeros((m, 1 + n))
    EPump_2D = np.zeros((m, 1 + n))
    if calculate_std:
        E1_std_2D = np.zeros((m, 1 + n))
        E2_std_2D = np.zeros((m, 1 + n))
        ERef_std_2D = np.zeros((m, 1 + n))
        EPump_std_2D = np.zeros((m, 1 + n))        

    # Copy time axis
    E1_2D[:, 0] = time_axis  
    E2_2D[:, 0] = time_axis  
    ERef_2D[:, 0] = time_axis
    EPump_2D[:, 0] = time_axis
    if calculate_std:
        E1_std_2D[:,0] = time_axis 
        E2_std_2D[:,0] = time_axis 
        ERef_std_2D[:,0] = time_axis 
        EPump_std_2D[:,0] = time_axis      

    # Write Pumpindex.dat
    np.savetxt('Pumpindex.dat', np.arange(1, n + 1), fmt='%d', newline='\n')


    # Read remaining files and populate E1 and E2 matrices
    for i in range(0, n):
        pump_times[i] = get_pump_time(os.path.join(filedir, filenames[i]))

        print(f'Importing delay {pump_times[i]} ps, at {filenames[i]}')
        strip_name = filenames[i].split(' ')
        _, E1_2D[:, i + 1], E2_2D[:, i + 1] = import_n_average_fd(f'{strip_name[0]} {strip_name[1]} ', strip_name[2], n_avgs, posfix, max_t_bg=max_t_bg, 
                                    ref_sign='0', multiply_ch1=multiply_ch1, multiply_ch2=multiply_ch2)

        _, ERef_2D[:, i + 1], EPump_2D[:, i + 1] = import_n_average_fd(f'{strip_name[0]} {strip_name[1]} ', strip_name[2], n_avgs, posfix, max_t_bg=max_t_bg, 
                                    ref_sign='-', multiply_ch1=multiply_ch1, multiply_ch2=multiply_ch2)


    # Write PumpTimes.dat if not already existing
    if not os.path.exists('PumpTimes.dat'):
        np.savetxt('PumpTimes.dat', pump_times, fmt='%.2f', newline='\n')
    
    # Write E1 and E2 to files
    np.savetxt('E1.dat', E1_2D, delimiter='\t', newline='\n', fmt='%-.7f')
    np.savetxt('E2.dat', E2_2D, delimiter='\t', newline='\n', fmt='%-.7f')
    np.savetxt('ERef.dat', ERef_2D, delimiter='\t', newline='\n', fmt='%-.7f')
    np.savetxt('EPump.dat', EPump_2D, delimiter='\t', newline='\n', fmt='%-.7f')

    print(f"Execution time: {time.time() - start_time:.2f} seconds")



if __name__ == "__main__":
    import_files()

    
