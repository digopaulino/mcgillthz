""" 
To run this code, copy it in the folder with all data files. Change the values below accordingly.
NOTE: calculate_std not implemented
"""
max_t_bg = 0.3          # Time range at which a background offset will be subtracted
fft_window  = ('tukey', 0.6) # Window function to apply to the data.
power_of_2 = 1         # Power of 2 the data will be padded before FFT 
posfix = '.d25'         # Posfix of the data files. Usually .d25, .d24, etc...
multiply_ch1 = 1/100    # I was reading in 0.1 V scale on the LIA
multiply_ch2 = 1/100    
ref_sign = '-'          # Signal used to obtain E_ref from E1 and E2. If '-', then E_ref = E1 - E2
subt_phase_offs = True  # If true, substracts 2pi phase offsets
subt_phase_freqs = [1.5, 3]   # Frequency region to perform fit for phase offset correction. Stay away from resonances.


import os
import numpy as np                              # type: ignore
from scipy.fft import rfft, rfftfreq, irfft     # type: ignore
from scipy.signal.windows import get_window     # type: ignore
from scipy.optimize import curve_fit            # type: ignore
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


def complex_std(a, x_values=None, ddof=1, axis=None):
    """
    Calculates the standard error for the real and imaginary parts of an array of complex values.
    If subt_phase_offs is True, it subtracts the 2pi offset before calculating.
    """
    global subt_phase_offs, subt_phase_freqs

    amp = np.abs(a)
    angle = np.unwrap(np.angle(a), axis=axis)

    if subt_phase_offs and (x_values is not None):
        mask = (x_values > subt_phase_freqs[0]) & (x_values < subt_phase_freqs[1])
        for i in range(len(a)):
            pars, _ = curve_fit(lambda x, a, b: a * x + b, x_values[mask], angle[i][mask])
            angle[i] = angle[i] - 2 * np.pi * round(pars[1] / (2 * np.pi))


    std_amp = np.std(amp, ddof=ddof, axis=axis)
    std_angle = np.std(angle, ddof=ddof, axis=axis)

    return std_amp * np.exp(-1j*std_angle)


def back_sub(data):
    """
    Subtracts the background from the data based on the average before the peak.
    """
    mask = (data[0] < max_t_bg)
    bg = np.mean(data[1][mask])
    if np.abs(bg) > 0:  # Checks if bg is a number
        return np.array([data[0], data[1] - bg, data[2]])
    else:
        return np.array([data[0], data[1], data[2]])

def get_asym_window(window, length, p_function=None):
    """
        Generate an asymmetric window using modulation of a symmetric window. Follows DOI: 10.1109/ICASSP.1991.150149.
        If the input window is not one of the pre-coded asymetric ones, it uses the scipy.signal.get_window() function.
    
        """
    def asymmetric_window(length, p_t):
        x = np.linspace(0, 1, length)
        modulated_x = p_t(x)  # Apply modulation function
        modulated_x = np.clip(modulated_x, 0, 1)  # Ensure values stay within [0,1]
        asymmetric_w = 0.5 * (1 - np.cos(2 * np.pi * modulated_x))  # Generate asymmetric hann window
        return asymmetric_w
    
    if window == 'hann-sin':        # Asymmetric window w/ peak at 0.33
        p = lambda t: np.sin(np.pi * t / 2)
        w = asymmetric_window(length, p)
    elif window == 'hann-log':      # Asymmetric window w/ peak at 0.42
        p = lambda t: np.log(t + 1)/np.log(2)
        w = asymmetric_window(length, p)
    else:
        if p_function is not None:
            w = asymmetric_window(length, p_function)
        else:
            w = get_window(window, length, fftbins=False)
    
    return w


def pad_to_power2(data, power_of_2=14):
    """
    Pads an array until the power of 2 determined.
    """
    N0 = len(data)
    N_pad = (2**power_of_2 - N0) 

    pad_data = np.append(data, np.zeros(N_pad))

    return pad_data

def center_and_pad(data):
    """
    Pads the data and the window.
    """
    global power_of_2, fft_window

    w = get_asym_window(fft_window, len(data[1]))
    
    if 2**power_of_2 < len(data[1]):
        power_of_2 = int(np.log2(len(data[1]))) + 1
    
    # Pads
    E1 = pad_to_power2(data[1], power_of_2)
    E2 = pad_to_power2(data[2], power_of_2)
    w = pad_to_power2(w, power_of_2)

    return np.array([E1, E2, w])
    


def fourier_transform(time, E1, E2, window=None):
        """
        Calculates the Fourier transform of time-domain data.
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
    Imports, processes, and averages multiple time-domain data files in the Fourier domain, and returns the averaged time-domain data.
    Processes them by performing background subtraction, calculating the Fourier transform, averaging the FT, and returns the inverse FT. 
    Does not return the standard deviation in this case.
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
    Imports, processes, and averages multiple time-domain data files in the Fourier domain. Returns the data in the Fourier domain with
    it's standard error as well.
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
    
    pad_list = [center_and_pad(d) for d in data_list]
    fft_list = np.array([ fourier_transform(data_list[0][0], d[0], d[1], window=d[2]) for d in pad_list])

    fft_freq = fft_list[0][0]                       # Calculate amplitude and angle before doing the statistics
    fft_a_avg = np.mean(fft_list[:,1], axis=0)
    fft_b_avg = np.mean(fft_list[:,2], axis=0)

    fft_a_std = complex_std(fft_list[:,1], axis=0, ddof=1, x_values=fft_freq) / np.sqrt(len(fft_list))      # Standard error, instead of standard deviation
    fft_b_std = complex_std(fft_list[:,2], axis=0, ddof=1, x_values=fft_freq) / np.sqrt(len(fft_list))

    return fft_freq, fft_a_avg, fft_b_avg, fft_a_std, fft_b_std

def open_files(filedir, filenames):
    """
    Opens all files and processes them. Creates the outputed files.
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

    
