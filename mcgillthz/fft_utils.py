import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import get_window
import pywt

def pad_to_power2(data, power_of_2=14):
    """
    Pads the data array with zeros so the number of data points is a power of 2.

    Parameters:
    data (ndarray): Array of data to be padded.
    power_of_2 (int): Power of 2 to pad the data to.

    Returns:
    ndarray: Padded data array.
    """
    N0 = len(data)
    N_pad = int((2**power_of_2 - N0) / 2)

    pad_data = np.append(np.zeros(N_pad), np.append(data, np.zeros(N_pad)))

    return pad_data

def do_fft(data, window='hann', min_time=-np.inf, max_time=np.inf, pad_power2=14):
    """
    Performs FFT on the data array.

    Parameters:
    data (ndarray): 2D array with time data in data[0] and corresponding values in data[1].
    window (str): Window function to apply to the data.
    min_time (float): Minimum time for FFT.
    max_time (float): Maximum time for FFT.
    pad_power2 (int): Power of 2 to pad the data to.

    Returns:
    ndarray: 2D array with frequency, FFT amplitude, and FFT phase.
    """
    mask = (data[0] > min_time) & (data[0] < max_time)
    t = data[0][mask]
    E = data[1][mask]
    dt = abs(t[1] - t[0])

    if 2**pad_power2 > len(E):
        peak_ind = np.argmax(E)
        N_right = len(E) - peak_ind
        N_pad = N_right - peak_ind

        if N_pad > 0:
            new_E = np.append(np.zeros(np.abs(N_pad)), E)
        else:
            new_E = np.append(E, np.zeros(np.abs(N_pad)))

        w = get_window(window, len(new_E), fftbins=False)
        
        # Pads
        E = pad_to_power2(new_E, power_of_2=14)
        w = pad_to_power2(w, power_of_2=14)
    else:
        w = get_window(window, len(E), fftbins=False)

    N = len(E)

    fft_result = rfft(E * w)
    fft_freq = rfftfreq(N, dt)

    fft_amp = np.abs(fft_result)

    # fft_phase = -np.angle(fft_result)  # We unwrap and add the time delayed phase later. See Jepsen https://doi.org/10.1063/1.5047659 for details
    fft_phase = np.angle(fft_result)  # We unwrap and add the time delayed phase later. See Jepsen https://doi.org/10.1063/1.5047659 for details
    

    return np.array([fft_freq, fft_amp, fft_phase])

def rms_fft(fft_array, min_f):
    """
    Calculates the noise floor, i.e., the RMS value of the FFT amplitude above a certain frequency.

    Parameters:
    fft_array (ndarray): 2D array with frequency in fft_array[0] and FFT amplitude in fft_array[1].
    min_f (float): Minimum frequency to consider for RMS calculation.

    Returns:
    float: RMS value of the FFT amplitude above min_f.
    """
    mask = (fft_array[0] > min_f)
    rms = np.sqrt(np.mean(fft_array[1][mask]**2))
    return rms


def wavelet(data, wavelet, min_time=-np.inf, max_time=np.inf):
    """
    Performs a Continuous Wavelet Transform (CWT) on the given data.

    Parameters:
    data (ndarray): 2D array with time data in data[0] and field values in data[1].
    wavelet (str): Wavelet to use for the CWT (e.g., 'mexh', 'morl', etc.). Testing several is recommended.
                        See https://pywavelets.readthedocs.io/en/latest/ref/cwt.html#continuous-wavelet-families 
    min_time (float): Minimum time for the CWT. Default is -infinity.
    max_time (float): Maximum time for the CWT. Default is infinity.

    Returns:
    tuple: Contains the time array, frequency array, and the 2D matrix with the CWT.
    """
    # Mask the data to include only the specified time range
    mask = (data[0] > min_time) & (data[0] < max_time)
    time = data[0][mask]
    field = data[1][mask]

    # Define the widths for the wavelet transform, using log spacing
    widths = np.geomspace(1, 1024, num=100)
    # Alternatively, use a linear spacing
    # widths = np.linspace(1, 1024, num=100)
    
    # Calculate the sampling period
    sampling_period = np.diff(time).mean()

    # Perform the Continuous Wavelet Transform
    cwt, freqs = pywt.cwt(field, widths, wavelet, sampling_period=sampling_period)
    
    # Take the absolute value of the CWT to get the magnitude
    cwt = np.abs(cwt[:-1, :-1])

    return time, freqs, cwt