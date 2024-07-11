from scipy.signal import hilbert
import numpy as np

from .fft_utils import *

# Define physical constants
c = 2.99792e8 # m/s
eps0 = 8.85419e-12 # F/m
Z0 = 376.730 # Ohm

def do_hilbert(data):
    """
    Computes the Hilbert transform of the data to obtain its amplitude envelope.

    Parameters:
    data (ndarray): 2D array with time data in data[0] and field values in data[1].

    Returns:
    ndarray: 2D array with time and amplitude envelope of the field.
    """
    t = data[0]
    E = data[1]

    analytic_signal = hilbert(E)
    amplitude_envelope = np.abs(analytic_signal)
    return np.array([t, amplitude_envelope])

def find_CEP(data):
    """
    Finds the Carrier-Envelope Phase (CEP) of the data.

    Parameters:
    data (ndarray): 2D array with time data in data[0] and field values in data[1].

    Returns:
    float: CEP phase in radians.
    """
    fft = do_fft(data)
    hilbert = do_hilbert(data)

    peak_freq = fft[0][np.argmax(fft[1])]
    peak_CEP_time = hilbert[0][np.argmax(hilbert[1])]
    peak_field_time = data[0][np.argmax(data[1])]

    dt = peak_field_time - peak_CEP_time
    phase = dt * 2 * np.pi * peak_freq  # See Reviews of Modern Physics, 81, 1, pp. 163-234, 2009-02-02
    return phase


def n_from_sig(sig, freq):
    """
    Calculates the complex refractive index (n_tilde) from the complex conductivity.

    Parameters:
    sig (complex or ndarray): Complex conductivity (sigma) values.
    freq (float or ndarray): Frequency values in THz.

    Returns:
    complex or ndarray: Complex refractive index (n_tilde).
    """
    w = 2 * np.pi * freq * 1e12  # Angular frequency in rad/s

    sig1, sig2 = np.real(sig), np.imag(sig)

    d1 = np.sqrt((1 - sig2 / (eps0 * w))**2 + (sig1 / (eps0 * w))**2) / 2
    d2 = 1/2 - sig2 / (2 * w * eps0)

    return np.sqrt(d1 + d2) + 1j * np.sqrt(d1 - d2)   

def eps_from_n(n_til):
    """
    Calculates the complex permittivity (epsilon) from the complex refractive index.

    Parameters:
    n_til (complex or ndarray): Complex refractive index (n_tilde).

    Returns:
    complex or ndarray: Complex permittivity (epsilon).
    """
    n, kappa = np.real(n_til), np.imag(n_til)

    eps1 = n**2 - kappa**2
    eps2 = 2 * n * kappa
    return eps1 + 1j * eps2

def sig_from_n(n_til, freq):
    """
    Calculates the complex conductivity (sigma) from the complex refractive index.

    Parameters:
    n_til (complex or ndarray): Complex refractive index (n_tilde).
    freq (float or ndarray): Frequency values in THz.

    Returns:
    complex or ndarray: Complex conductivity (sigma).
    """
    n, kappa = np.real(n_til), np.imag(n_til)
    w = 2 * np.pi * (freq * 1e12)  # Angular frequency in rad/s

    sig1 = 2 * n * kappa * w * eps0
    sig2 = (1 - n**2 - kappa**2) * eps0 * w
    return sig1 + 1j * sig2
