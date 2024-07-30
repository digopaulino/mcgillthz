from scipy.signal import hilbert    # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore

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
    fft = do_fft(data)       # type: ignore
    hilbert = do_hilbert(data)

    peak_freq = fft[0][np.argmax(fft[1])]
    peak_CEP_time = hilbert[0][np.argmax(hilbert[1])]
    peak_field_time = data[0][np.argmax(data[1])]

    dt = peak_field_time - peak_CEP_time
    phase = dt * 2 * np.pi * peak_freq  # See Reviews of Modern Physics, 81, 1, pp. 163-234, 2009-02-02
    return phase

def pdnp(df, time):
    """
    Converts a DataFrame to a numpy array with frequency and one column of time-domain data.

    Parameters:
    df (DataFrame): DataFrame containing the data.
    time (str): Column name for the time-domain data.

    Returns:
    ndarray: 2D array with frequency and time-domain data.
    """
    return np.array([df.iloc[:, 0], df[time]])



def residuals(x_exp, y_exp, x_fit, y_fit):
    """
    Computes residuals between experimental and fitted data.

    Parameters:
    x_exp (ndarray): Experimental x data.
    y_exp (ndarray): Experimental y data.
    x_fit (ndarray): Fitted x data.
    y_fit (ndarray): Fitted y data.

    Returns:
    ndarray: Residuals between interpolated experimental y data and fitted y data.
    """
    y_exp_interp = np.interp(x_fit, x_exp, y_exp)
    res = y_exp_interp - y_fit
    return res


def eps_from_sig(sig, freq, eps_inf=1.0):
    """
    Computes the complex permittivity from conductivity.

    Parameters:
    sig (ndarray): Complex conductivity.
    freq (ndarray): Frequency in THz.
    eps_inf (float or ndarray): High-frequency permittivity.

    Returns:
    ndarray: Complex permittivity.
    """
    w = 2 * np.pi * (freq * 1e12)
    return eps_inf + 1j * sig / (eps0 * w)

def n_from_sig(sig, freq, eps_inf=1.0, mu=1.0):
    """
    Computes the refractive index from conductivity.

    Parameters:
    sig (ndarray): Complex conductivity.
    freq (ndarray): Frequency in THz.
    eps_inf (float or ndarray): High-frequency permittivity.
    mu (float or ndarray): Magnetic permeability.

    Returns:
    ndarray: Complex refractive index.
    """
    w = 2 * np.pi * (freq * 1e12)
    return np.sqrt(mu * eps_inf + 1j * sig * mu / (eps0 * w))

def sig_from_n(n, freq, eps_inf=1.0, mu=1.0):
    """
    Computes the conductivity from the refractive index.

    Parameters:
    n (ndarray): Complex refractive index.
    freq (ndarray): Frequency in THz.
    eps_inf (float or ndarray): High-frequency permittivity.
    mu (float or ndarray): Magnetic permeability.

    Returns:
    ndarray: Complex conductivity.
    """
    w = 2 * np.pi * (freq * 1e12)
    return 1j * eps0 * w * (eps_inf - n**2 / mu)

def sig_from_eps(eps, freq, eps_inf=1.0):
    """
    Computes the conductivity from permittivity.

    Parameters:
    eps (ndarray): Complex permittivity.
    freq (ndarray): Frequency in THz.
    eps_inf (float or ndarray): High-frequency permittivity.

    Returns:
    ndarray: Complex conductivity.
    """
    w = 2 * np.pi * (freq * 1e12)
    return 1j * eps0 * w * (eps_inf - eps)

def eps_from_n(n, mu=1.0):
    """
    Computes the permittivity from the refractive index.

    Parameters:
    n (ndarray): Complex refractive index.
    mu (float or ndarray): Magnetic permeability.

    Returns:
    ndarray: Complex permittivity.
    """
    return n**2 / mu

def n_from_eps(eps, mu=1.0):
    """
    Computes the refractive index from permittivity.

    Parameters:
    eps (ndarray): Complex permittivity.
    mu (float or ndarray): Magnetic permeability.

    Returns:
    ndarray: Complex refractive index.
    """
    return np.sqrt(eps * mu)

def loss_from_eps(eps):
    """
    Computes the loss tangent from permittivity.

    Parameters:
    eps (ndarray): Complex permittivity.

    Returns:
    ndarray: Loss tangent.
    """
    return -np.imag(1 / eps)


def loss_from_n(n):
    """
    Computes the loss tangent from the refractive index.

    Parameters:
    n (ndarray): Complex refractive index.

    Returns:
    ndarray: Loss tangent.
    """
    eps = eps_from_n(n)
    return loss_from_eps(eps)



def sig_tinkham(amp, phase, d, n_sub, reflection=False):
    """
    Computes the conductivity from amplitude and phase using Tinkham's approximation.

    Parameters:
    amp (ndarray): Transmission amplitude.
    phase (ndarray): Transmission phase.
    d (float): Thickness of the material.
    n_sub (float or ndarray): Refractive index of the substrate.
    reflection (bool): If true, assumes the experiment was performed in reflection geometry. If falses, assumes it's in transmission.

    Returns:
    ndarray: Conductivity in S/m.
    """
    if reflection:
        R = amp * np.exp(1j * phase)
        return (1 - n_sub**2) * (1 - R) / (Z0 * d * ((1 + R) + n_sub * (1 - R)))
    else:
        return (n_sub + 1) / (Z0 * d) * (1 / (amp * np.exp(1j * phase)) - 1)


def n_from_sig_all(sig_df, eps_inf=1, mu=1, sig_eq=None):
    """
    Computes the refractive index from conductivity for all delays.

    Parameters:
    sig_df (DataFrame): DataFrame with frequency in the first column and conductivity (or change in conductivity) for each delay in subsequent columns.
    eps_inf (float): High-frequency permittivity. Default is 1.
    mu (float): Magnetic permeability. Default is 1.
    sig_eq (ndarray or None): Conductivity at equilibrium. If None, sig_df is assumed to be the total conductivity.

    Returns:
    DataFrame: DataFrame with frequency in the first column and refractive index for each delay in subsequent columns.
    """
    n_df = pd.DataFrame()
    
    if sig_eq is None:
        sig_eq = np.zeros(len(sig_df.iloc[:,0]))

    for i, time in enumerate(sig_df.columns[1:]):
        n_til = n_from_sig(sig_df[time] + sig_eq, sig_df.iloc[:,0], eps_inf=eps_inf, mu=mu)
        n_df.insert(i, time, n_til, True)
    
    n_df.insert(0, 'freq', sig_df.iloc[:,0], True)
    
    return n_df


def eps_from_sig_all(sig_df, eps_inf=1, sig_eq=None):
    """
    Computes the permittivity from conductivity for all delays.

    Parameters:
    sig_df (DataFrame): DataFrame with frequency in the first column and conductivity (or change in conductivity) for each delay in subsequent columns.
    eps_inf (float): High-frequency permittivity. Default is 1.
    sig_eq (ndarray or None): Conductivity at equilibrium. If None, sig_df is assumed to be the total conductivity.

    Returns:
    DataFrame: DataFrame with frequency in the first column and permittivity for each delay in subsequent columns. If sig_eq==None, this will be the differential permittivity.
    """
    eps_df = pd.DataFrame()
    
    if sig_eq is None:
        sig_eq = np.zeros(len(sig_df.iloc[:,0]))

    for i, time in enumerate(sig_df.columns[1:]):
        eps = eps_from_sig(sig_df[time] + sig_eq, sig_df.iloc[:,0], eps_inf=eps_inf)
        eps_df.insert(i, time, eps, True)
    
    eps_df.insert(0, 'freq', sig_df.iloc[:,0], True)
    
    return eps_df




