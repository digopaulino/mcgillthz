import numpy as np
import matplotlib.pyplot as plt

from .fft_utils import *
from .import_utils import *

# Define standard plotting parameters
params = {
    'font.family': 'DejaVu Sans',  
    'font.size': 18,

    'axes.linewidth': 2,
    'lines.linewidth': 2,
    'lines.markersize': 6,

    'xtick.direction': 'in',
    'xtick.minor.visible': True,
    'xtick.major.width': 2,
    'xtick.minor.width': 1.5,
    'xtick.major.size': 5,
    'xtick.minor.size': 3,
    'xtick.top': True,

    'ytick.direction': 'in',
    'ytick.minor.visible': True,
    'ytick.major.width': 2,
    'ytick.minor.width': 1.5,
    'ytick.major.size': 5,
    'ytick.minor.size': 3,
    'ytick.right': True,

    'legend.frameon': False,
    'legend.fontsize': 'small',

    'figure.dpi': 60,
    'figure.figsize': (8, 6),
    'figure.autolayout': True,
    'figure.facecolor': 'white',
    'animation.html': 'jshtml'
}
plt.rcParams.update(params)

def plot_spectrum(data, fft, axs=None, color='black', label=None, normalize=1, linestyle='-', marker='', alpha=1, dpi=80, dislocate0=False, max_freq=20, plot_err=False):
    """
    Plots the time domain and frequency domain spectrum of the data.

    Parameters:
    data (ndarray): 2D array with time data in data[0] and field values in data[1].
    fft (ndarray): 2D array with frequency in fft[0], FFT amplitude in fft[1], and optional error in fft[3].
    axs (array-like): Array of axes to plot on. If None, new axes are created.
    color (str): Color of the plot.
    label (str): Label for the plot.
    normalize (float or int): Normalization option. 1 leaves data as is, 0 normalizes to max, float divides by number.
    linestyle (str): Line style for the plot.
    marker (str): Marker style for the plot.
    alpha (float): Alpha value for the plot transparency.
    dpi (int): Dots per inch for the figure.
    dislocate0 (bool): If True, dislocate the time domaing data so that the peak of the data is at t=0.
    max_freq (float): Maximum frequency for the noise floor calculation.
    plot_err (bool): If True, plot error bands.

    Returns:
    array-like: Array of axes used for the plot.
    norm (float): If normalize=True, returns the used norm.
    """
    if axs is None:
        fig, axs = plt.subplots(ncols=2, figsize=(16, 6), dpi=dpi)

    if normalize == 0:
        norm_t = data[1].max()
        axs[0].set_ylabel('Normalized Electric Field (a.u.)')
    elif normalize == 1:
        norm_t = 1
        axs[0].set_ylabel('Electric Field (a.u.)')
    else:
        norm_t = normalize
        axs[0].set_ylabel('Normalized Electric Field (a.u.)')

    if dislocate0:
        peak_ind = np.argmax(np.abs(data[1]))
        disloc_time = data[0][peak_ind]
    else:
        disloc_time = 0

    axs[0].plot(data[0] - disloc_time, data[1]/norm_t, color=color, linestyle=linestyle, marker=marker, alpha=alpha, label=label)
    axs[0].axhline(0, linestyle='dashed', color='grey', linewidth=1)
    axs[0].set_xlabel('Time (ps)')

    axs[1].plot(fft[0], fft[1], color=color, label=label, linestyle=linestyle, marker=marker, alpha=alpha)
    noise_floor = rms_fft(fft, max_freq)
    axs[1].axhline(noise_floor, color=color, linestyle=linestyle, alpha=alpha/2)

    if plot_err and fft.shape[0] == 5:
        axs[1].fill_between(fft[0], (fft[1] - fft[3]), (fft[1] + fft[3]), color=color, alpha=alpha/5)
    
    if label is not None:
        axs[1].legend()

    axs[1].set_ylabel('FFT Amplitude (a.u.)')
    axs[1].set_xlabel('Frequency (THz)')
    axs[1].set_yscale('log')
    axs[1].set_ylim(bottom=(fft[1]).max()/1e4)
    axs[1].set_xlim(left=0)
    axs[1].grid()

    if normalize == 0:
        return axs, norm_t
    else:
        return axs
    


def plot_n_til(freqs, n_tils, ax=None, absorb=False):
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.plot(freqs, np.real(n_tils), color='red', linestyle='-', label=r'$n$')

    if absorb:
        ax2 = ax.twinx()
        wvl = c/(freqs*1e12)
        alphas = 4 * np.pi * np.imag(n_tils) / wvl / 100       # in cm-1

        ax2.plot(freqs, alphas, color='blue', linestyle='-')
        ax.plot([], [], color='blue', linestyle='-', label=r'$\alpha$')

        ax.set_ylabel(r'$n$')
        ax2.set_ylabel(r'$\alpha$ (cm$^{-1}$)')
    
    else:
        ax.plot(freqs, np.imag(n_tils), color='blue', linestyle='-', label=r'$\kappa$')
        ax.set_ylabel(r'$\tilde{n}$')

    ax2.axhline(0, color='darkblue', linewidth=1, linestyle='--')
    ax.axhline(1, color='grey', linewidth=1)

    ax.legend()

    ax.set_xlabel('Frequency (THz)')

    ax.set_xlim(left=0)

    return ax


def plot_eps(freqs, eps, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.plot(freqs, np.real(eps), color='red', linestyle='-', label=r'Re($\epsilon$)')

    ax.plot(freqs, np.imag(eps), color='blue', linestyle='-', label=r'Im($\epsilon$)')
    ax.set_ylabel(r'$\tilde{\epsilon}$')

    ax.legend()

    ax.set_xlabel('Frequency (THz)')

    ax.set_xlim(left=0)

    return ax


def plot_sigma(freqs, sigma, ax=None, norm=1e5):
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.plot(freqs, np.real(sigma)/norm, color='red', linestyle='-', label=r'Re($\sigma$)')

    ax.plot(freqs, np.imag(sigma)/norm, color='blue', linestyle='-', label=r'Im($\sigma$)')
    
    ax.set_ylabel(r'$\tilde{\sigma}$ '+rf'($10^{int(np.log10(norm))}$ S/m)')

    ax.legend()

    ax.set_xlabel('Frequency (THz)')

    ax.set_xlim(left=0)

    return ax

def plot_transmission(T, axs=None, delta_t=0, color='black', linestyle='-', label=None):
    if axs is None:
        fig, axs = plt.subplots(ncols=2, sharex=True, figsize=(14,6))
    
    if delta_t > 0:
        offset = 2*np.pi * delta_t * T[0]
        axs[1].set_ylabel(r'$\phi - \phi_{offset}$ (rad.)')
        axs[1].set_ylim(-6.28, 6.28)
    else:
        offset = np.zeros(len(T[0]))
        axs[1].set_ylabel(r'$\phi$ (rad.)')
    
    axs[0].plot(T[0], T[1], color=color, linestyle=linestyle)
    axs[1].plot(T[0], T[2] - offset, color=color, linestyle=linestyle, label=label)

    axs[0].set_xlabel('Frequency (THz)')
    axs[1].set_xlabel('Frequency (THz)')
    axs[0].set_ylabel('|T|')

    axs[0].set_xlim(0, 1.1*np.max(T[0]))
    axs[0].set_ylim(0, 1.1)
 

    if label is not None: axs[1].legend()

    return axs
