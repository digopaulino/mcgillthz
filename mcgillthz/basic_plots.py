import numpy as np           # type: ignore
import matplotlib.pyplot as plt         # type: ignore
import matplotlib.cm as cm          # type: ignore
from matplotlib import cycler        # type: ignore
import ipywidgets as ipw            # type: ignore
from scipy.interpolate import interpn       # type: ignore

from .fft_utils import *
from .import_utils import *

# Define standard plotting parameters
params = {
    'font.family': 'DejaVu Sans',  
    'font.size': 18,

    'axes.linewidth': 2,
    'axes.prop_cycle': cycler('color', ['0C5DA5', '00B945', 'FF9500', 'FF2C00', '845B97', '474747', '9e9e9e']),
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

def plot_spectrum(data, fft, axs=None, color='black', label=None, normalize=1, linestyle='-', marker='', alpha=1, dpi=80, dislocate0=False, max_freq=np.inf, plot_err=False):
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
    if max_freq is not np.inf:
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
    

def plot_spectrum_slider(data, fft, axs=None, fig=None, slider=None, color='black', label=None, normalize=1, linestyle='-', marker='', alpha=1, dpi=80):
    if axs is None:
        fig, axs = plt.subplots(ncols=2, figsize=(14,6), dpi=dpi)
    
    taus = data.columns[1:].values
    if slider is None:
        slider = ipw.widgets.SelectionSlider(
            options=taus,
            value=taus[0],
            description='tau (ps)',
            disabled=False,
            readout=True
        )

    if normalize == 0:
        norm_t = data[1].max()
        axs[0].set_ylabel('Normalized Electric Field (a.u.)')
    elif normalize == 1:
        norm_t = 1
        axs[0].set_ylabel('Electric Field (a.u.)')
    else:
        norm_t = normalize
        axs[0].set_ylabel('Normalized Electric Field (a.u.)')

    line1, = axs[0].plot(data.iloc[:,0], data.iloc[:,1]/norm_t, color=color, linestyle=linestyle, marker=marker, alpha=alpha, label=label)
    axs[0].axhline(0, linestyle='dashed', color='grey', linewidth=1)
    axs[0].set_xlabel('Time (ps)')

    line2, = axs[1].plot(fft.iloc[:,0], fft.iloc[:,1], color=color, label=label, linestyle=linestyle, marker=marker, alpha=alpha)



    
    if label is not None:
        axs[1].legend()

    axs[1].set_ylabel('FFT Amplitude (a.u.)')
    axs[1].set_xlabel('Frequency (THz)')
    axs[1].set_yscale('log')
    axs[1].set_ylim(bottom=(fft.iloc[:,1:]).max().max()/1e4)
    axs[1].set_xlim(0, fft.iloc[:,0].max())
    axs[1].grid()

    max_1 = data.iloc[:,1:].max().max()/norm_t
    min_1 = data.iloc[:,1:].min().min()/norm_t
    axs[0].set_ylim(min_1, max_1)
    axs[0].set_xlim(0, data.iloc[:,0].max())
    def update(change):
        line1.set_ydata(data[change.new]/norm_t)
        line2.set_ydata(fft[change.new])
        
        # if not fix_y:
        #     autoscale_y(axs[0])
        #     autoscale_y(axs[1])
        
        fig.canvas.draw_idle()
    
    slider.observe(update, 'value')

    return slider, fig, axs


def plot_n_til(freqs, n_tils, ax=None, absorb=False, label=''):
    """
    Plots the refractive index (n) and the absorption coefficient (alpha) if absorb=True,
    otherwise plots the refractive index (n) and the extinction coefficient (kappa).

    Parameters:
    freqs (ndarray): Array of frequencies.
    n_tils (ndarray): Array of complex refractive index values.
    ax (matplotlib.axes.Axes): Axes to plot on. If None, a new axes is created.
    absorb (bool): If True, plots the absorption coefficient. Otherwise plots the extinction coefficient.

    Returns:
    matplotlib.axes.Axes: Axes used for the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.plot(freqs, np.real(n_tils), color='red', linestyle='-', label=label+r'$n$')

    if absorb:
        ax2 = ax.twinx()
        alphas = abs_coef_from_n(n_tils, freqs)

        ax2.plot(freqs, alphas, color='blue', linestyle='-')
        ax.plot([], [], color='blue', linestyle='-', label=label+r'$\alpha$')

        ax2.axhline(0, color='darkblue', linewidth=1, linestyle='--')

        ax.set_ylabel(r'$n$')
        ax2.set_ylabel(r'$\alpha$ (cm$^{-1}$)')
    
    else:
        ax.plot(freqs, np.imag(n_tils), color='blue', linestyle='-', label=label+r'$\kappa$')

        ax.axhline(0, color='darkblue', linewidth=1, linestyle='--')
        ax.set_ylabel(r'$\tilde{n}$')

    ax.axhline(1, color='grey', linewidth=1)

    ax.legend()

    ax.set_xlabel('Frequency (THz)')

    ax.set_xlim(left=0)

    return ax



def plot_eps(freqs, eps, ax=None, label=''):
    """
    Plots the real and imaginary parts of the permittivity.

    Parameters:
    freqs (ndarray): Array of frequencies.
    eps (ndarray): Array of complex permittivity values.
    ax (matplotlib.axes.Axes): Axes to plot on. If None, a new axes is created.

    Returns:
    matplotlib.axes.Axes: Axes used for the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.plot(freqs, np.real(eps), color='red', linestyle='-', label=label+r'Re($\epsilon$)')

    ax.plot(freqs, np.imag(eps), color='blue', linestyle='-', label=label+r'Im($\epsilon$)')
    ax.set_ylabel(r'$\tilde{\epsilon}$')

    ax.legend()

    ax.set_xlabel('Frequency (THz)')

    ax.set_xlim(left=0)

    return ax



def plot_sigma(freqs, sigma, ax=None, norm=1e5, linestyle='-', label=''):
    """
    Plots the real and imaginary parts of the conductivity, normalized by norm.

    Parameters:
    freqs (ndarray): Array of frequencies.
    sigma (ndarray): Array of complex conductivity values.
    ax (matplotlib.axes.Axes): Axes to plot on. If None, a new axes is created.
    norm (float): Normalization factor for the conductivity. Use multiples of 10.

    Returns:
    matplotlib.axes.Axes: Axes used for the plot.
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.plot(freqs, np.real(sigma)/norm, color='red', linestyle=linestyle, label=label+r'Re($\sigma$)')

    ax.plot(freqs, np.imag(sigma)/norm, color='blue', linestyle=linestyle, label=label+r'Im($\sigma$)')
    
    ax.set_ylabel(r'$\tilde{\sigma}$ '+rf'($10^{int(np.log10(norm))}$ S/m)')

    ax.legend()

    ax.set_xlabel('Frequency (THz)')

    ax.set_xlim(left=0)

    return ax


def plot_loss(freqs, loss, ax=None, color='black', label=''):
    if ax is None:
        fig, ax = plt.subplots()
    

    ax.plot(freqs, loss, color=color, label=label)


    ax.set_xlim(0, 1.1*np.max(freqs))
    ax.set_ylim(0, 1.1*np.max(loss))

    ax.set_ylabel(r'Im(-$\frac{1}{\epsilon(\omega)}$)')
    ax.set_xlabel('Frequency (THz)')

    return ax


def plot_transmission(T, axs=None, delta_t=0, color='black', linestyle='-', label=''):
    """
    Plots the transmission amplitude and phase.

    Parameters:
    T (ndarray): 2D array with frequency in T[0], amplitude in T[1], and phase in T[2].
    axs (array-like, optional): Array of axes to plot on. If None, new axes are created.
    delta_t (float, optioanl): Time offset for phase correction.
    color (str, optional): Color of the plot.
    linestyle (str, optional): Line style for the plot.
    label (str, optional): Label for the plot.

    Returns:
    array-like: Array of axes used for the plot.
    """
    if axs is None:
        fig, axs = plt.subplots(ncols=2, sharex=True, figsize=(14,6))
    
    
    if delta_t != 0:
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
    axs[0].set_ylabel(r'$|T(\omega)|$')

    axs[0].set_xlim(0, 1.1*np.max(T[0]))
    axs[0].set_ylim(0, 1.1)
 

    if label is not None: axs[1].legend()

    return axs


# From https://stackoverflow.com/questions/29461608/fixing-x-axis-scale-and-autoscale-y-axis
def autoscale_y(ax, margin=0.1):
    """
    Rescales the y-axis based on the data that is visible given the current xlim of the axis.
    
    Parameters:
    ax -- a matplotlib axes object
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims
    """
    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo, hi = ax.get_xlim()
        y_displayed = yd[((xd > lo) & (xd < hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed) - margin * h
        top = np.max(y_displayed) + margin * h
        return bot, top

    lines = ax.get_lines()
    bot, top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot: bot = new_bot
        if new_top > top: top = new_top

    ax.set_ylim(bot, top)

def autoscale_y_all(obj, margin=0.1, obj_is_figure=False):
    """
    Wrapper for autoscale_y() function, applying for all axes in obj.
    
    Parameters:
    obj (Figure or list of Axes) -- a list of matplotlib axes or a matplotlib figure
    margin (float) -- the fraction of the total height of the y-data to pad the upper and lower ylims
    obj_if_figure (bool) -- tag to be turned on when obj is a figure
    """
    if obj_is_figure:
        axes = obj.axes
    else:
        axes = obj
    axes = axes.flatten()

    for a in axes:
        autoscale_y(a, margin=margin)


def plot_transmission_trts(T_amp, T_phase, time, color='black', label='', axs=None, xlim=None, reflection=False, sub_one=False):
    """
    Plots transmission amplitude and phase for a given pump-probe time delay.

    Parameters:
    T_amp (DataFrame): DataFrame with frequency in the first column and amplitude in the subsequent columns
    T_phase (DataFrame): DataFrame with frequency in the first column and phase in the subsequent columns
    time (str): The time at which the data is to be plotted
    color (str): The color of the plot lines. Default is 'black'.
    label (str or None): The label for the plot lines. Default is None.
    axs (list or None): List of matplotlib axes objects. Default is None.
    xlim (tuple or None): The x-axis limits. Default is None.
    reflection (bool): Whether the plot is for reflection instead of transmission. Default is False.
    sub_one (bool): Whether to subtract one from the amplitude. Default is False.
    """
    if axs is None:
        fig, axs = plt.subplots(ncols=2, figsize=(14,6), sharex=True)

    sub = 0
    if sub_one: sub = 1
    
    T = [T_amp.iloc[:, 0], T_amp[time] - sub, T_phase[time]]

    axs = plot_transmission(T, axs=axs, color=color, label=label)

    if reflection:
        if not sub == 0:
            axs[0].set_ylabel(r'$|R|$ - ' + str(sub))
        else:
            axs[0].set_ylabel(r'$|R|$')
    else:
        if not sub == 0:
            axs[0].set_ylabel(r'$|T|$ - ' + str(sub))
        else:
            axs[0].set_ylabel(r'$|T|$')
    
    if xlim is not None:
        axs[0].set_xlim(xlim)
        autoscale_y(axs[0])
        autoscale_y(axs[1])
    else:
        axs[0].set_ylim(-1, 1)
        axs[1].set_ylim(-3.14, 3.14)
    
    return axs


def plot_transmission_slider(T_amp, T_phase, fig=None, axs=None, slider=None, color='black', fix_y=False, linestyle='-', 
                             xlim=None, reflection=False, sub_one=False, label=None):
    """
    Plots a slider to dynamically visualize 2D transmission amplitude and phase over time.

    Parameters:
    T_amp (DataFrame): DataFrame with frequency in the first column and amplitude in the subsequent columns
    T_phase (DataFrame): DataFrame with frequency in the first column and phase in the subsequent columns
    fig (Figure or None): The matplotlib figure object. Default is None.
    axs (list or None): List of matplotlib axes objects. Default is None.
    color (str): The color of the plot lines. Default is 'black'.
    fix_y (bool): Whether to fix the y-axis limits. Default is False.
    linestyle (str): The linestyle of the plot lines. Default is '-'.
    xlim (tuple or None): The x-axis limits. Default is None.
    reflection (bool): Whether the plot is for reflection instead of transmission. Default is False.
    sub_one (bool): Whether to subtract one from the amplitude. Default is False.
    """
    if axs is None:
        fig, axs = plt.subplots(ncols=2, figsize=(14,6), sharex=True)
    
    taus = T_amp.columns[1:].values
    if slider is None:
        slider = ipw.widgets.SelectionSlider(
            options=taus,
            value=taus[0],
            description='tau (ps)',
            disabled=False,
            readout=True
        )

    sub = 0
    if sub_one: sub = 1

    if reflection:
        if not sub == 0:
            axs[0].set_ylabel(r'$|R|$ - ' + str(sub))
        else:
            axs[0].set_ylabel(r'$|R|$')
    else:
        if not sub == 0:
            axs[0].set_ylabel(r'$|T|$ - ' + str(sub))
        else:
            axs[0].set_ylabel(r'$|T|$')

    line1, = axs[0].plot(T_amp.iloc[:, 0], T_amp.iloc[:, 1] - sub, color=color, linestyle=linestyle)
    line2, = axs[1].plot(T_phase.iloc[:, 0], T_phase.iloc[:, 1], color=color, linestyle=linestyle, label=label)

    if label is not None:
        axs[1].legend()

    if xlim is not None:
        axs[0].set_xlim(xlim)
        max_1 = T_amp[ (T_amp.iloc[:,0] >= xlim[0]) & (T_amp.iloc[:,0] <= xlim[1])].iloc[:,1:].max().max()
        max_2 = T_phase[ (T_phase.iloc[:,0] >= xlim[0]) & (T_phase.iloc[:,0] <= xlim[1])].iloc[:,1:].max().max()
        min_1 = T_amp[ (T_amp.iloc[:,0] >= xlim[0]) & (T_amp.iloc[:,0] <= xlim[1])].iloc[:,1:].min().min()
        min_2 = T_phase[ (T_phase.iloc[:,0] >= xlim[0]) & (T_phase.iloc[:,0] <= xlim[1])].iloc[:,1:].min().min()

        axs[0].set_ylim(min_1 - sub, max_1 - sub)
        axs[1].set_ylim(min_2, max_2)
    else:
        axs[0].set_ylim(-1, 1.5)
        axs[1].set_ylim(-3.14, 3.14)
    
    def update(change):
        line1.set_ydata(T_amp[change.new] - sub)
        line2.set_ydata(T_phase[change.new])
        
        if not fix_y:
            autoscale_y(axs[0])
            autoscale_y(axs[1])
        
        fig.canvas.draw_idle()
    
    slider.observe(update, 'value')

    return slider, fig, axs


def interpolate_map(T, N=5, max_tau=5):
    """
    Interpolates the data for a finer grid.

    Parameters:
    T (DataFrame): DataFrame with frequency in the first column and amplitude or phase in the subsequent columns
    N (int): Factor by which to increase the resolution. Default is 5.
    max_tau (float): Maximum pump-probe time delay in ps to consider. Default is 5 ps.

    Returns:
    tuple: Interpolated X, Y, and Z arrays
    """
    X = T.iloc[:, 0].values.astype(float)
    Y = T.columns[1:].values.astype(float)
    Z = T.iloc[:, 1:].values

    Z = Z[:, Y <= max_tau]
    Y = Y[Y <= max_tau]

    if len(X) < len(Y):
        x_new = np.linspace(min(X), max(X), len(X) * N)
        y_new = Y
    else:
        x_new = X
        y_new = np.linspace(min(Y), max(Y), len(Y) * N)

    xx, yy = np.meshgrid(x_new, y_new)

    z_new = interpn((X, Y), Z, (xx, yy))

    return x_new, y_new, z_new


def interpolate_color_mesh(T, ax, N=5, max_tau=5, vmin=None, vmax=None, cmap=cm.plasma, subtract_one=False, xmin=-np.inf, xmax=np.inf, norm=1, real_value=True):
    """
    Plots a color mesh of the interpolated data.

    Parameters:
    T (DataFrame): DataFrame with frequency in the first column and amplitude or phase in the subsequent columns
    ax (Axes): The matplotlib axes object
    N (int): Factor by which to increase the resolution. Default is 5.
    max_tau (float): Maximum tau value to consider. Default is 5.
    vmin (float or None): Minimum value for the color scale. Default is None.
    vmax (float or None): Maximum value for the color scale. Default is None.
    cmap (Colormap): The colormap to use. Default is cm.plasma.
    subtract_one (bool): Whether to subtract one from the values. Default is False.
    xmin (float): Minimum x-axis value. Default is -np.inf.
    xmax (float): Maximum x-axis value. Default is np.inf.
    norm (float): Normalization factor. Default is 1.
    real_value (bool): Whether to plot the real part of the data. Default is True.

    Returns:
    QuadMesh: The plotted QuadMesh object
    """
    X, Y, Z = interpolate_map(T, N=N, max_tau=max_tau)
    inds = (X > xmin) & (X < xmax)

    if real_value:
        func = np.real
    else:
        func = np.imag

    if subtract_one:
        Z = Z - 1
    
    im = ax.pcolormesh(X[inds], Y, func(Z[:, inds]) / norm, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    
    return im

def plot_transmission_colormesh(T_amp, T_phase, N=5, max_tau=5, subtract_one=False, fig=None, axs=None, reflection=False, xmin=-np.inf, xmax=np.inf, cmap=cm.plasma):
    """
    Creates a color plot for the transmission amplitude and phase.

    Parameters:
    T_amp (DataFrame): DataFrame with frequency in the first column and amplitude in the subsequent columns
    T_phase (DataFrame): DataFrame with frequency in the first column and phase in the subsequent columns
    N (int): Factor by which to increase the resolution. Default is 5.
    max_tau (float): Maximum tau value to consider. Default is 5.
    subtract_one (bool): Whether to subtract one from the amplitude. Default is False.
    fig (Figure or None): The matplotlib figure object. Default is None.
    axs (list or None): List of matplotlib axes objects. Default is None.
    reflection (bool): Whether the plot is for reflection instead of transmission. Default is False.
    xmin (float): Minimum x-axis value. Default is -np.inf.
    xmax (float): Maximum x-axis value. Default is np.inf.
    cmap (Colormap): The colormap to use. Default is cm.plasma.

    Returns:
    tuple: The plotted figure and axes
    """
    if fig is None or axs is None:
        fig, axs = plt.subplots(ncols=2, figsize=(14,6), sharex=True)

    if reflection:
        ylabel = 'R'
    else:
        ylabel = 'T'

    if subtract_one: ylabel += ' - 1'


    im1 = interpolate_color_mesh(T_amp, axs[0], N=N, max_tau=max_tau, subtract_one=subtract_one, xmin=xmin, xmax=xmax, cmap=cmap)
    im2 = interpolate_color_mesh(T_phase, axs[1], N=N, max_tau=max_tau, xmin=xmin, xmax=xmax, cmap=cmap)

    fig.colorbar(im1, ax=axs[0])
    fig.colorbar(im2, ax=axs[1])

    axs[0].set_xlabel(r'Frequency (THz)')
    axs[0].set_ylabel(r'Time Delay (ps)')
    axs[1].set_xlabel(r'Frequency (THz)')
    axs[1].set_ylabel(r'Time Delay (ps)')

    axs[0].set_title(fr'$|{ylabel}|$')
    axs[1].set_title(fr'$\angle{ylabel}$')

    return fig, axs


def plot_slider(data, fig=None, ax=None, color='black', fix_y=False, norm=1, add_array=None, linestyle='-', label=''):
    """
    Creates a plot with a slider to visualize data at different time delays (tau values).

    Parameters:
    data (DataFrame): DataFrame with frequency in the first column and data values in the subsequent columns.
    fig (Figure, optional): The matplotlib figure object. Default is None.
    ax (Axes, optional): The matplotlib axes object. Default is None.
    color (str, optional): Color of the plot line. Default is 'black'.
    fix_y (bool, optional): Whether to fix the y-axis limits or auto-scale. Default is False.
    norm (float, optional): Normalization factor for the data. Default is 1.
    add_array (array-like, optional): Array to be added to the data before plotting. Default is None.
    linestyle (str, optional): Line style for the plot. Default is '-'.
    label (str, optional): Label for the plot line. Default is None.

    Returns:
    slider (SelectionSlider): An interactive slider widget to control the displayed time delay (tau).
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots()
    
    # Initialize add_array if not provided
    if add_array is None:
        add_array = np.zeros(len(data.iloc[:,0]))
    
    # Extract tau values from the columns of the DataFrame
    taus = data.columns[1:].values
    slider = ipw.widgets.SelectionSlider(
        options=taus,
        value=taus[0],
        description='tau (ps)',
        disabled=False,
        readout=True
    )

    # Plot the initial data
    line, = ax.plot(data.iloc[:,0], (data[taus[0]] + add_array) / norm, color=color, linestyle=linestyle, label=label)

    # Sets y-axis to fit maximum and min value within xlim
    max_y = data.iloc[:,1:].max().max()
    min_y = data.iloc[:,1:].min().min()
    ax.set_ylim(min_y - 0.1*np.abs(min_y) , max_y + 0.1*np.abs(max_y))
    
    def update(change):
        line.set_ydata((data[change.new] + add_array) / norm)
        if not fix_y: 
            autoscale_y(ax)
        fig.canvas.draw_idle()
    
    # Attach the update function to the slider
    slider.observe(update, 'value')

    return slider


def plot_transmission_fit(exp_freq, exp_amp, exp_phase, fit_freq, fit_amp, fit_phase, xlim=None, subtract_one=False, reflection=False, delta_t=0):
    """
    Plots the experimental and fitted transmission amplitude and phase, along with their residuals.

    Parameters:
    exp_freq (ndarray): Array of frequency values (in THz) corresponding to the experimental data.
    exp_amp (ndarray): Array of experimental amplitude values.
    exp_phase (ndarray): Array of experimental phase values (in radians).
    fit_freq (ndarray): Array of frequency values (in THz) corresponding to the fitted data.
    fit_amp (ndarray): Array of fitted amplitude values.
    fit_phase (ndarray): Array of fitted phase values (in radians).
    xlim (tuple, optional): Tuple specifying the x-axis limits (min, max). Default is None, which uses the full range.
    subtract_one (bool, optional): If True, subtracts one from the amplitude before plotting. Default is False.
    reflection (bool, optional): If True, plots reflection instead of transmission. Default is False.
    delta_t (float, optioanl): Time offset for phase correction.

    Returns:
    tuple: A tuple containing the matplotlib figure and axes objects for further customization or display.
    """
    fig, axs = plt.subplots(2, 2, figsize=(14,7), sharex=True, gridspec_kw={'height_ratios': [1, 2.5], 'hspace': 0})
    
    if delta_t > 0:
        offset_exp = 2*np.pi * delta_t * exp_freq
        offset_fit = 2*np.pi * delta_t * fit_freq
        axs[1,1].set_ylabel(r'$\Delta\phi - \phi_{offset}$ (rad.)')
    else:
        offset_exp = np.zeros(len(exp_freq))
        offset_fit = np.zeros(len(fit_freq))
        axs[1,1].set_ylabel(r'$\Delta \phi$ (rad.)')

    if reflection:
        ylabel = 'R'
    else:
        ylabel = 'T'

    if subtract_one: 
        ylabel += ' - 1'
        sub = 1
    else:
        sub = 0
    axs[1, 0].set_ylabel(fr'$|{ylabel}|$')

    axs[1, 0].plot(exp_freq, exp_amp - sub, color='black', linestyle='-')
    axs[1, 1].plot(exp_freq, exp_phase - offset_exp, color='black', linestyle='-')
    axs[1, 0].plot(fit_freq, fit_amp - sub, color='red', linestyle='--')
    axs[1, 1].plot(fit_freq, fit_phase - offset_fit, color='red', linestyle='--')

    
    axs[1, 1].yaxis.set_label_position('right')
    axs[1, 0].set_xlabel(r'Frequency (THz)')
    axs[1, 1].set_xlabel(r'Frequency (THz)')
    
    if xlim is not None:
        axs[1, 0].set_xlim(xlim)
        autoscale_y(axs[1, 0])
        autoscale_y(axs[1, 1])
    else:
        axs[1, 0].set_ylim(-1, 1.5)
        axs[1, 1].set_ylim(-3.14, 3.14)

    res_amp = residuals(exp_freq, exp_amp, fit_freq, fit_amp)
    res_phase = residuals(exp_freq, exp_phase, fit_freq, fit_phase)
    rms = np.sqrt(np.sum(res_amp**2) + np.sum(res_phase**2))

    axs[0, 0].plot(fit_freq, res_amp, 'or')
    axs[0, 1].plot(fit_freq, res_phase, 'or', label=f'RMS={rms:.3e}')
    
    axs[0, 0].axhline(0, linestyle='--')
    axs[0, 1].axhline(0, linestyle='--')
    axs[0, 0].set_ylabel('Residuals')
    axs[0, 1].legend()
    
    return fig, axs




def plot_transmission_fit_slider(exp_amp, exp_phase, fit_amp, fit_phase, fix_y=False, xlim=None, subtract_one=False, reflection=False):
    """
    Plots the experimental and fitted transmission amplitude and phase with a slider for adjusting the time delay (tau).

    Parameters:
    exp_amp (DataFrame): DataFrame containing experimental amplitude data, with the first column as frequency and subsequent columns for different time delays (tau).
    exp_phase (DataFrame): DataFrame containing experimental phase data, structured similarly to exp_amp.
    fit_amp (DataFrame): DataFrame containing fitted amplitude data, structured similarly to exp_amp.
    fit_phase (DataFrame): DataFrame containing fitted phase data, structured similarly to exp_phase.
    fix_y (bool, optional): If True, keeps the y-axis limits fixed during slider adjustment. Default is False.
    xlim (tuple, optional): Tuple specifying the x-axis limits (min, max). Default is None, which uses the full range.
    subtract_one (bool, optional): If True, subtracts one from the amplitude before plotting. Default is False.
    reflection (bool, optional): If True, plots reflection instead of transmission. Default is False.

    Returns:
    tuple: A tuple containing the slider widget, the matplotlib figure, and the axes objects for further customization or display.
    """
    fig, axs = plt.subplots(2, 2, figsize=(14,7), sharex=True, gridspec_kw={'height_ratios': [1, 2.5], 'hspace': 0})
    
    taus = exp_amp.columns[1:].values
    slider = ipw.widgets.SelectionSlider(
        options=taus,
        value=taus[0],
        description='tau (ps)',
        disabled=False,
        readout=True
    )

    if reflection:
        ylabel = 'R'
    else:
        ylabel = 'T'

    if subtract_one: 
        ylabel += ' - 1'
        sub = 1
    else:
        sub = 0
    axs[1, 0].set_ylabel(fr'$|{ylabel}|$')


    line1, = axs[1, 0].plot(exp_amp.iloc[:,0], exp_amp.iloc[:,1] - sub, color='black', linestyle='-')
    line2, = axs[1, 1].plot(exp_phase.iloc[:,0], exp_phase.iloc[:,1], color='black', linestyle='-')
    line3, = axs[1, 0].plot(fit_amp.iloc[:,0], fit_amp.iloc[:,1] - sub, color='red', linestyle='--')
    line4, = axs[1, 1].plot(fit_phase.iloc[:,0], fit_phase.iloc[:,1], color='red', linestyle='--')

    res_amp = residuals(exp_amp.iloc[:,0], exp_amp.iloc[:,1], fit_amp.iloc[:,0], fit_amp.iloc[:,1])
    res_phase = residuals(exp_amp.iloc[:,0], exp_phase.iloc[:,1], fit_amp.iloc[:,0], fit_phase.iloc[:,1])

    res_line1, = axs[0, 0].plot(fit_amp.iloc[:,0], res_amp, 'or')
    res_line2, = axs[0, 1].plot(fit_phase.iloc[:,0], res_phase, 'or')


    if xlim is not None:
        axs[1, 0].set_xlim(xlim)
        max_1 = exp_amp[ (exp_amp.iloc[:,0] >= xlim[0]) & (exp_amp.iloc[:,0] <= xlim[1])].iloc[:,1:].max().max()
        max_2 = exp_phase[ (exp_phase.iloc[:,0] >= xlim[0]) & (exp_phase.iloc[:,0] <= xlim[1])].iloc[:,1:].max().max()
        min_1 = exp_amp[ (exp_amp.iloc[:,0] >= xlim[0]) & (exp_amp.iloc[:,0] <= xlim[1])].iloc[:,1:].min().min()
        min_2 = exp_phase[ (exp_phase.iloc[:,0] >= xlim[0]) & (exp_phase.iloc[:,0] <= xlim[1])].iloc[:,1:].min().min()
        
        axs[1,0].set_ylim(min_1-sub, max_1-sub)
        axs[1,1].set_ylim(min_2, max_2)
    else:
        axs[1, 0].set_ylim(-1, 1.5)
        axs[1, 1].set_ylim(-3.14, 3.14)
    
    
    def update(change):
        line1.set_ydata(exp_amp[change.new]-sub )
        line2.set_ydata(exp_phase[change.new] )
        line3.set_ydata(fit_amp[change.new]-sub )
        line4.set_ydata(fit_phase[change.new] )

        res_line1.set_ydata( residuals(exp_amp.iloc[:,0], exp_amp[change.new], fit_amp.iloc[:,0], fit_amp[change.new]))
        res_line2.set_ydata( residuals(exp_phase.iloc[:,0], exp_phase[change.new], fit_phase.iloc[:,0], fit_phase[change.new]))
        
        if not fix_y: 
            autoscale_y(axs[1, 0])
            autoscale_y(axs[1, 1])
            autoscale_y(axs[0, 0])
            autoscale_y(axs[0, 1])
        
        fig.canvas.draw_idle()
    
    slider.observe(update, 'value')

    return slider, fig, axs


def plot_multiple_slider(datas, colors, linestyles, labels, fig=None, ax=None, fix_y=False, norm=1, add_array=None):
    """
    Plots multiple datasets with a slider for adjusting the time delay (tau) and updates the plot interactively.

    Parameters:
    datas (list of DataFrames): A list of DataFrames, each containing frequency data in the first column and amplitude data for different time delays (tau) in subsequent columns.
    colors (list of str): A list of color specifications for each dataset's plot line.
    linestyles (list of str): A list of linestyle specifications for each dataset's plot line.
    labels (list of str): A list of labels for each dataset, used in the plot legend.
    fig (Figure or None, optional): The matplotlib figure object. If None, a new figure is created. Default is None.
    ax (Axes or None, optional): The matplotlib axes object. If None, new axes are created. Default is None.
    fix_y (bool, optional): If True, keeps the y-axis limits fixed during slider adjustment. Default is False.
    norm (float, optional): A normalization factor to apply to the y-data. Default is 1.
    add_array (numpy.ndarray or None, optional): An array of values to be added to each dataset before normalization. If None, a zero array is created. Default is None.

    Returns:
    slider: The slider widget that allows selection of the time delay (tau).
    """
    if ax is None:
        fig, ax = plt.subplodatas()
    
    if add_array is None:
        add_array = np.zeros(len(datas[0].iloc[:,0]))
    
    taus = datas[0].columns[1:].values
    slider = ipw.widgedatas.SelectionSlider(
        options=taus,
        value='0',
        description='tau (ps)',
        disabled=False,
        readout=True
    )

    lines = [[]] * len(datas)
    for i, T in enumerate(datas):
        lines[i], _ = ax.plot(T.iloc[:,0], (T['0']+ add_array)/norm, color=colors[i], linestyle=linestyles[i], label=labels[i])


    
    def update(change):
        for i, T in enumerate(datas):
            lines[i].set_ydata((T[change.new] + add_array)/norm)
            if not fix_y: autoscale_y(ax)
            fig.canvas.draw_idle()
    
    slider.observe(update, 'value')

    return slider
    
    
def plot_complex_slider(data_c, fig, ax, fix_y=False, norm=1, add_array=None, linestyle='-', \
                        label='', ylabel=r'$\tilde{\sigma}$ (S/m)'):
    """
    Plots the real and imaginary parts of complex data with a slider for adjusting the time delay (tau) 
    and updates the plot interactively.

    Parameters:
    data_c (DataFrame): A DataFrame containing frequency data in the first column and complex values 
                        (amplitude and phase) in subsequent columns corresponding to different time delays (tau).
    fig (Figure): The matplotlib figure object where the plot will be displayed.
    ax (Axes): The matplotlib axes object where the plot will be drawn.
    fix_y (bool, optional): If True, keeps the y-axis limits fixed during slider adjustment. Default is False.
    norm (float, optional): A normalization factor to apply to the y-data. Default is 1.
    add_array (numpy.ndarray or None, optional): An array of values to be added to the complex data 
                                                  before normalization. If None, a zero array is created. Default is None.
    linestyle (str, optional): The style of the plot lines (e.g., solid, dashed). Default is '-'.
    label (str or None, optional): A label for the plot line(s). Default is None.
    ylabel (str, optional): The label for the y-axis. Default is r'$\tilde{\sigma}$ (S/m)'.

    Returns:
    slider: The slider widget that allows selection of the time delay (tau).
    """
    if add_array is None:
        add_array = np.zeros(len(data_c.iloc[:,0]))
        
    taus = data_c.columns[1:].values
    slider = ipw.widgets.SelectionSlider(
        options=taus,
        value=taus[0],
        description='tau (ps)',
        disabled=False,
        readout=True
    )

    line1, = ax.plot(data_c.iloc[:,0], np.real(data_c[taus[0]] + add_array)/norm, color='red', label=label, linestyle=linestyle)
    line2, = ax.plot(data_c.iloc[:,0], np.imag(data_c[taus[0]] + add_array)/norm, color='blue', linestyle=linestyle)

    ax.set_ylabel(ylabel)
    ax.set_xlabel('Frequency (THz)')

    def update(change):
        line1.set_ydata(np.real(data_c[change.new] + add_array)/norm)
        line2.set_ydata(np.imag(data_c[change.new] + add_array)/norm)
        if not fix_y: autoscale_y(ax)
        fig.canvas.draw_idle()
    
    slider.observe(update, 'value')

    return slider


def plot_multiple_complex_slider(Ts, fig, ax, linestyles, labels, fix_y=False, norm=1, xlim=None, plot_tau=True):
    """
    Plots multiple sets of complex data (real and imaginary parts) with a slider for adjusting the time delay (tau)
    and updates the plot interactively.

    Parameters:
    Ts (list of DataFrames): A list of DataFrames where each DataFrame contains frequency data in the first column
                             and complex values in the subsequent columns corresponding to different time delays (tau).
    fig (Figure): The matplotlib figure object where the plots will be displayed.
    ax (Axes): The matplotlib axes object where the plots will be drawn.
    linestyles (list of str): A list of line styles for each data set (e.g., solid, dashed).
    labels (list of str): A list of labels for each data set.
    fix_y (bool, optional): If True, keeps the y-axis limits fixed during slider adjustment. Default is False.
    norm (float, optional): A normalization factor to apply to the y-data. Default is 1.
    xlim (tuple or None, optional): The x-axis limits to set for the plot. Default is None (auto limits).
    plot_tau (bool, optional): If True, a label indicating the current tau value will be plotted. Default is True.

    Returns:
    slider: The slider widget that allows selection of the time delay (tau).
    """
    taus = Ts[0].columns[1:].values
    slider = ipw.widgets.SelectionSlider(
        options=taus,
        value=taus[0],
        description='tau (ps)',
        disabled=False,
        readout=True
    )

    lines = [[]] * len(Ts)*2
    if plot_tau:
        l_label, = ax.plot([],[], label=rf'$\tau = 0 ps$')
    for i, T in enumerate(Ts):
        lines[i], = ax.plot(T.iloc[:,0], np.real(T.iloc[:,1])/norm, color='red', label=labels[i], linestyle=linestyles[i])
        lines[i + len(Ts)], = ax.plot(T.iloc[:,0], np.imag(T.iloc[:,1])/norm, color='blue', linestyle=linestyles[i])

    ax.set_xlabel('Frequency (THz)')


    ax.set_xlim(xlim)
    ax.legend()

    def update(change):
        for i, T in enumerate(Ts):   
            if plot_tau:
                l_label.set_label(rf'$\tau =$ {change.new:.1f} ps')
            lines[i].set_ydata(np.real(T[change.new])/norm)
            lines[i + len(Ts)].set_ydata(np.imag(T[change.new])/norm)
    
        if not fix_y: autoscale_y(ax)
        fig.canvas.draw_idle()
    
    slider.observe(update, 'value')

    return slider


