import numpy as np      # type: ignore
import pandas as pd     # type:ignore
from scipy.optimize import minimize         # type: ignore
from scipy.interpolate import PchipInterpolator  # type: ignore
from tqdm import tqdm           # type: ignore
from numpy.fft import fft2, fftshift, fftfreq, ifftshift   # type: ignore
from scipy.interpolate import RectBivariateSpline   # type: ignore

from .fft_utils import *
from .import_utils import *
from .basic_plots import *

def get_2d_metadata(filename):
    with open(filename, 'r') as f:
        line = f.readline().strip()

    # Split by commas and extract values
    fields = line.split(',')

    # Map to desired keys
    metadata = {
        "scans": int(fields[0].split(':')[1]),
        "step": float(fields[1].split(':')[1]) / 1000,  # convert fs → ps
        "averaging_time": float(fields[2].split(':')[1]),
        "time_cal": float(fields[3].split(':')[1]),
        "stage1": float(fields[4].split(':')[1]),
        "stage2": float(fields[5].split(':')[1]),
        "notes": fields[6].split(':', 1)[1].strip()  # Keep entire notes string
    }

    return metadata



def subtract_leak_NL(df, tau_min, tau_max, t_min, t_max):
    """
    Removes a delay-dependent leakage signal (e.g., pump scatter or EOS artifacts) 
    by averaging the signal in a shifted time coordinate frame.

    This method isolates features that move linearly with the delay (tau) by:
    1. Shifting the data for each delay step so that the feature becomes stationary 
    (transforming from laboratory time to relative time).
    2. Averaging the signal within a specified window (defined by tau_min/max and t_min/max) 
    to compute the mean leakage profile.
    3. Shifting the calculated mean profile back to the original coordinate system.
    4. Subtracting this reconstructed background from the original data.

    Parameters
    ----------
    tau_min : float
        The minimum excitation delay (tau) to include in the averaging window.
    tau_max : float
        The maximum excitation delay (tau) to include in the averaging window.
    t_min : float
        The lower bound of the time window in the *shifted* coordinate system. 
        The averaging window includes times where t > t_min - tau.
    t_max : float
        Defines the upper bound of the time window in the *shifted* coordinate system. 
        The averaging window includes times where t > t_max - tau.
        
        Note: The effective window in the shifted frame is [-t_max, t_min].
        In the laboratory frame, this corresponds to a window of [(-t_max - tau), (t_min - tau)].

    Returns
    -------
    no_bg_data : ndarray
        The 2D spectroscopy data with the calculated leakage background subtracted. 
        Shape matches the original data input.
    """

    ts = df['time'].values
    taus = df.columns[1:].astype(float)
    data = df.drop(columns='time').values

    t_step = ts[1] - ts[0]
    shifted_data = np.zeros_like(data)

    for j, tau in enumerate(taus):
        shift_indices = int(round(tau / t_step))
        column_data = data[:, j]
        
        if shift_indices > 0:
            # Shift down (forward in time)
            shifted_data[:, j] = np.roll(column_data, shift_indices)
            shifted_data[:shift_indices, j] = 0 # Fill the gap with zeros
        elif shift_indices < 0:
            # Shift up (backward in time)
            shifted_data[:, j] = np.roll(column_data, shift_indices)
            shifted_data[shift_indices:, j] = 0 # Fill the gap with zeros
        else:
            shifted_data[:, j] = column_data

    tau_region = (taus > tau_min) & (taus < tau_max)
    t_region = (ts > t_min) & (ts < t_max)
    mask_2d = t_region[:, np.newaxis] & tau_region[np.newaxis, :]


    leak_filter = shifted_data.copy()
    leak_filter[~mask_2d] = 0
    avg_leak = np.mean(leak_filter[:, tau_region], axis=1)  # Makes array that has only average leak signal (t-shifted)

    shifted_background = np.zeros_like(data)
    for j, tau in enumerate(taus):
        shift_indices = int(round(-tau / t_step))
        column_background = np.roll(avg_leak, shift_indices)

        if shift_indices > 0:
            column_background[:shift_indices] = 0 
        elif shift_indices < 0:
            column_background[shift_indices:] = 0
            
        shifted_background[:, j] = column_background

    no_bg_data = data - shifted_background
    
    df_out = pd.DataFrame(no_bg_data, columns=df.columns[1:])
    df_out.insert(0, 'time', ts)
    return df_out



def subtract_still_leak_NL(df, tau_min, tau_max, t_min, t_max):
    """
    Removes a leakage signal in the pulse that's still (e.g., pump scatter or EOS artifacts) 
    by averaging the signal.

    This method isolates features that move linearly with the delay (tau) by:
    1. Averaging the signal within a specified window (defined by tau_min/max and t_min/max) 
    to compute the mean leakage profile.
    2. Subtracting this reconstructed background from the original data.

    Parameters
    ----------
    tau_min : float
        The minimum excitation delay (tau) to include in the averaging window.
    tau_max : float
        The maximum excitation delay (tau) to include in the averaging window.
    t_min : float
        The lower bound of the time window in the *shifted* coordinate system. 
        The averaging window includes times where t > t_min - tau.
    t_max : float
        Defines the upper bound of the time window in the *shifted* coordinate system. 
        The averaging window includes times where t > t_max - tau.
        
        Note: The effective window in the shifted frame is [-t_max, t_min].
        In the laboratory frame, this corresponds to a window of [(-t_max - tau), (t_min - tau)].

    Returns
    -------
    no_bg_data : ndarray
        The 2D spectroscopy data with the calculated leakage background subtracted. 
        Shape matches the original data input.
    """

    ts = df['time'].values
    taus = df.columns[1:].astype(float)
    data = df.drop(columns='time').values

    tau_region = (taus > tau_min) & (taus < tau_max)
    t_region = (ts > t_min) & (ts < t_max)
    mask_2d = t_region[:, np.newaxis] & tau_region[np.newaxis, :]


    leak_filter = data.copy()
    leak_filter[~mask_2d] = 0
    avg_leak = np.mean(leak_filter[:, tau_region], axis=1)  # Makes array that has only average leak signal 

    leaked_background = np.zeros_like(data)
    for j in range(len(taus)):
        leaked_background[:, j] = avg_leak

    no_bg_data = data - leaked_background
    
    df_out = pd.DataFrame(no_bg_data, columns=df.columns[1:])
    df_out.insert(0, 'time', ts)
    
    return df_out






def cosine_taper_window(M, flat_start, flat_end, rise_start=0, fall_end=None):
    """
    Generates a 1D window with a flat center, cosine-tapered edges, and 
    optional zero-padding (dead zones) at the start and end.

    Shape:
    [0...0] -> [Rise] -> [Flat (1.0)] -> [Fall] -> [0...0]

    Parameters:
    - M (int): Total length of the window.
    - flat_start (int): Index where the Rise ends and Flat region begins.
    - flat_end (int): Index where the Flat region ends and Fall begins.
    - rise_start (int, optional): Index where the Rise begins. 
                                  Indices 0 to rise_start will be 0. 
                                  Default is 0.
    - fall_end (int, optional): Index where the Fall ends. 
                                Indices from fall_end to M will be 0. 
                                Default is M.

    Returns:
    - window (ndarray): The constructed window of length M.
    """
    window = np.zeros(M)
    if fall_end is None:
        fall_end = M

    # Ensure specified values don't exceed bounds if the window is wider than the data
    rise_start = max(0, rise_start)
    fall_end = min(M, fall_end)
    flat_start = max(0, min(M, flat_start))
    flat_end = max(0, min(M, flat_end))

    window[flat_start:flat_end] = 1.0   # Flat region

    # --- 2. Rise Region (Left) ---
    rise_len = flat_start - rise_start
    if rise_len > 0:
        theta = np.linspace(0, np.pi, rise_len, endpoint=False) 
        window[rise_start:flat_start] = 0.5 * (1 - np.cos(theta))

    # --- 3. Fall Region (Right) ---
    fall_len = fall_end - flat_end
    if fall_len > 0:
        theta = np.linspace(0, np.pi, fall_len, endpoint=False)
        window[flat_end:fall_end] = 0.5 * (1 + np.cos(theta))

    return window





def window_2d(df, t_window=('tukey', 0.3), flat_tau_start=0, flat_tau_end=1, min_tau=-np.inf, max_tau=np.inf):
    """
    Applies a 2D window to spectroscopy data using a standard window for the probe axis
    and a variable-support cosine taper for the delay axis.

    This function generates a 2D mask by computing the outer product of two 1D windows:
    1. Probe Time (t): Uses a standard window function (e.g., Tukey) via scipy.signal.
    2. Excitation Delay (tau): Uses a custom asymmetric cosine window defined by 
       physical time coordinates.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing the 2D dataset. 
        - Column 0: Probe time ('time').
        - Columns 1..N: Excitation delays (tau).
    t_window : str, float, or tuple, optional
        Window type for the probe time axis passed to `scipy.signal.get_window`.
        Default is ('tukey', 0.3).
    flat_tau_start : float
        The delay time (tau) where the window reaches full amplitude (1.0).
        Defines the end of the rising edge.
    flat_tau_end : float
        The delay time (tau) where the window begins to decay from full amplitude.
        Defines the start of the falling edge.
    min_tau : float, optional
        The delay time (tau) where the window starts rising from 0.
        Data corresponding to tau < min_tau will be zeroed out. 
        Default is -inf (uses the first index).
    max_tau : float, optional
        The delay time (tau) where the window finishes decaying to 0.
        Data corresponding to tau > max_tau will be zeroed out.
        Default is inf (uses the last index).

    Returns
    -------
    windowed_data : numpy.ndarray
        The 2D array of data (excluding the time column) multiplied by the 
        generated 2D window. Shape corresponds to the values in `df`.
    
    Notes
    -----
    The function converts the provided physical time constants (min_tau, flat_tau_start, etc.) 
    into array indices using nearest-neighbor search (`np.argmin`).
    """
    ts = df['time'].values
    taus = df.columns[1:].astype(float)
    data = df.drop(columns='time').values

    x_window = get_window(t_window, len(ts))

    # Gets indices for the specified window times
    min_tau_ind = 0 if np.isinf(min_tau) else np.argmin(np.abs(taus - min_tau))
    max_tau_ind = len(taus) if np.isinf(max_tau) else np.argmin(np.abs(taus - max_tau))
    flat_tau_start_ind = np.argmin(np.abs(taus - flat_tau_start))
    flat_tau_end_ind = np.argmin(np.abs(taus - flat_tau_end))

    y_window = cosine_taper_window(len(taus), flat_tau_start_ind, flat_tau_end_ind, min_tau_ind, max_tau_ind)

    window_2d = x_window[:, np.newaxis] * y_window[np.newaxis, :] 
    windowed_data = data * window_2d

    df_out = pd.DataFrame(windowed_data, columns=df.columns[1:])
    df_out.insert(0, 'time', ts)
    return df_out





def compute_2d_fft(df, window=('tukey', 0.1), t_min=-np.inf, t_max=np.inf, 
                   subtract_baseline=False, t0_index=None, tau0_index=None, Nt=None, Ntau=None):
    """
    Computes the 2D FFT of time-resolved spectroscopy data with optional zero-padding 
    and phase correction.

    Parameters:
    - df : pandas DataFrame
        Input data with shape [N_delay, N_time+1]. The first column must be 'time'. 
        The remaining columns represent delay steps (tau).
    - window : str, tuple, or float, optional
        Window type passed to scipy.signal.get_window() (e.g., 'hann', ('tukey', 0.3)). 
        This window is only applied to t axis. Windowing beforehand is preferable
        Default is ('tukey', 0.3).
    - t_min : float, optional
        Minimum probe time (t) to include in the analysis.
    - t_max : float, optional
        Maximum probe time (t) to include in the analysis.
    - subtract_baseline : bool, optional
        If True, subtracts the mean of the data before processing.
    - t0_index : int, optional
        Index of t=0 in the *cropped* time axis. If provided, the data is zero-padded 
        such that this index becomes the geometric center of the array (index N/2). 
        This allows `ifftshift` to correctly align the phase origin to [0,0].
    - tau0_index : int, optional
        Index of tau=0 in the *cropped* delay axis. Similar behavior to t0_index 
        for the delay axis.
    - Nt : int, optional
        Target size for the probe time axis (e.g., 1024, 2048). If provided, the 
        data is padded to this length. If `t0_index` is also provided, the padding 
        is distributed to keep t0 centered. Must be larger than the data length.
        If None, no padding is applied.
    - Ntau : int, optional
        Target size for the delay axis. Similar behavior to Nt.

    Returns:
    - ft_data : ndarray (complex)
        2D FFT result, shifted such that the zero-frequency component is at the center.
    - freq_t : ndarray
        Frequency axis for probe time.
    - freq_tau : ndarray
        Frequency axis for delay time.
    """
    # --- Data Extraction ---
    time = df['time'].values
    mask = (time > t_min) & (time < t_max)
    t = time[mask]

    tau = df.columns[1:].astype(float)

    data = df.drop(columns='time').iloc[mask, :].values.T 

    if subtract_baseline:
        data = data - data.mean()

    # --- Windowing ---
    if window is not None:
        win_func = get_window(window, len(t)).reshape(1, -1)
        windowed_data = data * win_func
    else:
        windowed_data = data

    # --- Padding Calculations ---
    n_tau, n_t = windowed_data.shape

    def get_centered_padding(current_len, zero_idx, target_len):
        # Minimal length required to make zero_idx the center
        min_symmetric_len = 2 * max(zero_idx, current_len - zero_idx)
        # Use target_len if it fits the symmetry requirement, otherwise expand
        final_len = max(target_len, min_symmetric_len) if target_len else min_symmetric_len
        
        # Calculate pads to place zero_idx at final_len // 2
        pad_left = (final_len // 2) - zero_idx
        pad_right = final_len - current_len - pad_left
        return (pad_left, pad_right), final_len

    # Axis 0 (Tau) Padding
    if tau0_index is not None:
        (p_tau_l, p_tau_r), final_Ntau = get_centered_padding(n_tau, tau0_index, Ntau)
    else:
        final_Ntau = Ntau if (Ntau and Ntau > n_tau) else n_tau
        p_tau_l, p_tau_r = 0, final_Ntau - n_tau

    # Axis 1 (Time) Padding
    if t0_index is not None:
        (p_t_l, p_t_r), final_Nt = get_centered_padding(n_t, t0_index, Nt)
    else:
        final_Nt = Nt if (Nt and Nt > n_t) else n_t
        p_t_l, p_t_r = 0, final_Nt - n_t

    padded_data = np.pad(windowed_data, ((p_tau_l, p_tau_r), (p_t_l, p_t_r)), mode='constant')

    # --- FFT ---
    if (t0_index is not None) and (tau0_index is not None):
        ft_data = fftshift(fft2(ifftshift(padded_data)))
    else:
        ft_data = fftshift(fft2(padded_data))

    # --- Frequency Axes ---
    dt = t[1] - t[0]
    d_tau = tau[1] - tau[0]
    
    freq_t = fftshift(fftfreq(final_Nt, d=dt))
    freq_tau = fftshift(fftfreq(final_Ntau, d=d_tau))

    return ft_data, freq_t, freq_tau








class THzExp:
    def __init__(self, name, calibration_file=None, shift_times=True):
        """
        Initialize a 2D THz Experiment instance.
        
        Parameters
        ----------
        name : str
            Base name of the experiment files (without channel suffix).
        calibration_file : str, optional
            Path to a calibration file (if any).
        """
        self.name = name
        self.calibration_file = calibration_file
        
        # Load all four dataframes
        self.NL, self.AB, self.A, self.B = self._get_all_channels()

        # Definying t0 and tau0 by looking at maximum AB signal
        self.t0_index, self.tau0_index = np.unravel_index(np.argmax(np.abs(self.AB.drop(columns='time').values)), self.AB.drop(columns='time').values.shape)
        if shift_times:
            self._shift_times()

        # FFT-related properties (to be calculated later)
        self.ft_data = None
        self.freq_t = None
        self.freq_tau = None

        self.summary()




    def _get_2d_df(self, channel):
        """Load and process a single channel."""
        self.metadata = get_2d_metadata(f'{self.name}_{channel}.csv')

        df = pd.read_csv(f'{self.name}_{channel}.csv', header=None, skiprows=1).T

        if self.calibration_file is not None:
            calibration = np.genfromtxt(self.calibration_file, skip_header=1)
            df = df.divide(calibration[1], axis=0)
            df.insert(0, 'time', calibration[0])
        else:
            df.insert(0, 'time', np.arange(0, len(df[0])) * self.metadata['time_cal'])
        
        if self.metadata['step'] == 0:
            df.columns = ['time', '0.0']
        else:
            step = np.abs(self.metadata['step'])
            df.columns = ['time'] + [round(x, 2) for x in np.arange(0, round(step * (df.shape[1]-1), 1), step)]

        return df




    def _get_all_channels(self):
        """Load all channels (AB, A, B) and compute the nonlinear response (NL)."""
        AB = self._get_2d_df('AB')
        A = self._get_2d_df('A')
        B = self._get_2d_df('B')

        NL = AB - A - B
        NL['time'] = A['time']
        return NL, AB, A, B
    
    def _shift_times(self):
        """Considers the maximum AB position, and shifts t and tau so that is zeroed"""
        # Getting new arrays
        ts = self.AB['time'].values - self.AB['time'].values[self.t0_index]
        taus = self.AB.columns[1:].astype(float) - self.AB.columns[1:].astype(float)[self.tau0_index]

        # Relabel datasets
        self.NL['time'] = ts
        self.AB['time'] = ts
        self.A['time'] = ts
        self.B['time'] = ts
        
        self.NL.columns = ['time'] + list(np.round(taus, 2))
        self.AB.columns = ['time'] + list(np.round(taus, 2))
        self.A.columns = ['time'] + list(np.round(taus, 2))
        self.B.columns = ['time'] + list(np.round(taus, 2))




    def do_fft(self, data=None, window=('tukey', 0.3), t_min=-np.inf, t_max=np.inf, 
                    subtract_baseline=False, t0_index=None, tau0_index=None, Nt=None, Ntau=None):
        """
        Compute and store 2D FFT results for a given dataframe (default: NL).

        Parameters
        ----------
        data : pandas.DataFrame, optional
            Data to transform (defaults to NL)
        window : tuple
            Window type and parameters for scipy.signal.get_window
        t_min, t_max : float
            Probe time range to include
        subtract_baseline : bool, optional
            If True, subtracts the mean of the data before processing.
        t0_index : int, optional
            Index of t=0 in the *cropped* time axis. If provided, the data is zero-padded 
            such that this index becomes the geometric center of the array (index N/2). 
            This allows `ifftshift` to correctly align the phase origin to [0,0].
        tau0_index : int, optional
            Index of tau=0 in the *cropped* delay axis. Similar behavior to t0_index 
            for the delay axis.
        Nt : int, optional
            Target size for the probe time axis (e.g., 1024, 2048). If provided, the 
            data is padded to this length. If `t0_index` is also provided, the padding 
            is distributed to keep t0 centered. Must be larger than the data length.
        Ntau : int, optional
            Target size for the delay axis. Similar behavior to Nt.
        """
        if data is None:
            df = self.NL
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            df = pd.DataFrame(data, columns=self.NL.columns[1:], index = self.NL.index)
            df.insert(0, 'time', self.NL['time'].values)
        
        if (t0_index is None) and (tau0_index is None):
            t0_index = self.t0_index
            tau0_index = self.tau0_index

        self.ft_data, self.freq_t, self.freq_tau = compute_2d_fft(df, window=window, t_min=t_min, t_max=t_max,
                                                                    subtract_baseline=subtract_baseline, t0_index=t0_index, tau0_index=tau0_index,
                                                                    Nt=Nt, Ntau=Ntau)

        print("2D FFT computed and stored in class properties.")







    def summary(self):
        """Print basic info about the experiment."""
        print(f"Experiment: {self.name}")
        print(f"Notes: {self.metadata['notes']}")
        print(f"Channels: AB ({self.AB.shape}), A ({self.A.shape}), B ({self.B.shape}), NL ({self.NL.shape})")
        if self.calibration_file:
            print(f"Calibrated with: {self.calibration_file}")




    # -----------------------------
    # Plotting methods
    # -----------------------------
    def plot_2x2_grid(self, suptitle='', min_t=None, max_t=None, vmax=None):
        """Plot AB, NL, A, B dataframes in a 2x2 grid."""
        dfs = [self.AB, self.NL, self.A, self.B]
        titles = ['AB', 'Nonlinear', 'A', 'B']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        t = dfs[0]['time'].values
        tau = dfs[0].columns[1:].astype(float)

        for i, (df, ax) in enumerate(zip(dfs, axes)):
            data = df.drop(columns='time').values.T
            if (min_t is not None) and (max_t is not None):
                time_mask = (t > min_t) & (t < max_t)
            else:
                time_mask = (t > -np.inf)
            
            vmax_abs = np.max(np.abs(data[:, time_mask]))

            if i == 1 and vmax is not None:
                im = ax.imshow(
                    data, aspect='auto',
                    extent=[t.min(), t.max(), tau.min(), tau.max()],
                    origin='lower', cmap='bwr',
                    vmin=-vmax, vmax=vmax
                )
            else:
                im = ax.imshow(
                    data, aspect='auto',
                    extent=[t.min(), t.max(), tau.min(), tau.max()],
                    origin='lower', cmap='bwr',
                    vmin=-vmax_abs, vmax=vmax_abs
                )

            ax.set_xlim(min_t, max_t)
            ax.set_title(titles[i])
            ax.set_xlabel('THz time t (ps)')
            ax.set_ylabel('Excitation delay τ (ps)')
            fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)

        plt.suptitle(suptitle or f"{self.name} – Time-domain signals", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig, axes




    def plot_2d_fft(self, vmax=30, vmin=0, title='2D FFT', cmap='magma', log_scale=False, Nt=None, Ntau=None, value_to_plot='abs'):
        """Plot stored 2D FFT result."""
        if self.ft_data is None:
            raise ValueError("No FFT data found. Run add_fft() first.")

        fig, ax = plt.subplots(figsize=(8, 6))

        freq_t = self.freq_t
        freq_tau = self.freq_tau
        if value_to_plot == 'real':
            ft_data = np.real(self.ft_data)
        elif value_to_plot == 'imag': 
            ft_data = np.imag(self.ft_data) 
        else:
            ft_data = np.abs(self.ft_data)

        if (Nt is not None) and (Ntau is not None):
            interp = RectBivariateSpline(freq_tau, freq_t, ft_data)
            freq_t_fine = np.linspace(freq_t.min(), freq_t.max(), Nt)
            freq_tau_fine = np.linspace(freq_tau.min(), freq_tau.max(), Ntau)
            Z_fine = interp(freq_tau_fine, freq_t_fine)
            T, Tau = np.meshgrid(freq_t_fine, freq_tau_fine)
            ft_data = Z_fine
        else:
            T, Tau = np.meshgrid(freq_t, freq_tau)

        if log_scale:
            pcm = ax.pcolormesh(T, Tau, np.log10(ft_data), shading='auto',
                                cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(pcm, ax=ax, label='Log(FFT Amplitude)')
        else:
            pcm = ax.pcolormesh(T, Tau, ft_data, shading='auto',
                                cmap=cmap, vmin=vmin, vmax=vmax)
            fig.colorbar(pcm, ax=ax, label='FFT Amplitude')

        ax.set_xlim(0, 3)
        ax.set_xlabel(r'Probe frequency $\nu_t$ (THz)')
        ax.set_ylabel(r'Excitation frequency $\nu_\tau$ (THz)')
        ax.set_title(title or f"{self.name} – 2D FFT")
        fig.tight_layout()
        return fig, ax





    def plot_time_traces_slider(self, NL_factor=1.0, normalize=False, 
                                    t_min=None, t_max=None, title="Time-domain traces (interactive)"):
            """
            Interactive plot of time-domain traces (A, B, AB, and optionally NL × NL_factor)
            with a slider to move along the delay axis (τ).

            Parameters
            ----------
            NL_factor : float
                Multiplicative factor for the nonlinear (NL) signal.
                If 0, the NL trace is not plotted.
            normalize : bool
                If True, normalize each trace to its maximum absolute value.
            t_min, t_max : float, optional
                Limits for time axis.
            title : str
                Figure title.
            """
            import numpy as np

            t = self.A['time'].values
            tau = self.A.columns[1:].astype(float)

            # Initial index
            tau_idx = 0
            tau_val = tau[tau_idx]

            # Extract initial traces
            A_trace = self.A.iloc[:, tau_idx + 1].values
            B_trace = self.B.iloc[:, tau_idx + 1].values
            AB_trace = self.AB.iloc[:, tau_idx + 1].values
            NL_trace = self.NL.iloc[:, tau_idx + 1].values if NL_factor != 0 else None

            if normalize:
                def norm(x): return x / np.max(np.abs(x))
                A_trace, B_trace, AB_trace = map(norm, [A_trace, B_trace, AB_trace])
                if NL_trace is not None:
                    NL_trace = norm(NL_trace)

            # --- Plot setup ---
            fig, ax = plt.subplots(figsize=(10, 7))
            plt.subplots_adjust(top=0.85, bottom=0.15)
            
            line_AB, = ax.plot(t, AB_trace, label='AB', color='black')
            line_A, = ax.plot(t, A_trace, label='A', color='red')
            line_B, = ax.plot(t, B_trace, label='B', color='blue')
            line_NL = None
            if NL_trace is not None:
                line_NL, = ax.plot(t, NL_trace * NL_factor, label=f'NL × {NL_factor:g}', linestyle='-', color='black', alpha=0.5)

            ax.set_xlabel("Time t (ps)")
            ax.set_ylabel("Signal (a.u.)")
            ax.set_xlim(t_min or t.min(), t_max or t.max())
            ax.set_title(f"{title}\nτ = {tau_val:.2f} ps")
            ax.legend()

            # --- Slider ---
            ax_tau = plt.axes([0.2, 0.92, 0.6, 0.03], facecolor='lightgray')
            tau_slider = Slider(ax_tau, 'τ (ps)', tau.min(), tau.max(), valinit=tau_val)

            # --- Update function ---
            def update(val):
                tau_current = tau_slider.val
                idx = np.argmin(np.abs(tau - tau_current))

                A_trace = self.A.iloc[:, idx + 1].values
                B_trace = self.B.iloc[:, idx + 1].values
                AB_trace = self.AB.iloc[:, idx + 1].values
                NL_trace = self.NL.iloc[:, idx + 1].values if NL_factor != 0 else None

                if normalize:
                    A_trace, B_trace, AB_trace = map(norm, [A_trace, B_trace, AB_trace])
                    if NL_trace is not None:
                        NL_trace = norm(NL_trace)

                line_A.set_ydata(A_trace)
                line_B.set_ydata(B_trace)
                line_AB.set_ydata(AB_trace)
                if line_NL is not None:
                    line_NL.set_ydata(NL_trace * NL_factor)

                ax.set_title(f"{title}\nτ = {tau[idx]:.2f} ps")
                fig.canvas.draw_idle()

            tau_slider.on_changed(update)

            plt.show()

            return tau_slider, ax