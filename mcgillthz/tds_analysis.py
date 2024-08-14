
import numpy as np  # type: ignore
from scipy.optimize import curve_fit, minimize # type: ignore
from scipy.interpolate import PchipInterpolator  # type: ignore

from .fft_utils import *
from .import_utils import *
from .misc import *


def get_T_tds(ref_td, ref_fft, samp_td, samp_fft, freqs_for_fit=[2, 4], min_freq=0, max_freq=np.inf):
    """
    Calculate the transmission function T(ω) from reference and sample time-domain and frequency-domain data. Requires same frequencies
    for both reference and sample data.

    Parameters:
    ref_td (ndarray): 2D array with time-domain data for the reference.
    ref_fft (ndarray): 2D array with frequency-domain data for the reference.
    samp_td (ndarray): 2D array with time-domain data for the sample.
    samp_fft (ndarray): 2D array with frequency-domain data for the sample.
    freqs_for_fit (list, optional): List containing the minimum and maximum frequencies for fitting. Defauls it [2,4]

    Returns:
    ndarray: Array containing frequency, transmission amplitude, phase, amplitude error, and phase error.
        If ref_fft does not contain the errors, the output also won't contain the errors.
    """
    freq = samp_fft[0]
    T_amp = samp_fft[1] / ref_fft[1]  # Amplitude transmission

    t0_ref, t0_samp = ref_td[0][np.argmax(ref_td[1])], samp_td[0][np.argmax(samp_td[1])]
    delta_t = t0_samp - t0_ref
    phase_offset = 2 * np.pi * delta_t * freq

    # # Calculate reduced unwrapped phase following Jepsen's method
    phi0_ref, phi0_samp = 2*np.pi*ref_fft[0]*t0_ref, 2*np.pi*samp_fft[0]*t0_samp
    dphi0_reduced = np.unwrap( np.angle(np.exp(1J*(samp_fft[2] - phi0_samp))) - np.angle(np.exp(1J*(ref_fft[2] - phi0_ref))  ) )   
    mask = (freq < freqs_for_fit[1]) & (freq > freqs_for_fit[0])
    pars, _ = curve_fit(lambda x, a, b: a * x + b, freq[mask], dphi0_reduced[mask])    # Fit phase to a linear function to remove offset
    phase0 = dphi0_reduced - 2 * np.pi * round(pars[1] / (2 * np.pi))
    phase = phase0 - phi0_ref + phi0_samp + phase_offset
    


    if len(ref_fft) > 3:
        T_amp_err = T_amp * np.sqrt((samp_fft[3] / samp_fft[1])**2 + (ref_fft[3] / ref_fft[1])**2)
        phase_err = np.sqrt(samp_fft[4]**2 + ref_fft[4]**2)

        T = np.array([freq, T_amp, phase, T_amp_err, phase_err])
    else:
        T = np.array([freq, T_amp, phase])
    
    return T[:, (T[0]>min_freq) & (T[0]<max_freq)]

def error_function(n_til, freq, T_function, T_function_pars, exp_amp, exp_phase):
    """
    Compute the error between the experimental and theoretical transmission functions.

    Parameters:
    n_til (complex): Complex refractive index.
    freq (float): Frequency value.
    T_function (callable): Transmission function.
    T_function_pars (tuple): Parameters for the transmission function.
    exp_amp (float): Experimental amplitude.
    exp_phase (float): Experimental phase.

    Returns:
    float: Error value.
    """
    amp, phase = T_function(n_til, freq, *T_function_pars)

    chi1 = (np.log(amp) - np.log(exp_amp))**2
    chi2 = (phase - exp_phase)**2

    return chi1 + chi2

def minimize_err_at_freq_tds(n_til, freq, T_function, T_function_pars, exp_amp, exp_phase, method='L-BFGS-B'):
    """
    Minimize the error at a specific frequency to find the best-fit complex refractive index.

    Parameters:
    n_til (complex): Initial guess for the complex refractive index.
    freq (float): Frequency value.
    T_function (callable): Transmission function.
    T_function_pars (tuple): Parameters for the transmission function.
    exp_amp (float): Experimental amplitude.
    exp_phase (float): Experimental phase.
    method (str, optional): Optimization method. Options include L-BFGS-B, Powell or Nelder-Mead
                    See scipy.optimize.minimize function for explanation.

    Returns:
    OptimizeResult: Result of the optimization.
    """
    err_func_simp = lambda x: error_function(x[0] + 1j * x[1], freq, T_function, T_function_pars, exp_amp, exp_phase)

    result = minimize(err_func_simp, [np.real(n_til), np.imag(n_til)], method=method, bounds=((0, np.inf), (-np.inf, np.inf)))
    return result

def minimize_err_tds(freqs, T_function, T_function_pars, exp_freqs, exp_amp, exp_phase, start_from='simple', method='L-BFGS-B', n_til0=1, d=None):
    """
    Minimize the error over a range of frequencies to find the best-fit complex refractive index for each frequency.

    Parameters:
    freqs (ndarray): Frequency values to calculate the index, in THz.
    T_function (callable): Transmission function.
    T_function_pars (tuple): Parameters for the transmission function.
    exp_freqs (ndarray): Experimental frequency values.
    exp_amp (ndarray): Experimental amplitude values.
    exp_phase (ndarray): Experimental phase values.
    start_from (str, optional): 'low', 'high', 'simple'. Indicates the starting point for optimization. 'low' goes from low to high frequencies, 
        'high' goes from high to low frequencies, 'simple' considers the sample is transparent, with no substrate and no Fabry-Perot reflection.
    method (str, optional): Optimization method. Options include L-BFGS-B, Powell or Nelder-Mead. Default for TDS is 'L-BFGS-B'
                    See scipy.optimize.minimize function for explanation.
    n_til0 (complex, optional): Initial guess for the complex refractive index. Highly recommended when 'start_from' is 'low' or 'high'.
    d (float, optional): Thickness of the sample. Required for 'start_from == simple'.

    Returns:
    ndarray: Best-fit complex refractive indices for each frequency.
    """
    # Interpolating the experimental amplitude and phase to the given frequencies
    amp_interp = PchipInterpolator(exp_freqs, exp_amp)
    phase_interp = PchipInterpolator(exp_freqs, exp_phase)

    if (start_from == 'simple') or (start_from == 'Simple'):
        n_tils = np.zeros(len(freqs), dtype=complex)

        if d is None:
            print('Please insert an estimated thickness using the variable "d".')
            return n_tils

        for i in range(len(freqs)):       
            freq = freqs[i]
            amp = amp_interp(freq)
            phase = phase_interp(freq)
            n0 = n_thick_transp(freq, amp, phase, d, nsub=1)

            result = minimize_err_at_freq_tds(n0, freq, T_function, T_function_pars, amp, phase, method=method)
            n_tils[i] = result.x[0] + 1J * result.x[1]

        return n_tils


    else:
        if (start_from == 'low') or (start_from == 'Low'):
            func = np.min
        elif (start_from == 'high') or (start_from == 'High'):
            func = np.max
        else:
            print('Invalid "start_from". Starting from high frequencies instead.')
            func = np.max
        
        if n_til0 == 1:
            print('Using n = 1 as initial guess. If optimization fails, please insert an initial guess for variable "n_til0".')

        amp0 = exp_amp[np.argmin(np.abs(exp_freqs - func(freqs)))]
        phase0 = exp_phase[np.argmin(np.abs(exp_freqs - func(freqs)))]

        result = minimize_err_at_freq_tds(n_til0, func(freqs), T_function, T_function_pars, amp0, phase0, method=method)

        if not result.success:
            print('Optimization failed! Change initial guess.')
            return result
        else:
            n_tils = np.zeros(len(freqs), dtype=complex)
            
            for i in range(len(freqs)):       
                if (start_from == 'low') or (start_from == 'Low'):
                    j = i
                else:
                    j = len(freqs) - 1 - i
        
                freq = freqs[j]
                amp = amp_interp(freq)
                phase = phase_interp(freq)

                result = minimize_err_at_freq_tds(result.x[0] + 1J * result.x[1], freq, T_function, T_function_pars, amp, phase, method=method)
                n_tils[j] = result.x[0] + 1J * result.x[1]

            return n_tils

def get_total_err_tds(n_tils, freqs, T_function, T_function_pars, exp_freqs, exp_amps, exp_phases):
    """
    Calculate the total error for the fitted complex refractive indices over a range of frequencies.

    Parameters:
    n_tils (ndarray): Best-fit complex refractive indices.
    freqs (ndarray): Frequency values, in THz.
    T_function (callable): Transmission function.
    T_function_pars (tuple): Parameters for the transmission function.
    exp_freqs (ndarray): Experimental frequency values.
    exp_amps (ndarray): Experimental amplitude values.
    exp_phases (ndarray): Experimental phase values.

    Returns:
    float: Total error.
    """
    total_err = 0
    for i, f in enumerate(freqs):
        ind_of_freq = np.argmin(np.abs(exp_freqs - freqs[i]))
        amp = exp_amps[ind_of_freq]
        phase = exp_phases[ind_of_freq]

        total_err += error_function(n_tils[i], f, T_function, T_function_pars, amp, phase)
    
    return total_err

def get_R2(n_tils, freqs, T_function, T_function_pars, exp_freqs, exp_amps, exp_phases):
    """
    Calculate the R2 parameter for the fitted complex refractive indices over a range of frequencies.

    Parameters:
    n_tils (ndarray): Best-fit complex refractive indices.
    freqs (ndarray): Frequency values.
    T_function (callable): Transmission function.
    T_function_pars (tuple): Parameters for the transmission function.
    exp_freqs (ndarray): Experimental frequency values.
    exp_amps (ndarray): Experimental amplitude values.
    exp_phases (ndarray): Experimental phase values.

    Returns:
    float: Total error.
    """
    sum_sqr_res = 0
    sum_sqr = 0
    for i, f in enumerate(freqs):
        ind_of_freq = np.argmin(np.abs(exp_freqs - freqs[i]))
        amp = exp_amps[ind_of_freq]
        phase = exp_phases[ind_of_freq]

        sum_sqr_res += error_function(n_tils[i], f, T_function, T_function_pars, amp, phase)
        sum_sqr += (np.log(amp) - np.log(np.mean(exp_amps)))**2 + (phase - np.mean(exp_phases))**2
    
    R2 = 1 - sum_sqr_res/sum_sqr
    return R2
