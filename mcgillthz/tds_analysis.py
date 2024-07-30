
import numpy as np
from scipy.optimize import curve_fit, minimize

from .fft_utils import *
from .import_utils import *
from .misc import *


def get_T_tds(ref_td, ref_fft, samp_td, samp_fft, freqs_for_fit=[2, 4]):
    """
    Calculate the transmission function T(ω) from reference and sample time-domain and frequency-domain data.

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
    freq = ref_fft[0]
    T_amp = samp_fft[1] / ref_fft[1]  # Amplitude transmission

    min_freq, max_freq = freqs_for_fit
    
    t0_ref, t0_samp = ref_td[0][np.argmax(ref_td[1])], samp_td[0][np.argmax(samp_td[1])]
    delta_t = t0_samp - t0_ref
    phase_offset = 2 * np.pi * delta_t * freq

    # # Calculate reduced unwrapped phase following Jepsen's method
    phi0_ref, phi0_samp = 2*np.pi*freq*t0_ref, 2*np.pi*freq*t0_samp
    dphi0_reduced = np.unwrap( np.angle(np.exp(1J*(samp_fft[2] - phi0_samp))) - np.angle(np.exp(1J*(ref_fft[2] - phi0_ref))  ) )   
    mask = (freq < max_freq) & (freq > min_freq)
    pars, _ = curve_fit(lambda x, a, b: a * x + b, freq[mask], dphi0_reduced[mask])    # Fit phase to a linear function to remove offset
    phase0 = dphi0_reduced - 2 * np.pi * round(pars[1] / (2 * np.pi))
    phase = phase0 - phi0_ref + phi0_samp + phase_offset
    


    if len(ref_fft) > 3:
        T_amp_err = T_amp * np.sqrt((samp_fft[3] / samp_fft[1])**2 + (ref_fft[3] / ref_fft[1])**2)
        phase_err = np.sqrt(samp_fft[4]**2 + ref_fft[4]**2)

        return np.array([freq, T_amp, phase, T_amp_err, phase_err])

    else:
        return np.array([freq, T_amp, phase])

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

def minimize_err_tds(n_til0, freqs, T_function, T_function_pars, exp_freqs, exp_amp, exp_phase, method='L-BFGS-B', start_from_low=False):
    """
    Minimize the error over a range of frequencies to find the best-fit complex refractive index for each frequency.

    Parameters:
    n_til0 (complex): Initial guess for the complex refractive index.
    freqs (ndarray): Frequency values.
    T_function (callable): Transmission function.
    T_function_pars (tuple): Parameters for the transmission function.
    exp_freqs (ndarray): Experimental frequency values.
    exp_amp (ndarray): Experimental amplitude values.
    exp_phase (ndarray): Experimental phase values.
    method (str, optional): Optimization method. Options include L-BFGS-B, Powell or Nelder-Mead. Default for TDS is 'L-BFGS-B'
                    See scipy.optimize.minimize function for explanation.
    start_from_low (bool, optional): Whether to start optimization from the lowest frequency. Default is False

    Returns:
    ndarray: Best-fit complex refractive indices for each frequency.
    """
    if start_from_low:
        func = np.min
    else:
        func = np.max
    
    amp0 = exp_amp[np.argmin(np.abs(exp_freqs - func(freqs)))]
    phase0 = exp_phase[np.argmin(np.abs(exp_freqs - func(freqs)))]

    result = minimize_err_at_freq_tds(n_til0, func(freqs), T_function, T_function_pars, amp0, phase0, method=method)

    if not result.success:
        print('Optimization failed! Change initial guess.')
        return result
    else:
        n_tils = np.zeros(len(freqs), dtype=complex)
        
        for i in range(len(freqs)):       
            if start_from_low:
                j = i
            else:
                j = len(freqs) - 1 - i
        
            ind_of_freq = np.argmin(np.abs(exp_freqs - freqs[j]))
            amp = exp_amp[ind_of_freq]
            phase = exp_phase[ind_of_freq]

            result = minimize_err_at_freq_tds(result.x[0] + 1j * result.x[1], freqs[j], T_function, T_function_pars, amp, phase, method=method)
            n_tils[j] = result.x[0] + 1j * result.x[1]

        return n_tils

def get_total_err_tds(n_tils, freqs, T_function, T_function_pars, exp_freqs, exp_amps, exp_phases):
    """
    Calculate the total error for the fitted complex refractive indices over a range of frequencies.

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
