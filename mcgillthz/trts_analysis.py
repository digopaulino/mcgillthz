import numpy as np      # type: ignore
from scipy.optimize import minimize         # type: ignore
from scipy.interpolate import PchipInterpolator  # type: ignore
from tqdm import tqdm           # type: ignore


from .fft_utils import *
from .import_utils import *
from .misc import *
from .tds_analysis import *


def get_T_trts(ref_df, ref_amp, ref_phase, pump_df, pump_amp, pump_phase, freqs_for_fit=[2,4], 
               ref_amp_std=None, ref_phase_std=None, pump_amp_std=None, pump_phase_std=None):
    """
    Calculates the transmission or reflection amplitude and phase using reference and pumped data. 
    TO-DO: Remove ref_df and pump_df.

    Parameters:
    ref_df (DataFrame): DataFrame containing reference data, where the first column is frequency and subsequent
                        columns correspond to time delays.
    ref_amp (DataFrame): DataFrame containing the amplitude of the reference signal, where the first column
                         is frequency and subsequent columns correspond to time delays.
    ref_phase (DataFrame): DataFrame containing the phase of the reference signal, where the first column
                           is frequency and subsequent columns correspond to time delays.
    pump_df (DataFrame): DataFrame containing the pumped data, where the first column is frequency and subsequent
                         columns correspond to time delays.
    pump_amp (DataFrame): DataFrame containing the amplitude of the pumped signal, where the first column
                          is frequency and subsequent columns correspond to time delays.
    pump_phase (DataFrame): DataFrame containing the phase of the pumped signal, where the first column
                            is frequency and subsequent columns correspond to time delays.
    freqs_for_fit (list, optional): List of indices corresponding to frequencies used for fitting. Default is [2, 4].

    Returns:
    T_df (DataFrame): DataFrame containing the calculated transmission (or reflection) amplitudes for each time delay.
    phase_df (DataFrame): DataFrame containing the calculated transmission (or reflection) phases for each time delay.
    """
    T_df = pd.DataFrame()
    phase_df = pd.DataFrame()

    if ref_amp_std is not None:
        T_std_df = pd.DataFrame()
        phase_std_df = pd.DataFrame()

        for i, time in enumerate(ref_amp.columns[1:]):
            T_amp = pump_amp[time] / ref_amp[time]
            T_phase_raw = np.unwrap(  pump_phase[time] - ref_phase[time]  )

            # Remove 2pi offset
            mask = (ref_amp.iloc[:,0] > freqs_for_fit[0]) & (ref_amp.iloc[:,0] < freqs_for_fit[1])
            pars, _ = curve_fit(lambda x, a, b: a * x + b, ref_amp.iloc[:,0][mask], T_phase_raw[mask])    # Fit phase to a linear function to remove offset
            T_phase = T_phase_raw - 2 * np.pi * round(pars[1] / (2 * np.pi))

            T_amp_std = T_amp * np.sqrt((pump_amp_std[time] / pump_amp[time])**2 + (ref_amp_std[time] / ref_amp[time])**2)
            T_phase_std = np.sqrt(pump_phase_std[time]**2 + ref_phase_std[time]**2)

            T_df.insert(i, time, T_amp, True)
            phase_df.insert(i, time, T_phase, True)
            T_std_df.insert(i, time, T_amp_std, True)
            phase_std_df.insert(i, time, T_phase_std, True)

        T_df.insert(0, 'freq', ref_amp.iloc[:,0], True)
        phase_df.insert(0, 'freq', ref_amp.iloc[:,0], True)
        T_std_df.insert(0, 'freq', ref_amp.iloc[:,0], True)
        phase_std_df.insert(0, 'freq', ref_amp.iloc[:,0], True)

        return T_df, phase_df, T_std_df, phase_std_df

    else:
        for i, time in enumerate(ref_amp.columns[1:]):
            T_amp = pump_amp[time] / ref_amp[time]
            T_phase = np.unwrap(  pump_phase[time] - ref_phase[time]  )

            T_df.insert(i, time, T_amp, True)
            phase_df.insert(i, time, T_phase, True)

        T_df.insert(0, 'freq', ref_amp.iloc[:,0], True)
        phase_df.insert(0, 'freq', ref_amp.iloc[:,0], True)

        return T_df, phase_df



def sig_tinkham_all(Amps, Phases, d, n, reflection=False, Amps_err=None, Phases_err=None):
    """
    Calculates the change in conductivity using Tinkham's formula for a given set of transmission amplitudes and phases.

    Parameters:
    Amps (DataFrame): DataFrame containing transmission amplitudes, where the first column is frequency
                        and subsequent columns correspond to different time delays.
    Phases (DataFrame): DataFrame containing transmission phases, where the first column is frequency
                          and subsequent columns correspond to different time delays.
    d (float): Thickness of the sample.
    n (float or ndarray): Complex efractive index of the substrate.
    reflection (bool, optional): If true, assumes the experiment was performed in reflection geometry. If falses, assumes it's in transmission.

    Returns:
    dsig_df (DataFrame): DataFrame containing the calculated change in conductivity for each time delay,
                         with the first column as frequency.
    """
    dsig_df = pd.DataFrame()

    freqs = Amps.iloc[:,0] 
    for i, time in enumerate(Amps.columns[1:]):
        exp_amp = Amps[time]
        exp_phase = Phases[time]
        if reflection:
            R = exp_amp * np.exp(1j*exp_phase)
            dsig = (1 - n**2) * (1-R) / (Z0 * d * ( (1+R) + n*(1-R) ))
        else:
            dsig = n/(Z0*d) * (1/(exp_amp*np.exp(1J*exp_phase)) - 1)
        
        dsig_df.insert(i, time, dsig, True)

    dsig_df.insert(0, 'freq', freqs, True)

    if (Amps_err is not None) and (Phases_err is not None):
        dsig_err_df = pd.DataFrame()
        for i, time in enumerate(Amps.columns[1:]):
            amp = Amps[time]
            phase = Phases[time]
            amp_err = Amps_err[time]
            phase_err = Phases_err[time]
            sig1 = np.real(dsig_df[time])
            sig2 = np.imag(dsig_df[time])
            
            if reflection:  
                F = 1 + 2*np.real(n) + n**2 + 2*amp*(np.cos(phase)*(1 - n**2) + 2*np.sin(phase)*np.imag(n)) + amp**2 * (1 - 2*np.real(n) + n**2)
                dFdR = 2*(np.cos(phase)*(1 - n**2) + 2*np.sin(phase)*np.imag(n)) + 2*amp*(1 - 2*np.real(n) + n**2)
                dFdphase = 2*np.cos(phase)*np.imag(n) - 2*amp*np.sin(phase)*(1-n**2)

                sig1_err_sqr = ((1-n**2)/(Z0*d) * -2 *(amp*(1 - np.conj(n)) + np.conj(n)*np.cos(phase)  )/F - sig1*dFdR/F)**2 * amp_err**2 + \
                                    ((1-n**2)/(Z0*d) * 2*np.conj(n)*amp*np.sin(phase)/F  - sig1 * dFdphase/F   )**2 * phase_err**2

                sig2_err_sqr = ((1-n**2)/(Z0*d) * -2 *np.sin(phase)/F - sig2*dFdR/F)**2 * amp_err**2 + \
                                    ((1-n**2)/(Z0*d) * -2*amp*np.cos(phase)/F  - sig2 * dFdphase/F   )**2 * phase_err**2

                sig_err = np.sqrt(sig1_err_sqr) + 1J*np.sqrt(sig2_err_sqr)

            else:
                raise ValueError('Error propagation in transmission not implemented.')   

            dsig_err_df.insert(i, time, sig_err, True)
        
        dsig_err_df.insert(0, 'freq', freqs, True)
        
        return dsig_df, dsig_err_df

    else:
        return dsig_df



def minimize_err_at_freq_trts(dsig, freq, T_function, sig0, T_pars, exp_amp, exp_phase, method='Powell', bound_offset=1e2):
    """
    Minimizes the error between the calculated and experimental transmission values at a specific frequency.

    Parameters:
    dsig (complex): The complex change in conductivity at the given frequency.
    freq (float): The frequency at which to minimize the error.
    T_function (callable): The transmission (or reflection) function used to calculate theoretical transmission (or reflection) values.
    sig0 (complex): The equilibrium conductivity value.
    T_pars (list): Parameters to be passed to the transmission (or reflection) function.
    exp_amp (float): Experimental amplitude value.
    exp_phase (float): Experimental phase value.
    method (str, optional): The optimization method to use (default is 'Powell').
    bound_offset (float, optional): The offset for bounds on the real part of the conductivity (default is 1e2).

    Returns:
    result (OptimizeResult): The result of the optimization process, containing information about the minimum found,
                             the value of the objective function at that point, and success/failure status.
    """
    err_func_simp = lambda x: error_function(x[0] + 1j*x[1], freq, T_function, [sig0, *T_pars], exp_amp, exp_phase)
    result = minimize(err_func_simp, [np.real(dsig), np.imag(dsig)], method=method, bounds=((-np.real(sig0) - bound_offset,np.infty), (-np.infty, np.infty)))

    return result



def minimize_err_trts(dsig0, freqs, T_function, sig0s, T_pars, exp_amp, exp_phase, exp_freqs, method='Powell', start_from='high', bound_offset=1e2, d_tink=100e-9, nsub_tink=1, txt=True):
    """
    Minimizes the error between calculated and experimental transmission values over all frequencies.

    Parameters:
    dsig0 (complex): The initial guess for the complex change in conductivity.
    freqs (ndarray): Frequencies at which to minimize the error.
    T_function (callable): User defined transmission (or reflection) function used to calculate theoretical transmission (or reflection).
        Must have conductivity as 1st parameter, frequency as 2nd, and equilibrium conductivity as 3rd.
    sig0s (ndarray): Equilibrium conductivity values calculated at frequencies "freqs".
    T_pars (list): Additional parameters to be passed to the transmission function.
    exp_amp (ndarray): Experimental amplitude values.
    exp_phase (ndarray): Experimental phase values.
    exp_freqs (ndarray): Experimental frequencies values.
    method (str, optional): The optimization method to use (default is 'Powell').
    start_from (str, optional): 'low', 'high', 'tinkham' or 'tinkham-ref'. Indicates the starting point for optimization. 'low' goes from low to high frequencies, 
        'high' goes from high to low frequencies, 'tinkham' uses the Tinkham approximation to give the initial conductivities in each frequency, and
        'tinkham-ref' does the same but in reflection instead of transmission. 
    bound_offset (float, optional): The offset for bounds on the real part of the conductivity (default is 1e2).
    d_tink (float, optional): Thickness parameter used in Tinkham approximation (default is 100 nm).
    nsub_tink (float, optional): Substrate index used in Tinkham approximation (default is 1).
    txt (bool, optional): If True, print information about the process.

    Returns:
    dsigs (ndarray): The optimized complex change in conductivity for each frequency.
    T_amps (ndarray): The calculated transmission amplitudes at the optimized conductivity values.
    T_phases (ndarray): The calculated transmission phases at the optimized conductivity values.
    """
    # Interpolating the experimental amplitude and phase to the given frequencies
    amp_interp = PchipInterpolator(exp_freqs, exp_amp)
    phase_interp = PchipInterpolator(exp_freqs, exp_phase)

    # start_from can be "low", "high", "tinkham" or "tinkham-ref"
    dsigs = np.zeros(len(freqs), dtype=complex)
    T_amps = np.zeros(len(freqs))
    T_phases = np.zeros(len(freqs))

    if (start_from == 'tinkham') or (start_from == 'Tinkham'):
        if txt: print(f'Using Tinkham as initial guess with d={d_tink/1e-9:.0f} nm, and n_sub={nsub_tink:.2f}')

        amps = amp_interp(freqs)
        phases = phase_interp(freqs)
        for j in range(len(freqs)):       
            result = minimize_err_at_freq_trts(sig_tinkham(amps[j], phases[j], d_tink, nsub_tink), freqs[j], T_function, sig0s[j], T_pars, amps[j], phases[j], method=method, bound_offset=bound_offset)
            dsigs[j] = result.x[0] + 1J*result.x[1]

            T_amps[j], T_phases[j] = T_function(dsigs[j], freqs[j], sig0s[j], *T_pars)
        
        return dsigs, T_amps, T_phases
    
    elif (start_from == 'tinkham-ref') or (start_from == 'Tinkham-Ref'):
        if txt: print(f'Using Tinkham in reflection as initial guess with d={d_tink/1e-9:.0f} nm, and n_sub={nsub_tink:.2f}')

        amps = amp_interp(freqs)
        phases = phase_interp(freqs)
        for j in range(len(freqs)):       
            result = minimize_err_at_freq_trts(sig_tinkham(amps[j], phases[j], d_tink, nsub_tink, reflection=True), freqs[j], T_function, sig0s[j], T_pars, amps[j], phases[j], method=method, bound_offset=bound_offset)
            dsigs[j] = result.x[0] + 1J*result.x[1]

            T_amps[j], T_phases[j] = T_function(dsigs[j], freqs[j], sig0s[j], *T_pars)
        
        return dsigs, T_amps, T_phases

    else:
        if (start_from == 'low') or (start_from == 'Low'):
            func = np.min
            argfunc = np.argmin
        elif (start_from == 'high') or (start_from == 'High'):
            func = np.max
            argfunc = np.argmax
        else:
            if txt: print('Invalid "start_from". Options are "low", "high", "tinkham" or "tinkham-ref". Starting from high frequencies instead.')
            func = np.max
            argfunc = np.argmax

        freq0 = func(freqs)
        amp0 = amp_interp(freq0)
        phase0 = phase_interp(freq0)       

        sig0 = sig0s[ argfunc(freqs) ]
        result = minimize_err_at_freq_trts(dsig0, freq0, T_function, sig0, T_pars, amp0, phase0, method=method, bound_offset=bound_offset)

        if not result.success:
            if txt: print('Initial guess was too off. Trying with Tinkham approximation as initial guess.')

            dsig0 = sig_tinkham(amp0, phase0, d_tink, nsub_tink)

            result = minimize_err_at_freq_trts(dsig0, freq0, T_function, sig0, T_pars, amp0, phase0, method=method, bound_offset=bound_offset)

            if not result.success:
                print('OPTIMIZATION FAILED. Change initial guess, transfer function or solver method.')

                return dsigs, T_amps, T_phases
        

        for i in range(len(freqs)):       
            if (start_from == 'low') or (start_from == 'Low'):
                j = i
            else:
                j = len(freqs)-1 - i
        
            amp = amp_interp(freqs[j])
            phase = phase_interp(freqs[j]) 

            result = minimize_err_at_freq_trts(result.x[0] + 1j*result.x[1], freqs[j], T_function, sig0s[j], T_pars, amp, phase, method=method, bound_offset=bound_offset)
            dsigs[j] = result.x[0] + 1J*result.x[1]

            T_amps[j], T_phases[j] = T_function(dsigs[j], freqs[j], sig0s[j], *T_pars)


        return dsigs, T_amps, T_phases


    

def minimize_err_trts_all(T_amps, T_phases, freqs, T_function, sig0, T_pars, method='Powell', dsig0=0, start_from='tinkham', bound_offset=1e2, d_tink=100e-9, nsub_tink=1):
    """
    Minimizes the error and calculates the conductivity for all time delays between calculated and experimental transmission values over a range of frequencies.

    Parameters:
    T_amps (pd.DataFrame): Experimental amplitude values for each time step and frequency.
    T_phases (pd.DataFrame): Experimental phase values for each time step and frequency.
    freqs (ndarray): Frequencies at which to minimize the error.
    T_function (callable): The transmission function used to calculate theoretical transmission values.
    sig0 (ndarray): Initial conductivity values for each frequency.
    T_pars (list): Parameters to be passed to the transmission function.
    method (str, optional): The optimization method to use (default is 'Powell').
    dsig0 (complex, optional): The initial guess for the complex change in conductivity.
    start_from (str, optional): Indicates the starting point for optimization ('low', 'high', 'tinkham', or 'tinkham-ref'). See minimize_err_trts for details.
    bound_offset (float, optional): The offset for bounds on the real part of the conductivity (default is 1e2).
    d_tink (float, optional): Thickness parameter used in Tinkham approximation (default is 100 nm).
    nsub_tink (float, optional): Substrate index used in Tinkham approximation (default is 1).

    Returns:
    dsig_df (pd.DataFrame): The optimized complex change in conductivity for each time step and frequency.
    T_amps_fit (pd.DataFrame): The calculated transmission amplitudes at the optimized conductivity values for each time step and frequency.
    T_phases_fit (pd.DataFrame): The calculated transmission phases at the optimized conductivity values for each time step and frequency.
    """    
    dsig_df = pd.DataFrame()
    T_amps_fit = pd.DataFrame()
    T_phases_fit = pd.DataFrame()

    exp_freqs = T_amps['freq']    
    for i, time in enumerate(tqdm(T_amps.columns[1:])):
        exp_amp = T_amps[time]
        exp_phase = T_phases[time]

        dsig, T_amp_fit, T_phase_fit = minimize_err_trts(dsig0, freqs, T_function, sig0, T_pars, exp_amp, exp_phase, exp_freqs, method=method, \
                                            start_from=start_from, bound_offset=bound_offset, d_tink=d_tink, nsub_tink=nsub_tink, txt=False)

        dsig_df.insert(i, time, dsig, True)
        T_amps_fit.insert(i, time, T_amp_fit, True)
        T_phases_fit.insert(i, time, T_phase_fit, True)

    
    
    dsig_df.insert(0, 'freq', freqs, True)
    T_amps_fit.insert(0, 'freq', freqs, True)
    T_phases_fit.insert(0, 'freq', freqs, True)


    return dsig_df, T_amps_fit, T_phases_fit



