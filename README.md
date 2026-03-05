# McGill THz Spectroscopy package

This package was made to facilitate the analysis of different types of THz spectroscopy experiments, be it a simple scan, time-domain spectroscopy (TDS) or 2D optical pump-THz probe spectroscopy (TRTS).

A full documentation is not yet available, but all functions are thoroughly described in the files they are defined. Combined with the tutorials available in the *tutorials/* folder, it should be more than enough to get you going in your analysis.

# Installation 

To use this package, simply download all the files and extract them into a folder (like "mcgillthz/"). You'll need an installation of Python 3 (v3.9 or higher), but I recommend using the functions on a jupyter notebook.

The following packages must also be installed:
- numpy, pandas, matplotlib, scipy.
- tqdm (optional) - Adds a progress bar to the code
- PyWavelets (optional) - Does continous wavelet transforms.
- ipympl (optional) - Allows for iterative plots on a jupyter notebook.
<!-- - lmfit - Wraps scipy.optimize to allow for easier manipulation of the parameters. Very useful for fitting optical functions with many parameters. -->
- Streamlit, Plotly (optional) - Necessary for all GUIs.


# File Tree

In the *mcgillthz/* folder, the separate files are organized as follows:

- fft_utils.py - All Fourier transform related functions, including padding, wavelet transforms and more.
- import_utils.py - Functions to easily import experimental data, including error estimation and averaging.
- utils.py - Contains miscellaneous functions like a function to do Hilbert transform, calculate the Carrier Envelope phase (CEP) and to convert between response functions. Also includes physical constants
- tds_analysis.py - All functions needed for time-domain spectroscopy, from determining the amplitude and phase of the transmission coefficiet, to numerically find the complex refractive index.
- trts_analysis.py - All functions needed for 2d time-resolved THz spectroscopy
- basic_plots.py - Basic functions to do quick standard plots and some iterative plots, including 2d color plots and plots with a slider to help visualizing different pump-probe time delays.
- twodim_analysis.py - All functions and classes needed to do 2D coherent THz spectroscopy.

The *tutorials/* folder contain Jupyter notebooks with all the essential steps for analyzing TDS and 2D TRTS data. 

The *GUIs/* folder contain GUIs (and scripts to run them natively) using Streamlit and the functions defined in the mcgillthz/ folder. Right now, this is mostly focused in the 2D coherent THz spectroscopy.

# To-Do:

- Make documentation, at least detailing the physics and numerical methods behind the code.
- Choose license.
