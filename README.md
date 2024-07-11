# About this package

This package was made to facilitate the analysis of different types of THz spectroscopy experiments, be it a simple scan, time-domain spectroscopy (TDS) or 2D optical pump-THz probe spectroscopy (TRTS).

A full documentation is not yet available, but all functions are thoroughly described in the files they are defined. Combined with the tutorials available in the *tutorial/* folder, it should be more than enough to get you going in your analysis.

# Installation 

To use this package, simply download all the files and extract them. You'll need an installation of Python 3 (this was coded using Python 3.9.13, but others should work). The packages required are

- numpy, pandas, matplotlib, scipy.
- PyWavelets - Does continous wavelet transforms.
<!-- - lmfit - Wraps scipy.optimize to allow for easier manipulation of the parameters. Very useful for fitting optical functions with many parameters. -->
- ipympl (optional) - Allows for iterative plots on a jupyter notebook.


# Organization

In the *src/* folder, the separate files are organized as follows:

- fft_utils.py - All Fourier transform related functions, including padding, wavelet transforms and more.
- import_utils.py - Functions to easily import experimental data, including error estimation and averaging.
- utils.py - Contains miscellaneous functions like a function to do Hilbert transform, calculate the Carrier Envelope phase (CEP) and to convert between response functions. Also includes physical constants
- tds_analysis.py - All functions needed for time-domain spectroscopy, from determining the amplitude and phase of the transmission coefficiet, to numerically find the complex refractive index.
- basic_plots.py - Basic functions to do quick standard plots.

