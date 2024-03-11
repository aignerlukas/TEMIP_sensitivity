#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 15:21:05 2022

utility functions for empymod_frwrd_ip and empymod_frwrd

@author: laigner
"""

# %% modules
import logging
import numpy as np
import pandas as pd

from scipy.constants import epsilon_0
from scipy.special import roots_legendre
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline

from scipy.constants import mu_0


# %% logging ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.WARNING)
# logger.setLevel(logging.DEBUG)


# %% general ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def getR_fromSquare(a):
    """
    function to get the radius of circle with equal area as a given square

    Parameters
    ----------
    a : float
        square side length.

    Returns
    -------
    float
        radius of areally equivalent circle.

    """
    return np.sqrt((a*a) / np.pi)


def scaling(signal):
    """
    function to rescale a signal as preparation for the area sinus hyperbolicus
    transformation. from: Seidel and Tezkan, (2017)

    Parameters
    ----------
    signal : np.array (float)
        signal to be transformed.

    Returns
    -------
    s : np.array (float)
        transformed signal.

    """
    signal_min = np.min(np.abs(signal))
    signal_max = np.max(np.abs(signal))
    s = np.abs(signal_max / (10 * (np.log10(signal_max) - np.log10(signal_min))))
    return s


def arsinh(signal):
    """
    function to rescale and transform a signal to an area sinus hyperbolicus.
    helps to keep the shape of a signal that contains negative values with 
    a huge dynamic range containing also positive values; e.g., TEM-IP data
    from: Seidel and Tezkan, (2017)

    Parameters
    ----------
    signal : np.array (float)
        signal to be transformed.

    Returns
    -------
    np.array (float)
        transformed signal.

    """
    s = scaling(signal)
    return np.log((signal/s) + np.sqrt((signal/s)**2 + 1))


def kth_root(x, k):
    """
    kth root, returns only real roots

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    k : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if k % 2 != 0:
        res = np.power(np.abs(x),1./k)
        return res*np.sign(x)
    else:
        return np.power(np.abs(x),1./k)


def reshape_model(model, nLayer, nParam):
    """
    function to reshape a 1D bel1d style model to a n-D model containing as
    many rows as layers and as many columns as parameters (thk + nParams)

    Parameters
    ----------
    model : np.array
        bel1D vector model:
            thk_lay_0
            thk_lay_1
            ...
            thk_lay_n-1
            paramID-0_lay_0
            ...
            paramID-0_lay_n
            ...
            paramID-n_lay_0
            ...
            paramID-n_lay_n
    nLayer : int
        number of layers in the model.
    nParam : int
        number of parameters in the model, thk also counts!!

    Returns
    -------
    model : np.ndarray
        nLayer by nParam array with the model params, where the rows have the 
        different layers and the column the different parameters, the first 
        column is always the thickness

    """
    mdlRSHP = np.zeros((nLayer,nParam))
    i = 0
    for col in range(0, nParam):
        for row in range(0, nLayer):
            if col == 0 and row == nLayer-1:
                pass
            else:
                mdlRSHP[row, col] = model[i]
                i += 1
    model = mdlRSHP
    return model


def simulate_error(relerr, abserr, data):
    """
    function to simulate an error vector using the classical approach with an 
    relative and an absolute error

    Parameters
    ----------
    relerr : float
        relative error in %/100.
    abserr : float
        absolute error in the same unit as the data.
    data : np.ndarray (n x 1)
        data vector for which the absolute error will be calculated at each 
        vector entry.

    Returns
    -------
    rand_error_abs : np.ndarray (n x 1)
        random absolute data error in the same unit as the given data vector.

    """
    np.random.seed(42)
    rndm = np.random.randn(len(data))

    rand_error_abs = (relerr * np.abs(data) +
                  abserr) * rndm

    return rand_error_abs


# %% forward preparations ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_TEMFAST_rampdata(location, current_key='1A'):
    """
    function to get the TEM-FAST 48 ramp data as measured by the TU Wien at
    different field sites.

    Parameters
    ----------
    location : string
        name of the field site at which the data where measured. Currently 
        available are:
            'donauinsel', shallow resistivity ca. 80 Ohmm
            'salzlacken', shallow resistivity ca. 20 Ohmm
            'hengstberg', shallow resistivity ca. 50 Ohmm
            'sonnblick', shallow resistivity ca. 500 Ohmm
            
    current_key : string, optional
        Current setting in the TEM-FAST 48 device, can be either 1A or 4A and 
        is not related to the actually induced current. The default is '1A'.

    Raises
    ------
    ValueError
        raised if the location is not correct or if the current_key is not available.

    Returns
    -------
    ramp_data : pd.DataFrame
        n x 4 matrix, where the rows have different cables and the columns the 
        ramp information, which are related to:
            cable: full cable length (m)
            side: square loop side length (m)
            turns: turns in the loop ()
            ramp_off: turn-off ramp(s) for the selected current (microseconds)

    """

    column_list = ['cable', 'side', 'turns', 'ramp_off']

    if current_key == '1A':
        if location == 'donauinsel':
            ramp_data_array = np.array([[  6. ,  1.5 , 1. , 0.15],
                                        [ 12. ,  3.  , 1. , 0.23],
                                        [ 25. ,  6.25, 1. , 0.4 ],
                                        [ 50. , 12.5 , 1. , 0.8 ],
                                        [100. , 25.  , 1. , 1.3 ]])

        elif location == 'salzlacken':
            ramp_data_array = np.array([[  8. ,  2.  , 1.  ,  0.21],
                                        [ 25. ,  6.25, 1.  ,  0.44],
                                        [ 50. , 12.5 , 1.  ,  0.8 ],
                                        [100. , 25.  , 1.  ,  1.3 ]])

        elif location == 'hengstberg':
            ramp_data_array = np.array([[ 25. ,  6.25 , 1. ,  0.44],
                                        [ 50. ,  12.5 , 1. ,  0.82]])

        elif location == 'sonnblick':
            ramp_data_array = np.array([[ 50. ,  12.5 , 1. ,  1.0],
                                        [100. ,  25.  , 1. ,  2.5],
                                        [200. ,  50.  , 1. ,  4.2]])

        else:
            raise ValueError('location of ramp data measurements is not available ...')

    elif current_key == '4A':
        if location == 'donauinsel':
            ramp_data_array = np.array([[  6. ,  1.5  , 1. ,  0.17],
                                        [ 25. ,  6.25 , 1. ,  0.45],
                                        [ 50. ,  12.5 , 1. ,  0.95],
                                        [100. ,  25.  , 1. ,  1.5],
                                        [400. , 100.  , 1. , 10.0]])

        elif location == 'salzlacken':
            ramp_data_array = np.array([[  8. ,  2.  , 1. ,  0.21],
                                        [ 25. ,  6.25, 1. ,  0.5 ],
                                        [ 50. , 12.5 , 1. ,  0.95],
                                        [100. , 25.  , 1. ,  1.5 ],
                                        [200. , 50.  , 1. ,  4.3 ],
                                        [400. ,100.  , 1. , 10.0]])

        elif location == 'hengstberg':
            ramp_data_array = np.array([[ 25. ,  6.25 , 1. ,  0.48],
                                        [ 50. ,  12.5 , 1. ,  0.98]])

        elif location == 'sonnblick':
            ramp_data_array = np.array([[ 50. ,  12.5 , 1. ,  1.15],
                                        [100. ,  25.  , 1. ,  2.70],
                                        [200. ,  50.  , 1. ,  5.10]])

        else:
            raise ValueError('location of ramp data measurements is not available ...')

    else:
        raise ValueError('current key is not available ... \nPlease use either 1A or 4A')

    ramp_data = pd.DataFrame(ramp_data_array, columns=column_list)

    return ramp_data


def get_TEMFAST_timegates():
    """
    function to get the the TEM-FAST 48 receiver timegates from a hard coded 
    list of timegates

    Returns
    -------
    pd.DataFrame(48 x n)
        Pandas DataFrame with 48 rows for the 48 different time gates to sample
        the TEM decay curve. The columns are related to:
            id: numerical id for each row
            startT: starting time for each time gate (microseconds)
            endT: end time for each time gate (microseconds)
            centerT: center of each time gate (microseconds)
            deltaT: width of each timegate (microseconds)

    """
    
    tg_raw = np.array([[1.00000e+00, 3.60000e+00, 4.60000e+00, 4.06000e+00, 1.00000e+00],
       [2.00000e+00, 4.60000e+00, 5.60000e+00, 5.07000e+00, 1.00000e+00],
       [3.00000e+00, 5.60000e+00, 6.60000e+00, 6.07000e+00, 1.00000e+00],
       [4.00000e+00, 6.60000e+00, 7.60000e+00, 7.08000e+00, 1.00000e+00],
       [5.00000e+00, 7.60000e+00, 9.60000e+00, 8.52000e+00, 2.00000e+00],
       [6.00000e+00, 9.60000e+00, 1.16000e+01, 1.05300e+01, 2.00000e+00],
       [7.00000e+00, 1.16000e+01, 1.36000e+01, 1.25500e+01, 2.00000e+00],
       [8.00000e+00, 1.36000e+01, 1.56000e+01, 1.45600e+01, 2.00000e+00],
       [9.00000e+00, 1.56000e+01, 1.96000e+01, 1.74400e+01, 4.00000e+00],
       [1.00000e+01, 1.96000e+01, 2.36000e+01, 2.14600e+01, 4.00000e+00],
       [1.10000e+01, 2.36000e+01, 2.76000e+01, 2.54900e+01, 4.00000e+00],
       [1.20000e+01, 2.76000e+01, 3.16000e+01, 2.95000e+01, 4.00000e+00],
       [1.30000e+01, 3.16000e+01, 3.96000e+01, 3.52800e+01, 8.00000e+00],
       [1.40000e+01, 3.96000e+01, 4.76000e+01, 4.33000e+01, 8.00000e+00],
       [1.50000e+01, 4.76000e+01, 5.56000e+01, 5.14000e+01, 8.00000e+00],
       [1.60000e+01, 5.56000e+01, 6.36000e+01, 5.94100e+01, 8.00000e+00],
       [1.70000e+01, 6.36000e+01, 7.96000e+01, 7.16000e+01, 1.60000e+01],
       [1.80000e+01, 7.96000e+01, 9.56000e+01, 8.76000e+01, 1.60000e+01],
       [1.90000e+01, 9.56000e+01, 1.11600e+02, 1.03600e+02, 1.60000e+01],
       [2.00000e+01, 1.11600e+02, 1.27600e+02, 1.19600e+02, 1.60000e+01],
       [2.10000e+01, 1.27600e+02, 1.59600e+02, 1.43600e+02, 3.20000e+01],
       [2.20000e+01, 1.59600e+02, 1.91600e+02, 1.75600e+02, 3.20000e+01],
       [2.30000e+01, 1.91600e+02, 2.23600e+02, 2.07600e+02, 3.20000e+01],
       [2.40000e+01, 2.23600e+02, 2.55600e+02, 2.39600e+02, 3.20000e+01],
       [2.50000e+01, 2.55600e+02, 3.19600e+02, 2.85000e+02, 6.40000e+01],
       [2.60000e+01, 3.19600e+02, 3.83600e+02, 3.50000e+02, 6.40000e+01],
       [2.70000e+01, 3.83600e+02, 4.47600e+02, 4.14000e+02, 6.40000e+01],
       [2.80000e+01, 4.47600e+02, 5.11600e+02, 4.78000e+02, 6.40000e+01],
       [2.90000e+01, 5.11600e+02, 6.39600e+02, 5.70000e+02, 1.28000e+02],
       [3.00000e+01, 6.39600e+02, 7.67600e+02, 6.99000e+02, 1.28000e+02],
       [3.10000e+01, 7.67600e+02, 8.95600e+02, 8.28000e+02, 1.28000e+02],
       [3.20000e+01, 8.95600e+02, 1.02360e+03, 9.56000e+02, 1.28000e+02],
       [3.30000e+01, 1.02360e+03, 1.27960e+03, 1.15200e+03, 2.56000e+02],
       [3.40000e+01, 1.27960e+03, 1.53560e+03, 1.40800e+03, 2.56000e+02],
       [3.50000e+01, 1.53560e+03, 1.79160e+03, 1.66400e+03, 2.56000e+02],
       [3.60000e+01, 1.79160e+03, 2.04760e+03, 1.92000e+03, 2.56000e+02],
       [3.70000e+01, 2.04760e+03, 2.55960e+03, 2.30400e+03, 5.12000e+02],
       [3.80000e+01, 2.55960e+03, 3.07160e+03, 2.81600e+03, 5.12000e+02],
       [3.90000e+01, 3.07160e+03, 3.58360e+03, 3.32800e+03, 5.12000e+02],
       [4.00000e+01, 3.58360e+03, 4.09560e+03, 3.84000e+03, 5.12000e+02],
       [4.10000e+01, 4.09560e+03, 5.11960e+03, 4.60800e+03, 1.02400e+03],
       [4.20000e+01, 5.11960e+03, 6.14360e+03, 5.63200e+03, 1.02400e+03],
       [4.30000e+01, 6.14360e+03, 7.16760e+03, 6.65600e+03, 1.02400e+03],
       [4.40000e+01, 7.16760e+03, 8.19160e+03, 7.68000e+03, 1.02400e+03],
       [4.50000e+01, 8.19160e+03, 1.02396e+04, 9.21600e+03, 2.04800e+03],
       [4.60000e+01, 1.02396e+04, 1.22876e+04, 1.12640e+04, 2.04800e+03],
       [4.70000e+01, 1.22876e+04, 1.43356e+04, 1.33120e+04, 2.04800e+03],
       [4.80000e+01, 1.43356e+04, 1.63836e+04, 1.53600e+04, 2.04800e+03]])

    return pd.DataFrame(tg_raw, columns=['id', 'startT', 'endT', 'centerT', 'deltaT'])


def get_time(times, wf_times):
    """Additional time for ramp.
    from: https://empymod.emsig.xyz/en/stable/gallery/tdomain/tem_walktem.html

    Because of the arbitrary waveform, we need to compute some times before and
    after the actually wanted times for interpolation of the waveform.

    Some implementation details: The actual times here don't really matter. We
    create a vector of time.size+2, so it is similar to the input times and
    accounts that it will require a bit earlier and a bit later times. Really
    important are only the minimum and maximum times. The Fourier DLF, with
    `pts_per_dec=-1`, computes times from minimum to at least the maximum,
    where the actual spacing is defined by the filter spacing. It subsequently
    interpolates to the wanted times. Afterwards, we interpolate those again to
    compute the actual waveform response.

    Note: We could first call `waveform`, and get the actually required times
          from there. This would make this function obsolete. It would also
          avoid the double interpolation, first in `empymod.model.time` for the
          Fourier DLF with `pts_per_dec=-1`, and second in `waveform`. Doable.
          Probably not or marginally faster. And the code would become much
          less readable.

    Parameters
    ----------
    times : ndarray
        Desired times

    wf_times : ndarray
        Waveform times

    Returns
    -------
    time_req : ndarray
        Required times
    """
    tmin = np.log10(max(times.min()-wf_times.max(), 1e-10))
    tmax = np.log10(times.max()-wf_times.min())
    return np.logspace(tmin, tmax, times.size+2)


def waveform(times, resp, times_wanted, wave_time, wave_amp, nquad=3):
    """Apply a source waveform to the signal.
    from: https://empymod.emsig.xyz/en/stable/gallery/tdomain/tem_walktem.html

    Parameters
    ----------
    times : ndarray
        Times of computed input response; should start before and end after
        `times_wanted`.

    resp : ndarray
        EM-response corresponding to `times`.

    times_wanted : ndarray
        Wanted times. Rx-times at which the decay is observed

    wave_time : ndarray
        Time steps of the wave. i.e current pulse

    wave_amp : ndarray
        Amplitudes of the wave corresponding to `wave_time`, usually
        in the range of [0, 1].

    nquad : int
        Number of Gauss-Legendre points for the integration. Default is 3.

    Returns
    -------
    resp_wanted : ndarray
        EM field for `times_wanted`.

    """

    # Interpolate on log.
    PP = iuSpline(np.log10(times), resp)

    # Wave time steps.
    dt = np.diff(wave_time)
    dI = np.diff(wave_amp)
    dIdt = dI/dt

    # Gauss-Legendre Quadrature; 3 is generally good enough.
    # (Roots/weights could be cached.)
    g_x, g_w = roots_legendre(nquad)

    # Pre-allocate output.
    resp_wanted = np.zeros_like(times_wanted)

    # Loop over wave segments.
    for i, cdIdt in enumerate(dIdt):

        # We only have to consider segments with a change of current.
        if cdIdt == 0.0:
            continue

        # If wanted time is before a wave element, ignore it.
        ind_a = wave_time[i] < times_wanted
        if ind_a.sum() == 0:
            continue

        # If wanted time is within a wave element, we cut the element.
        ind_b = wave_time[i+1] > times_wanted[ind_a]

        # Start and end for this wave-segment for all times.
        ta = times_wanted[ind_a]-wave_time[i]
        tb = times_wanted[ind_a]-wave_time[i+1]
        tb[ind_b] = 0.0  # Cut elements

        # Gauss-Legendre for this wave segment. See
        # https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval
        # for the change of interval, which makes this a bit more complex.
        logt = np.log10(np.outer((tb-ta)/2, g_x)+(ta+tb)[:, None]/2)
        fact = (tb-ta)/2*cdIdt
        resp_wanted[ind_a] += fact*np.sum(np.array(PP(logt)*g_w), axis=1)

    return resp_wanted
