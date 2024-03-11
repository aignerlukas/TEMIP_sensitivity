#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 14:50:15 2021

@author: laigner
"""

# %% import modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from matplotlib.backends.backend_pdf import PdfPages

from scipy.constants import mu_0
from scipy.constants import epsilon_0

from pygimli.viewer.mpl import drawModel1D


# %% function_lib
def prep_mdl_para_names(param_names, n_layers):
    mdl_para_names = []
    for pname in param_names:
        for n in range(0, n_layers):
            if 'thk' in pname and n == n_layers-1:
                break
            mdl_para_names.append(f'{pname}_{n:02d}')
    return mdl_para_names


def CCM(rho0, con0, con8, tau, c, f):
    """
    Cole and Cole Model

    Parameters
    ----------
    rho0 : float
        resistivity (Ohmm).
    con0 : float
        conductivity at baseline/zero frequency (mS/m ??).
    con8 : float
        conductivity at high/inf frequency (mS/m ??).
    tau : float
        relaxation time ().
    c : float (0 ->1)
        dispersion coefficient ().
    f : float ndarray
        frequency in Hz.

    Returns
    -------
    complex_con : ndarray with complex numbers
        complex conductivity array.

    """
    
    iotc = (2j*np.pi*f * tau)**c
    complex_con = (con8 + (con0-con8) / (1 + iotc))

    return complex_con


def PEM_res(rho0, m, tau, c, f):
    """
    Pelton et al. model
    in terms of resistivity

    Parameters
    ----------
    rho0 : float (Ohmm)
        DC or low frequency limit of resistivity.
    m : float ()
        chargeability (0-1).
    tau : float (s)
        relaxation time.
    c : float
        dispersion coefficient (0-1).
    f : np.array (float)
        frequencies at which the model should be evaluated.

    Returns
    -------
    complex_res : np.array
        complex resistivity at the given frequencies.

    """

    iotc = (2j*np.pi*f * tau)**c
    complex_res = (rho0 * (1 - m*(1 - 1/(1 + iotc))))

    return complex_res


def PEM_con(sig_inf, m, tau, c, f):
    """
    Pelton et al. model
    in terms of conductivity

    Parameters
    ----------
    sig_inf : float
        high frequency limit of the electrical conductivity.
    m : float ()
        chargeability (0-1).
    tau : float (s)
        relaxation time.
    c : float
        dispersion coefficient (0-1).
    f : np.array (float)
        frequencies at which the model should be evaluated.

    Returns
    -------
    complex_con : np.array
        complex conductivity at the given frequencies.

    """
    
    iotc = (2j*np.pi*f * tau)**c
    complex_con = sig_inf - sig_inf * (m / (1 + (iotc)*(1 - m)))
    
    return complex_con


def PEM_fia_con0(sig_0, m, tau_sig, c, f):
    """
    from fiandaca et al. (2018)
    Formula 3 (erroneous sign in manuscript version)

    Parameters
    ----------
    sig_0 : float
        DC conductivity.
    m : float (0 - 1)
        chargeability.
    tau : float
        relaxation time (s).
    c : float (0 - 1)
        dispersion coefficient.
    f : float, array-like, or single value
        frequency(ies) at which the complex conductivity should be calculated.

    Returns
    -------
    complex_con : complex, array-like, or single value, depending on f
        complex conductivity.

    """
    iotc = (2j*np.pi*f * tau_sig)**c
    complex_con = sig_0 * (1 + (m / (1 - m)) * (1 - (1 / (1 + (iotc))))) # correct version!!
    # complex_con = sig_0 * (1 - (m / (1 - m)) * (1 - (1 / (1 + (iotc)))))  wrong in paper!!
    
    return complex_con


def PEM_tar_con0(sig_0, m, tau, c, f):
    """
    from Tarasov and Titov (2013)
    equations 23, last version

    Parameters
    ----------
    sig_0 : float
        DC conductivity.
    m : float (0 - 1)
        chargeability.
    tau : float
        relaxation time (s).
    c : float (0 - 1)
        dispersion coefficient.
    f : float, array-like, or single value
        frequency(ies) at which the complex conductivity should be calculated.

    Returns
    -------
    complex_con : complex, array-like, or single value, depending on f
        complex conductivity.

    """
    iotc = (2j*np.pi*f * tau)**c
    complex_con = sig_0 * (1 + (m / ((1 - m)) * (1 - (1 / (1 + (iotc)*(1 - m))))))
    
    return complex_con


def CCM_K(con0, eps_s, eps_8, tau, c, f):

    io = 2j * np.pi * f  # i*omega --> from frequency to angular frequency
    iotc = (io * tau)**c
    complex_con = (con0 + (io * epsilon_0) * (eps_8 +
            ((eps_s - eps_8) / (1 + iotc)))
           )

    return complex_con


def get_omega_peak_PE(m, tau, c):
    # om_peak = (1 / (np.pi*tau)) * (1 / ((1-m)**(1/2*c)))
    om_peak = (1 / tau) * (1 / ((1-m)**(1/c)))
    # om_peak = (1 / tau) * (1 / ((1-m)**(1/2*c)))
    return om_peak



# %% MPA #############################################################################
def CC_MPA(rho_0, phi_max, tau_phi, c, f):
    """
    maximum phase angle model (after Fiandaca et al, 2018)
    Formula 8 - 11, appendix A.1 - A.08

    Parameters
    ----------
    rho_0 : float
        DC resistivity.
    phi_max : float
        maximum phase angle, peak value of the phase of complex res (rad).
    tau_phi : float
        relaxation time, specific for mpa model, see Formula 10 (s).
    c : float (0 - 1)
        dispersion coefficient.
    f : float, array-like, or single value
        frequency(ies) at which the complex conductivity should be calculated.

    Returns
    -------
    complex_res : complex, array-like, or single value, depending on f
        complex resistivity at the given frequencies.

    """
    m, tau_rho = get_m_taur_MPA(phi_max, tau_phi, c, verbose=True)
    iotc = (2j*np.pi*f * tau_rho)**c
    complex_res = rho_0 * (1 + (m / ((1 - m)) * (1 - (1 / (1 + (iotc)*(1 - m))))))
    
    return complex_res


def get_m_taur_MPA(phi_max, tau_phi, c, verbose=True):
    """
    function to obtain teh classical cc params from the mpa ones:
        uses an iterative approach and stops once the difference between
        two consecutive m values equals 0
    (after Fiandaca et al, 2018), appendix A.1 - A.08

    Parameters
    ----------
    phi_max : float
        maximum phase angle, peak value of the phase of complex res (rad).
    tau_phi : float
        relaxation time, specific for mpa model, see Formula 10 (s).
    c : float (0 - 1)
        dispersion coefficient.

    Raises
    ------
    ValueError
        in case the iteration doesn't converge after 100 iters.

    Returns
    -------
    m : float ()
        chargeability (0-1).
    tau_rho : float (s)
        relaxation time.

    """
    mns = []
    tau_rs = []
    areal = []
    bimag = []
    delta_ms = []
    
    n_iters = 1000
    th = 1e-9

    for n in range(0, n_iters):
        if n == 0:
            mns.append(0)
            tau_rs.append(mpa_get_tau_rho(m=mns[n],
                                          tau_phi=tau_phi,
                                          c=c))
            areal.append(mpa_get_a(tau_rs[n], tau_phi, c))
            bimag.append(mpa_get_b(tau_rs[n], tau_phi, c))
            mns.append(mpa_get_m(a=areal[n],
                                 b=bimag[n],
                                 phi_max=phi_max))
            delta_ms.append(mpa_get_deltam(mn=mns[n+1], mp=mns[n]))
        else:
            tau_rs.append(mpa_get_tau_rho(m=mns[n],
                                          tau_phi=tau_phi,
                                          c=c))
            areal.append(mpa_get_a(tau_rs[n], tau_phi, c))
            bimag.append(mpa_get_b(tau_rs[n], tau_phi, c))
            mns.append(mpa_get_m(a=areal[n],
                                 b=bimag[n],
                                 phi_max=phi_max))
            delta_ms.append(mpa_get_deltam(mn=mns[n+1], mp=mns[n]))
            print('delta_m: ', delta_ms[n])
            if delta_ms[n] <= th:  # stop if the difference is below 1e-9
                if verbose:
                    print(f'iteration converged after {n} iters')
                    print('solved m:', mns[-1])
                    print('solved tau_rho:', tau_rs[-1])
                    
                m = mns[-1]
                tau_rho = tau_rs[-1]
                break
    if delta_ms[n] > th:
        raise ValueError(f'the iterations did not converge after {n_iters} iterations, please check input!')
    return m, tau_rho


def get_tauphi_from_ts(m, tau_sig, c):
    """
    after fiandaca et al (2018), Formula 10
    obtain tau_phi from cole-cole conductivity formula (clasic CC)
    
    Parameters
    ----------
    m : float ()
        chargeability (0-1).
    tau_sig : float (s)
        relaxation time.
    c : float (0 - 1)
        dispersion coefficient.

    Returns
    -------
    tau_phi : float
        relaxation time, specific for mpa model, see Formula 10 (s).

    """
    tau_phi = tau_sig * (1 - m)**(-1/(2*c))
    return tau_phi


def get_tauphi_from_tr(m, tau_rho, c):
    """
    after fiandaca et al (2018), Formula 10
    obtain tau_phi from cole-cole resisitvity formula (PEM)

    Parameters
    ----------
    m : float ()
        chargeability (0-1).
    tau_rho : float (s)
        relaxation time.
    c : float (0 - 1)
        dispersion coefficient.

    Returns
    -------
    tau_phi : float
        relaxation time, specific for mpa model, see Formula 10 (s).

    """
    tau_phi = tau_rho * (1 - m)**(1/(2*c))
    return tau_phi


def get_phimax_from_CCC(sig_0, m, tau_sig, c):
    """
    after fiandaca et al (2018), Formula 9
    obtain phi_max from cole-cole resisitvity formula (PEM)

    Parameters
    ----------
    sig_0 : float
        DC conductivity.
    m : float ()
        chargeability (0-1).
    tau_sig : float (s)
        relaxation time.
    c : float (0 - 1)
        dispersion coefficient.

    Returns
    -------
    phi_max : float
        maximum phase angle, peak value of the phase of complex res (rad).

    """
    tau_phi = get_tauphi_from_ts(m, tau_sig, c)
    cmplx_con = PEM_fia_con0(sig_0=sig_0, m=m, tau_sig=tau_sig, 
                             c=c, f=1 / (2*np.pi*tau_phi))
    phi_max = np.arctan(np.imag(cmplx_con) / np.real(cmplx_con))
    return phi_max


def get_phimax_from_CCR(rho_0, m, tau_rho, c):
    """
    

    Parameters
    ----------
    rho_0 : float
        DC resistivity.
    m : float ()
        chargeability (0-1).
    tau_rho : float (s)
        relaxation time.
    c : float (0 - 1)
        dispersion coefficient.

    Returns
    -------
    phi_max : float
        maximum phase angle, peak value of the phase of complex res (rad).

    """
    tau_phi = get_tauphi_from_tr(m, tau_rho, c)
    cmplx_res = PEM_res(rho0=rho_0, m=m, tau=tau_rho,
                        c=c, f=1 / (2*np.pi*tau_phi))
    phi_max = -np.arctan(np.imag(cmplx_res) / np.real(cmplx_res))
    return phi_max


def mpa_get_deltam(mn, mp):
    """
    after Fiandaca et al. (2018), Appendix A.04

    Parameters
    ----------
    mn : float
        m of current iteration.
    mp : TYPE
        m of previous iteration.

    Returns
    -------
    float
        delta_m, difference between current and previous m.

    """
    return np.abs(mn - mp) / mn


def mpa_get_tau_rho(m, tau_phi, c):
    """
    after Fiandaca et al. (2018), Appendix A.05
    needs |1 - m|, otherwise values of m > 1 will result in nan!!

    Parameters
    ----------
    m : float ()
        chargeability (0-1).
    tau_phi : float
        relaxation time, specific for mpa model, see Formula 10 (s).
    c : float (0 - 1)
        dispersion coefficient.

    Returns
    -------
    tau_rho : float (s)
        relaxation time.

    """
    tau_rho = tau_phi * (abs(1 - m)**(-1/(2*c)))  # abs is essential here
    return tau_rho


def mpa_get_a(tau_rho, tau_phi, c):
    """
    after Fiandaca et al. (2018), Appendix A.06

    Parameters
    ----------
    tau_rho : float (s)
        relaxation time.
    tau_phi : float
        relaxation time, specific for mpa model, see Formula 10 (s).
    c : float (0 - 1)
        dispersion coefficient.

    Returns
    -------
    a : float
        real part of complex variable.

    """
    a = np.real(1 / (1 + (1j*(tau_rho / tau_phi))**c))
    return a


def mpa_get_b(tau_rho, tau_phi, c):
    """
    after Fiandaca et al. (2018), Appendix A.07

    Parameters
    ----------
    tau_rho : float (s)
        relaxation time.
    tau_phi : float
        relaxation time, specific for mpa model, see Formula 10 (s).
    c : float (0 - 1)
        dispersion coefficient.

    Returns
    -------
    b : float
        imaginary part of complex variable.

    """
    b = np.imag(1 / (1 + (1j*(tau_rho / tau_phi))**c))
    return b


def mpa_get_m(a, b, phi_max):
    """
    after Fiandaca et al. (2018), Appendix A.08

    Parameters
    ----------
    a : float
        real part of complex variable. see mpa_get_a
    b : float
        imaginary part of complex variable. see mpa_get_b
    phi_max : float
        maximum phase angle, peak value of the phase of complex res (rad).

    Returns
    -------
    m : float ()
        chargeability (0-1).

    """
    tan_phi = np.tan(-phi_max)
    m = tan_phi / ((1 - a) * tan_phi + b)
    return m



# %% general stuff
def mdl2steps(mdl, extend_bot=25, cum_thk=True):
    """
    function to re-order a thk, res model for plotting

    Parameters
    ----------
    mdl : np.array
        thk, res model.
    extend_bot : float (m), optional
        thk which will be added to the bottom layer for plotting.
        The default is 25 m.
    cum_thk : bool, optional
        switch to decide whether or not to calculate the
        cumulative thickness (i.e depth).
        The default is True.

    Returns
    -------
    mdl_cum : np.array (n x 2)
        thk,res model.
    model2plot : np.array (n x 2)
        step model for plotting.

    """
    mdl_cum = mdl.copy()
    
    if cum_thk:
        cum_thk = np.cumsum(mdl[:,0]).copy()
        mdl_cum[:,0] = cum_thk  # calc cumulative thickness for plot

    mdl_cum[-1, 0] = mdl_cum[-2, 0] + extend_bot

    thk = np.r_[0, mdl_cum[:,0].repeat(2, 0)[:-1]]
    model2plot = np.column_stack((thk, mdl_cum[:,1:].repeat(2, 0)))
    
    return mdl_cum, model2plot


def parse_TEMfastFile(filename, path):
    """
    read a .tem file and return as a pd.dataframe !!Convert first!!

    Parameters
    ----------
    filename : string
        name of file including extension.
    path : string
        path to file - either absolute or relative.

    Returns
    -------
    rawData : pd.DataFrame
        contains all data in one file.
    nLogs : int
        number of soundings in the .tem data file.
    indices_hdr : pd.DataFrame
        Contains the indices where the header block of 
        each sounding starts and ends.
    indices_dat : pd.DataFrame
        Contains the indices where the data block of 
        each sounding starts and ends..

    """
    headerLines = 8
    # properties = generate_props('TEMfast')

    # fName = filename[:-4] if filename[-4:] == '_txt' else filename
    fin = path + '/' + filename

    # Start of file reading
    myCols = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
    rawData = pd.read_csv(fin,
                          names=myCols,
                          sep='\\t',
                          engine="python")
    rawData = rawData[~pd.isnull(rawData).all(1)].fillna('')
    lengthData = len(rawData.c1)

    # create start and end indices of header and data lines
    start_hdr = np.asarray(np.where(rawData.loc[:]['c1'] ==
                                    'TEM-FAST 48 HPC/S2  Date:'))
    nLogs = np.size(start_hdr)
    start_hdr = np.reshape(start_hdr, (np.size(start_hdr),))
    end_hdr = start_hdr + headerLines

    start_dat = end_hdr
    end_dat = np.copy(start_hdr)
    end_dat = np.delete(end_dat, 0)
    end_dat = np.append(end_dat, lengthData)

    # create new dataframe which contains all indices
    indices_hdr = pd.DataFrame({'start': start_hdr, 'end': end_hdr},
                               columns=['start', 'end'])
    indices_dat = pd.DataFrame({'start': start_dat, 'end': end_dat},
                               columns=['start', 'end'])

    return rawData, nLogs, indices_hdr, indices_dat


def simulate_error(relerr, abserr, data):
    np.random.seed(42)
    rndm = np.random.randn(len(data))

    rand_error_abs = (relerr * np.abs(data) + 
                  abserr) * rndm

    return rand_error_abs


def calc_rhoa(setup_device, signal, times):
    """
    
    Function that calculates the apparent resistivity of a TEM sounding
    using equation from Christiansen et al (2006)

    Parameters
    ----------
    forward : instance of forward class
        instance of wrapping class for TEM inductive loop measurements.
    signal : np.array
        signal in V/m².

    Returns
    -------
    rhoa : np.array
        apparent resistivity.

    """
    sub0 = (signal <= 0)
    turns = 1
    M = (setup_device['current_inj'] *
         setup_device['txloop']**2 * turns)
    rhoa = ((1 / np.pi) *
            (M / (20 * (abs(signal))))**(2/3) *
            (mu_0 / (times))**(5/3))
    rhoa[sub0] = rhoa[sub0]*-1
    return rhoa



def plot_signal(axis, time, signal, sub0color='aqua', sub0marker='s', sub0label=None, **kwargs):
    sub0 = (signal <= 0)
    sub0_sig = signal[sub0]
    sub0_time = time[sub0]
    
    line, = axis.loglog(time, abs(signal), **kwargs)
    if any(sub0):
        if sub0label is not None:
            line_sub0, = axis.loglog(sub0_time, abs(sub0_sig), 
                                     marker=sub0marker, ls='none',
                                     mfc='none', markersize=6,
                                     mew=1.2, mec=sub0color,
                                     label=sub0label)
        else:
            line_sub0, = axis.loglog(sub0_time, abs(sub0_sig), 
                                     marker=sub0marker, ls='none',
                                     mfc='none', markersize=6,
                                     mew=1.2, mec=sub0color)
        return axis, line, line_sub0
    else:
        return axis, line


def plot_rhoa(axis, time, rhoa, sub0color='k', **kwargs):
    sub0 = (rhoa <= 0)
    sub0_rhoa = rhoa[sub0]
    sub0_time = time[sub0]
    
    line, = axis.loglog(time, abs(rhoa), **kwargs)
    if any(sub0):
        line_sub0, = axis.loglog(sub0_time, abs(sub0_rhoa), 's',
                                 mfc='none', ms=6,
                                 mew=1.2, mec=sub0color,
                                 label='sub0 vals')
        return axis, line, line_sub0
    else:
        return axis, line


def get_diffs(response, measured):
    diffs = abs(response - measured)
    diffs_rel = abs((diffs / response) * 100)
    return diffs, diffs_rel


def plot_diffs(ax, times, response, measured, relative=True, max_diff=30):
    diffs, diffs_rel = get_diffs(response, measured)
    
    axt = ax.twinx()
    if relative == True:
        axt.plot(times, diffs_rel, '.:', color='gray', zorder=0, label='rel. diffs')
        axt.set_ylabel('diff resp-data (%)')
        axt.grid(False)
        axt.set_ylim((0, max_diff))
    else:
        axt.plot(times, diffs, '.:', color='gray', zorder=0, label='diffs')
        axt.set_ylabel('diff resp-data ()')
        axt.grid(False)
        axt.set_ylim((0, max_diff))
    return axt


def plot_pem_stepmodel(axes, model2d, depth_limit=(40, 0), **kwargs):
    thickness = model2d[:, 0]
    rho0 = model2d[:, 1]
    chargeability = model2d[:, 2]
    relaxtime = model2d[:, 3]
    dispersion = model2d[:, 4]

    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(9, 9), constrained_layout=True)
        axes = axes.flatten()
    else:
        if len(axes.shape) != 1:  #check if there are more than one dimensions
            if axes.shape[1] > 1:  # check if second dimension is larger than 1
                axes = axes.flatten()
        fig = axes[0].get_figure()

    drawModel1D(axes[0], thickness, rho0, **kwargs)
    # axes[0].legend()
    axes[0].set_ylim(depth_limit)
    axes[0].set_xlabel(r'$\rho$ ($\Omega$m)')
    axes[0].set_ylabel('z (m)')
    axes[0].set_xscale('log')
    # axes[0].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    drawModel1D(axes[1], thickness, chargeability, **kwargs)
    # axes[1].legend()
    axes[1].set_ylim(depth_limit)
    axes[1].set_xlabel(r'chargeability m ()')
    # axes[1].set_ylabel('z (m)')

    drawModel1D(axes[2], thickness, relaxtime, **kwargs)
    # axes[2].legend()
    axes[2].set_ylim(depth_limit)
    axes[2].set_xlabel(r'rel. time (s)')
    # axes[2].set_ylabel('z (m)')
    axes[2].set_xscale('log')
    # axes[2].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    drawModel1D(axes[3], thickness, dispersion, **kwargs)
    axes[3].legend()
    axes[3].set_ylim(depth_limit)
    axes[3].set_xlabel(r'disp. coefficient c ()')
    axes[3].set_ylabel('z (m)')
    axes[3].yaxis.tick_right()
    axes[3].yaxis.set_label_position('right')
    
    # plt.suptitle('pelton model', fontsize=16)
    
    return fig, axes


def plot_mpa_stepmodel(axes, model2d, depth_limit=(40, 0), **kwargs):
    thickness = model2d[:, 0]
    rho0 = model2d[:, 1]
    max_phase_angle = model2d[:, 2]
    tau_phi = model2d[:, 3]
    dispersion = model2d[:, 4]
    
    if axes is None:
        fig, axes = plt.subplots(2, 2, figsize=(9, 9), constrained_layout=True)
        axes = axes.flatten()
    else:
        if len(axes.shape) != 1:  #check if there are more than one dimensions
            if axes.shape[1] > 1:  # check if second dimension is larger than 1
                axes = axes.flatten()
        fig = axes[0].get_figure()

    drawModel1D(axes[0], thickness, rho0, **kwargs)
    # axes[0].legend()
    axes[0].set_ylim(depth_limit)
    axes[0].set_xlabel(r'$\rho$ ($\Omega$m)')
    axes[0].set_ylabel('z (m)')
    axes[0].set_xscale('log')
    # axes[0].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    drawModel1D(axes[1], thickness, max_phase_angle, **kwargs)
    # axes[1].legend()
    axes[1].set_ylim(depth_limit)
    axes[1].set_xlabel(r'mpa $\phi_{max}$ (rad)')
    axes[1].set_ylabel('')

    drawModel1D(axes[2], thickness, tau_phi, **kwargs)
    # axes[2].legend()
    axes[2].set_ylim(depth_limit)
    axes[2].set_xlabel(r'rel. time $\tau_{\phi}$ (s)')
    axes[2].set_ylabel('')
    axes[2].set_xscale('log')
    # axes[2].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    drawModel1D(axes[3], thickness, dispersion, **kwargs)
    axes[3].legend(title='Models:', title_fontsize=14)
    axes[3].set_ylim(depth_limit)
    axes[3].set_xlabel(r'disp. coefficient c ()')
    axes[3].set_ylabel('z (m)')
    axes[3].yaxis.tick_right()
    axes[3].yaxis.set_label_position('right')


    # drawModel1D(axes[0], thickness, rho0, **kwargs)
    # axes[0].legend()
    # axes[0].set_ylim((40, 0))
    # axes[0].set_xlabel(r'$\rho (\Omega m)$')
    # axes[0].set_ylabel('z (m)')

    # drawModel1D(axes[1], thickness, max_phase_angle, **kwargs)
    # axes[1].legend()
    # axes[1].set_ylim((40, 0))
    # axes[1].set_xlabel(r'max. phase angle $\phi_{max}$ (rad)')
    # axes[1].set_ylabel('z (m)')

    # drawModel1D(axes[2], thickness, tau_phi, **kwargs)
    # axes[2].legend()
    # axes[2].set_ylim((40, 0))
    # axes[2].set_xlabel(r'rel. time $\tau_{\phi}$ (s)')
    # axes[2].set_ylabel('z (m)')
    # axes[2].set_xscale('log')
    # axes[2].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    # drawModel1D(axes[3], thickness, dispersion, **kwargs)
    # axes[3].legend()
    # axes[3].set_ylim((40, 0))
    # axes[3].set_xlabel(r'disp. coefficient c ()')
    # axes[3].set_ylabel('z (m)')

    # plt.suptitle('maximum phase angle model', fontsize=16)
    # axes[2].set_title('maximum phase angle (mpa) model')
    
    return fig, axes



# %% converting
def mtrxMdl2vec(mtrx):
    """
    reshapes a multicolumn model to a 1D vector using the structure
    as required by bel1d

    Parameters
    ----------
    mtrx : 2D - np.array
        array containing parameter values in the rows and different params in columns.
        uses thk of each individual layer in such structure that:
            thk_lay_0,     param1_lay_0,    param2_lay_0,   ....  param_n_lay_0
            thk_lay_1,     param1_lay_1,    param2_lay_1,   ....  param_n_lay_1
            .              .                .               ....  .            
            .              .                .               ....  .            
            thk_lay_n-1,   param1_lay_n-1,  param2_lay_n-1, .... param_n_lay_n-1
            0,             param1_lay_n,    param2_lay_n,   .... param_n-1_lay_n
         

    Returns
    -------
    mtrx_1D : np.array (1D)
        1D array (or vector) containing the same info as mtrx
        but reshaped to:
            thk_lay_0
            thk_lay_1
            .
            .
            thk_lay_n-1 
            param1_lay_0
            param1_lay_1
            .
            .
            param1_lay_n
            .
            .
            .
            param_n_lay_0
            param_n_lay_1
            .
            .
            param_n_lay_n
    """
    nLayers = mtrx.shape[0]
    nParams = mtrx.shape[1]
    for par in range(nParams):
        if par == 0:
            mtrx_1D = mtrx[:-1,0]
        else:
            mtrx_1D = np.hstack((mtrx_1D, mtrx[:,par]))
    return mtrx_1D


def vectorMDL2mtrx(model, nLayer, nParam):
    """
    function to reshape a 1D bel1d style model to a n-D model containing as
    many rows as layers and as many columns as parameters (thk + nParams)

    Parameters
    ----------
    model : np.array
        bel1D vector model:
            thk_lay_0
            thk_lay_1
            .
            .
            thk_lay_n-1 
            param1_lay_0
            param1_lay_1
            .
            .
            param1_lay_n
            .
            .
            .
            param_n_lay_0
            param_n_lay_1
            .
            .
            param_n_lay_n
    nLayer : int
        number of layers in the model.
    nParam : int
        number of parameters in the model, thk also counts!!

    Returns
    -------
    model : np.array
        n-D array with the model params.

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


def plot_ip_model(axis, ip_model, layer_ip,
                  ip_modeltype='pelton',
                  rho2log=False, **kwargs):
    
    # plot resistivity model
    # depth = np.r_[0, pe_ip_model[:-1,0]]
    # rA_model = reArr_mdl(np.column_stack((depth, pe_ip_model[:,1])))
    # TODO add extend bottom!!!
    
    _, rA_model = mdl2steps(ip_model, extend_bot=15, cum_thk=True)
    
    axis.plot(rA_model[:,1], rA_model[:,0], '-k.')
    
    if ip_modeltype == 'cole_cole':  # "$\\rho_\infty$" + u"\u221e" + " = {:.3f}\n" + \

        coleparams = ("$\sigma_0$ = {:.2e} S/m\n" +
                      "$\sigma_\infty$ = {:.2e} S/m\n" +
                      "$\\tau$ = {:.2e} s\n" +  # TODO fix formatting, maybe better scientific notation??
                      "c = {:.3f}").format(ip_model[layer_ip, 2],
                                           ip_model[layer_ip, 3],
                                           ip_model[layer_ip, 4],
                                           ip_model[layer_ip, 5])

    elif ip_modeltype == 'pelton':
        coleparams = ("m = {:.3f}\n" + \
                      "$\\tau$ = {:.2e} s\n" + \
                      "c = {:.3f}").format(ip_model[layer_ip, 2],
                                           ip_model[layer_ip, 3],
                                           ip_model[layer_ip, 4],)

    elif ip_modeltype == 'cc_kozhe':  # con_0, eps_0, eps_8, taus, cs)

        coleparams = ("$\sigma_0$ = {:.2e}\n" +
                      "$\epsilon_s$ = {:.1f}\n" +
                      "$\epsilon_\infty$ = {:.1f}\n" +
                      "$\\tau$ = {:.2e} s\n" +  # TODO fix formatting, maybe better scientific notation??
                      "c = {:.3f}").format(ip_model[layer_ip, 2],
                                           ip_model[layer_ip, 3],
                                           ip_model[layer_ip, 4],
                                           ip_model[layer_ip, 5],
                                           ip_model[layer_ip, 6])
        pass

    else:
        raise TypeError('Requested IP model is not implemented.')

    # xtxt = min(rA_model[:,1]) * 2
    xtxt = rA_model[layer_ip, 1] * 0.95
    ytxt = (rA_model[layer_ip, 0] + rA_model[layer_ip+2, 0]) / 2 # intermediate height between upper and lower layer
    
    axis.text(xtxt, ytxt, 
              coleparams, 
              {'color': 'C2', 'fontsize': 12}, 
              va="center", ha="center", # correspond to both alignment and position?!?!? TODO understand
              transform=axis.transData)
    
    if rho2log:
        axis.set_xscale('log')

    axis.set_xlabel(r'$\rho$ ($\Omega$m)')
    axis.set_ylabel('h (m)')
    
    axis.grid(which='major', alpha=0.75, ls='-')
    axis.grid(which='minor', alpha=0.75, ls=':')
    
    axis.legend(['Model'])

    return axis, coleparams


# %% general stuff
def multipage(filename, figs=None, dpi=200):
    """
    function to save all plots to multipage pdf
    https://stackoverflow.com/questions/26368876/saving-all-open-matplotlib-figures-in-one-file-at-once

    Parameters
    ----------
    filename : string
        name of the the desired pdf file including the path and file extension.
    figs : list, optional
        list of instances of figure objects. If none automatically retrieves currently opened figs.
        The default is None.
    dpi : int, optional
        dpi of saved figures. The default is 200.

    Returns
    -------
    None.

    """
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf', dpi=dpi)
    pp.close()


def query_yes_no(question, default='no'):
    """
    yes no query for terminal usage
    from: https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    
    Parameters
    ----------
    question : string
        query to ask the user.
    default : string, optional
        default answer. The default is 'no'.

    Raises
    ------
    ValueError
        if the expected variations of yes/no are not in the answer...

    Returns
    -------
    none

    """
    from distutils.util import strtobool
    if default is None:
        prompt = " (y/n)? "
    elif default == 'yes':
        prompt = " ([y]/n)? "
    elif default == 'no':
        prompt = " (y/[n])? "
    else:
        raise ValueError(f"Unknown setting '{default}' for default.")

    while True:
        try:
            resp = input(question + prompt).strip().lower()
            if default is not None and resp == '':
                return default == 'yes'
            else:
                return strtobool(resp)
        except ValueError:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


def get_temfast_date():
    """
    get current date and time and return temfast date string.
    eg. Thu Dec 30 09:34:11 2021

    Returns
    -------
    temf_datestr : str
        current date including name of day, month and adding year at the end.

    """
    import datetime
    tdy = datetime.datetime.today()
    time_fmt = ('{:02d}:{:02d}:{:02d}'.format(tdy.hour,
                                              tdy.minute,
                                              tdy.second))
    temf_datestr = ('{:s} '.format(tdy.strftime('%a')) +  # uppercase for long name of day
                    '{:s} '.format(tdy.strftime('%b')) +  # uppercase for long name of month
                    '{:d} '.format(tdy.day) + 
                    '{:s} '.format(time_fmt) + 
                    '{:d}'.format(tdy.year))
    return temf_datestr


def save_as_tem(path_data, template_fid,
                model_name, metadata, frwrd,
                times, signal, error, rhoa,
                save_as_internal=True):
    
    myCols = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
    template = pd.read_csv(template_fid, names=myCols,
                           sep='\\t', engine="python")
    
    tf_date = get_temfast_date()
    template.iat[0,1] = tf_date                          # set date
    template.iat[1,1] = f'ERT at {metadata["location"]} m'             # set location
    template.iat[2,1] = metadata["snd_name"]
    
    template.iat[3,1] = f'{frwrd.setup_device["timekey"]}'
    template.iat[3,4] = 'ramp={:.2f} us'.format(frwrd.properties_snd['rampoff']*1e6)
    template.iat[3,5] = 'I={:.1f} A'.format(frwrd.properties_snd['current_inj'])
    template.iat[3,6] = 'FILTR={:d} Hz'.format(frwrd.setup_device['filter_powerline'])
    
    template.iat[4,1] = '{:.3f}'.format(frwrd.setup_device['txloop'])
    template.iat[4,3] = '{:.3f}'.format(frwrd.setup_device['rxloop'])
    
    template.iat[5,1] = metadata["comments"]
    
    template.iat[6,1] = '{:+.3f}'.format(metadata["x"])  # x
    template.iat[6,3] = '{:+.3f}'.format(metadata["y"])       # y
    template.iat[6,5] = '{:+.3f}'.format(metadata["z"])       # z
    
    template.iat[7,1] = 'Time[us]'
    
    chnls_act = np.arange(1, len(times)+1)
    data_norm = signal * (frwrd.setup_device['txloop']**2) / frwrd.setup_device['current_inj']
    err_norm = error * (frwrd.setup_device['txloop']**2) / frwrd.setup_device['current_inj']
    
    # clear data first:
    chnls_id = len(times) + 8
    template.iloc[8:, :] = np.nan
    
    # add new data
    template.iloc[8:chnls_id, 0] = chnls_act
    template.iloc[8:chnls_id, 1] = times*1e6  # to us
    template.iloc[8:chnls_id, 2] = data_norm
    template.iloc[8:chnls_id, 3] = abs(err_norm)

    if save_as_internal:
        template.iloc[8:chnls_id, 4] = rhoa
        exp_fmt = '%d\t%.2f\t%.5e\t%.5e\t%.2f'
        data_fid = path_data + f'{model_name}.tem'
    else:
        exp_fmt = '%d\t%.2f\t%.5e\t%.5e'
        data_fid = path_data + f'{model_name}.tem'
    
    # write to file
    print('saving data to: ', data_fid)
    header = template.iloc[:8, :]
    header.to_csv(data_fid, header=None,
                    index=None, sep='\t', mode='w')
    
    data4exp = np.asarray(template.iloc[8:, :], dtype=np.float64)
    data4exp = data4exp[~np.isnan(data4exp).all(axis=1)]
    data4exp = data4exp[:, ~np.isnan(data4exp).all(axis=0)]
    
    with open(data_fid, 'a') as fid:
        np.savetxt(fid, X=data4exp,
                   header='', comments='',
                   delimiter='\t', fmt=exp_fmt)
    with open(data_fid) as file: # remove trailing spaces of each line
        lines = file.readlines()
        lines_clean = [l.strip() for l in lines if l.strip()]
    with open(data_fid, "w") as f:
        f.writelines('\n'.join(lines_clean))
        
    return template