#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 14:50:15 2021

library for tools associated with the IP effect and complex resistivity

@author: laigner
"""

# %% import modules
import logging

import numpy as np
import matplotlib.pyplot as plt

from scipy.constants import epsilon_0


# %% logging ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.WARNING)
# logger.setLevel(logging.DEBUG)


# %% function_lib
def prep_mdl_para_names(param_names, n_layers):
    """
    Function to add layer numbering info to model parameter names
    if thk is in the list of model parameter names, it will be only used
    n_layer - 1 times.

    Parameters
    ----------
    param_names : list
        list with the names of the model parameters.
    n_layers : int
        number of layers in the model.

    Returns
    -------
    mdl_para_names : list
        contains mdl parameter names for each layer, layer id written with 
        up to two leading zeros.

    """
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
    #complex_res = rho_0 * (1 + (m / ((1 - m)) * (1 - (1 / (1 + (iotc)*(1 - m))))))
    complex_res = (rho_0 * (1 - m*(1 - 1/(1 + iotc))))

    return complex_res


def get_m_taur_MPA(phi_max, tau_phi, c, verbose=True, backend='loop',
                   max_iters=42, threshhold=1e-7):
    """
    function to obtain the classical cc params from the mpa ones:
        uses an iterative approach and stops once the difference between
        two consecutive m values equals 0
    (after Fiandaca et al, 2018), appendix A.1 - A.08

    Parameters
    ----------
    phi_max : float, numpy.ndarray
        maximum phase angle, peak value of the phase of complex res (rad).
    tau_phi : float, numpy.ndarray
        relaxation time, specific for mpa model, see Formula 10 (s).
    c : float (0 - 1), numpy.ndarray
        dispersion coefficient.
    verbose : boolean, default is True
        suppress output?
    backend : str, default is loop
        only for array-like inputs, whether to loop or
    max_iters : int, default is 42
        maximum number of iterations
    threshhold : float, default is 1e-7
        difference between consecutive iterations at which the solution
        will be accepted

    Raises
    ------
    ValueError
        in case the iteration doesn't converge after max_iters iters.

    Returns
    -------
    m : float ()
        chargeability (0-1).
    tau_rho : float (s)
        relaxation time.

    """

    th = threshhold

    m0s = []
    tau_rs = []
    areal = []
    bimag = []
    delta_ms = []

    if isinstance(phi_max, np.ndarray) and isinstance(tau_phi, np.ndarray) and isinstance(c, np.ndarray):
        if verbose:
            logger.info('input MPA model params are all numpy array, converting all of them . . .')

        if backend == 'loop':
            phi_max_sub = np.copy(phi_max)
            tau_phi_sub = np.copy(tau_phi)
            c_sub = np.copy(c)

            if any(phi_max == 0):
                logger.info('encountered phi_max == 0, assuming no-IP effect, setting m also to 0')
                mask = phi_max == 0
                phi_max_sub = phi_max[~mask]
                tau_phi_sub = tau_phi_sub[~mask]
                c_sub = c_sub[~mask]

            m, tau_rho = np.zeros_like(phi_max), np.zeros_like(tau_phi)
            ms, tau_rhos = np.zeros_like(phi_max_sub), np.zeros_like(tau_phi_sub)

            for i, phimax in enumerate(phi_max_sub):
                mns = []
                tau_rs = []
                areal = []
                bimag = []
                delta_ms = []

                for n in range(0, max_iters):
                    if n == 0:
                        mns.append(0)

                    tau_rs.append(mpa_get_tau_rho(m=mns[n],
                                                  tau_phi=tau_phi_sub[i],
                                                  c=c_sub[i]))
                    areal.append(mpa_get_a(tau_rs[n], tau_phi_sub[i], c_sub[i]))
                    bimag.append(mpa_get_b(tau_rs[n], tau_phi_sub[i], c_sub[i]))
                    mns.append(mpa_get_m(a=areal[n],
                                          b=bimag[n],
                                          phi_max=phimax))
                    delta_ms.append(mpa_get_deltam(mc=mns[n+1], mp=mns[n]))

                    logger.info('delta_m: ', delta_ms[n])

                    if delta_ms[n] <= th:  # stop if the difference is below th
                        if verbose:
                            logger.info(f'iteration converged after {n} iters')
                            logger.info('solved m:', mns[n])
                            logger.info('solved tau_rho:', tau_rs[n])
                        ms[i] = mns[n]
                        tau_rhos[i] = tau_rs[n]
                        break

            if any(phi_max == 0):
                m[~mask] = ms
                tau_rho[~mask] = tau_rhos
                
                tau_rho[mask] = mpa_get_tau_rho(m=0,
                                                tau_phi=tau_phi[mask],
                                                c=c[mask])
            else:
                m = ms
                tau_rho = tau_rhos

        elif backend == 'vectorized':
            # TODO
            raise ValueError('not yet implemented . . .')
            pass

        else:
            raise ValueError("Please select either 'loop' or 'vectorized' for the backend kwarg")

    elif isinstance(phi_max, float) and isinstance(tau_phi, float) and isinstance(c, float):
        if verbose:
            print('input MPA model params are all single floats . . .')

        if phi_max == 0:
            m = 0
            tau_rho = mpa_get_tau_rho(m=m, tau_phi=tau_phi, c=c)

        elif (phi_max > 0):
            for n in range(0, max_iters):
                if n == 0:
                    m0s.append(0)

                tau_rs.append(mpa_get_tau_rho(m=m0s[n],
                                              tau_phi=tau_phi,
                                              c=c))
                areal.append(mpa_get_a(tau_rs[n], tau_phi, c))
                bimag.append(mpa_get_b(tau_rs[n], tau_phi, c))
                m0s.append(mpa_get_m(a=areal[n],
                                      b=bimag[n],
                                      phi_max=phi_max))
                delta_ms.append(mpa_get_deltam(mc=m0s[n+1], mp=m0s[n]))

                if verbose:
                    logger.info('delta_m: ', delta_ms[n])

                if delta_ms[n] <= th:  # stop if the difference is below the th
                    if verbose:
                        logger.info(f'iteration converged after {n} iters')
                        logger.info('solved m:', m0s[n])
                        logger.info('solved tau_rho:', tau_rs[n])

                    m = m0s[n]
                    tau_rho = tau_rs[n]
                    break

            if delta_ms[n] > th:
                raise ValueError(f'the iterations did not converge after {max_iters} iterations, please check input!')

    else:
        raise ValueError('imputs have to be all floats or all numpy arrays')

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
    Function to get the phi_may (i.e., the maximum phase angle in rad) from the
    Pelton resistivity model.

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


def mpa_get_deltam(mc, mp):
    """
    after Fiandaca et al. (2018), Appendix A.04

    Parameters
    ----------
    mc : float
        m of current iteration.
    mp : TYPE
        m of previous iteration.

    Returns
    -------
    float
        delta_m, difference between current and previous m.

    """
    return np.abs(mc - mp) / mc


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


# %% models for empymod
def cole_cole(inp, p_dict):
    """Cole and Cole (1941).
    code from: https://empymod.emsig.xyz/en/stable/examples/time_domain/cole_cole_ip.html#sphx-glr-examples-time-domain-cole-cole-ip-py
    """

    # Compute complex conductivity from Cole-Cole
    iotc = np.outer(2j*np.pi*p_dict['freq'], inp['tau'])**inp['c']
    condH = inp['cond_8'] + (inp['cond_0']-inp['cond_8']) / (1 + iotc)
    condV = condH/p_dict['aniso']**2

    # Add electric permittivity contribution
    etaH = condH + 1j*p_dict['etaH'].imag
    etaV = condV + 1j*p_dict['etaV'].imag

    return etaH, etaV



def pelton_res(inp, p_dict):
    """ Pelton et al. (1978).
    code from: https://empymod.emsig.xyz/en/stable/examples/time_domain/cole_cole_ip.html#sphx-glr-examples-time-domain-cole-cole-ip-py
    """

    # Compute complex resistivity from Pelton et al.
    # print('\n   shape: p_dict["freq"]\n', p_dict['freq'].shape)
    iotc = np.outer(2j*np.pi*p_dict['freq'], inp['tau'])**inp['c']

    rhoH = inp['rho_0'] * (1 - inp['m']*(1 - 1/(1 + iotc)))
    rhoV = rhoH*p_dict['aniso']**2

    # Add electric permittivity contribution
    etaH = 1/rhoH + 1j*p_dict['etaH'].imag
    etaV = 1/rhoV + 1j*p_dict['etaV'].imag

    return etaH, etaV


def mpa_model(inp, p_dict):
    """
    maximum phase angle model (Fiandaca et al 2018)
    Formula 8 - 11, appendix A.1 - A.08

    Parameters
    ----------
    inp : dictionary
        dictionary containing the cole-cole parameters:
            'rho_0' - DC resistivity
            'phi_max' - maximum phase angle, peak value of the phase of complex res (rad).
            'tau_phi' - relaxation time, specific for mpa model, see Formula 10 (s).
            'c' - dispersion coefficient
    p_dict : dictionary
        additional dictionary with empymod specific parameters.

    Returns
    -------
    etaH, etaV : numpy array

    """

    # obtain chargeability and tau from mpa model
    m, tau_rho = get_m_taur_MPA(inp['phi_max'], inp['tau_phi'], inp['c'], verbose=True)

    iotc = np.outer(2j*np.pi*p_dict['freq'], tau_rho)**inp['c']

    # Compute complex resistivity using the pelton resistivity model
    rhoH = inp['rho_0'] * (1 - m*(1 - 1/(1 + iotc)))
    rhoV = rhoH*p_dict['aniso']**2

    # Add electric permittivity contribution
    etaH = 1/rhoH + 1j*p_dict['etaH'].imag
    etaV = 1/rhoV + 1j*p_dict['etaV'].imag

    return etaH, etaV


# %% plotting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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


def plot_ip_model(axis, ip_model, layer_ip,
                  ip_modeltype='pelton',
                  rho2log=False, **kwargs):
    """
    funciton to plot an TEM-IP inversion result as a step model
    only intended for models where a single layer has IP effect.
    plots into a single subplot

    Parameters
    ----------
    axis : matplotlib axis object
        axis into which you would like to plot. only one subplot needed
    ip_model : np.ndarray
        array containing the model that will be plotted.
    layer_ip : int
        id of the layer that has the IP effect.
    ip_modeltype : str, optional
        name of the IP model. The default is 'pelton'. also available 'mpa', 'cole_cole'
    rho2log : boolean, optional
        switch to plot resistivity logarithmic. The default is False.
    **kwargs : key-word arguments
        for the plt.plot method.

    Raises
    ------
    TypeError
        If an ip_modeltype is not available. Please select either:
            'mpa', 'cole_cole', or 'pelton'

    Returns
    -------
    axis : matplotlib axis object
        axis into which the model was visualized.
    coleparams : tuple
        contains the names of each IP (cole-cole type) model parameter.

    """

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

    elif ip_modeltype == 'mpa':
        coleparams = ("$\phi_{max}$" + " = {:.2f} rad\n".format(ip_model[layer_ip, 2]) + \
                      "$\\tau_{\phi}$" + " = {:.2e} s\n".format(ip_model[layer_ip, 3]) + \
                      "c = {:.2f}".format(ip_model[layer_ip, 4],))

    elif ip_modeltype == None:
        coleparams = ''

    else:
        raise TypeError('Requested IP model is not implemented.')

    xtxt = rA_model[layer_ip, 1] * 2.2
    ytxt = (rA_model[layer_ip, 0] + rA_model[layer_ip+2, 0]) / 2 # intermediate height between upper and lower layer
    axis.text(xtxt, ytxt,
              coleparams,
              {'color': 'k', 'fontsize': 12},
              va="center", ha="center", # correspond to both alignment and position?!?!? TODO understand
              transform=axis.transData)

    if rho2log:
        axis.set_xscale('log')

    axis.set_xlabel(r'$\rho$ ($\Omega$m)')
    axis.set_ylabel('h (m)')
    axis.grid(which='major', alpha=0.75, ls='-')
    axis.grid(which='minor', alpha=0.75, ls=':')

    return axis, coleparams


def plot_pem_stepmodel(axes, model2d, depth_limit=(40, 0),
                       add_bottom=50, **kwargs):
    """
    Function to plot a pelton model into four subplots (one for each model parameter)
    i.e., thk vs rho, thk vs m, thk vs tau and thk vs c

    Parameters
    ----------
    axes : list containing axes
        axes into which you would like to plot. requires 4 subplots
    model2d : np.ndarray (5, n)
        n rows for the layers, 4 columns - one for each parameter.
    depth_limit : tuple, optional
        y-axis limits. The default is (40, 0).
    add_bottom : float, optional
        extend the bottom layer by this number. The default is 50.
    **kwargs : key-word arguments
        for the plt.plot method.

    Returns
    -------
    fig : mpl Figure object
        .
    axes : mpl axis object
        .

    """
    from pygimli.viewer.mpl import drawModel1D
    
    if add_bottom is not None:
        thickness = np.r_[model2d[:, 0], model2d[-1, 0] + add_bottom]
        rho0 = np.r_[model2d[:, 1], model2d[-1, 1]]
        chargeability = np.r_[model2d[:, 2], model2d[-1, 2]]
        relaxtime = np.r_[model2d[:, 3], model2d[-1, 3]]
        dispersion = np.r_[model2d[:, 4], model2d[-1, 4]]
    else:
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
    axes[0].set_xlabel(r'$\rho_0$ ($\Omega$m)')
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
    axes[2].set_xlabel(r'rel. time $\tau$ (s)')
    # axes[2].set_ylabel('z (m)')
    axes[2].set_xscale('log')
    # axes[2].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    drawModel1D(axes[3], thickness, dispersion, **kwargs)
    axes[3].legend()
    axes[3].set_ylim(depth_limit)
    axes[3].set_xlabel(r'disp. coeff. c ()')
    axes[3].set_ylabel('z (m)')
    axes[3].yaxis.tick_right()
    axes[3].yaxis.set_label_position('right')
    
    plt.suptitle('pelton model', fontsize=16)
    
    return fig, axes


def plot_mpa_stepmodel(axes, model2d, depth_limit=(40, 0),
                       add_bottom=50, legend_loc='lower right', **kwargs):
    """
    Function to plot a pelton model into four subplots (one for each model parameter)
    i.e., thk vs rho, thk vs mpa, thk vs tau_phi and thk vs c

    Parameters
    ----------
    axes : list containing axes
        axes into which you would like to plot. requires 4 subplots
    model2d : np.ndarray (5, n)
        n rows for the layers, 4 columns - one for each parameter.
    depth_limit : tuple, optional
        y-axis limits. The default is (40, 0).
    add_bottom : float, optional
        extend the bottom layer by this number. The default is 50.
    **kwargs : key-word arguments
        for the plt.plot method.

    Returns
    -------
    fig : mpl Figure object
        .
    axes : mpl axis object
        .

    """
    from pygimli.viewer.mpl import drawModel1D
    
    if add_bottom is not None:
        thickness = np.r_[model2d[:, 0], model2d[-1, 0] + add_bottom]
        rho0 = np.r_[model2d[:, 1], model2d[-1, 1]]
        max_phase_angle = np.r_[model2d[:, 2], model2d[-1, 2]]
        tau_phi = np.r_[model2d[:, 3], model2d[-1, 3]]
        dispersion = np.r_[model2d[:, 4], model2d[-1, 4]]
    else:
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
    axes[0].set_xlabel(r'$\rho_0$ ($\Omega$m)')
    axes[0].set_ylabel('z (m)')
    axes[0].set_xscale('log')
    # axes[0].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    drawModel1D(axes[1], thickness, max_phase_angle, **kwargs)
    # axes[1].legend()
    axes[1].set_ylim(depth_limit)
    axes[1].set_xlabel(r'$\phi_{\mathrm{max}}$ (rad)')
    axes[1].set_ylabel('')

    drawModel1D(axes[2], thickness, tau_phi, **kwargs)
    # axes[2].legend()
    axes[2].set_ylim(depth_limit)
    axes[2].set_xlabel(r'$\tau_{\phi}$ (s)')
    axes[2].set_ylabel('')
    axes[2].set_xscale('log')
    # axes[2].xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))

    drawModel1D(axes[3], thickness, dispersion, **kwargs)
    # axes[3].legend(title='Models:', title_fontsize=14)
    # axes[3].legend(title='Models:', title_fontsize=10, loc=legend_loc)
    axes[3].legend(loc=legend_loc)
    axes[3].set_ylim(depth_limit)
    axes[3].set_xlabel(r'$c$ ()')
    axes[3].set_ylabel('')
    # axes[3].yaxis.tick_right()
    # axes[3].yaxis.set_label_position('right')

    return fig, axes
