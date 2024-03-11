#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 15:21:05 2022

utility function for empymod_frwrd_ip

@author: laigner
"""

# %% modules
import numpy as np

from scipy.constants import epsilon_0


# %% general
def getR_fromSquare(a):
    return np.sqrt((a*a) / np.pi)


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


def simulate_error(relerr, abserr, data):
    np.random.seed(42)
    rndm = np.random.randn(len(data))

    rand_error_abs = (relerr * np.abs(data) + 
                  abserr) * rndm

    return rand_error_abs


# %% plotting
def plot_signal(axis, time, signal, sub0color='aqua', **kwargs):
    sub0 = (signal <= 0)
    sub0_sig = signal[sub0]
    sub0_time = time[sub0]
    
    line, = axis.loglog(time, abs(signal), **kwargs)
    line_sub0, = axis.loglog(sub0_time, abs(sub0_sig), 's',
                             markerfacecolor='none', markersize=5,
                             markeredgewidth=0.8, markeredgecolor=sub0color)
    return axis, line, line_sub0



# %% CC type models
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



def pelton_et_al(inp, p_dict):
    """ Pelton et al. (1978).
    code from: https://empymod.emsig.xyz/en/stable/examples/time_domain/cole_cole_ip.html#sphx-glr-examples-time-domain-cole-cole-ip-py
    """

    # Compute complex resistivity from Pelton et al.
    # print('\n   shape: p_dict["freq"]\n', p_dict['freq'].shape)
    iotc = np.outer(2j*np.pi*p_dict['freq'], inp['tau'])**inp['c']

    # print('\n   shape: inp["rho_0"]\n', inp["rho_0"].shape)
    # print('\n   shape: iotc\n', iotc.shape)
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
    etaH, etaV : dtype??
        

    """
    # obtain chargeability and tau from mpa model
    m, tau_rho = get_m_taur_MPA(inp['phi_max'], inp['tau_phi'], inp['c'], verbose=True)
    
    iotc = np.outer(2j*np.pi*p_dict['freq'], tau_rho)**inp['c']
    # Compute complex resistivity
    rhoH = inp['rho_0'] * (1 - m*(1 - 1/(1 + iotc)))
    rhoV = rhoH*p_dict['aniso']**2
    
    # Add electric permittivity contribution
    etaH = 1/rhoH + 1j*p_dict['etaH'].imag
    etaV = 1/rhoV + 1j*p_dict['etaV'].imag
    
    return etaH, etaV


# TODO: Test and finish function!!
# def cc_eps(inp, p_dict):
#     """
#     Mudler et al. (2020) - after Zorin and Ageev, 2017 with HF EM part - dielectric permittivity
#     """

#     # Compute complex permittivity
#     iotc = np.outer(2j*np.pi*p_dict['freq'], inp['tau'])**inp['c']
#     iwe0rhoDC = np.outer(2j*np.pi*p_dict['freq'], epsilon_0, inp["rho_DC"])
#     eta_c_r = inp["epsilon_hf"] + ((["epsilon_DC"] - ["epsilon_HF"]) / (1 + iotc)) + (1 / iwe0rhoDC)

#     etaH = eta_c_r
#     etaV = eta_c_r

#     return etaH, etaV


def cc_con_koz(inp, p_dict):
    """
    compl. con from Kozhevnikov, Antonov (2012 - JaGP)
    using perm0 and perm8 - Formula 5
    # TODO: Test and finish method!!
    """

    # Compute complex permittivity,
    # print('\n   shape: p_dict["freq"]\n', p_dict['freq'].shape)
    io = 2j * np.pi * p_dict['freq'] ## i*omega --> from frequency to angular frequency
    iotc = np.outer(io, inp['tau'])**inp['c']

    # print('\n   shape: inp["sigma_0"]\n', inp["sigma_0"].shape)
    # print('\n   shape: iotc\n', iotc.shape)
    # print('\n   shape: io\n', io.shape)
    # print('1st term:', (inp["sigma_0"] + np.outer(io, epsilon_0)).shape)
    # print('2nd term:', (inp["epsilon_8"] + ((inp["epsilon_s"] - inp["epsilon_8"]) / (1 + iotc))).shape)
    etaH = (inp["sigma_0"] + np.outer(io, epsilon_0) *
            (inp["epsilon_8"] + ((inp["epsilon_s"] - inp["epsilon_8"]) / (1 + iotc)))
           )

    etaV = etaH * p_dict['aniso']**2

    return etaH, etaV





# %% functions for MPA
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
    n_iters = 1000
    th = 1e-9
    
    if (phi_max.dtype == c.dtype) and (tau_phi.dtype == c.dtype):
        if hasattr(phi_max, '__len__') and (not isinstance(phi_max, str)):
            m, tau_rho = np.zeros_like(tau_phi), np.zeros_like(tau_phi)
            if phi_max[0] == 0:
                print('encountered phi_max == 0, assuming no-IP effect, setting m also to 0')
                # starting from the 2nd layer, id 1
                start_id = 1
            else:
                start_id = 0
            
            for i in range(start_id, len(phi_max)):
                mns = []
                tau_rs = []
                areal = []
                bimag = []
                delta_ms = []

                for n in range(0, n_iters):
                    if n == 0:
                        mns.append(0)
                        tau_rs.append(mpa_get_tau_rho(m=mns[n],
                                                      tau_phi=tau_phi[i],
                                                      c=c[i]))
                        areal.append(mpa_get_a(tau_rs[n], tau_phi[i], c[i]))
                        bimag.append(mpa_get_b(tau_rs[n], tau_phi[i], c[i]))
                        mns.append(mpa_get_m(a=areal[n],
                                             b=bimag[n],
                                             phi_max=phi_max[i]))
                        delta_ms.append(mpa_get_deltam(mn=mns[n+1], mp=mns[n]))
                    else:
                        tau_rs.append(mpa_get_tau_rho(m=mns[n],
                                                      tau_phi=tau_phi[i],
                                                      c=c[i]))
                        areal.append(mpa_get_a(tau_rs[n], tau_phi[i], c[i]))
                        bimag.append(mpa_get_b(tau_rs[n], tau_phi[i], c[i]))
                        mns.append(mpa_get_m(a=areal[n],
                                             b=bimag[n],
                                             phi_max=phi_max[i]))
                        delta_ms.append(mpa_get_deltam(mn=mns[n+1], mp=mns[n]))
                        print('delta_m: ', delta_ms[n])
                        if delta_ms[n] <= th:  # stop if the difference is below 1e-9
                            if verbose:
                                print(f'iteration converged after {n} iters')
                                print('solved m:', mns[-1])
                                print('solved tau_rho:', tau_rs[-1])
                                
                            m[i] = mns[-1]
                            tau_rho[i] = tau_rs[-1]
                            break
                if delta_ms[n] > th:
                    raise ValueError(f'the iterations did not converge after {n_iters} iterations, please check input!')

        else:
            mns = []
            tau_rs = []
            areal = []
            bimag = []
            delta_ms = []

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
    else:
        raise TypeError('please make sure that all 3 input params are of the same dtype!!')

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

