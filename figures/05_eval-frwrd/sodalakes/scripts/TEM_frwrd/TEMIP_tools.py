#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 27 14:50:15 2021

@author: laigner
"""

# %% import modules
# import os
# import sys
# import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from matplotlib.offsetbox import AnchoredText
# from matplotlib.ticker import LogLocator, NullFormatter
# from matplotlib.lines import Line2D
# from matplotlib.patches import Polygon
# from matplotlib.collections import PatchCollection
# from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.backends.backend_pdf import PdfPages

from scipy.constants import mu_0
from scipy.constants import epsilon_0


# %% function_lib

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


def PEM(rho0, m, tau, c, f):
    """
    Pelton et al. model

    Parameters
    ----------
    rho0 : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.
    tau : TYPE
        DESCRIPTION.
    c : TYPE
        DESCRIPTION.
    f : TYPE
        DESCRIPTION.

    Returns
    -------
    complex_con : TYPE
        DESCRIPTION.

    """

    iotc = (2j*np.pi*f * tau)**c
    complex_con = 1 / (rho0 * (1 - m*(1 - 1/(1 + iotc))))

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


def calc_rhoa(forward, signal):
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
    M = (forward.setup_device['current_inj'] *
         forward.setup_device['txloop']**2 * turns)
    rhoa = ((1 / np.pi) *
            (M / (20 * (abs(signal))))**(2/3) *
            (mu_0 / (forward.times_rx))**(5/3))
    rhoa[sub0] = rhoa[sub0]*-1
    return rhoa



def plot_signal(axis, time, signal, sub0color='aqua', **kwargs):
    sub0 = (signal <= 0)
    sub0_sig = signal[sub0]
    sub0_time = time[sub0]
    
    line, = axis.loglog(time, abs(signal), **kwargs)
    line_sub0, = axis.loglog(sub0_time, abs(sub0_sig), 's',
                             markerfacecolor='none', markersize=5,
                             markeredgewidth=0.8, markeredgecolor=sub0color)
    return axis, line, line_sub0


def plot_rhoa(axis, time, rhoa, sub0color='k', **kwargs):
    sub0 = (rhoa <= 0)
    sub0_rhoa = rhoa[sub0]
    sub0_time = time[sub0]
    
    axis.loglog(time, abs(rhoa), **kwargs)
    axis.loglog(sub0_time, abs(sub0_rhoa), 's',
                markerfacecolor='none', markersize=6,
                markeredgewidth=1, markeredgecolor=sub0color)
    return axis


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