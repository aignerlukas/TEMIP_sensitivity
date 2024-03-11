# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:18:06 2021

additional tools specifically for BEL1D TEM usage

@author: lukas


"""

# TODO do I actually need this library at all?
# if only for a few functions, maybe move to sean lib?


# %% import modules
import logging
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from .universal_tools import calc_rhoa
from .universal_tools import plot_rhoa
from .universal_tools import plot_signal


# %% set logging level
logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)
# logging.getLogger().setLevel(logging.DEBUG)
# logging.getLogger().setLevel(logging.INFO)


# %% functions_bel
def get_min(mean, factor):
    min_val = mean - mean/factor
    return min_val


def get_max(mean, factor):
    max_val = mean + mean/factor
    return max_val


def reArr_mdl(Mdl2_reArr, add_bottom=10):
    """
    function to rearrange thickness/res model structure to
    plot a step-model.
    """
    Mdl2_reArr = np.asarray(Mdl2_reArr, dtype='float')

    MdlElev = np.zeros((len(Mdl2_reArr)*2,2)); r=1;k=0
    for i in range(0,len(Mdl2_reArr)*2):
        if i == len(Mdl2_reArr):
            MdlElev[-1,1] = MdlElev[-2,1]
            break
        if i == 0:
            MdlElev[i,0] = Mdl2_reArr[i,0]
            MdlElev[i:i+2,1] = Mdl2_reArr[i,1]
        else:
            MdlElev[k+i:k+i+2,0] = -Mdl2_reArr[i,0]    #elevation
            MdlElev[r+i:r+i+2,1] = Mdl2_reArr[i,1]     #resistivity
            k+=1; r+=1
    
    if not add_bottom is None:
        pass
    else:
        MdlElev = np.delete(MdlElev,-1,0) # delete last row!!
    return MdlElev


def mtrxMDL2vec(mtrx):
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


def vecMDL2mtrx(model, nLayer, nParam):
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



def get_stduniform_fromPrior(prior_space):
    """
    

    Parameters
    ----------
    prior_space : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    prior_elems = prior_space.shape[0]* prior_space.shape[1]
    stdUniform = lambda a,b: (b-a)/np.sqrt(prior_elems)
    nLayers = prior_space.shape[0]
    nParams = int(prior_space.shape[1] / 2)
    stdTrue = []
    for par in range(nParams):
        par *= 2
        for lay in range(nLayers):
            if (prior_space[lay,0] == 0) and (par == 0):  # only skip for thk param
                pass
            else:
                stdTrue.append(stdUniform(prior_space[lay,par],
                                          prior_space[lay,par+1]))
    return np.asarray(stdTrue)


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



# %% plotting_functions
def plot_mdl_memima(mdl_mean, mdl_min, mdl_max, extend_bot=25, ax=None):
    """
    function to plot the min and max of a model space.
    mainly intended for bugfixing the plot_prior space function

    Parameters
    ----------
    mdl_mean : np.array
        means of prior space.
    mdl_min : np.array
        mins of prior space.
    mdl_max : np.array
        maxs of prior space.
    extend_bot : int, optional
        extension of bottom layer. The default is 25.
    ax : pyplot axis object, optional
        for plotting to an already existing axis. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    prior_min_cum, min2plot = mdl2steps(mdl_min,
                                        extend_bot=extend_bot, cum_thk=True)
    prior_max_cum, max2plot = mdl2steps(mdl_max,
                                        extend_bot=extend_bot, cum_thk=True)
    prior_mea_cum, mean2plot = mdl2steps(mdl_mean,
                                         extend_bot=extend_bot, cum_thk=True)
    
    if ax is None:
        fig, ax = plt.subplots(1,1)
    
    ax.plot(mean2plot[:,1], mean2plot[:,0],
            color='k', ls='-', lw=2,
            zorder=10)
    ax.plot(min2plot[:,1], min2plot[:,0],
            color='c', ls='--', lw=2,
            zorder=5)
    ax.plot(max2plot[:,1], max2plot[:,0],
            color='m', ls=':', lw=2,
            zorder=5)
    ax.invert_yaxis()
    if ax is None:
        return fig, ax
    else:
        return ax


def plot_prior_space(prior, show_patchcorners=False, show_mean=False):
    """
    function to visualize the prior model space
    updated version to plot multiple parameters to subplots
    could be also used to visualize the solved means and stds
    
    use with caution - might be buggy still...
        TODO - fix possible bugs!!
    
    Parameters
    ----------
    prior : np.array
        prior model matrix.
    show_patchcorners : boolean, optional
        switch to decide whether to show the corners in the plot.
        useful for debugging. The default is False.

    Returns
    -------
    fig_pr : pyplot figure object
        figure object.
    ax_pr : pyplot axis object
        1 subplot

    """
    
    prior_min = prior[:,::2]
    prior_max = prior[:,1::2]
    
    ncols = int(prior.shape[1] / 2) - 1  # depth doesn't count
    
    prior_mean = np.mean(np.array([prior_min, prior_max]), axis=0)
    
    prior_min_cum_all, min2plot = mdl2steps(prior_min,
                                        extend_bot=25, cum_thk=True)
    prior_max_cum_all, max2plot = mdl2steps(prior_max,
                                        extend_bot=25, cum_thk=True)
    prior_mea_cum_all, mean2plot = mdl2steps(prior_mean,
                                         extend_bot=25,cum_thk=True)
    
    fig_pr, axes = plt.subplots(nrows=1,ncols=ncols,
                                figsize=(3.2*ncols,6), squeeze=False)
    axes = axes.flat
    
    xcols = np.arange(1,ncols+1)
    xcolid = 1
    for idx, ax_pr in enumerate(axes):
        k = 0
        logging.info('\n\n--------------------------')
        logging.info('preparing patch creation:\n')
        
        prior_min_cum = prior_min_cum_all[:,(0,xcolid)]
        prior_max_cum = prior_max_cum_all[:,(0,xcolid)]
        logging.info('prior_min_cum', prior_min_cum)
        logging.info('prior_max_cum', prior_max_cum)
        logging.info('--------------------------\n')
        
        
        patches_min = []
        for prior_cum in [prior_min_cum, prior_max_cum]:
            logging.debug(prior_cum)
            for i, thk in enumerate(prior_cum[:,0]):
                logging.debug('layerID', i)
                logging.debug('curr_thickness', thk)
                r_rli_min = prior_min_cum[i,1]
                r_rli_max = prior_max_cum[i,1]
                logging.debug('minmax_rho', r_rli_min, r_rli_max)
                
                color = 'r' if k % 2 == 0 else 'g'
                mrkr = '.' if k % 2 == 0 else 'o'
                mfc = None if k % 2 == 0 else 'None'
            
                if i == 0:
                    thk_top = 0
                else:
                    thk_top = prior_min_cum[i-1,0]
                corners_xy = [[r_rli_min, thk_top],
                              [r_rli_max, thk_top],
                              [r_rli_max, thk],
                              [r_rli_min, thk],
                              ]
                logging.debug(corners_xy)
                if show_patchcorners:
                    ax_pr.plot(np.asarray(corners_xy)[:,0],
                                np.asarray(corners_xy)[:,1],
                                mrkr, color=color,
                                mfc=mfc)
                    alpha = 0.7
                else:
                    alpha = 1
                patches_min.append(Polygon(corners_xy,
                                           color='darkgray', alpha=alpha,
                                           zorder=0))
            k += 1

        p = PatchCollection(patches_min, match_original=True)
        ax_pr.add_collection(p)
        
        if show_mean:
            logging.debug('\n\n--------------------------')
            logging.debug('plotting mean model:\n')
            logging.debug(mean2plot)
            logging.debug(xcolid)
            logging.debug(mean2plot[:,xcolid])
            ax_pr.plot(mean2plot[:,xcolid], mean2plot[:,0],
                       'k.--', label='prior mean')
        ax_pr.invert_yaxis()
        ax_pr.grid(which='major', color='white', linestyle='-', zorder=10)

        if idx > 0:
            ax_pr.tick_params(axis='y', which='both', labelleft=False)

        xcolid += 1

    return fig_pr, axes


def plot_frwrd_comp(obs_data_norm, frwrd_response, frwrd_times, frwrd_rhoa,
                   filter_times, show_rawdata=True,
                   show_noise_and_estim=True, est_err=None, **kwargs):
    """

    Parameters
    ----------
    obs_data_norm : pd.DataFrame
        Normalized to V/m² with columns:
        ['channel', 'time', 'signal', 'err', 'rhoa'].
    frwrd_response : np.array (V/m²)
        response from forward calc.
    frwrd_times : np.array (s)
        times at which the frwrd response was calculated.
    filter_times : tuple (minT, maxT) (s)
        time range at which the data where filtered.
    show_rawdata: Boolean
        whether or not to show the rawdata.
    show_noise_and_estim: Boolean
        whether or not to show the noise that was measured by the device.
    est_err: 2x1 np.array
        if not None, then est_err[0] is the relative error and est_err[1] is the absolute error
    
    **kwargs : mpl kwargs
        for setting markersize, linewidth, etc...

    Returns
    -------
    ax : pyplot axis object
        1 row, 2 columns - signal, app. Res.

    """
    minT = filter_times[0]
    maxT = filter_times[1]
    
    obs_sub = obs_data_norm[(obs_data_norm.time>minT) &
                            (obs_data_norm.time<maxT)]

    fig, ax = plt.subplots(nrows=1, ncols=2,
                           figsize=(12, 6),
                           sharex=True)

    if show_rawdata:
        ax[0].loglog(obs_data_norm.time, obs_data_norm.signal,
                     ':o', alpha=0.7,
                     color='k', label='raw data observed',
                     **kwargs)
    
    ax[0].loglog(obs_sub.time, obs_sub.signal,
                 ':o', alpha=0.7,
                  color='g', label='data observed (filtered)',
                  **kwargs)
    
    ax[0].loglog(frwrd_times, 
                 frwrd_response,
                  ':d', color='b', 
                  label='forward response - prior mean',
                  **kwargs)

    if show_noise_and_estim:
        ax[0].loglog(obs_data_norm.time, obs_data_norm.error,
                     ':k', alpha=0.3, label='noise observed')
        if est_err is not None:
            # print('plotting error estimation')
            np.random.seed(42)
            rndm = np.random.randn(len(obs_data_norm.signal))
            noise_calc_rand = (est_err[0] * np.abs(obs_data_norm.signal) + 
                               est_err[1]) * rndm
            # print(noise_calc_rand)
            ax[0].loglog(obs_data_norm.time, abs(noise_calc_rand),
                         '--k', alpha=0.5, label='noise estimated')
    
    ax[0].set_title('measured signal vs. frwrd sol')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel(r"$\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t$ (V/m²)")
    ax[0].set_xlim((2e-6, 5e-3))
    ax[0].legend(loc='best')

    if show_rawdata:
        ax[1].loglog(obs_data_norm.time, obs_data_norm.rhoa,
                     ':o', alpha=0.7,
                     color='k', label='raw rhoa observed',
                     **kwargs)
    
    ax[1].loglog(obs_sub.time, obs_sub.rhoa,
                 ':o', alpha=0.7,
                  color='g', label='rhoa observed (filtered)',
                  **kwargs)
    
    ax[1].loglog(frwrd_times, 
                 frwrd_rhoa,
                  ':d', color='b', 
                  label='forward response - prior mean',
                  **kwargs)
    
    # ax[1].set_title('apparent resistivity after christiansen paper...')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel(r'$\rho_a$ ($\Omega$m)')
    # ax[1].set_ylim((min(frwrd_rhoa)*0.8, max(frwrd_rhoa)*1.2))
    ax[1].yaxis.set_label_position("right")
    ax[1].yaxis.tick_right()
    ax[1].legend(loc='best')

    return fig, ax


def plot_fit(setup_device, rx_times, data_pre, data_post, response_scale=None, **kwargs):
    """
    function to plot the data fit after finishing a bel1d run

    Parameters
    ----------
    forward : simpeg forward operator
        DESCRIPTION.
    data_pre : np.array (float)
        tem data to solve for (V/m²).
    mdl_means_post : np.array
        solved models.
    **kwargs : matplotlib kwargs

    Returns
    -------
    fig : pyplot figure object
        figure object.
    ax : pyplot axis object
        1 row, 2 columns - signal, app. Res.
    post_sig : np.array (float)
        tem data calculated from mdl means post (V/m²).
    rms_sig : np.array (float)
        contains the absolute and relative rms of the misfit between observed and calculated data.
    pre_rhoa : np.array (float)
        tem data to solve for in terms of app. res - calculated here (Ohmm).
    post_rhoa : np.array (float)
        tem data calculated from mdl means post (Ohmm).
    rms_rhoa : np.array (float)
        contains the absolute and relative rms of the misfit between observed and calculated rhoa.

    """
    # print('###############################################################')
    # print('Post data fit plotting.... at receiver times:')
    # print(rx_times)
    # forward.calc_response(mdl_means_post)
    # post_sig = forward.response
    # print('reponse...')
    # print(post_sig)
    
    # calc RMS of misfit
    misfit_sig = data_post - data_pre
    misfit_sig_norm = misfit_sig / data_pre
    arms_sig = np.sqrt(np.mean(misfit_sig**2))
    rrms_sig = np.sqrt(np.mean(misfit_sig_norm**2))*100
    rms_sig = np.r_[arms_sig, rrms_sig]
    
    fig, ax = plt.subplots(nrows=1, ncols=2,
                           figsize=(10, 5),
                           sharex=True)
    
    ax[0].semilogx(rx_times, data_pre,
                 'd', color='g',
                 label='simulated data pre',
                  **kwargs)
    ax[0].semilogx(rx_times, data_post,
                  '-', alpha=0.7,
                  color='k', 
                  label='fitted data post',
                  **kwargs)
    
    if response_scale == None:
        ax[0].set_yscale('log')
        pre_rhoa = calc_rhoa(setup_device, data_pre, rx_times)
        post_rhoa = calc_rhoa(setup_device, data_post, rx_times)
    elif response_scale == 'log10':
        pre_rhoa = calc_rhoa(setup_device, 10**data_pre, rx_times)
        post_rhoa = calc_rhoa(setup_device, 10**data_post, rx_times)


    ax[0].set_title(('abs RMS: {:10.3e} V/m² \n'.format(arms_sig) + 
                     'rel RMS: {:6.2f} % '.format(rrms_sig)))
    ax[0].set_xlabel('time (s)')
    ax[0].set_ylabel(r"$\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t$ (V/m²)")
    ax[0].set_xlim((1e-6, 1e-3))
    ax[0].legend(loc='best')

    # rhoa - calc RMS of misfit
    misfit_rhoa = post_rhoa - pre_rhoa
    misfit_rhoa_norm = misfit_rhoa / pre_rhoa
    arms_rhoa = np.sqrt(np.mean(misfit_rhoa**2))
    rrms_rhoa = np.sqrt(np.mean(misfit_rhoa_norm**2))*100
    rms_rhoa = np.r_[arms_rhoa, rrms_rhoa]
    
    ax[1].loglog(rx_times, pre_rhoa,
                 'd', color='g', label='rhoa pre',
                 **kwargs)

    ax[1].loglog(rx_times, post_rhoa,
                  '-', alpha=0.7,
                  color='k', label='fitted rhoa post',
                  **kwargs)

    ax[1].set_title(('abs RMS: {:6.2f} ($\Omega$m) \n'.format(arms_rhoa) +
                     'rel RMS: {:6.2f} % '.format(rrms_rhoa)))
    ax[1].set_xlabel('time (s)')
    ax[1].set_ylabel(r'$\rho_a$ ($\Omega$m)')
    ax[1].yaxis.set_label_position('right')
    ax[1].yaxis.tick_right()
    # ax[1].set_ylim((min(post_rhoa)*0.8, max(post_rhoa)*1.2))
    ax[1].legend(loc='best')
    
    return fig, ax, rms_sig, pre_rhoa, post_rhoa, rms_rhoa


def plot_simdata(times_rx, dbdt_clean, setup_device,
                 color_dbdt='k', color_rhoa='teal',
                 show_rhoa=False, show_noise=True,
                 relerr=0.003, abserr=1e-10):
    """
    

    Parameters
    ----------
    times_rx : TYPE
        DESCRIPTION.
    dbdt_clean : TYPE
        DESCRIPTION.
    forward_solver : TYPE
        DESCRIPTION.
    show_rhoa : TYPE, optional
        DESCRIPTION. The default is False.
    show_noise : TYPE, optional
        DESCRIPTION. The default is True.
    relerr : TYPE, optional
        DESCRIPTION. The default is 0.003.
    abserr : TYPE, optional
        DESCRIPTION. The default is 1e-10.

    Returns
    -------
    None.

    """
    np.random.seed(42)
    rndm = np.random.randn(len(dbdt_clean))

    noise_calc_rand = (relerr * np.abs(dbdt_clean) + 
                       abserr) * rndm
    dbdt_noisy_calc = noise_calc_rand + dbdt_clean
    
    ## PLOT synthetic data ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig, ax = plt.subplots(nrows=1, ncols=2,
                           figsize=(12, 6),
                           sharex=True, sharey=True)
    ax1 = ax[0]
    ax2 = ax[1]

    rhoa_clean = calc_rhoa(setup_device, dbdt_clean, times_rx)
    rhoa_noisy_calc = calc_rhoa(setup_device, dbdt_noisy_calc, times_rx)


    _ = plot_signal(ax1, times_rx, dbdt_clean,
                    marker='.', ls='-', color=color_dbdt,
                    label='data clean trueMDL')
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel(r"$\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t$ (V/m²)")

    if show_rhoa:
        ax1_ra = ax1.twinx()
        _ = plot_rhoa(ax1_ra, times_rx, rhoa_clean,
                      marker='.', ls='-', color=color_rhoa,
                      label=r'$\rho_a$ clean')
        ax1_ra.tick_params(axis='y', which='both', labelright=False)
        ax1_ra.set_ylim((100, 10000))
    ax1.set_title('clean data')


    _ = plot_signal(ax2, times_rx, dbdt_noisy_calc,
                    marker='.', ls='-', color=color_dbdt,
                    label='data noisy trueMDL')
    ax2.set_xlabel('time (s)')
    ax2.tick_params(axis='y', which='both', labelleft=False)

    if show_noise:
        ax2.loglog(times_rx, abs(noise_calc_rand),
                    '.', ms=4, ls='-', color='grey', alpha=0.75,
                    zorder=0, label='sim noise')
    if show_rhoa:
        ax2_ra = ax2.twinx()
        _ = plot_rhoa(ax2_ra, times_rx, rhoa_noisy_calc,
                      marker='.', ls='-', color=color_rhoa,
                      label=r'$\rho_a$ noisy')
        ax2_ra.tick_params(axis='y', which='both', labelright=True)
        ax2_ra.set_ylim((100, 10000))
        ax2_ra.tick_params('y', colors='tomato')
        ax2_ra.set_ylabel(r'$\rho_a$', color='tomato')

    ax2.set_title(('noise:  ' + 
                   're_{:.1e}  '.format(relerr) + 
                   'ae_{:.1e}'.format(abserr)))

    plt.tight_layout()
    
    return ax1, ax2


def save_result_bel1d(setup_device, means, stds,
                      ip_modeltype, position,
                      rxtimes_sub, nparams, nlayers,
                      pred_data, obsdat_sub, 
                      obserr_sub, abs_err, obsrhoa_sub,
                      savepath_csv, savename):
    """
    

    Parameters
    ----------
    setup_device : TYPE
        DESCRIPTION.
    model_inv : TYPE
        DESCRIPTION.
    ip_modeltype : TYPE
        DESCRIPTION.
    position : TYPE
        DESCRIPTION.
    rxtimes_sub : TYPE
        DESCRIPTION.
    nparams : TYPE
        DESCRIPTION.
    nlayers : TYPE
        DESCRIPTION.
    pred_data : TYPE
        DESCRIPTION.
    obsdat_sub : TYPE
        DESCRIPTION.
    obserr_sub : TYPE
        DESCRIPTION.
    abs_err : TYPE
        DESCRIPTION.
    obsrhoa_sub : TYPE
        DESCRIPTION.
    savepath_csv : TYPE
        DESCRIPTION.
    savename : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    posX = position[0]
    posY = position[1]
    posZ = position[2]

    res_mean, thk_mean = means[nlayers-1:nlayers*2-1], means[0:nlayers-1]
    means_mtrx = vecMDL2mtrx(means, nlayers, nparams)
    
    res_std, thk_std = stds[nlayers-1:nlayers*2-1], stds[0:nlayers-1]
    stds_mtrx = vecMDL2mtrx(stds, nlayers, nparams)

    pred_rhoa = calc_rhoa(setup_device, pred_data,
                          rxtimes_sub)

    if ip_modeltype != None:
        inv_m = means_mtrx[:, 2]
        inv_tau = means_mtrx[:, 3]
        inv_c = means_mtrx[:, 4]

    if ip_modeltype == 'pelton':
        header_result = 'X,Y,Z,thk(m),rho(Ohmm),m(),tau(s),c()'
        labels_CC = ['chargeability m ()', r'rel. time $\tau$ (s)']
        result_arr = np.column_stack((np.r_[thk_mean, 0], res_mean,
                                      inv_m, inv_tau, inv_c))

    elif ip_modeltype == 'mpa':
        header_result = 'X,Y,Z,thk(m),rho(Ohmm),mpa(rad),tau_p(s),c()'
        labels_CC = ['mpa (rad)', r'rel. time $\tau_{\phi}$ (s)']
        result_arr = np.column_stack((np.r_[thk_mean, 0], res_mean,
                                      inv_m, inv_tau, inv_c))

    elif ip_modeltype == None:
        header_result = 'X,Y,Z,thk(m),rho(Ohmm)'
        result_arr_mean = np.column_stack((np.r_[thk_mean, 0], res_mean))
        result_arr_stds = np.column_stack((np.r_[thk_std, 0], res_std))

    else:
        raise ValueError('this ip modeltype is not implemented here ...')
    export_array_means = np.column_stack((np.full((len(result_arr_mean),), posX),
                                          np.full((len(result_arr_mean),), posY),
                                          np.full((len(result_arr_mean),), posZ),
                                          result_arr_mean))
    export_array_stds = np.column_stack((np.full((len(result_arr_stds),), posX),
                                          np.full((len(result_arr_stds),), posY),
                                          np.full((len(result_arr_stds),), posZ),
                                          result_arr_stds))

    header_fit = ('time(s), signal_pred(V/m2), ' +
                  'signal_obs(V/m2), err_obs(V/m2), err_scl(V/m2),' +
                  'rhoa_pred(V/m2), rhoa_obs(V/m2)')
    export_fit = np.column_stack((rxtimes_sub, pred_data,
                                  obsdat_sub, obserr_sub, abs_err,
                                  pred_rhoa, obsrhoa_sub))

    if ip_modeltype != None:
        formatting = '%.3f,%.3f,%.3f,%.3f,%.3f,%.1f,%.1e,%.3f'
    else:
        formatting = '%.3f,%.3f,%.3f,%.3f,%.3f'

    np.savetxt(savepath_csv + savename +'_means.csv',
               export_array_means, comments='',
               header=header_result,
               fmt=formatting)
    np.savetxt(savepath_csv + savename +'_stds.csv',
               export_array_stds, comments='',
               header=header_result,
               fmt=formatting)
    np.savetxt(savepath_csv + savename +'_fit.csv',
               export_fit,
               comments='',
               header=header_fit,
               fmt='%.6e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e')

    return
