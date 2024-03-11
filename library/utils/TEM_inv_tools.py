# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 12:37:17 2018
collection of functions to parse, plot and generate a protocol from TEMfast data.
can be adapted for other devices in the future...
Version2 ... still some Work in Progress to finish...

ToDo:
add log scale to plotting routines and
change labels to number instead of scientific notation

@author: LAigner
"""

# %% import modules
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import root
from scipy.constants import mu_0


# %% function_lib
def prep_mdl_para_names(param_names, n_layers):
    """
    function to prepare a list of model paramter names for each layer

    Parameters
    ----------
    param_names : list with strings
        list with model parameter names.
    n_layers : int
        number of layers, thickness will always be added n-1 
        times (bottom has inf thk).

    Returns
    -------
    mdl_para_names : list with strings
        format: f'{pname}_{n:02d}'

    """
    mdl_para_names = []
    for pname in param_names:
        for n in range(0, n_layers):
            if 'thk' in pname and n == n_layers-1:
                break
            mdl_para_names.append(f'{pname}_{n:02d}')
    return mdl_para_names


def rho_average(doi, lr_thk, lr_rho):
    """
    function to calculate the resistivity average to a given doi.
    required for calc_doi function.

    Parameters
    ----------
    doi : float
        depth of investigation.
    lr_thk : np.ndarray
        thicknesses of inversion result.
    lr_rho : np.ndarray
        resistivities of inversion result.

    Returns
    -------
    rho_av : float
        average resistivity down to the DOI.

    """
    rho_int = 0
    for i in range(1,len(lr_rho)):
        # print(doi)
        # print(lr_thk[0:i])
        if all(doi > lr_thk[0:i]):
            rho_int += np.log10(lr_rho[i]) * (lr_thk[i] - lr_thk[i-1])
        else:
            rho_int += np.log10(lr_rho[i]) * (doi - lr_thk[i-1])
            break
    rho_av = 10**(rho_int / doi)
    return rho_av


def calc_doi(current, tx_area, eta, mdl_rz,
             x0=100, verbose=False):
    """
    classical DOI calculation using EM diffusion
    check manuscript: e.g., Yogeshwar et al., (2020)
    uses root search from scipy.optimize

    Parameters
    ----------
    current : float
        injected current in A.
    tx_area : float
        transmitter loop area in m².
    eta : float
        voltage level of last reading (late time) in V/m².
    mdl_rz : np.array
        resistivity and depth of layers in Ohmm and m.
    x0 : float, optional
        initial value for the optimization. The default is 100.
    verbose : boolean, optional
        decide if yu want to see explicit messages. The default is False.

    Returns
    -------
    DOI : float
        depth of investigation in m.
    OPT : object
        instance of root class.

    """

    doi_fun = lambda x: 0.55*(current*tx_area*rho_average(x, mdl_rz[:,0], mdl_rz[:,1]) / (eta))**(1/5) - x
    OPT = root(doi_fun, x0)
    DOI = OPT['x'][0]

    if verbose:
        print('volt-level: ', eta)
        print('TX-area: ', tx_area)
        print(OPT['message'])
        print('DOI: ', DOI)

    return DOI, OPT


def reArr_zondMdl(Mdl2_reArr):
    """
    function to rearrange model value structure in order to
    plot a step-model.
    """
    Mdl2_reArr = np.asarray(Mdl2_reArr, dtype='float')
    FileLen=len(Mdl2_reArr)
    MdlElev = np.zeros((FileLen*2,2)); r=1;k=0
    for i in range(0,FileLen*2):
        if i == FileLen:
            MdlElev[-1,1] = MdlElev[-2,1]
            break
        if i == 0:
            MdlElev[i,0] = Mdl2_reArr[i,4]
            MdlElev[i:i+2,1] = Mdl2_reArr[i,1]
        else:
            MdlElev[k+i:k+i+2,0] = -Mdl2_reArr[i,4]    #elevation
            MdlElev[r+i:r+i+2,1] = Mdl2_reArr[i,1]     #resistivity
            k+=1; r+=1

    MdlElev = np.delete(MdlElev,-1,0) # delete last row!!
    return MdlElev


def get_diffs(response, measured):
    """
    function to get the difference between the model response (e.g., of the 
    inverted model) and the measured data. need to be in the same units, 
    otherwise the difference makes no sense!!

    Parameters
    ----------
    response : np.ndarray
        model response.
    measured : np.ndarray
        measured data.

    Returns
    -------
    diffs : np.ndarray
        differences.
    diffs_rel : np.ndarray
        differences relative to the response ndarray.

    """
    diffs = abs(response - measured)
    diffs_rel = abs((diffs / response) * 100)
    return diffs, diffs_rel


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


def calc_rhoa(setup_device, signal, times):
    """
    Function that calculates the apparent resistivity of a TEM sounding
    using the late time approximation equation from Christiansen et al (2006)

    Parameters
    ----------
    setup_device : instance of forward class
        instance of wrapping class for TEM inductive loop measurements.
    signal : np.array
        signal in V/m².
    times : np.array
        measurement times (time gates) in s
    Returns
    -------
    rhoa : np.array
        apparent resistivity in Ohmm.

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




# %% plotting
def plot_signal(axis, time, signal, log10_y=True,
                sub0color='aqua', sub0marker='s', sub0label=None, **kwargs):
    """
    

    Parameters
    ----------
    axis : axis object, optional
        use this if you want to plot to an existing axis outside of this method.
    time : np.ndarray
        time gates at which the apparent resistivity should be visualized.
    signal : np.ndarray
        TEM signal at given time gates, needs to have same length as time.
    log10_y : boolean, optional
        switch to plot the signal in log scale. The default is True.
    sub0col : string, optional
        color for the sub0 signal markers. The default is 'aqua'.
    sub0marker : string, optional
        marker-type for the sub0 signal markers. The default is 's' (squares).
    sub0label : boolean, optional
        decide whether a label for the sub0 markers should be added 
        to the legend. The default is True.
    **kwargs : key-word arguments
        for the plt.plot method.

    Returns
    -------
    fig, ax or only ax
        figure and axis object, only ax will be returned if you decided to
        provide an axis to the method.

    """

    sub0 = (signal <= 0)

    line, = axis.semilogx(time, abs(signal), **kwargs)
    if any(sub0):
        sub0_sig = signal[sub0]
        sub0_time = time[sub0]
        if sub0label is not None:
            line_sub0, = axis.semilogx(sub0_time, abs(sub0_sig), 
                                     marker=sub0marker, ls='none',
                                     mfc='none', markersize=6,
                                     mew=1.2, mec=sub0color,
                                     label=sub0label)
        else:
            line_sub0, = axis.semilogx(sub0_time, abs(sub0_sig), 
                                     marker=sub0marker, ls='none',
                                     mfc='none', markersize=6,
                                     mew=1.2, mec=sub0color)
        if log10_y:
            axis.set_yscale('log')
        return axis, line, line_sub0
    else:
        if log10_y:
            axis.set_yscale('log')
        return axis, line


def plot_rhoa(axis, time, rhoa, log10_y=True,
              sub0color='aqua', sub0marker='s', sub0label=None, **kwargs):
    """
    Function to plot the apparent resistivity with a predefined style.

    Parameters
    ----------
    axis : axis object, optional
        use this if you want to plot to an existing axis outside of this method.
    time : np.ndarray
        time gates at which the apparent resistivity should be visualized.
    log10_y : boolean, optional
        switch to plot the signal in log scale. The default is True.
    sub0col : string, optional
        color for the sub0 signal markers. The default is 'aqua'.
    sub0marker : string, optional
        marker-type for the sub0 signal markers. The default is 's' (squares).
    sub0label : boolean, optional
        decide whether a label for the sub0 markers should be added 
        to the legend. The default is True.
    **kwargs : key-word arguments
        for the plt.plot method.

    Returns
    -------
    fig, ax or only ax
        figure and axis object, only ax will be returned if you decided to
        provide an axis to the method.

    """
    sub0 = (rhoa <= 0)

    line, = axis.semilogx(time, abs(rhoa), **kwargs)
    if any(sub0):
        sub0_sig = rhoa[sub0]
        sub0_time = time[sub0]
        if sub0label is not None:
            line_sub0, = axis.semilogx(sub0_time, abs(sub0_sig), 
                                     marker=sub0marker, ls='none',
                                     mfc='none', markersize=6,
                                     mew=1.2, mec=sub0color,
                                     label=sub0label)
        else:
            line_sub0, = axis.semilogx(sub0_time, abs(sub0_sig), 
                                     marker=sub0marker, ls='none',
                                     mfc='none', markersize=6,
                                     mew=1.2, mec=sub0color)
        if log10_y:
            axis.set_yscale('log')
        return axis, line, line_sub0
    else:
        if log10_y:
            axis.set_yscale('log')
        return axis, line


def plot_diffs(ax, times, response, measured, relative=True, max_diff=30):
    """
    Function to plot the differences between the model response and 
    measured data

    Parameters
    ----------
    ax : axis object, optional
        use this if you want to plot to an existing axis outside of this method.
    times : np.ndarray
        time gates at which the differences should be visualized.
    response : np.ndarray
        Model response at given time gates, needs to have same length as times.
    measured : np.ndarray
        Measured data at given time gates, needs to have same length as time.
    relative : boolean, optional
        Switch to decide if the diffs will be shown as absolute or relative 
        values. The default is True.
    max_diff : float, optional
        upper limit for the differences on the y-axis. The default is 30 (%).

    Returns
    -------
    axt : TYPE
        DESCRIPTION.

    """
    
    diffs, diffs_rel = get_diffs(response, measured)
    
    axt = ax.twinx()
    if relative == True:
        axt.plot(times, diffs_rel, ':', color='gray', zorder=0, label='$\delta_{rel}$')
        axt.set_ylabel('$\delta$ resp-data (%)', color='gray')
        plt.setp(axt.get_yticklabels(), color="gray")
        axt.grid(False)
        axt.set_ylim((0, max_diff))
    else:
        axt.plot(times, diffs, ':', color='gray', zorder=0, label='$\delta_{abs}$')
        axt.set_ylabel('$\delta$ resp-data ()', color='gray')
        plt.setp(axt.get_yticklabels(), color="gray")
        axt.grid(False)
        axt.set_ylim((0, max_diff))
    return axt


# %% file IO
def save_result_and_fit(tem_inv, setup_device,
                        model_inv, jac_df, 
                        ip_modeltype, position,
                        rxtimes_sub, nparams, nlayers,
                        initmdl_arr, obsdat_sub,
                        obserr_sub, abs_err, obsrhoa_sub,
                        savepath_csv, savename, truemdl_arr=None):
    """
    function to save the inversion result and data fit of a TEM inversion

    Parameters
    ----------
    tem_inv : pg.inv object
        pyGIMLi inversion object that was used to run the inversion of the data.
    ip_modeltype : string, or None
        type of the cc model to get the complex res and model the IP effect.
        None if there was no IP effect considered
    position : tuple
        x,y,z coordinates of the sounding postion.
    rxtimes_sub : np.ndarray
        filtered time gates at which the model response is calculated.
    nparams : int
        number of model paramters.
    nlayers : int
        number of layers.
    initmdl_arr : np.ndarray
        initial model (nlayers x nparams).
    obsdat_sub : np.ndarray
        filtered data vector.
    obserr_sub : np.ndarray
        filtered error vector.
    pred_rhoa : np.ndarray
        apparent resistivit of the model response.
    obsrhoa_sub : np.ndarray
        filtered apparent resistivity of the measured datat.
    savepath_csv : string
        path to the folder where the files should be stored.
    savename : string
        name for the file.

    Raises
    ------
    ValueError
        if the ip_modeltype is not available.

    Returns
    -------
    None.

    """

    posX = position[0]
    posY = position[1]
    posZ = position[2]

    inv_res, inv_thk = model_inv[nlayers-1:nlayers*2-1], model_inv[0:nlayers-1]
    model_inv_mtrx = vecMDL2mtrx(model_inv, nlayers, nparams)
    pred_data = np.asarray(tem_inv.response())
    pred_rhoa = calc_rhoa(setup_device, pred_data,
                          rxtimes_sub)

    if ip_modeltype != None:
        inv_m = model_inv_mtrx[:, 2]
        inv_tau = model_inv_mtrx[:, 3]
        inv_c = model_inv_mtrx[:, 4]

    if ip_modeltype == 'pelton':
        header_result = 'X,Y,Z,depth(m),rho(Ohmm),m(),tau(s),c()'
        result_arr = np.column_stack((np.r_[inv_thk, 0], inv_res,
                                      inv_m, inv_tau, inv_c))
    elif ip_modeltype == 'mpa':
        header_result = 'X,Y,Z,depth(m),rho(Ohmm),mpa(rad),tau_p(s),c()'
        result_arr = np.column_stack((np.r_[inv_thk, 0], inv_res,
                                      inv_m, inv_tau, inv_c))
    elif ip_modeltype == None:
        header_result = 'X,Y,Z,depth(m),rho(Ohmm)'
        result_arr = np.column_stack((np.r_[inv_thk, 0], inv_res))
    else:
        raise ValueError('this ip modeltype is not implemented here ...')
    export_array = np.column_stack((np.full((len(result_arr),), posX),
                                    np.full((len(result_arr),), posY),
                                    np.full((len(result_arr),), posZ),
                                    result_arr))

    header_fit = ('time(s), signal_pred(V/m2), ' +
                  'signal_obs(V/m2), err_obs(V/m2), err_scl(V/m2),' +
                  'rhoa_pred(Ohmm), rhoa_obs(Ohmm)')
    export_fit = np.column_stack((rxtimes_sub, pred_data,
                                  obsdat_sub, obserr_sub, abs_err,
                                  pred_rhoa, obsrhoa_sub))

    exportSM_array = np.column_stack((np.full((len(result_arr),), posX),
                                      np.full((len(result_arr),), posY),
                                      np.full((len(result_arr),), posZ),
                                      initmdl_arr))
    
    if truemdl_arr is not None:
        exportTM_array = np.column_stack((np.full((len(result_arr),), posX),
                                          np.full((len(result_arr),), posY),
                                          np.full((len(result_arr),), posZ),
                                          truemdl_arr))

    if ip_modeltype != None:
        formatting = '%.3f,%.3f,%.3f,%.3f,%.3f,%.1f,%.1e,%.3f'
    else:
        formatting = '%.3f,%.3f,%.3f,%.3f,%.3f'
    np.savetxt(savepath_csv + savename +'.csv',
               export_array, comments='',
               header=header_result,
               fmt=formatting)
    np.savetxt(savepath_csv + savename +'_startmodel.csv',
               exportSM_array, comments='',
               header=header_result,
               fmt=formatting)
    if truemdl_arr is not None:
        np.savetxt(savepath_csv + savename +'_truemodel.csv',
                   exportTM_array, comments='',
                   header=header_result,
                   fmt=formatting)

    np.savetxt(savepath_csv + savename +'_fit.csv',
               export_fit,
               comments='',
               header=header_fit,
               fmt='%.6e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e')
    jac_df.to_csv(savepath_csv + savename + '_jac.csv')

    return