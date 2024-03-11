#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 12:52:04 2022

function and class library for blocky IP inversion

uses inversion class from joint inversion:
    https://github.com/florian-wagner/four-phase-inversion/blob/master/code/fpinv/lsqrinversion.py

@author: florian wagner
"""

# %% import modules
import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import norm
from math import sqrt

import pygimli as pg
from pygimli.utils import boxprint

from pygimli.viewer.mpl import drawModel1D


# %% functions
def lsqr(A, b, damp=0.0, x=None, maxiter=200, show=False):
    """Solve A x = b in a Least-Squares sense using LSQR algorithm.
    
    After Paige and Saunders (1982)
    from: https://github.com/florian-wagner/four-phase-inversion/blob/master/code/fpinv/mylsqr.py
    """
    if x is None:  # no starting vector
        x = pg.Vector(A.cols())
        u = b
    else:
        u = A.mult(x) - b

    beta = norm(u)
    u /= beta
    v = A.transMult(u)
    alfa = norm(v)
    v /= alfa
    Arnorm0 = alfa * 1.0
    Arnorm = Arnorm0 * 1.0
    w = v.copy()
    phiU = beta
    rhoU = alfa
    for i in range(maxiter):
        if show and (i % 10 == 0):
            print(i, Arnorm, Arnorm/Arnorm0)

        u = A.mult(v) - alfa * u
        beta = norm(u)
        if np.isclose(beta, 0.0):
            break
        u /= beta
        v = A.transMult(u) - beta * v
        alfa = norm(v)
        v /= alfa
        rho = sqrt(rhoU**2 + beta**2)
        c = rhoU / rho
        s = beta / rho
        theta = s * alfa
        rhoU = - c * alfa
        phi = c * phiU
        phiU = s * phiU
        x += (phi/rho) * w
        w = v - (theta/rho) * w
        Arnorm = phiU * alfa * abs(c)
        if Arnorm / Arnorm0 < 1e-8:
            if show:
                print(i, Arnorm, Arnorm/Arnorm0)
            break

    return x


def setup_initialmdl_constraints(constr_thk, constr_res,
                                 init_layer_thk, init_layer_res):
    """
    Function to prepare initial model constraints as required for the inversion
    of TEM data using the pyGIMLi routines by Wagner et al., (2019) which were 
    originally prepared for the four phase model inversion.

    Parameters
    ----------
    constr_thk : np.ndarray [0, 1]
        array with length=nlayers-1 containing zeros and ones only. Ones mark 
        the position where the thickness should be constrained to the 
        initial model value
    constr_res : np.ndarray [0, 1]
        array with length=nlayers-1 containing zeros and ones only. Ones mark 
        the position where the resistivity should be constrained to the 
        initial model value.
    init_layer_thk : np.ndarray, float, (m)
        initial layer thickness.
    init_layer_res : np.ndarray, float, (Ohmm)
        initial layer resistivity.

    Returns
    -------
    initmdl_pgvec : pg.vector
        vector with all 0 or 1 marking the positions of the constrained model p
        arameters.
    initmdl_arr : np.ndarray
        array containing the initial model values.
    Gi : pg.matrix
        rows equal the number of parameters that are constrained in the model 
        and the 1 mark the position of the constrained parameters.
        the number of columns are equal to the number of model parameters.
    constrain_mdl_params : boolean
        switch to indicate whether there are any constaints at all.
    param_names : list with strings
        containing the model parameter names. ['thk', 'rho']

    """

    constr_1d = np.hstack((constr_thk, constr_res))
    param_names = ['thk', 'rho']
    initmdl_pgvec = pg.Vector(np.hstack((init_layer_thk, init_layer_res)))
    initmdl_arr = np.column_stack((np.r_[init_layer_thk, 0], init_layer_res))

    constrain_mdl_params = False
    if any(constr_1d == 1):
        constrain_mdl_params = True
        print('any constrain switch==1 --> using constraints')

    ones_pos = np.where(constr_1d==1)[0]
    Gi = pg.core.RMatrix(rows=len(ones_pos), cols=len(initmdl_pgvec))

    for i, pos in enumerate(ones_pos):
        Gi.setVal(i, pos, 1)
    # cstr_vals = Gi * true_model_1d  # use true model values
    # if None it will use the start model values

    return initmdl_pgvec, initmdl_arr, Gi, constrain_mdl_params, param_names


def setup_initialipmdl_constraints(ip_modeltype,constr_thk, constr_res,
                                 constr_charg, constr_tau, constr_c,
                                 init_layer_thk, init_layer_res,
                                 init_layer_m, init_layer_tau, init_layer_c):
    """
    Function to prepare initial model constraints as required for the inversion
    of TEM data including IP effects (i.e., additional model parameters) using
    the pyGIMLi routines by Wagner et al., (2019) which were originally published
    for the four phase model inversion.

    Parameters
    ----------
    ip_modeltype : str
        Name of the CC-type model used to calculate the complex resistivity.
        available are: 
            Pelton Model ('pelton')
            Maximum phase angle model ('mpa')
            None
    constr_thk : np.ndarray [0, 1]
        array with length=nlayers-1 containing zeros and ones only. Ones mark 
        the position where the thickness should be constrained to the 
        initial model value
    constr_res : np.ndarray [0, 1]
        array with length=nlayers containing zeros and ones only. Ones mark 
        the position where the resistivity should be constrained to the 
        initial model value.
    constr_charg : np.ndarray [0, 1]
        chargeability or maximum phase angle constraints.
    constr_tau : np.ndarray [0, 1]
        time constant constraints.
    constr_c : np.ndarray [0, 1]
        dispersion coefficient constraints.
    init_layer_thk : np.ndarray, float, (m)
        initial layer thickness.
    init_layer_res : np.ndarray, float, (Ohmm)
        initial layer resistivity.
    init_layer_m : np.ndarray, float [0.0 - 1.0] (s) or (rad)
        chargeability or mpa initial model parameters.
    init_layer_tau : np.ndarray, float (s)
        time constant initial model parameters.
    init_layer_c : np.ndarray, float [0.0 - 1.0] ()
        dispersion coefficent initial model parameters.

    Raises
    ------
    ValueError
        if the selected ip_modeltype is not available.

    Returns
    -------
    initmdl_pgvec : pg.vector
        vector with all 0 or 1 marking the positions of the constrained model p
        arameters.
    initmdl_arr : np.ndarray
        array containing the initial model values.
    Gi : pg.matrix
        rows equal the number of parameters that are constrained in the model 
        and the 1 mark the position of the constrained parameters.
        the number of columns are equal to the number of model parameters.
    constrain_mdl_params : boolean
        switch to indicate whether there are any constaints at all.
    param_names : list with strings
        containing the model parameter names depending on the selected ip model

    """
    
    if ip_modeltype == 'pelton':
        constr_1d = np.hstack((constr_thk, constr_res, constr_charg, constr_tau, constr_c))
        param_names = ['thk', 'rho0','m', 'tau', 'c']
        initmdl_pgvec = pg.Vector(np.hstack((init_layer_thk, init_layer_res, init_layer_m, init_layer_tau, init_layer_c)))
        initmdl_arr = np.column_stack((np.r_[init_layer_thk, 0], init_layer_res,
                                          init_layer_m, init_layer_tau, init_layer_c))

    elif ip_modeltype == 'mpa':
        constr_1d = np.hstack((constr_thk, constr_res, constr_charg, constr_tau, constr_c))
        param_names = ['thk', 'rho0','max_pha', 'tau_phi', 'c']
        initmdl_pgvec = pg.Vector(np.hstack((init_layer_thk, init_layer_res, init_layer_m, init_layer_tau, init_layer_c)))
        initmdl_arr = np.column_stack((np.r_[init_layer_thk, 0], init_layer_res,
                                          init_layer_m, init_layer_tau, init_layer_c))

    elif ip_modeltype == None:
        constr_1d = np.hstack((constr_thk, constr_res))
        param_names = ['thk', 'rho']
        initmdl_pgvec = pg.Vector(np.hstack((init_layer_thk, init_layer_res)))
        initmdl_arr = np.column_stack((np.r_[init_layer_thk, 0], init_layer_res))

    else:
        raise ValueError('this ip modeltype is not implemented here ...')

    constrain_mdl_params = False
    if any(constr_1d == 1):
        constrain_mdl_params = True
        print('any constrain switch==1 --> using constraints')

    ones_pos = np.where(constr_1d==1)[0]
    Gi = pg.core.RMatrix(rows=len(ones_pos), cols=len(initmdl_pgvec))

    for i, pos in enumerate(ones_pos):
        Gi.setVal(i, pos, 1)
    # cstr_vals = Gi * true_model_1d  # use true model values
    # if None it will use the start model values

    return initmdl_pgvec, initmdl_arr, Gi, constrain_mdl_params, param_names


def plot_pem_stepmodel(axes, model2d, depth_limit=(40, 0), **kwargs):
    """
    function to plot a pelton model that was obtained from TEM data as 
    1D step model in 4 subplots.

    Parameters
    ----------
    axes : mpl axes object
        have to contain 4 axes in a flattened configuration.
    model2d : np.ndarray
        the pelton model params in a 2D array including:
            thickness, rho_0, m, tau and c
    depth_limit : tuple with floats, optional
        max and min depth for the y limit. The default is (40, 0).
    **kwargs : key-word arguments
        key-word arguments for the plt.plot function.

    Returns
    -------
    fig : plt.figure
        Figure object that contains the subplots.
    axes : plt.axes
        Axis object that contains the individual parameter plots.

    """
    
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


def plot_mpa_stepmodel(axes, model2d, depth_limit=(40, 0), **kwargs):
    """
    

    function to plot a maximum phase angle model that was obtained from TEM
    data as 1D step model in 4 subplots.

    Parameters
    ----------
    axes : mpl axes object
        have to contain 4 axes in a flattened configuration.
    model2d : np.ndarray
        the pelton model params in a 2D array including:
            thickness, rho_0, m, tau and c
    depth_limit : tuple with floats, optional
        max and min depth for the y limit. The default is (40, 0).
    **kwargs : key-word arguments
        key-word arguments for the plt.plot function.

    Returns
    -------
    fig : plt.figure
        Figure object that contains the subplots.
    axes : plt.axes
        Axis object that contains the individual parameter plots.

    """
    
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
    axes[1].set_xlabel(r'mpa $\phi_{\mathrm{max}}$ (rad)')
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
    axes[3].set_xlabel(r'disp. coeff. c ()')
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

    axes[2].set_title('maximum phase angle (mpa) model')

    return fig, axes





# %% data filtering
def filter_data(dmeas_norm, time_range, ip_modeltype):
    """
    function to filter TEM data to a given time range
    if there are negative voltage readings in the measured signal and the IP 
    model type is None, this function will also remove all negative readings.

    Parameters
    ----------
    dmeas_norm : pd.DataFrame
        dmeas_norm.time : observed/measured time gates before filtering (s)
        dmeas_norm.err : observed data error in V/m²
        dmeas_norm.signal : measured data normalized to V/m²
    time_range : np.ndarray [2,1], (s)
        lower and upper limit for the time range used for the filtering.
    ip_modeltype : str or None
        to decide whether negative readings are to be considered erroneous.

    Returns
    -------
    rxtimes_sub : np.ndarray
        time gates of the receiver after filtering.
    obsdat_sub : np.ndarray
        filtered observed data.
    obserr_sub : np.ndarray
        filtered error vector.
    dmeas_sub : pd.DataFrame
        dmeas_sub.time : observed/measured time gates before filtering (s)
        dmeas_sub.err : observed data error in V/m²
        dmeas_sub.signal : measured data normalized to V/m²
    time_range : np.ndarray [2,1], (s)
        lower and upper limit for the time range used for the filtering.
        slightly wider than the initial one to be reused for other functions.

    """
    
    tr0 = time_range[0]
    trN = time_range[1]

    dmeas_sub = dmeas_norm[(dmeas_norm.time>tr0) & (dmeas_norm.time<trN)]
    obsdat_sub = dmeas_sub.signal.values

    # additional filtering of negative values:
    if any(obsdat_sub < 0) and ip_modeltype == None:
        # check first if one negative reading is in the main necessary range
        # main_range = (15, 100)
        # data_main_range = dmeas_sub[(dmeas.time>main_range[0]) & (dmeas.time<main_range[1])]
        # if any(data_main_range < 0):
        #     continue                      # if so skip the snd entirely!!
        pos_first0 = np.where(obsdat_sub < 0)[0][0]
        dmeas_sub = dmeas_sub.iloc[:pos_first0-1, :]  # remove one additional point
        rxtimes_sub = dmeas_sub.time.values
        obserr_sub = dmeas_sub.err.values
        obsdat_sub = dmeas_sub.signal.values
    else:
        rxtimes_sub = dmeas_sub.time.values
        obserr_sub = dmeas_sub.err.values
        obsdat_sub = dmeas_sub.signal.values

    time_range = np.r_[np.floor(rxtimes_sub[0]*0.99 * 1e6),
                       np.ceil(rxtimes_sub[-1]*1.01 * 1e6)] * 1e-6

    return rxtimes_sub, obsdat_sub, obserr_sub, dmeas_sub, time_range



# %% forward classes

class tem_block1D_fwd(pg.frameworks.Block1DModelling):
    """
    class for blocky forward modeling of tem data with pyGIMLi and empymod, 
    preparing for a blocky inversion approach which means that the model vector
    holds both the layer thickness and the layer resistivity
    
    inherits methods from pg.frameworks.Block1DModelling, see:
        https://www.pygimli.org/pygimliapi/_generated/pygimli.frameworks.html#pygimli.frameworks.Block1DInversion

    """

    def __init__(self, empy_frwrd, return_rhoa=False,
                 nPara=1, nLayers=4, **kwargs):
        """
        Constructor

        Parameters
        ----------
        return_rhoa : boolean
            decide to return rhoa or dBz/dt
        nPara : int
            Number of parameters per layer, default is 1.
            (e.g. nPara=2 for resistivity and phase)
        nLayers : int
            Number of layers. the default is 4.
            **kwargs : key-word arguments
                key-word arguments for pg.frameworks.Block1DModelling

        """
        self._nLayers = 0
        self._nPara = nPara  # number of parameters per layer
        self._return_rhoa = return_rhoa

        self._thk = None
        self._res = None
        self._dpt = None
        self.empy_frwrd = empy_frwrd

        super(pg.frameworks.Block1DModelling, self).__init__(**kwargs)
        self._withMultiThread = True

        self.initModelSpace(nLayers)

    def response(self, model_vec):
        """Forward response for given model vector
        """
        return self.empy_frwrd.calc_response(model_vec, return_rhoa=self._return_rhoa)


class temip_block1D_fwd(pg.frameworks.Block1DModelling):
    """
    class for blocky forward modeling of tem data including IP effects with 
    pyGIMLi and empymod, preparing for a blocky inversion approach which means 
    that the model vector holds both the layer thickness and the layer resistivity,
    as well as the IP cole-cole model parameters
    
    inherits methods from pg.frameworks.Block1DModelling, see:
        https://www.pygimli.org/pygimliapi/_generated/pygimli.frameworks.html#pygimli.frameworks.Block1DInversion
    
    """
    def __init__(self, empy_frwrd, nPara=4, nLayers=4,
                 ip_mdltype='pelton', return_rhoa=False,
                 resp_trafo=None, **kwargs):
        """

        Parameters
        ----------
        empy_frwrd : empymod_forwrd class
            holds all logic to generate the TEM forward response for a given 
            instrument (currently only TEM-FAST) based on given input 
            parameters (i.e., device setup).
        nPara : int, optional
            number of model parameters excluding the layer thickness.
            The default is 4 (rho0, m, tau, c).
        nLayers : int, optional
            number of layers in the model. The default is 4.
        ip_modeltype : string, optional
            Type of CC model to calc. the complex resistivity.
            The default is 'pelton'.
        return_rhoa : boolean, optional
            decide to return rhoa or dBz/dt. 
            The default is False - i.e. dBz/dt will be returned.
        resp_trafo : None or string, optional
            choice of response transformation, 
            e.g.: lin (None), log, arsinh. The default is None.
        **kwargs : key-word arguments
            key-word arguments for pg.frameworks.Block1DModelling

        Returns
        -------
        None.

        """
        self._nLayers = 0
        self._nPara = nPara  # number of parameters per layer

        self._thk = None
        self._res = None
        self._dpt = None

        self.empy_frwrd = empy_frwrd
        self.ip_mdltype = ip_mdltype
        self.return_rhoa = return_rhoa
        self.resp_trafo = resp_trafo

        # super(temip_block1D_fwd, self).__init__(**kwargs)
        # pg.frameworks.Block1DModelling.__init__(self, **kwargs)
        super(pg.frameworks.Block1DModelling, self).__init__(**kwargs)

        self._withMultiThread = True
        self.initModelSpace(nLayers)

    def response(self, model_vec):
        """Forward response for a given model vector
        """
        return self.empy_frwrd.calc_response(model_vec,
                                             ip_modeltype=self.ip_mdltype,
                                             return_rhoa=self.return_rhoa,
                                             resp_trafo=self.resp_trafo)


# %% inversion classes
class LSQRInversion(pg.core.RInversion):
    """
    LSQR solver based inversion
    from: https://github.com/florian-wagner/four-phase-inversion/blob/master/code/fpinv/lsqrinversion.py
    
    """

    def __init__(self, *args, **kwargs):
        """Init."""
        pg.core.RInversion.__init__(self, *args, **kwargs)
        self.G = None
        self.c = None
        self.n_iters = None  # store number of iterations
        self.my = 1.0

    def setParameterConstraints(self, G, c, my=1.0):
        """Set parameter constraints G*p=c.
        """
        self.G = G
        self.c = c
        self.my = my

    def run(self, **kwargs):
        """Run the inversion
        """
        print("model (min, max)", min(self.model()), max(self.model()))
        oldphi = 0
        strtMdl = self.model()
        if (self.c is None) and (self.G is not None):
            self.c = self.G * strtMdl
            print('INFO: constraining to selected initial model values')

        for i in range(self.maxIter()):
            print('\n')
            boxprint("Iteration #%d" % i, width=80, sym="+")
            self.oneStep()
            boxprint("Iteration: %d | Lam: %d | Chi^2: %.2f | RMS: %.2f%%" %
                     (i, self.getLambda(), self.chi2(), self.relrms())
                     )
            if self.chi2() <= 1.0:
                print("Done. Reached target data misfit of chi^2 <= 1.")
                self.n_iters = i
                break
            phi = self.getPhi()
            if i > 2:
                print("Phi / oldphi", phi / oldphi)
            if (i > 10) and (phi / oldphi >
                             (1 - self.deltaPhiAbortPercent() / 100)):
                print("Done. Reached data fit criteria of delta phi < %.2f%%."
                      % self.deltaPhiAbortPercent())
                self.n_iters = i
                break
            if i + 1 == self.maxIter():
                print("Done. Maximum number of iterations reached.")
                self.n_iters = i
                break
            oldphi = phi
        return self.model()

    def oneStep(self):
        """One inversion step.
        """
        model = self.model()
        # print(model)

        if len(self.response()) != len(self.data()):
            self.setResponse(self.forwardOperator().response(model))

        self.forwardOperator().createJacobian(model)
        self.checkTransFunctions()
        tD = self.transData()
        tM = self.transModel()
        nData = self.data().size()
        #        nModel = len(model)
        self.A = pg.matrix.BlockMatrix()  # to be filled with scaled J and C matrices
        
        # part 1: data part
        J = self.forwardOperator().jacobian()
        # self.dScale = 1.0 / pg.log(self.error()+1.0)
        self.dScale = 1.0 / (
            tD.deriv(self.data()) * self.error() * self.data())
        self.leftJ = tD.deriv(self.response()) * self.dScale
        #        self.leftJ = self.dScale / tD.deriv(self.response())
        
        print(tM.deriv(model))
        
        self.rightJ = 1.0 / tM.deriv(model)
        self.JJ = pg.matrix.MultLeftRightMatrix(J, self.leftJ, self.rightJ)
        #        self.A.addMatrix(self.JJ, 0, 0)
        self.mat1 = self.A.addMatrix(self.JJ)
        self.A.addMatrixEntry(self.mat1, 0, 0)
        
        # part 2: normal constraints
        self.checkConstraints()
        self.C = self.forwardOperator().constraints()
        self.leftC = pg.Vector(self.C.rows(), 1.0)
        self.rightC = pg.Vector(self.C.cols(), 1.0)
        self.CC = pg.matrix.MultLeftRightMatrix(self.C, self.leftC,
                                                self.rightC)
        self.mat2 = self.A.addMatrix(self.CC)
        lam = self.getLambda()
        self.A.addMatrixEntry(self.mat2, nData, 0, sqrt(lam))
        
        # % part 3: parameter constraints
        if self.G is not None:
            self.rightG = 1.0 / tM.deriv(model)
            self.GG = pg.matrix.MultRightMatrix(self.G, self.rightG)
            self.mat3 = self.A.addMatrix(self.GG)
            nConst = self.C.rows()
            self.A.addMatrixEntry(self.mat3, nData + nConst, 0, sqrt(self.my))
        self.A.recalcMatrixSize()
        # right-hand side vector
        deltaD = (tD.fwd(self.data()) - tD.fwd(self.response())) * self.dScale
        deltaC = -(self.CC * tM.fwd(model) * sqrt(lam))
        deltaC *= 1.0 - self.localRegularization()  # operates on DeltaM only
        rhs = pg.cat(deltaD, deltaC)
        if self.G is not None:
            print('cstr-vals: ', self.c)
            print('cstr at 1 using mtrx:\n', np.array(self.G))
            # print('c - g*model', (self.c - self.G * model))
            deltaG = (self.c - self.G * model) * sqrt(self.my)
            # print('scaling with sqrt(my): ', sqrt(self.my))
            # print(deltaG)
            
            rhs = pg.cat(pg.cat(deltaD, deltaC), deltaG)
            # print('rhs: ', rhs)

        dM = lsqr(self.A, rhs)
        tau, responseLS = self.lineSearchInter(dM)
        if tau < 0.1:  # did not work out
            tau = self.lineSearchQuad(dM, responseLS)
        if tau > 0.9:  # save time and take 1
            tau = 1.0
        else:
            self.forwardOperator().response(self.model())

        if tau < 0.1:  # still not working
            tau = 0.1  # try a small value

        self.setModel(tM.update(self.model(), dM * tau))
        # print("model", min(self.model()), max(self.model()))
        if tau == 1.0:
            self.setResponse(responseLS)
        else:  # compute new response
            self.setResponse(self.forwardOperator().response(self.model()))

        self.setLambda(self.getLambda() * self.lambdaFactor())
        return True


    def lineSearchInter(self, dM, nTau=100):
        """Optimizes line search parameter by linear responses interpolation.
        
        """
        tD = self.transData()
        tM = self.transModel()
        model = self.model()
        response = self.response()
        modelLS = tM.update(model, dM)
        responseLS = self.forwardOperator().response(modelLS)
        taus = np.linspace(0.0, 1.0, nTau)
        phi = np.ones_like(taus) * self.getPhi()
        phi[-1] = self.getPhi(modelLS, responseLS)
        t0 = tD.fwd(response)
        t1 = tD.fwd(responseLS)
        for i in range(1, len(taus) - 1):
            tau = taus[i]
            modelI = tM.update(model, dM * tau)
            responseI = tD.inv(t1 * tau + t0 * (1.0 - tau))
            phi[i] = self.getPhi(modelI, responseI)

        return taus[np.argmin(phi)], responseLS

    def lineSearchQuad(self, dM, responseLS):
        """Optimize line search by fitting parabola by Phi(tau) curve.
        
        """
        return 0.1