#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 11:54:03 2022

uses MPA model for creating the dispersion of el. resistivity
uses inversion class from joint inversion:
    https://github.com/florian-wagner/four-phase-inversion/blob/master/code/fpinv/lsqrinversion.py

TODO:
    [Done] add versioning
    [Done] test other startmodel
    [Done] abort criteria
    [Done] constraints!!
    [] add log

    [done] iterate over two different tau values and plot into two rows!
    [] save stuff
    [] reload for plotting

    [] increase error around negative peak!?!
    [] estimate tau from peak position?!
        [] function estimate_tau()
    [] add doi
    [] save results for dgsa


generate data without noise, add noise, trafo afterwards
    [done] generate clean data
    [done] add noise
    [done] trafo afterwards

test new data transformations
    [done] kth root
    [done] arsinh

Graphite test case with negative and positive IP effect
(few readings with reversed sign, two different tau values!)


v10: same number of layers for the inversion as for the true model

@author: laigner
"""

# %% import modules
import os
import sys
import logging

rel_path_to_libs = '../../../'
if not rel_path_to_libs in sys.path:
    sys.path.append('../../../')  # add relative path to folder that contains all custom modules


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.colors as clr
import seaborn as sns

from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.offsetbox import AnchoredText
from numpy.linalg import norm
from math import sqrt

import pygimli as pg
from pygimli.utils import boxprint

# import custom modules from additional sys.path
from library.TEM_frwrd.empymod_frwrd_ip import empymod_frwrd
from library.TEM_frwrd.utils import arsinh
from library.TEM_frwrd.utils import kth_root

from library.utils.TEM_ip_tools import prep_mdl_para_names
# from library.utils.TEM_ip_tools import get_phimax_from_CCR
# from library.utils.TEM_ip_tools import get_tauphi_from_tr
# from library.utils.TEM_ip_tools import plot_diffs

from library.utils.universal_tools import plot_signal
from library.utils.universal_tools import plot_rhoa
from library.utils.universal_tools import calc_rhoa
from library.utils.universal_tools import simulate_error
from library.utils.universal_tools import query_yes_no

from library.utils.TEM_inv_tools import vecMDL2mtrx
from library.utils.TEM_inv_tools import mtrxMDL2vec
from library.utils.TEM_inv_tools import save_result_and_fit

from library.utils.timer import Timer

# from pg_temip_invtools import lsqr
from library.TEM_inv.pg_temip_inv import temip_block1D_fwd
from library.TEM_inv.pg_temip_inv import LSQRInversion


# %% plot appearance
dpi = 200
plt.style.use('ggplot')

fs_shift = -5
plt.rcParams['axes.labelsize'] = 18 + fs_shift
plt.rcParams['axes.titlesize'] = 18 + fs_shift
plt.rcParams['xtick.labelsize'] = 16 + fs_shift
plt.rcParams['ytick.labelsize'] = 16 + fs_shift
plt.rcParams['legend.fontsize'] = 18 + fs_shift

lims_time = (2e-6, 3e-3)
lims_signal = (1e-10, 1e-2)
lims_rhoa = (1e0, 1e4)
lims_depth = (40, 0)

# lims_rho0 = (0.5, 1e3)
lims_rho0 = (1, 1e3)
lims_mpa = (-0.1, 1.1)
lims_tau = (5e-7, 8e-2)
lims_c = (-0.1, 1.1)


# %% setup script
savefigs = True
# savefigs = False

show_simulated_data = True
# show_simulated_data = False

start_inv = True
# start_inv = False

max_depth = 55
nlayers = 3


scriptname = os.path.basename(sys.argv[0])
print(f'running {scriptname} ...')
version = scriptname.split('.')[0].split('_')[-1]
inv_typ = scriptname.split('.')[0].split('_')[-2]
case = scriptname.split('.')[0].split('_')[-3]

main = './'
savepath = main + f'04-vis/{case}/{version}/'
if not os.path.exists(savepath):
    os.makedirs(savepath)

savepath_data = main + f'03-inv_results/{inv_typ}/{version}/'
if not os.path.exists(savepath_data):
    os.makedirs(savepath_data)

t = Timer()


# %% setup solver
device = 'TEMfast'
setup_device = {"timekey": 5,
                "currentkey": 4,
                "txloop": 12.5,  #6.25, 12.5, 25
                "rxloop": 12.5,
                "current_inj": 4.0,
                "filter_powerline": 50,
                "ramp_data": 'donauinsel'}

# 'ftarg': 'key_81_CosSin_2009', 'key_201_CosSin_2012', 'ftarg': 'key_601_CosSin_2009'
setup_solver = {'ft': 'dlf',                     # type of fourier trafo
                  'ftarg': 'key_601_CosSin_2009',  # ft-argument; filter type # https://empymod.emsig.xyz/en/stable/api/filters.html#module-empymod.filters -- for filter names
                  'verbose': 0,                    # level of verbosity (0-4) - larger, more info
                  'srcpts': 3,                     # Approx. the finite dip. with x points. Number of integration points for bipole source/receiver, default is 1:, srcpts/recpts < 3 : bipole, but calculated as dipole at centre
                  'recpts': 3,                     # Approx. the finite dip. with x points. srcpts/recpts >= 3 : bipole
                  'ht': 'dlf',                     # type of fourier trafo
                  'htarg': 'key_401_2009',         # hankel transform filter type, 'key_401_2009', 'key_101_2009'
                  'nquad': 3,                     # Number of Gauss-Legendre points for the integration. Default is 3.
                  'cutoff_f': 1e8,               # cut-off freq of butterworthtype filter - None: No filter applied, WalkTEM 4.5e5
                  'delay_rst': 0,                 # ?? unknown para for walktem - keep at 0 for fasttem
                  'rxloop': 'vert. dipole'}       # or 'same as txloop' - not yet operational


# %% setup inversion parameters
max_iter = 25  # 25
lam = 1000
cooling_factor = 0.8
noise_floor = 0.04  # lowest relative error accepted in the error vector (%/100)
my = 1e2

relerr = 0.04
abserr = 1e-10
noise_factor = 1

tx_loop = 12.5

# resp_trafo = 'oddroot_7'
# resp_trafo = 'areasinhyp'
resp_trafo = None

return_rhoa = False


# %% setup figure! (two rows)
fig_result = plt.figure(figsize=(14, 10), constrained_layout=False)
gs = fig_result.add_gridspec(2, 6)

ax_fit = fig_result.add_subplot(gs[0, 0:2])
ax_rho = fig_result.add_subplot(gs[0, 2:3])
ax_mpa = fig_result.add_subplot(gs[0, 3:4])
ax_tau = fig_result.add_subplot(gs[0, 4:5])
ax_c = fig_result.add_subplot(gs[0, 5:6])

ax_fit1 = fig_result.add_subplot(gs[1, 0:2])
ax_rho1 = fig_result.add_subplot(gs[1, 2:3])
ax_mpa1 = fig_result.add_subplot(gs[1, 3:4])
ax_tau1 = fig_result.add_subplot(gs[1, 4:5])
ax_c1 = fig_result.add_subplot(gs[1, 5:6])

axes_fit = np.array([ax_fit, ax_fit1])
axes_mdl = np.array([[ax_rho, ax_mpa, ax_tau, ax_c],
                     [ax_rho1, ax_mpa1, ax_tau1, ax_c1]])

ip_effect_names = ['IP_p', 'IP_m']
tau_vals = [5e-2, 5e-4]

# rho_bg_s = [200, 20]
# rho_ip_s = [30, 30]
rho_bg_s = [50, 50]
rho_ip_s = [50, 50]


# prepare inversion protokoll
main_prot_fid = savepath_data + f'{case}_{version}.log'
invprot_hdr = ('name\tri\tminT(us)\tmaxT(us)\tlam\tmy\tcf\tnoifl\t' +
               'max_iter\tn_iter\taRMS\trRMS\tchi2\truntime(min)')
if start_inv:
    with open(main_prot_fid, 'w') as main_prot:
        main_prot.write(invprot_hdr + '\n')


# %% iterate over two different tau values
for j, tau_val in enumerate(tau_vals):

    # %% setup model
    # ip_modeltype = 'pelton'  # mpa
    ip_modeltype = 'mpa'  # mpa

    thks = np.r_[8, 12, 0]
    rho_0 = np.r_[50, 10, 500]

    phi_max = np.r_[0.0, 0.8, 0.0]
    tau_phi = np.r_[1e-6, tau_val, 1e-6]
    cs = np.r_[0.01, 0.9, 0.01]

    ip_depth_range = (0.1, 1.0)
    bottom_ip_switch = False

    # init_layer_thk = np.diff(np.r_[np.logspace(-1, 0.0, nlayers, endpoint=True)] * max_depth)
    init_layer_thk = np.r_[10, 10]
    init_thkcum = np.cumsum(init_layer_thk)

    lower_bound_ip = np.quantile(init_thkcum, ip_depth_range[0])
    upper_bound_ip = np.quantile(init_thkcum, ip_depth_range[1])

    condition = (init_thkcum >= lower_bound_ip) & (init_thkcum <= upper_bound_ip)
    mask_ip_layers = np.r_[condition, bottom_ip_switch]

    init_rho_background = rho_bg_s[j]
    init_rho_ip = rho_ip_s[j]
    initpmax = 0.5
    # inittaup = 5e-3
    inittaup = tau_val / 5
    
    initc = 0.5

    inv_run = 0
    name_snd = ip_effect_names[j]


    #%% setup start model
    # thickness
    constr_thk = np.zeros_like(init_layer_thk)
    
    # resistivity
    init_layer_res = np.ones((len(init_layer_thk) + 1,)) * init_rho_background
    init_layer_res[mask_ip_layers] = init_rho_ip
    constr_res = np.zeros_like(init_layer_res)
    
    # chargeability, mpa
    init_layer_mpa = np.full_like(constr_res, 0.0)
    init_layer_mpa[mask_ip_layers] = initpmax
    constr_mpa = np.ones_like(init_layer_mpa)
    constr_mpa[mask_ip_layers] = 0.0
    
    # tau and tau phi
    init_layer_taup = np.full_like(constr_res, 1e-6)
    init_layer_taup[mask_ip_layers] = inittaup
    constr_taup = np.ones_like(init_layer_taup)
    constr_taup[mask_ip_layers] = 0.0
    
    # dispersion coefficient
    init_layer_c = np.full_like(constr_res, 0.01)
    init_layer_c[mask_ip_layers] = initc
    constr_c = np.ones_like(init_layer_c)
    constr_c[mask_ip_layers] = 0.0
    
    
    if ip_modeltype == 'pelton':
        truemdl_arr = np.column_stack((thks, rho_0, charg, taus, cs))  # pelton model
        constr_vec = np.hstack((constr_thk, constr_res, constr_m, constr_tau, constr_c))
        initmdl_vec = pg.Vector(np.hstack((init_layer_thk, init_layer_res, init_layer_m, init_layer_tau, init_layer_c)))
        param_names = ['thk', 'rho0','m', 'tau', 'c']
    elif ip_modeltype == 'mpa':
        truemdl_arr = np.column_stack((thks, rho_0, phi_max, tau_phi, cs))  # mpa model
        constr_vec = np.hstack((constr_thk, constr_res, constr_mpa, constr_taup, constr_c))
        initmdl_vec = pg.Vector(np.hstack((init_layer_thk, init_layer_res, init_layer_mpa, init_layer_taup, init_layer_c)))
        param_names = ['thk', 'rho0','max_pha', 'tau_phi', 'c']
    else:
        raise ValueError('this ip modeltype is not implemented here ...')
    
    nlayers_tm = truemdl_arr.shape[0]
    nlayers = len(init_layer_res)
    nparams = truemdl_arr.shape[1]  # same for true model and initial model
    
    truemdl_vec = mtrxMDL2vec(truemdl_arr)
    initmdl_arr = vecMDL2mtrx(initmdl_vec, nLayer=nlayers, nParam=nparams)
    mdl_para_names = prep_mdl_para_names(param_names, n_layers=len(init_layer_res))
    
    transThk = pg.trans.TransLogLU(3, 30)  # log-transform ensures thk>0
    transRho = pg.trans.TransLogLU(5, 1000)  # lower and upper bound
    transM = pg.trans.TransLogLU(0.0001, 1.3)  # lower and upper bound
    transTau = pg.trans.TransLogLU(1e-6, 1e-1)  # lower and upper bound
    transC = pg.trans.TransLogLU(0.0001, 1.0)  # lower and upper bound
    # transData = pg.trans.TransLog()  # log transformation for data
    transData = pg.trans.TransLin()  # lin transformation for data


    # %% prepare constraints:
    constrain_mdl_params = False
    if any(constr_vec == 1):
        constrain_mdl_params = True
        print('any constrain switch=1 --> using constraints')
    ones_pos = np.where(constr_vec==1)[0]
    Gi = pg.core.RMatrix(rows=len(ones_pos), cols=len(initmdl_vec))

    for idx, pos in enumerate(ones_pos):
        Gi.setVal(idx, pos, 1)
    # cstr_vals = Gi * true_model_1d  # use true model values
    cstr_vals = None  # if None it will use the start model values


    # %% plot start model
    fig_im, ax_im = plt.subplots(2, 2, figsize=(9, 9), constrained_layout=True)

    if ip_modeltype == 'pelton':
        from library.utils.TEM_ip_tools import plot_pem_stepmodel
        plot_pem_stepmodel(axes=ax_im, model2d=truemdl_arr, label='true model',
                           color='black', ls='-')
        plot_pem_stepmodel(axes=ax_im, model2d=initmdl_arr, label='initial model',
                           color='green', ls='--')

    elif ip_modeltype == 'mpa':
        from library.utils.TEM_ip_tools import plot_mpa_stepmodel
        plot_mpa_stepmodel(axes=ax_im, model2d=truemdl_arr, label='true model',
                           color='black', ls='-')
        plot_mpa_stepmodel(axes=ax_im, model2d=initmdl_arr, label='initial model',
                           color='green', ls='--')

    else:
        raise ValueError('this ip modeltype is not implemented here ...')

    plt.suptitle(f'model with tau = {tau_val} s')
    if savefigs:
        fig_im.savefig(savepath + f'01a-initial_model_{ip_modeltype}_mdl{j:02d}.png', dpi=dpi)


    # %% simulate data and add noise
    # if query_yes_no('continue with forward testing?', default='no'):
    # setup system and forward solver
    frwrd_empymod_tm = empymod_frwrd(setup_device=setup_device,
                            setup_solver=setup_solver,
                            time_range=None, device='TEMfast',
                            relerr=1e-6, abserr=1e-28,
                            nlayer=nlayers_tm, nparam=nparams)
    times_rx = frwrd_empymod_tm.times_rx


    # setup system and forward solver for inversion
    frwrd_empymod = empymod_frwrd(setup_device=setup_device,
                            setup_solver=setup_solver,
                            time_range=None, device='TEMfast',
                            relerr=1e-6, abserr=1e-28,
                            nlayer=nlayers, nparam=nparams)
    times_rx = frwrd_empymod.times_rx


    numdat = frwrd_empymod_tm.calc_response(model=truemdl_vec,
                                           ip_modeltype=ip_modeltype,  # 'cole_cole', 'cc_kozhe'
                                           show_wf=False, return_rhoa=return_rhoa,
                                           resp_trafo=None)

    numnoise = simulate_error(relerr, abserr, data=numdat)
    numdat_noisy = numnoise + numdat
    simrhoa_noisy = calc_rhoa(setup_device, numdat_noisy, times_rx)

    relerr_est = abs(numnoise) / numdat_noisy
    if any(relerr_est < noise_floor):
        logging.warning(f'Encountered rel. error below {noise_floor*100}% - setting those to {noise_floor*100}%')
        relerr_est[relerr_est < noise_floor] = noise_floor
    error_abs = abs(numdat_noisy * relerr_est)


    # transform data and error!
    if return_rhoa:
        pass

    else:
        if resp_trafo == None:
            print('no transformation done ...')

        elif resp_trafo == 'min_to_1':
            numdat_trafo = numdat + np.min(numdat) + 1
            # numnoise_trafo = numnoise + np.min(numnoise) + 1
            # numdat_noisy_trafo = numdat_noisy + np.min(numdat_noisy) + 1
            # error_abs_trafo = error_abs + np.min(error_abs) + 1

        elif resp_trafo == 'areasinhyp':
            numdat_trafo = arsinh(numdat)
            # numnoise_trafo = arsinh(numnoise)
            # numdat_noisy_trafo = arsinh(numdat_noisy)
            # error_abs_trafo = arsinh(error_abs)

        elif 'oddroot' in resp_trafo:
            root = int(resp_trafo.split('_')[1])
            numdat_trafo = kth_root(numdat, root)
            
            numnoise_trafo = simulate_error(relerr, abserr, data=numdat_trafo)
            numdat_noisy_trafo = numnoise_trafo + numdat_trafo

            relerr_est_tf = abs(numnoise_trafo) / numdat_noisy_trafo
            if any(relerr_est_tf < noise_floor):
                logging.warning(f'Encountered rel. error below {noise_floor*100}% - setting those to {noise_floor*100}%')
                relerr_est_tf[relerr_est_tf < noise_floor] = noise_floor
            error_abs_trafo = abs(numdat_noisy_trafo * relerr_est_tf)
            # numnoise_trafo = kth_root(numnoise, root)
            # numdat_noisy_trafo = kth_root(numdat_noisy, root)
            # error_abs_trafo = kth_root(error_abs, root)

        else:
            raise ValueError('This response transformation is not available!')


    if show_simulated_data:
        fig_simd, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6),
                                            sharex=True)

        plot_signal(axis=ax1, time=times_rx, signal=numdat, color='gray', ls='-',
                    label='clean data (true model)')
        plot_signal(axis=ax1, time=times_rx, signal=numdat_noisy,
                    ls='--', marker='.', color='black', label='noisy data')

        ax1.set_title(f'noise: rel-err: {relerr*100}%, abs-err: {abserr}')
        ax1.loglog(times_rx, abs(numnoise), color='gray', ls=':', label='numerical noise')
        ax1.loglog(times_rx, abs(error_abs), color='gray', ls='--', label=f'noise floor {noise_floor*100} %')

        if resp_trafo is not None:
            plot_signal(axis=ax2, time=times_rx, signal=numdat_trafo, color='gray', ls='-',
                        label='clean data (true model)')
            plot_signal(axis=ax2, time=times_rx, signal=numdat_noisy_trafo,
                        ls='--', marker='.', color='black', label='noisy data')

            ax2.set_title(f'transformed data and error - {resp_trafo}')
            ax2.loglog(times_rx, abs(numnoise_trafo), color='gray', ls=':', label='numerical noise')
            ax2.loglog(times_rx, abs(error_abs_trafo), color='gray', ls='--', label=f'noise floor {noise_floor*100} %')

            ax1.set_ylabel(r"$\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t$ (V/m²)")
            ax2.set_ylabel(r"transformed $\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t$ (V/m²)")

        ax1.set_xlabel('time (s)')
        ax2.set_xlabel('time (s)')

        plt.suptitle(f'model with tau = {tau_val} s')
        if savefigs:
            plt.savefig(savepath + f'01b-frwrd-comp_{ip_modeltype}_mdl{j:02d}.png', dpi=dpi)

    mean_rhoa = abs(np.mean(calc_rhoa(setup_device, numdat_noisy, times_rx)))



    # %% prepare start model and fop
    fop = temip_block1D_fwd(frwrd_empymod, nPara=nparams-1,
                            nLayers=nlayers, verbose=True,
                            return_rhoa=return_rhoa,
                            resp_trafo=resp_trafo)

    t.start()
    startmdl_response = fop.response(initmdl_vec)
    frwrd_time = t.stop(prefix='forward-')

    if show_simulated_data:
        _ = plot_signal(axis=ax2, time=times_rx, signal=startmdl_response,
                        marker='.', ls='--', color='green', label='response initial model')
        ax1.legend()
        ax2.legend()

    plt.savefig(savepath + 'data_temip_simdata.png', dpi=dpi)

    fop.region(0).setTransModel(transThk)  # 0=thickness
    fop.region(1).setTransModel(transRho)  # 1=resistivity
    fop.region(2).setTransModel(transM)    # 2=m
    fop.region(3).setTransModel(transTau)  # 3=tau
    fop.region(4).setTransModel(transC)    # 4=c

    fop.setMultiThreadJacobian(1)


    savename = ('invrun{:03d}_{:s}'.format(inv_run, name_snd))
    print(f'saving data from inversion run: {inv_run}')
    position = (0, 0, 0)  # location of sounding

    savepath_csv = savepath_data + f'{name_snd}/csv/'
    if not os.path.exists(savepath_csv):
        os.makedirs(savepath_csv)

    # prepare inversion protokoll for individual sounding
    snd_prot_fid = savepath_csv.replace('csv/', '') + f'{case}_snd{name_snd}.log'
    
    tr0 = times_rx[0]
    trN = times_rx[1]


    # %% setup inversion
    if start_inv:          # if query_yes_no('continue with inversion?', default='no'):
        with open(snd_prot_fid, 'w') as snd_prot:
            snd_prot.write(invprot_hdr + '\n')

        inv = LSQRInversion(verbose=True)
        inv.setTransData(transData)
        inv.setForwardOperator(fop)

        inv.setMaxIter(max_iter)
        inv.setLambda(lam)  # (initial) regularization parameter
        inv.setMarquardtScheme(cooling_factor)  # decrease lambda by factor 0.9
        inv.setModel(initmdl_vec)  # set start model

        if resp_trafo == None:
            inv.setData(numdat_noisy)
            inv.setAbsoluteError(error_abs)  # use the numerical noise
        else:
            inv.setData(numdat_noisy_trafo)
            inv.setAbsoluteError(error_abs_trafo)  # use the transformed numerical noise

        inv.saveModelHistory(True)


        # %% set constraints
        if constrain_mdl_params:
            inv.setParameterConstraints(G=Gi, c=cstr_vals, my=my)
        else:
            print('no constraints used ...')
    
        # %% start inversion and obtain relevant info
        t.start()
        invmdl_vec = inv.run()
        inv_time = t.stop(prefix='inv-')
    
        res_inv, thk_inv = invmdl_vec[nlayers-1:nlayers*2-1], invmdl_vec[0:nlayers-1]
        chi2 = inv.chi2()
        relrms = inv.relrms()
        absrms = inv.absrms()
        n_iter = inv.n_iters  # get number of iterations
    
        response = inv.response()
    
        invmdl_arr = vecMDL2mtrx(invmdl_vec, nlayers, nparams)
        inv_m = invmdl_arr[:, 2]
        inv_tau = invmdl_arr[:, 3]
        inv_c = invmdl_arr[:, 4]
    
        fop = inv.fop()
        col_names = mdl_para_names
        row_names = [f'dB/dt tg{i:02d}' for i in range(1, len(numdat_noisy)+1)]
        jac_df = pd.DataFrame(np.array(fop.jacobian()), columns=col_names, index=row_names)
    
        # TODO convert to m, tau again!!
        # plot separately!!
    
    
        # %% save result and fit
        save_result_and_fit(inv, setup_device, invmdl_vec, jac_df, ip_modeltype, position,
                            rxtimes_sub=times_rx, nparams=nparams, nlayers=nlayers,
                            initmdl_arr=initmdl_arr, obsdat_sub=numdat_noisy,
                            obserr_sub=numnoise, abs_err=error_abs, 
                            obsrhoa_sub=np.full_like(simrhoa_noisy, np.nan),
                            savepath_csv=savepath_csv, savename=savename, truemdl_arr=truemdl_arr)


        # %% compare result to start model and true model
        np.set_printoptions(precision=3, linewidth=300, suppress=False)
        print('\n', '###'*10, '\n')
        print(f'chi2: {chi2:.2f}')
        print(f'relRMS: {relrms:.2f}')  #pg.sum((0, nlay - 1))
        print('start model: ', np.asarray(initmdl_vec))
        print('cstr at ones: ', constr_vec)
        print('solved model: ', np.asarray(invmdl_vec))
        print('true model: ', np.asarray(truemdl_vec))
        # print('model diff: ', np.asarray(invmdl_vec - true_model_1d))


        # %% save a log file
        logline = ("%s\t" % (name_snd) +
                   "r%03d\t" % (inv_run) +
                   "%.1f\t" % (tr0) +
                   "%.1f\t" % (trN) +
                   "%8.1f\t" % (lam) +
                   "%.1e\t" % (my) +
                   "%.1e\t" % (cooling_factor) +  # cooling factor
                   "%.2f\t" % (noise_floor) +
                   "%d\t" % (max_iter) +
                   "%d\t" % (n_iter) +
                   "%.2e\t" % (absrms) +
                   "%7.3f\t" % (relrms) +
                   "%7.3f\t" % (chi2) +
                   "%4.1f\n" % (inv_time[0]/60))  # to min
        with open(main_prot_fid,'a+') as f:
            f.write(logline)
        with open(snd_prot_fid,'a+') as f:
            f.write(logline)


        # %% prepare the figure for the data fit and the model params

        # plot the data fit with trafo and rhoa
        if resp_trafo is not None:
            print('\noverriding response with the non-trafo response ...')
            response_trafo = np.copy(response)
            
            if 'oddroot' in resp_trafo:
                root = int(resp_trafo.split('_')[1])
                response = response_trafo**root
            else:
                response = frwrd_empymod.calc_response(invmdl_arr, ip_modeltype=ip_modeltype,
                                                       show_wf=False, return_rhoa=return_rhoa,
                                                       resp_trafo=resp_trafo)

            fig_tf, ax_tf = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True,
                                    sharex=True)

            _ = plot_signal(ax_tf[0], time=times_rx, signal=numdat_noisy_trafo, label='data',
                            marker='o', color='k', sub0color='gray',
                            sub0marker='d', sub0label='negative data')
            ax_tf[0].loglog(times_rx, abs(error_abs_trafo), color='gray', ls='--', label='noise floor')
            _ = plot_signal(ax_tf[0], time=times_rx, signal=response_trafo, label='response',
                            marker='.', ls='--', color='crimson',
                            sub0marker='s', sub0label='negative response')

            ax_tf[0].legend()
            ax_tf[0].set_xlabel('Time (s)')
            ax_tf[0].set_ylabel(r"$\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t$ (V/m²)")
            ax_tf[0].set_title(f'data transformed - {resp_trafo}')


            rhoa_response = calc_rhoa(setup_device, np.asarray(response), times_rx)
            _ = plot_rhoa(ax_tf[1], time=times_rx, rhoa=simrhoa_noisy, label='sim rhoa',
                          marker='o', color='k', sub0color='gray')
            _ = plot_rhoa(ax_tf[1], time=times_rx, rhoa=rhoa_response, label='response rhoa',
                          marker='.', ls='--', color='crimson', sub0color='orange')

            ax_tf[1].legend()
            ax_tf[1].set_xlabel('Time (s)')
            ax_tf[1].set_ylabel(r'$\rho_a (\Omega m)$')
            ax_tf[1].set_ylim(lims_rhoa)

            fig_tf.savefig(savepath + f'temIP_numdata-trafo-rhoa_{ip_modeltype}_mdl{j:02d}.png', dpi=dpi)


        # %% plot the data fit - without trafo into result plot
        _ = plot_signal(axes_fit[j], time=times_rx, signal=numdat_noisy, label='data',
                        marker='o', color='k', sub0color='gray',
                        sub0marker='d', sub0label='negative data')
        axes_fit[j].loglog(times_rx, abs(error_abs), color='gray', ls='--', label='noise floor')

        _ = plot_signal(axes_fit[j], time=times_rx, signal=response, label='response',
                        marker='.', ls='--', color='crimson',
                        sub0marker='s', sub0label='negative response')
        
        lines, labels = axes_fit[j].get_legend_handles_labels()

        axes_fit[j].legend(loc='lower left')
        axes_fit[j].set_xlabel('Time (s)')
        axes_fit[j].set_ylabel(r"$\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t$ (V/m²)")
        axes_fit[j].set_xlim(lims_time)
        axes_fit[j].set_ylim(lims_signal)
        axes_fit[j].set_title(f'$\chi_2$ = {chi2:0.1f}, rRMS = {relrms:0.1f}%')


        # %% plot the model
        if ip_modeltype == 'pelton':
            plot_pem_stepmodel(axes=axes_mdl[j, :], model2d=truemdl_arr, label='true',
                               color='black', ls='-', depth_limit=lims_depth)
            plot_pem_stepmodel(axes=axes_mdl[j, :], model2d=initmdl_arr, label='initial',
                               color='green', ls='--', marker='.', depth_limit=lims_depth)
            plot_pem_stepmodel(axes=axes_mdl[j, :], model2d=invmdl_arr, label='inverted',
                               color='green', ls='--', marker='.', depth_limit=lims_depth)

        elif ip_modeltype == 'mpa':
            plot_mpa_stepmodel(axes=axes_mdl[j, :], model2d=truemdl_arr, label='true',
                               color='black', ls='-', depth_limit=lims_depth)
            plot_mpa_stepmodel(axes=axes_mdl[j, :], model2d=initmdl_arr, label='initial',
                               color='green', ls='--', marker='.', depth_limit=lims_depth)
            plot_mpa_stepmodel(axes=axes_mdl[j, :], model2d=invmdl_arr, label='inverted',
                               color="crimson", ls='-.', marker='.', depth_limit=lims_depth)

        else:
            raise ValueError('this ip modeltype is not implemented here ...')
        
        axes_mdl[j, 0].set_xlim()
        axes_mdl[j, 1].set_xlim(lims_mpa)
        axes_mdl[j, 2].set_xlim(lims_tau)
        axes_mdl[j, 3].set_xlim(lims_c)

        gs.tight_layout(fig_result)


        # %% plot the jacobian matrix
        vmin = -1e-5  # todo automatize
        vmax = abs(vmin)
        norm_jac = SymLogNorm(linthresh=3, linscale=3,
                          vmin=vmin, vmax=vmax, base=10)
        # norm = SymLogNorm(linthresh=0.3, linscale=0.3, base=10)

        plt.figure(figsize=(12, 8))
        axj = sns.heatmap(jac_df, cmap="BrBG", annot=True,
                          fmt='.2g', robust=True, center=0,
                          vmin=vmin, vmax=vmax, norm=norm_jac)  # 
        axj.set_title('jacobian last iteration')
        axj.set_xlabel('model parameters')
        axj.set_ylabel('data parameters')
        plt.tight_layout()
        figj = axj.get_figure()
        
        if savefigs:
            figj.savefig(savepath + 'jacobian_temip_simdata.png', dpi=dpi, bbox_inches='tight')

    axes_mdl[0, 2].set_title('maximum phase angle (mpa) model')
    # add labels:
    all_axs = fig_result.get_axes()
    tags = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)']
    for idx, tag in enumerate(tags):
        at = AnchoredText(tag,
                          prop={'color': 'k', 'fontsize': 14}, frameon=True,
                          loc='upper right')
        at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        all_axs[idx].add_artist(at)

    if savefigs:
        savename = f'temIP_numdata-result_{ip_modeltype}.png'
        print('saving to: ', savepath + savename)
        fig_result.savefig(savepath + savename, dpi=dpi, bbox_inches='tight')




# TODO reload results, plot

# plot from class?
# or use old routines?

# update DGSA, reload result here and save full figure

# save sensitivity, add inv version and dgsa version to scriptname







