#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 13:34:02 2021

script for a basic blocky inversion of synthetic tem data with empymod
based upon empymod (forward solver) and pygimli (inversion framework)
uses MPA model for creating the dispersion of el. resistivity

uses inversion class from joint inversion:
    https://github.com/florian-wagner/four-phase-inversion/blob/master/code/fpinv/lsqrinversion.py

TODO:
    [Done] add versioning
    [Done] test other startmodel
    [Done] abort criteria
    [Done] constraints!!

    [] save stuff
    [] reload for plotting
    [] add log

v00:
    new version, preparing for merging with dgsa results

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
import seaborn as sns
# import matplotlib.colors as clr

from matplotlib.offsetbox import AnchoredText
from numpy.linalg import norm
from matplotlib.colors import SymLogNorm

import pygimli as pg
# from pygimli.utils import boxprint

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

fs_shift = -7
plt.rcParams['axes.labelsize'] = 18 + fs_shift
plt.rcParams['axes.titlesize'] = 18 + fs_shift
plt.rcParams['xtick.labelsize'] = 16 + fs_shift
plt.rcParams['ytick.labelsize'] = 16 + fs_shift
plt.rcParams['legend.fontsize'] = 18 + fs_shift

lims_time = (2e-6, 2e-3)
lims_signal = (1e-11, 1e-1)
lims_rhoa = (1e0, 1e4)

lims_depth = (45, 0)
lims_rho = (1e2, 1e4)
lims_pol = (-0.1, 1.1)
lims_tau = (1e-7, 1e-2)
lims_c = (-0.1, 1.1)


# %% setup script
savefigs = True
# savefigs = False

start_inv = True
# start_inv = False

show_simulated_data = True

return_rhoa = False

resp_trafo = None

t = Timer()


scriptname = os.path.basename(sys.argv[0])
print(f'running {scriptname} ...')
version = scriptname.split('.')[0].split('_')[-1]
inv_typ = scriptname.split('.')[0].split('_')[-2]
case = scriptname.split('.')[0].split('_')[-3]

main = './'
savepath = main + f'plots/{case}_{inv_typ}/{version}/'
if not os.path.exists(savepath):
    os.makedirs(savepath)

savepath_data = main + f'30_inv-results/{case}_{inv_typ}/{version}/'
if not os.path.exists(savepath_data):
    os.makedirs(savepath_data)


# %% setup solver
device = 'TEMfast'
setup_device = {"timekey": 5,
            "currentkey": 4,
            "txloop": 50,
            "rxloop": 50,
            "current_inj": 4.1,
            "filter_powerline": 50,
            "ramp_data": 'sonnblick'}

# 'ftarg': 'key_81_CosSin_2009', 'key_201_CosSin_2012', 'ftarg': 'key_601_CosSin_2009'
setup_solver = {'ft': 'dlf',                     # type of fourier trafo
                'ftarg': 'key_601_CosSin_2009',  # ft-argument; filter type # https://empymod.emsig.xyz/en/stable/api/filters.html#module-empymod.filters -- for filter names
                'verbose': 0,                    # level of verbosity (0-4) - larger, more info
                'srcpts': 3,                     # Approx. the finite dip. with x points. Number of integration points for bipole source/receiver, default is 1:, srcpts/recpts < 3 : bipole, but calculated as dipole at centre
                'recpts': 3,                     # Approx. the finite dip. with x points. srcpts/recpts >= 3 : bipole
                'ht': 'dlf',                     # type of fourier trafo
                'htarg': 'key_401_2009',         # hankel transform filter type, 'key_401_2009', 'key_101_2009'
                'nquad': 3,                      # Number of Gauss-Legendre points for the integration. Default is 3.
                'cutoff_f': 1e8,                 # cut-off freq of butterworthtype filter - None: No filter applied, WalkTEM 4.5e5, TEM-FAST either 5e6 (large loop) or 1e8 (small loop)
                'delay_rst': 0,                  # ?? unknown para for walktem - keep at 0 for fasttem
                'rxloop': 'vert. dipole'}        # or 'same as txloop' - not yet operational


# %% setup inversion parameters
lam = 420
cooling_factor = 0.9
max_iter = 25

relerr = 0.03
abserr = 1e-10
noise_factor = 1

noise_floor = 0.07  # lowest relative noise accepted in the error (%/100)
my = 1e2


# %% setup model
ip_modeltype = 'pelton'  # mpa
ip_modeltype = 'mpa'  # mpa

thks = np.r_[10, 15, 0]
rho_0 = np.r_[500, 5000, 300]
cs = np.r_[0.01, 0.9, 0.01]
phi_max = np.r_[0.0, 0.8, 0.0]
tau_phi = np.r_[1e-6, 5e-4, 1e-6]


#%% setup start model, 3 layers
# todo find useful start model values for convergence, estimate tau from data curve??, get initial resistivity values??
init_layer_thk = np.r_[11, 14]
constr_thk = np.r_[0, 0]  # set to 1 if parameter should be fixed

init_layer_res = np.r_[400, 6000, 400]
constr_res = np.r_[0, 0, 0]

init_layer_mpa = np.r_[0.0, 0.6, 0.0]
constr_mpa = np.r_[1, 0, 1]

init_layer_taup = np.array([1e-6, 1e-4, 1e-6])
constr_taup = np.r_[1, 0, 1]

init_layer_c = np.r_[0.01, 0.6, 0.01]
constr_c = np.r_[1, 0, 1]


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

transThk = pg.trans.TransLogLU(8, 24)  # log-transform ensures thk>0
transRho = pg.trans.TransLogLU(200, 8000)  # lower and upper bound
transM = pg.trans.TransLogLU(0.0001, 1.1)  # lower and upper bound
transTau = pg.trans.TransLogLU(1e-6, 1e-3)  # lower and upper bound
transC = pg.trans.TransLogLU(0.01, 1.0)  # lower and upper bound

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

if savefigs:
    fig_im.savefig(savepath + f'01a-initial_model_{ip_modeltype}.png', dpi=dpi)


# prepare inversion protokoll
main_prot_fid = savepath_data + f'{case}_{version}.log'
invprot_hdr = ('name\tri\tminT(us)\tmaxT(us)\tlam\tmy\tcf\tnoifl\t' +
               'max_iter\tn_iter\taRMS\trRMS\tchi2\truntime(min)')
if start_inv:
    with open(main_prot_fid, 'w') as main_prot:
        main_prot.write(invprot_hdr + '\n')

inv_run = 0
name_snd = 'T00'


 # %% simulate data and add noise
if query_yes_no('continue with forward testing?', default='no'):
    # %% setup system and forward solver
    frwrd_empymod_tm = empymod_frwrd(setup_device=setup_device,
                            setup_solver=setup_solver,
                            time_range=None, device='TEMfast',
                            relerr=1e-6, abserr=1e-28,
                            nlayer=nlayers_tm, nparam=nparams)
    times_rx = frwrd_empymod_tm.times_rx
    
    forward_empymod = empymod_frwrd(setup_device=setup_device,
                            setup_solver=setup_solver,
                            time_range=None, device='TEMfast',
                            relerr=1e-6, abserr=1e-28,
                            nlayer=nlayers, nparam=nparams)


    # %% generate data and add artificial noise
    numdat = frwrd_empymod_tm.calc_response(model=truemdl_vec,
                                           ip_modeltype=ip_modeltype,  # 'cole_cole', 'cc_kozhe'
                                           show_wf=False, return_rhoa=False,
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

        if savefigs:
            plt.savefig(savepath + f'01b-frwrd-comp_{ip_modeltype}_{name_snd}.png', dpi=dpi)

    mean_rhoa = abs(np.mean(calc_rhoa(setup_device, numdat_noisy, times_rx)))



    # %% prepare start model and fop
    fop = temip_block1D_fwd(forward_empymod, nPara=nparams-1,
                            nLayers=nlayers, verbose=True)

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

    name_snd = 'T00'
    inv_run = 0

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

    # if query_yes_no('continue with inversion?', default='no'):
    if start_inv:
        with open(snd_prot_fid, 'w') as snd_prot:
            snd_prot.write(invprot_hdr + '\n')

        # %% setup inversion
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
        fig = plt.figure(figsize=(15, 5))

        gs = fig.add_gridspec(1, 6)
        
        ax_fit = fig.add_subplot(gs[0, 0:2])
        
        ax_rho = fig.add_subplot(gs[0, 2:3])
        ax_mpa = fig.add_subplot(gs[0, 3:4])
        ax_tau = fig.add_subplot(gs[0, 4:5])
        ax_c = fig.add_subplot(gs[0, 5:6])
        
        ax_mdl = np.array([ax_rho, ax_mpa, ax_tau, ax_c])


        # %% plot the data fit
        # fig, ax = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True,
        #                        sharex=True)

        _ = plot_signal(ax_fit, time=times_rx, signal=numdat_noisy, label='data',
                        marker='o', color='k', sub0color='gray',
                        sub0marker='d', sub0label='negative data')
        _ = plot_signal(ax_fit, time=times_rx, signal=response, label='response',
                        marker='.', ls='--', color='crimson',
                        sub0marker='s', sub0label='negative response')
        # axt = plot_diffs(ax=ax_fit, times=times_rx,
        #                  response=response, measured=numdat_noisy,
        #                  relative=True, max_diff=30)
        # ask matplotlib for the plotted objects and their labels
        lines, labels = ax_fit.get_legend_handles_labels()
        # lines2, labels2 = axt.get_legend_handles_labels()
        # ax[1].legend(lines + lines2, labels + labels2, loc=0)
        ax_fit.legend(loc='lower left')
        ax_fit.set_xlabel('Time (s)')
        ax_fit.set_ylabel(r"$\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t$ (V/m²)")
        ax_fit.set_xlim(lims_time)
        ax_fit.set_ylim(lims_signal)
        ax_fit.set_title(f'chi2 = {chi2:0.2f}\n rrms = {relrms:0.1f}%')


        if savefigs:
            plt.savefig(savepath + f'02-result-fit_temip_simdata_{ip_modeltype}.png', dpi=dpi)


        # %% plot the model
        # fig_mdl, ax_mdl = plt.subplots(2, 2, figsize=(9, 9), constrained_layout=True)
        
        if ip_modeltype == 'pelton':
            plot_pem_stepmodel(axes=ax_mdl, model2d=truemdl_arr, label='true',
                               color='black', ls='-', depth_limit=lims_depth)
            plot_pem_stepmodel(axes=ax_mdl, model2d=initmdl_arr, label='initial',
                               color='green', ls='--', marker='.', depth_limit=lims_depth)
            plot_pem_stepmodel(axes=ax_mdl, model2d=invmdl_arr, label='inverted',
                               color='green', ls='--', marker='.', depth_limit=lims_depth)

        elif ip_modeltype == 'mpa':
            plot_mpa_stepmodel(axes=ax_mdl, model2d=truemdl_arr, label='true',
                               color='black', ls='-', depth_limit=lims_depth)
            plot_mpa_stepmodel(axes=ax_mdl, model2d=initmdl_arr, label='initial',
                               color='green', ls='--', marker='.', depth_limit=lims_depth)
            plot_mpa_stepmodel(axes=ax_mdl, model2d=invmdl_arr, label='inverted',
                               color="crimson", ls='-.', marker='.', depth_limit=lims_depth)

        else:
            raise ValueError('this ip modeltype is not implemented here ...')

        ax_mdl[0].set_xlim(lims_rho)
        ax_mdl[1].set_xlim(lims_pol)
        ax_mdl[2].set_xlim(lims_tau)
        ax_mdl[3].set_xlim(lims_c)



        # add labels:
        all_axs = fig.get_axes()
        tags = ['(a)', '(b)', '(c)', '(d)', '(e)']
        for idx, tag in enumerate(tags):
            at = AnchoredText(tag,
                              prop={'color': 'k', 'fontsize': 14}, frameon=True,
                              loc='upper right')
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            all_axs[idx].add_artist(at)

        gs.tight_layout(fig)
        if savefigs:
            print('saving to: ', savepath + f'02-temip_simdata_{ip_modeltype}.png')
            fig.savefig(savepath + f'02-temip_simdata_{ip_modeltype}.png', dpi=dpi)



        # %% plot the jacobian matrix
        vmin = -1e-5  # todo automatize
        vmax = abs(vmin)
        norm = SymLogNorm(linthresh=3, linscale=3,
                          vmin=vmin, vmax=vmax, base=10)
        # norm = SymLogNorm(linthresh=0.3, linscale=0.3, base=10)

        plt.figure(figsize=(12, 8))
        axj = sns.heatmap(jac_df, cmap="BrBG", annot=True,
                          fmt='.2g', robust=True, center=0,
                          vmin=vmin, vmax=vmax, norm=norm)  # 
        axj.set_title('jacobian last iteration')
        axj.set_xlabel('model parameters')
        axj.set_ylabel('data parameters')
        plt.tight_layout()
        figj = axj.get_figure()
        
        if savefigs:
            figj.savefig(savepath + 'jacobian_temip_simdata.png', dpi=dpi)

