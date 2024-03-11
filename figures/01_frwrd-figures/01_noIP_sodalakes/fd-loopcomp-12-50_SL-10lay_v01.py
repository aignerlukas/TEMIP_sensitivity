#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 18:12:19 2022

fielddata inversion, sodalalkes, 12 and 50 m loop
blocky inversion
10 layers up to same depth

v00:
    [done] same initial model for 12 and 50 m loop
    [done] log scaled thk values increasing to max depth
    
    [done] new starting model shape:
        [done] conductive anomalies increase in resistivity with depth
        [done] slightly more resistive bottom layer

    [] add bottom resistivity



@author: laigner
"""


# %% import modules
import os
import sys

rel_path_to_libs = '../../../'
if not rel_path_to_libs in sys.path:
    sys.path.append('../../../')  # add realtive path to folder that contains all custom modules

import logging

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from matplotlib.offsetbox import AnchoredText

import pygimli as pg
# from pygimli.viewer.mpl import drawModel1D

from library.TEM_frwrd.empymod_frwrd import empymod_frwrd as empyfrwrd

from library.TEM_inv.pg_temip_inv import tem_block1D_fwd
from library.TEM_inv.pg_temip_inv import LSQRInversion
from library.TEM_inv.pg_temip_inv import setup_initialmdl_constraints
from library.TEM_inv.pg_temip_inv import filter_data

from library.utils.universal_tools import round_up

from library.utils.TEM_inv_tools import plot_signal
from library.utils.TEM_inv_tools import plot_rhoa
from library.utils.TEM_inv_tools import calc_rhoa
from library.utils.TEM_inv_tools import calc_doi
from library.utils.TEM_inv_tools import prep_mdl_para_names
from library.utils.TEM_inv_tools import vecMDL2mtrx
from library.utils.TEM_inv_tools import mtrxMDL2vec
from library.utils.TEM_inv_tools import save_result_and_fit

from library.tem_tools.survey import Survey

from library.utils.timer import Timer
t = Timer()


# %% save path structure
scriptname = os.path.basename(sys.argv[0])
print(f'running {scriptname} ...')
version_new = scriptname.split('.')[0].split('_')[-1]
typ = scriptname.split('.')[0].split('_')[-2]
lid = scriptname.split('.')[0].split('_')[0]

savepath = f'./04-vis/{lid}_{typ}/{version_new}/'
if not os.path.exists(savepath):
    os.makedirs(savepath)


# %% setup plotting
dpi = 200
plt.style.use('ggplot')

fs_shift = -4
plt.rcParams['axes.labelsize'] = 18 + fs_shift
plt.rcParams['axes.titlesize'] = 18 + fs_shift
plt.rcParams['xtick.labelsize'] = 16 + fs_shift
plt.rcParams['ytick.labelsize'] = 16 + fs_shift
plt.rcParams['legend.fontsize'] = 16 + fs_shift

lims_time = (1e-6, 1e-3)
lims_sign = (1e-10, 1e0)
lims_rhoa = (1e0, 1e2)
lims_rho = (5e0, 0.6e2)
lims_rho_fld = (1e0, 0.8e2)
lims_depth = (90, -2)

min_time = 3e-6
lw = 2.5


# %% setup script
save_fig = True
# save_fig = False

show_simulated_data = True
# show_simulated_data = False

start_inv = True
# start_inv = False

show_results = True
# show_results = False

save_data = True
# save_data = False

#  (inversion settings)
lambdas = [50]  # in blocky case: initial lambdas
cooling_factor = [0.9]
max_iter = 25
mys = [1e2]  # regularization for parameter constraint
noise_floors = [0.015]  # 1.0 %


# %% read rawdata ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
main = './'
name_file = '2022052-loop-comp'

rawdata_path = main + '01-data/selected/'
rawdata_fname = f'{name_file}.tem'

survey_blk = Survey()
survey_blk.parse_temfast_data(filename=rawdata_fname, path=rawdata_path)
survey_blk.select_soundings_by(prop='tx_side1', vals=[12.5, 50.0])

snd12_blk = survey_blk.soundings[survey_blk.sounding_names[1]]
snd50_blk = survey_blk.soundings[survey_blk.sounding_names[0]]


# %% run blocky inversion using LSQRInv
ip_modeltype = None

filters = [(5e-6, 2e-4), (12e-6, 5e-3)]  # 12.5m loop and 50.0 m loop
soundings = [snd12_blk, snd50_blk]
loop_sizes = np.r_[snd12_blk.tx_loop, snd50_blk.tx_loop]
depths = loop_sizes * 4
depths = np.r_[100, 100]

# setup
init_rho = np.median(np.r_[snd12_blk.rhoa_o, snd50_blk.rhoa_o])

general_max_depth = np.max(loop_sizes) * 4
general_nlayers = 10

general_init_thk = np.diff(np.r_[np.logspace(-1, 0.0, general_nlayers, endpoint=True)] * general_max_depth)
general_init_thkcum = np.cumsum(general_init_thk)

# use_same_thk = True
use_same_thk = False

thk_lu = (1, 30)
res_lu = (1, 1000)

transThk = pg.trans.TransLogLU(thk_lu[0], thk_lu[1])  # log-transform ensures thk>0
transRho = pg.trans.TransLogLU(res_lu[0], res_lu[1])  # lower and upper bound
transData = pg.trans.TransLog()  # log transformation for data, possible, because we dont expect any negative values in the data


# %% 
savepath_result = main + f'03-inv_results/{lid}/{version_new}/'
if not os.path.exists(savepath_result):
    os.makedirs(savepath_result)

# prepare inversion protokoll
fID_savename = f'{typ}'
main_prot_fid = savepath_result + f'{fID_savename}.log'
invprot_hdr = ('name\tri\tminT(us)\tmaxT(us)\tlam\tlam_fin\tmy\tcf\tnoifl\t' +
               'max_iter\tn_iter\taRMS\trRMS\tchi2\truntime(min)')
if start_inv:
    with open(main_prot_fid, 'w') as main_prot:
        main_prot.write(invprot_hdr + '\n')


# pre = f'{inv_type}_{batch_type}'
for i, sounding in enumerate(soundings):

    name_snd = sounding.name.strip()
    print('\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f' - starting preparation for inversion of sounding: {name_snd}\n')
    
    if use_same_thk:
        mask_thk = general_init_thkcum < depths[i]
        init_layer_thk = general_init_thk[mask_thk]
        nlayer = init_layer_thk.shape[0] + 1  # +1 for bottom layer
    else:
        # init_layer_thk = np.diff(np.r_[np.logspace(-1, 0.0, general_nlayers, endpoint=True)] * depths[i])
        init_layer_thk = np.diff(np.r_[np.logspace(-1, 0.0, general_nlayers, endpoint=True)] * np.max(depths))
        nlayer = init_layer_thk.shape[0] + 1  # +1 for bottom layer
    
    # setup constraints, basic parameters
    # init_layer_thk = np.full((n_layers, ), thickness)
    constr_thk = np.zeros_like(init_layer_thk)  # set to 1 if parameter should be fixed
    init_layer_res = np.full((nlayer, ), init_rho)
    constr_res = np.zeros_like(init_layer_res)

    Tx_loop = sounding.tx_loop
    Rx_loop = sounding.rx_loop
    current_inj = sounding.current
    currentkey = sounding.currentkey
    time_key = sounding.timekey

    name_snd_save = f'Loop_{int(Tx_loop)}'
    savepath_csv = savepath_result + f'{name_snd}/csv/'
    if not os.path.exists(savepath_csv):
        os.makedirs(savepath_csv)

    rawdata = sounding.sgnl_o
    time_range = filters[i]

    # setup data frame
    dmeas = sounding.get_obsdata_dataframe()
    
    times_all = dmeas.time.values
    max_time = round_up(max(times_all)*1e6, 100) / 1e6


    # %% prepare constraints and initial values
    (initmdl_pgvec, initmdl_arr,
     Gi, constrain_mdl_params,
     param_names) = setup_initialmdl_constraints(constr_thk, constr_res,
                                                 init_layer_thk, init_layer_res)
    cstr_vals = None
    # (initmdl_pgvec, initmdl_arr,
    #  Gi, constrain_mdl_params,
    #  param_names) = setup_initialipmdl_constraints(ip_modeltype, constr_thk, constr_res,
    #                                              constr_charg, constr_tau, constr_c,
    #                                              init_layer_thk, init_layer_res,
    #                                              init_layer_m, init_layer_tau, init_layer_c)

    mdl_para_names = prep_mdl_para_names(param_names, n_layers=len(init_layer_res))
    nlayers = initmdl_arr.shape[0]
    nparams = initmdl_arr.shape[1]


    # %% filtering the data - # select subset according to time range
    (rxtimes_sub, obsdat_sub, obserr_sub,
     dmeas_sub, time_range) = filter_data(dmeas,
                                          time_range, ip_modeltype=ip_modeltype)

    tr0 = time_range[0]
    trN = time_range[1]
    relerr_sub = abs(obserr_sub) / obsdat_sub
    # rhoa_median = np.round(np.median(dmeas_sub.rhoa.values), 2)


    # %% setup system and forward solver
    device = 'TEMfast'
    cutoff_f = 1e8
    setup_device = {"timekey": time_key,
                    "currentkey": currentkey,
                    "txloop": Tx_loop,
                    "rxloop": Rx_loop,
                    "current_inj": current_inj,
                    "filter_powerline": 50,
                    "ramp_data": 'salzlacken'}

    # 'ftarg': 'key_81_CosSin_2009', 'key_201_CosSin_2012', 'ftarg': 'key_601_CosSin_2009'
    setup_solver = {'ft': 'dlf',                     # type of fourier trafo
                    'ftarg': 'key_601_CosSin_2009',  # ft-argument; filter type # https://empymod.emsig.xyz/en/stable/api/filters.html#module-empymod.filters -- for filter names
                    'verbose': 0,                    # level of verbosity (0-4) - larger, more info
                    'srcpts': 3,                     # Approx. the finite dip. with x points. Number of integration points for bipole source/receiver, default is 1:, srcpts/recpts < 3 : bipole, but calculated as dipole at centre
                    'recpts': 3,                     # Approx. the finite dip. with x points. srcpts/recpts >= 3 : bipole
                    'ht': 'dlf',                     # type of fourier trafo
                    'htarg': 'key_401_2009',         # hankel transform filter type, 'key_401_2009', 'key_101_2009'
                    'nquad': 3,                      # Number of Gauss-Legendre points for the integration. Default is 3.
                    'cutoff_f': 1e8,                 # TODO add automatisation for diff loops;  cut-off freq of butterworthtype filter - None: No filter applied, WalkTEM 4.5e5
                    'delay_rst': 0,                  # ?? unknown para for walktem - keep at 0 for fasttem
                    'rxloop': 'vert. dipole'}        # or 'same as txloop' - not yet operational

    empy_frwrd = empyfrwrd(setup_device=setup_device,
                                setup_solver=setup_solver,
                                time_range=time_range, device='TEMfast',
                                nlayer=nlayers, nparam=nparams)
    fop = tem_block1D_fwd(empy_frwrd, nPara=nparams-1, nLayers=nlayers,
                          verbose=True)

    fop.region(0).setTransModel(transThk)  # 0=thickness
    fop.region(1).setTransModel(transRho)  # 1=resistivity
    fop.setMultiThreadJacobian(1)


    # %% visualize rawdata and filtering plus error!!
    t.start()
    simdata = fop.response(initmdl_pgvec)  # simulate start model response
    frwrd_time = t.stop(prefix='forward-')
    print('\n\n #############################################################')
    
    if show_simulated_data:
        fg_raw, ax_raw = plt.subplots(1, 2, figsize=(12,6))
    
        # rawdata first
        _ = plot_signal(ax_raw[0], times_all, sounding.sgnl_o,
                        marker='o', ls=':', color='grey', sub0color='orange',
                        label='data raw')
        
        ax_raw[0].loglog(times_all, sounding.error_o,  # noise
                          'd', ms=4, ls=':',
                          color='grey', alpha=0.5,
                          label='noise raw')
    
        _ = plot_rhoa(ax_raw[1], times_all, sounding.rhoa_o,
                      marker='o', ls=':', color='grey',
                      label='rhoa raw')
    
        # filtered data
        _ = plot_signal(ax_raw[0], rxtimes_sub, obsdat_sub,
                        marker='d', ls=':', color='k', sub0color='orange',
                        label='data subset')
    
        _ = plot_rhoa(ax_raw[1], rxtimes_sub, dmeas_sub.rhoa,
                      marker='d', ls=':', color='k', sub0color='orange',
                      label='rhoa subset')
    
        # and comparing it to the measured (observed) subset of the data
        _ = plot_signal(ax_raw[0], rxtimes_sub, simdata,
                        marker='x', color='crimson', ls='None', sub0color='orange',
                        label='data sim')
        
        sim_rhoa = calc_rhoa(setup_device, simdata, rxtimes_sub)
        _ = plot_rhoa(ax_raw[1], rxtimes_sub, sim_rhoa,
                      marker='x', color='crimson', ls='None', sub0color='orange',
                      label='rhoa sim')
    
        max_time = round_up(max(times_all)*1e6, 100) / 1e6
        ax_raw[0].set_xlabel('time (s)')
        ax_raw[0].set_ylabel(r'$\frac{\delta B}{\delta t}$ (V/m²)')
        ax_raw[0].set_ylim((lims_sign[0], lims_sign[1]))
        # ax_raw[0].set_xlim((min_time, max_time))
        ax_raw[0].grid(True, which='major', color='white', linestyle='-')
        ax_raw[0].grid(True, which='minor', color='white',  linestyle=':')
        ax_raw[0].legend()
        # ax_raw[0].set_title(f'{name_snd}')
    
        ax_raw[1].set_xlabel('time (s)')
        ax_raw[1].set_ylabel(r'$\rho_a$ ($\Omega$m)')
        ax_raw[1].set_ylim((lims_rhoa[0], lims_rhoa[1]))
        ax_raw[1].set_xlim((min_time, max_time))
        ax_raw[1].yaxis.set_label_position("right")
        ax_raw[1].yaxis.tick_right()
        ax_raw[1].yaxis.set_ticks_position('both')
        ax_raw[1].yaxis.set_minor_formatter(ticker.FuncFormatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y))))
        for label in ax_raw[1].yaxis.get_minorticklabels()[1::2]:
            label.set_visible(False)  # remove every second label
        ax_raw[1].grid(True, which='major', color='white', linestyle='-')
        ax_raw[1].grid(True, which='minor', color='white',  linestyle=':')
    
        ax_raw[1].legend()
        plt.tight_layout()


    # %% inversion
    # prepare inversion protokoll for individual sounding
    snd_prot_fid = savepath_csv.replace('csv/', '') + f'{fID_savename}_snd-{name_snd}.log'
    
    total_runs = len(lambdas) * len(mys) * len(cooling_factor) * len(noise_floors)
    message = (f'proceed with inversion using n={total_runs:.0f} different settings\n' +
               '"no" proceeds with plotting - only if inversion was done already ...')

    if start_inv:
        inv_run = 0
        with open(snd_prot_fid, 'w') as snd_prot:
            snd_prot.write(invprot_hdr + '\n')

        for lam in lambdas:
            for my in mys:
                for cf in cooling_factor:
                    for noise_floor in noise_floors:

                        tem_inv = LSQRInversion(verbose=True)
                        tem_inv.setTransData(transData)
                        tem_inv.setForwardOperator(fop)

                        tem_inv.setMaxIter(max_iter)
                        tem_inv.setLambda(lam)  # (initial) regularization parameter

                        tem_inv.setMarquardtScheme(cf)  # decrease lambda by factor 0.9
                        tem_inv.setBlockyModel(True)
                        tem_inv.setModel(initmdl_pgvec)  # set start model
                        tem_inv.setData(obsdat_sub)

                        print('\n\n####################################################################')
                        print('####################################################################')
                        print('####################################################################')
                        print(f'--- about to start the inversion run: ({inv_run}) ---')
                        print('lambda - cool_fac - my - noise_floor(%) - max_iter:')
                        print((f'{lam:6.3f} - ' +
                               f'{cf:.1e} - ' +
                               f'{my:.1e} - ' +
                               f'{noise_floor*100:.2f} - ' +
                               f'{max_iter:.0f} - '))
                        print('and initial model:\n', initmdl_pgvec)

                        rel_err = np.copy(relerr_sub)
                        if any(rel_err < noise_floor):
                            logging.warning(f'Encountered rel. error below {noise_floor*100}% - setting those to {noise_floor*100}%')
                            rel_err[rel_err < noise_floor] = noise_floor
                            abs_err = abs(obsdat_sub * rel_err)

                        tem_inv.setAbsoluteError(abs(abs_err))
                        if constrain_mdl_params:
                            tem_inv.setParameterConstraints(G=Gi, c=cstr_vals, my=my)
                        else:
                            print('no constraints used ...')

                        # break

                        t.start()
                        model_inv = tem_inv.run()
                        inv_time = t.stop(prefix='inv-')

                        inv_res, inv_thk = model_inv[nlayers-1:nlayers*2-1], model_inv[0:nlayers-1]
                        model_inv_mtrx = vecMDL2mtrx(model_inv, nlayers, nparams)

                        if ip_modeltype != None:
                            inv_m = model_inv_mtrx[:, 2]
                            inv_tau = model_inv_mtrx[:, 3]
                            inv_c = model_inv_mtrx[:, 4]

                        fop = tem_inv.fop()
                        col_names = mdl_para_names
                        row_names = [f'dB/dt tg{i:02d}' for i in range(1, len(obsdat_sub)+1)]
                        jac_df = pd.DataFrame(np.array(fop.jacobian()), columns=col_names, index=row_names)

                        print('\ninversion runtime: {:.1f} min.'.format(inv_time[0]))
                        print('--------------------   INV finished   ---------------------')


                        # %% save result and fit
                        chi2 = tem_inv.chi2()
                        rrms = tem_inv.relrms()
                        arms = tem_inv.absrms()
                        lam_fin = tem_inv.getLambda()
                        n_iter = tem_inv.n_iters  # get number of iterations

                        pred_data = np.asarray(tem_inv.response())
                        pred_rhoa = calc_rhoa(setup_device, pred_data,
                                              rxtimes_sub)

                        # %% save result and fit
                        savename = ('invrun{:03d}_{:s}'.format(inv_run, name_snd))
                        print(f'saving data from inversion run: {inv_run}')
                        position = (0, 0, 0)  # location of sounding

                        save_result_and_fit(tem_inv, setup_device, model_inv, jac_df, ip_modeltype, position,
                                            rxtimes_sub=rxtimes_sub, nparams=nparams, nlayers=nlayer,
                                            initmdl_arr=initmdl_arr, obsdat_sub=obsdat_sub,
                                            obserr_sub=obserr_sub, abs_err=abs_err, 
                                            obsrhoa_sub=np.full_like(obsdat_sub, np.nan),
                                            savepath_csv=savepath_csv, savename=savename)


                        # %% save main log
                        logline = ("%s\t" % (name_snd) +
                                    "r%03d\t" % (inv_run) +
                                    "%.1f\t" % (tr0) +
                                    "%.1f\t" % (trN) +
                                    "%8.1f\t" % (lam) +
                                    "%8.1f\t" % (lam_fin) +
                                    "%.1e\t" % (my) +
                                    "%.1e\t" % (cf) +  # cooling factor
                                    "%.2f\t" % (noise_floor) +
                                    "%d\t" % (max_iter) +
                                    "%d\t" % (n_iter) +
                                    "%.2e\t" % (arms) +
                                    "%7.3f\t" % (rrms) +
                                    "%7.3f\t" % (chi2) +
                                    "%4.1f\n" % (inv_time[0]/60))  # to min
                        with open(main_prot_fid,'a+') as f:
                            f.write(logline)
                        with open(snd_prot_fid,'a+') as f:
                            f.write(logline)

                        inv_run += 1


# %% reload for plotting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if show_results:
    invrun = '000'
    version = version_new

    survey_blk_n = Survey()
    survey_blk_n.parse_temfast_data(filename=rawdata_fname, path=rawdata_path)
    survey_blk_n.parse_inv_results(result_folder=savepath_result, invrun=invrun)
    survey_blk_n.select_soundings_by(prop='tx_side1', vals=[12.5, 50.0])

    snd12_blk_n = survey_blk_n.soundings[survey_blk_n.sounding_names[1]]
    filter_12 = np.r_[np.asarray(snd12_blk_n.time_f)[0]*0.99, np.asarray(snd12_blk_n.time_f)[-1]*1.1]
    
    log_12 = survey_blk_n.invlog_info[survey_blk_n.invlog_info.name == snd12_blk_n.name.strip()]
    rrms_12 = log_12.rRMS.values[0]
    chi2_12 = log_12.chi2.values[0]
    lam_12 = log_12.lam.values[0]
    lamfin_12 = log_12.lam_fin.values[0]
    cf_12 = log_12.cf.values[0]
    noifl_12 = log_12.noifl.values[0]
    niter_12 = log_12.n_iter.values[0]
    
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
    
    model_inv = mtrxMDL2vec(snd12_blk_n.inv_model)
    res_inv_blk, thk_inv_blk = model_inv[nlayer-1:nlayer*2-1], model_inv[0:nlayer-1]
    mdl4doi = np.column_stack((np.r_[0, thk_inv_blk], res_inv_blk))
    doi12 = calc_doi(current=snd12_blk_n.current,
                    tx_area=snd12_blk_n.tx_loop**2,
                    eta=snd12_blk_n.sgnl_c.iloc[-1],
                    mdl_rz=mdl4doi, x0=30,
                    verbose=True)
    
    
    snd12_blk_n.plot_dBzdt(which='observed', ax=ax[0, 0], xlimits=(3e-6, 5e-3),
                          ylimits=(1e-10, 1e-1), label='data', color='k', show_sub0_label=False)
    snd12_blk_n.plot_dBzdt(which='calculated', ax=ax[0, 0], color='dodgerblue',
                         marker='x', ls=':', label='response', show_sub0_label=False)
    ax[0, 0].set_title(f'field data 12.5 m loop\n$\chi^2$ = {chi2_12:0.1f}, rRMS = {rrms_12:0.1f}%')
    ax[0, 0].legend()

    snd12_blk_n.plot_initial_model(ax=ax[0, 1], color='green', 
                                   marker='.', ls='--', label='init. model'
                                   )
    snd12_blk_n.plot_inv_model(ax=ax[0, 1], color='dodgerblue',
                               marker='.', label='inv. model'
                               )
    ax[0, 1].axhline(y=doi12[0], ls='--', color='magenta', label='DOI')

    ax[0, 1].set_title((f'init. lam {lam_12:.1f} cooling: {cf_12}, iters {niter_12}\n' +
                        f'fin. lam {lamfin_12:.1f}, noise floor = {noifl_12}%'))
    
    ax[0, 1].invert_yaxis()
    ax[0, 1].set_xlim(lims_rho_fld)
    ax[0, 1].set_ylim(lims_depth)
    ax[0, 1].set_xlabel(r'$\rho$ ($\Omega$m)')
    ax[0, 1].set_ylabel('z (m)')
    ax[0, 1].legend(loc='upper right')
    
    tag = f'DOI = {doi12[0]:.1f} m'
    at = AnchoredText(tag, prop={'color': 'k', 'fontsize': 14},
                        frameon=True, loc='lower right')
    at.patch.set_boxstyle("round, pad=0.0, rounding_size=0.2")
    ax[0, 1].add_artist(at)

    snd50_blk_n = survey_blk_n.soundings[survey_blk_n.sounding_names[0]]
    filter_50 = np.r_[np.asarray(snd50_blk_n.time_f)[0]*0.99, np.asarray(snd50_blk_n.time_f)[-1]*1.1]
    
    model_inv = mtrxMDL2vec(snd50_blk_n.inv_model)
    res_inv_blk, thk_inv_blk = model_inv[nlayer-1:nlayer*2-1], model_inv[0:nlayer-1]
    mdl4doi = np.column_stack((np.r_[0, thk_inv_blk], res_inv_blk))
    doi50 = calc_doi(current=snd50_blk_n.current,
                    tx_area=snd50_blk_n.tx_loop**2,
                    eta=snd50_blk_n.sgnl_c.iloc[-1],
                    mdl_rz=mdl4doi, x0=30,
                    verbose=True)
    
    
    log_50 = survey_blk_n.invlog_info[survey_blk_n.invlog_info.name == snd50_blk_n.name.strip()]
    rrms_50 = log_50.rRMS.values[0]
    chi2_50 = log_50.chi2.values[0]
    lam_50 = log_50.lam.values[0]
    lamfin_50 = log_50.lam_fin.values[0]
    cf_50 = log_50.cf.values[0]
    noifl_50 = log_50.noifl.values[0]
    niter_50 = log_50.n_iter.values[0]
    
    snd50_blk_n.plot_dBzdt(which='observed', ax=ax[1, 0], xlimits=(3e-6, 5e-3),
                          ylimits=(1e-10, 1e-1), label='field data 50.0 m loop',
                          color='k', show_sub0_label=False
                          )
    snd50_blk_n.plot_dBzdt(which='calculated', ax=ax[1, 0],
                           color='dodgerblue', marker='x', ls=':',
                           show_sub0_label=False
                           )
    ax[1, 0].set_title(f'field data 50.0 m loop\n$\chi^2$ = {chi2_50:0.1f}, rRMS = {rrms_50:0.1f}%')
    
    snd50_blk_n.plot_initial_model(ax=ax[1, 1], color='green', marker='.', ls='--')
    snd50_blk_n.plot_inv_model(ax=ax[1, 1], color='dodgerblue', marker='.')
    ax[1, 1].axhline(y=doi50[0], ls='--', color='magenta', label='DOI')
    
    tag = f'DOI = {doi50[0]:.1f} m'
    at = AnchoredText(tag, prop={'color': 'k', 'fontsize': 14},
                        frameon=True, loc='lower right')
    at.patch.set_boxstyle("round, pad=0.0, rounding_size=0.2")
    ax[1, 1].add_artist(at)
    
    
    ax[1, 1].set_title(f'lambda: {lam_50}')
    ax[1, 1].set_title((f'init. lam {lam_50:.1f} cooling: {cf_50}, iters {niter_50}\n' +
                        f'fin. lam {lamfin_50:.1f}, noise floor = {noifl_50}%'))
    ax[1, 1].invert_yaxis()
    ax[1, 1].set_xlim(lims_rho_fld)
    ax[1, 1].set_ylim(lims_depth)
    ax[1, 1].set_xlabel(r'$\rho$ ($\Omega$m)')
    ax[1, 1].set_ylabel('z (m)')



# %% add labels and save fig
all_axs = fig.get_axes()
tags = ['(a)', '(b)', '(c)', '(d)']

for i, tag in enumerate(tags):
    at = AnchoredText(tag,
                      prop={'color': 'k', 'fontsize': 14}, frameon=True,
                      loc='lower left')
    at.patch.set_boxstyle("round, pad=0.0, rounding_size=0.2")
    all_axs[i].add_artist(at)


if save_fig:
    fig.savefig(savepath + f'blk_field_12-50_{version_new}.png', dpi=300)


# old version, breaks because model trafos cant be set inside of a function and returned succesfully
# # %% inversion setup and test startmodel using first entry in mesh related lists
# def setup_fop(ip_modeltype, nparams, nlayers, time_range,
#               setup_device, setup_solver, verbose=True,
#               thk_lu=(1, 30), res_lu=(1, 1000),
#               ip_params_lu=np.array([[0.0001, 1], [1e-6, 1e-3], [0.0001, 1]])):

#     transThk = pg.trans.TransLogLU(thk_lu[0], thk_lu[1])  # log-transform ensures thk>0
#     transRho = pg.trans.TransLogLU(res_lu[0], res_lu[1])  # lower and upper bound

#     if ip_modeltype != None:
#         # from library.TEM_frwrd.empymod_frwrd_ip import empymod_frwrd
#         transM = pg.trans.TransLogLU(ip_params_lu[0, :])  # lower and upper bound chargeability, mpa
#         transTau = pg.trans.TransLogLU(ip_params_lu[1, :])  # lower and upper bound tau
#         transC = pg.trans.TransLogLU(ip_params_lu[2, :])  # lower and upper bound dispersion coeff

#         empy_frwrd = empyfrwrd_ip(setup_device=setup_device,
#                                   setup_solver=setup_solver,
#                                   filter_times=time_range, device='TEMfast',
#                                   nlayer=nlayers, nparam=nparams)
#         fop = temip_block1D_fwd(empy_frwrd, ip_mdltype=ip_modeltype,
#                                 nPara=nparams-1, nLayers=nlayers,
#                                 shift=None, verbose=verbose)

#         fop.region(0).setTransModel(transThk)  # 0=thickness
#         fop.region(1).setTransModel(transRho)  # 1=resistivity
#         fop.region(2).setTransModel(transM)    # 2=m
#         fop.region(3).setTransModel(transTau)  # 3=tau
#         fop.region(4).setTransModel(transC)    # 4=c

#         fop.setMultiThreadJacobian(1)

#     else:
#         # from library.TEM_frwrd.empymod_frwrd import empymod_frwrd
#         empy_frwrd = empyfrwrd(setup_device=setup_device,
#                                setup_solver=setup_solver,
#                                filter_times=time_range, device='TEMfast',
#                                nlayer=nlayers, nparam=nparams)
#         fop = tem_block1D_fwd(empy_frwrd, nPara=nparams-1, nLayers=nlayers,
#                               verbose=verbose)

#         fop.region(0).setTransModel(transThk)  # 0=thickness
#         fop.region(1).setTransModel(transRho)  # 1=resistivity

#         fop.setMultiThreadJacobian(1)
        
#         return fop
