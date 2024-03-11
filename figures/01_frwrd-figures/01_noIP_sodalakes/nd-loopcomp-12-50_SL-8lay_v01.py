#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 18:12:19 2022

script for basic simdata inversion
blocky inversion
number of layers based upon log scaled valules

v60:
    [done] same initial model for 12 and 50 m loop
    [done] log scaled thk values increasing to max depth

    [done] new starting model shape:
        conductive anomalies increase in resistivity with depth
        slightly more resistive bottom layer
        no layer > 150

    [] add DOIs
        [done] classical
        [] from jacobian

    [done] test double resistivity v69

@author: laigner
"""


# %% import modules
import os
import sys

rel_path_to_libs = '../../../'
if not rel_path_to_libs in sys.path:
    sys.path.append('../../../')  # add realtive path to folder that contains all custom mudules

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
# import matplotlib.colors as clr

import pygimli as pg
from pygimli.viewer.mpl import drawModel1D

from library.TEM_frwrd.empymod_frwrd import empymod_frwrd

from library.utils.TEM_inv_tools import plot_simdata
from library.utils.TEM_inv_tools import plot_signal
from library.utils.TEM_inv_tools import calc_rhoa
from library.utils.TEM_inv_tools import get_diffs
from library.utils.TEM_inv_tools import calc_doi

from library.utils.TEM_inv_tools import prep_mdl_para_names
from library.utils.TEM_inv_tools import vecMDL2mtrx
from library.utils.TEM_inv_tools import mtrxMDL2vec
from library.utils.TEM_inv_tools import save_result_and_fit

from library.TEM_inv.pg_temip_inv import tem_block1D_fwd
from library.TEM_inv.pg_temip_inv import LSQRInversion

from library.tem_tools.survey import Survey
from library.tem_tools.survey import Sounding
from library.utils.universal_tools import save_as_tem
from library.utils.timer import Timer

t = Timer()


# %% save path structure
scriptname = os.path.basename(sys.argv[0])
print(f'running {scriptname} ...')
version = scriptname.split('.')[0].split('_')[-1]
typ = scriptname.split('.')[0].split('_')[-2]

invtyp = scriptname.split('_')[0]

savepath = f'./04-vis/{invtyp}_{typ}/{version}/'
if not os.path.exists(savepath):
    os.makedirs(savepath)

inv_run = '000'


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
lims_signal = (1e-10, 1e0)
lims_rhoa = (1e0, 1e2)
lims_rho = (1e0, 2e2)
lims_depth = (81, -1)
lw = 2.5


# %% setup script (inversion settings)
# start_inv = False
start_inv = True

# show_simulated_data = True
show_simulated_data = False

show_results = True

save_fig = True
# save_fig = False

save_data_astem = True
# save_data_astem = False


# setup forward
device = 'TEMfast'
cutoff_f = 1e8

# data error
relerr = 0.025
abserr = 1e-10

# setup inversion
max_iter = 25
lam_mrqrdt = 50
cooling_factor = 0.9
noise_floor = 0.025  # 2.5%
my = 1e-2  # no effect of there are no constraints!


# %% setup model
ip_modeltype = None

thk = np.r_[4, 10, 15, 25, 0]
# res = np.r_[25, 100, 15, 150, 15] * 2
res = np.r_[30, 120, 35, 100, 55]
depth = np.cumsum(np.r_[0, thk])[:-1]

# construct model vector thk (nLayer-1), followed by res
model_vec = np.r_[thk[:-1], res]
model1 = np.column_stack((depth, res))

nlayer_true = res.shape[0]
nparam = model1.shape[1]


# %% loop preparations
loop_sizes = np.r_[12.50, 50.00]
timekeys = [3, 5]
# depths = loop_sizes*3

depths = loop_sizes * 3
depths = np.r_[100, 100]

general_max_depth = np.max(loop_sizes) * 4
general_nlayers = 8

general_init_thk = np.diff(np.r_[np.logspace(-1, 0.0, general_nlayers, endpoint=True)] * general_max_depth)
general_init_thkcum = np.cumsum(general_init_thk)

param_names = ['thk', 'res']

# use_same_thk = True
use_same_thk = False


# %% field data visualizations
# rawdata
main = './'
filters = [(1e-6, 5e-3), (1e-6, 5e-3)]
fig, ax = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)


# %% prepare inversion protokoll
savepath_result = f'03-inv_results/{invtyp}/{version}/'
if not os.path.exists(savepath_result):
    os.makedirs(savepath_result)

fID_savename = f'{typ}'
main_prot_fid = savepath_result + f'{fID_savename}.log'
invprot_hdr = ('name\tri\tminT(us)\tmaxT(us)\tlam\tlam_fin\tmy\tcf\tnoifl\t' +
               'max_iter\tn_iter\taRMS\trRMS\tchi2\truntime(min)')
if start_inv:
    with open(main_prot_fid, 'w') as main_prot:
        main_prot.write(invprot_hdr + '\n')


# %% inversion of simulated data
for i, loop_size in enumerate(loop_sizes):

    savepath_csv = savepath_result + f'snd-{int(loop_size)}/csv/'
    if not os.path.exists(savepath_csv):
        os.makedirs(savepath_csv)

    if use_same_thk:
        mask_thk = general_init_thkcum < depths[i]
        init_layer_thk = general_init_thk[mask_thk]
        nlayer = init_layer_thk.shape[0] + 1  # +1 for bottom layer
    else:
        # init_layer_thk = np.diff(np.r_[np.logspace(-1, 0.0, general_nlayers, endpoint=True)] * depths[i])
        init_layer_thk = np.diff(np.r_[np.logspace(-1, 0.0, general_nlayers, endpoint=True)] * np.max(depths))
        nlayer = init_layer_thk.shape[0] + 1  # +1 for bottom layer
    mdl_para_names = prep_mdl_para_names(param_names, n_layers=nlayer)


    setup_device = {"timekey": timekeys[i],
                    "currentkey": 4,
                    "txloop": loop_size,
                    "rxloop": loop_size,
                    "current_inj": 4.1,
                    "filter_powerline": 50,
                    "ramp_data": 'salzlacken'}

    setup_empymod = {'ft': 'dlf',                     # type of fourier trafo
                      'ftarg': 'key_601_CosSin_2009',  # ft-argument; filter type # https://empymod.emsig.xyz/en/stable/api/filters.html#module-empymod.filters -- for filter names
                      'verbose': 0,                    # level of verbosity (0-4) - larger, more info
                      'srcpts': 3,                     # Approx. the finite dip. with x points. Number of integration points for bipole source/receiver, default is 1:, srcpts/recpts < 3 : bipole, but calculated as dipole at centre
                      'recpts': 3,                     # Approx. the finite dip. with x points. srcpts/recpts >= 3 : bipole
                      'ht': 'dlf',                     # type of fourier trafo
                      'htarg': 'key_401_2009',         # hankel transform filter type, 'key_401_2009', 'key_101_2009'
                      'nquad': 3,                     # Number of Gauss-Legendre points for the integration. Default is 3.
                      'cutoff_f': cutoff_f,           # cut-off freq of butterworthtype filter - None: No filter applied, WalkTEM 4.5e5
                      'delay_rst': 0,                 # ?? unknown para for walktem - keep at 0 for fasttem
                      'rxloop': 'vert. dipole'}       # or 'same as txloop' - not yet operational

    empy_frwrd_raw = empymod_frwrd(setup_device=setup_device,
                            setup_solver=setup_empymod,
                            time_range=None, device='TEMfast',
                            relerr=1e-6, abserr=1e-28,
                            nlayer=nlayer_true, nparam=nparam)
    times_rx_raw = empy_frwrd_raw.times_rx

    empy_frwrd = empymod_frwrd(setup_device=setup_device,
                            setup_solver=setup_empymod,
                            time_range=filters[i], device='TEMfast',
                            relerr=1e-6, abserr=1e-28,
                            nlayer=nlayer_true, nparam=nparam)
    times_rx = empy_frwrd.times_rx


    # %% generate data and add artificial noise
    sim_data_raw = empy_frwrd_raw.calc_response(model=model_vec, show_wf=False)
    sim_data = empy_frwrd.calc_response(model=model_vec, show_wf=False)

    np.random.seed(42)
    rndm = np.random.randn(len(sim_data))
    noise_calc_rand = (relerr * np.abs(sim_data) +
                        abserr) * rndm
    sim_data_noisy = noise_calc_rand + sim_data

    if show_simulated_data:
        ax1, ax2 = plot_simdata(times_rx=times_rx, dbdt_clean=sim_data, forward_solver=empy_frwrd,
                                show_rhoa=False, show_noise=True,
                                relerr=relerr, abserr=abserr)
        plt.savefig(savepath + f'data-{int(np.round(loop_size, 0)):02d}_{version}.png', dpi=300)

    sim_rhoa = calc_rhoa(setup_device, sim_data_raw, times_rx_raw)
    sim_rhoa_noisy = calc_rhoa(setup_device, sim_data_noisy, times_rx)
    med_rhoa = np.median(sim_rhoa_noisy)


    rel_err_estimates = abs(noise_calc_rand) / sim_data_noisy
    if any(rel_err_estimates < noise_floor):
        logging.warning(f'Encountered rel. error below {noise_floor*100}% - setting those to {noise_floor*100}%')
        rel_err_estimates[rel_err_estimates < noise_floor] = noise_floor
    abs_err_estimates = sim_data_noisy * rel_err_estimates

    # save numerical data as .tem file
    if save_data_astem:
        template_fid = rel_path_to_libs + 'library/utils/template.tem'
        savepath_temdata = main + '01-data/modelled/sodalakes/'
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        name_snd = f'snd-{int(loop_size)}'
        filename = f'test-loops_{version}'
        metadata = {'snd_name': f'{name_snd}',
                    'location': 'soda lakes',
                    'comments': 'for testing with commercial software',
                    'x': 0.0,
                    'y': 0.0,
                    'z': 0.0}
        properties_snd = empy_frwrd.properties_snd

        append_to_existing = False
        if i == 1:
            append_to_existing = True

        save_as_tem(savepath_temdata, template_fid,
                    filename, metadata, setup_device, properties_snd,
                    times_rx, sim_data_noisy, abs_err_estimates, sim_rhoa_noisy,
                    save_rhoa=True, append_to_existing=append_to_existing)


    if start_inv:
        # %% inv blocky
        # preparation, fop, mdl thk, etc ##############################################
        empy_frwrd_inv = empymod_frwrd(setup_device=setup_device,
                                setup_solver=setup_empymod,
                                time_range=filters[i], device='TEMfast',
                                nlayer=nlayer, nparam=nparam)
        times_rx = empy_frwrd_inv.times_rx

        fop = tem_block1D_fwd(empy_frwrd_inv, nPara=1, nLayers=nlayer)

        init_layer_res = np.full((nlayer,), med_rhoa)
        start_model = pg.Vector(pg.cat(init_layer_thk, init_layer_res))
        initmdl_arr = vecMDL2mtrx(start_model, nlayer, nparam)

        t.start()
        test_response = fop.response(start_model)
        frwrd_time = t.stop(prefix='forward-')

        transThk = pg.trans.TransLogLU(1, 50)  # log-transform ensures thk>0
        transRho = pg.trans.TransLogLU(1, 1000)  # lower and upper bound
        transData = pg.trans.TransLog()  # log transformation for data

        fop.region(0).setTransModel(transThk)  # 0=thickness
        fop.region(1).setTransModel(transRho)  # 1=resistivity
        fop.setMultiThreadJacobian(1)


        ###############################################################################
        inv = LSQRInversion(verbose=True)

        inv.setTransData(transData)
        inv.setForwardOperator(fop)

        inv.setBlockyModel(True)  # necessary?

        inv.setMaxIter(max_iter)
        inv.setLambda(lam_mrqrdt)  # (initial) regularization parameter

        inv.setMarquardtScheme(cooling_factor)  # decrease lambda by factor 0.9
        inv.setModel(start_model)  # set start model
        inv.setData(sim_data_noisy)
        inv.setAbsoluteError(abs_err_estimates)

        t.start()

        model_inv = inv.run()
        inv_time = t.stop(prefix='inv-')

        res_inv, thk_inv = model_inv[nlayer-1:nlayer*2-1], model_inv[0:nlayer-1]

        chi2 = inv.chi2()
        rrms = inv.relrms()
        arms = inv.absrms()
        lam_fin = inv.getLambda()
        n_iter = inv.n_iters  # get number of iterations

        resp = inv.response()


        # %% calc DOI ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print('\n######## DOI calc #########')
        # mdl4doi = np.column_stack((np.r_[0, np.cumsum(thk_inv)], res_inv))
        mdl4doi = np.column_stack((np.r_[0, thk_inv], res_inv))
        doi = calc_doi(current=setup_device['current_inj'],
                       tx_area=setup_device['txloop']**2,
                       eta=np.array(resp)[-1],
                       mdl_rz=mdl4doi, x0=30,
                       verbose=True)


        # %% calc DOI from jacobian
        fop_inv = inv.fop()
        col_names = mdl_para_names
        row_names = [f'dB/dt tg{i:02d}' for i in range(1, len(resp)+1)]

        jac = np.array(fop_inv.jacobian())
        jac_df = pd.DataFrame(jac, columns=col_names, index=row_names)
        # TODO!!


        # %% save result and fit
        savename = ('invrun{:s}_{:s}'.format(inv_run, name_snd))
        print(f'saving data from inversion run: {inv_run}')

        position = (42, 42, 42)  # x, y, z
        save_result_and_fit(inv, setup_device, model_inv, jac_df, ip_modeltype, position,
                            rxtimes_sub=times_rx, nparams=nparam, nlayers=nlayer,
                            initmdl_arr=initmdl_arr, obsdat_sub=sim_data_noisy,
                            obserr_sub=noise_calc_rand, abs_err=abs_err_estimates, 
                            obsrhoa_sub=np.full_like(sim_data_noisy, np.nan),
                            savepath_csv=savepath_csv, savename=savename)

        # %% save main log
        logline = ("%s\t" % (name_snd) +
                    "r%03d\t" % (int(inv_run)) +
                    "%.1f\t" % (filters[i][0]) +
                    "%.1f\t" % (filters[i][1]) +
                    "%8.1f\t" % (lam_mrqrdt) +
                    "%8.1f\t" % (lam_fin) +
                    "%.1e\t" % (my) +
                    "%.1e\t" % (cooling_factor) +
                    "%.2f\t" % (noise_floor) +
                    "%d\t" % (max_iter) +
                    "%d\t" % (n_iter) +
                    "%.2e\t" % (arms) +
                    "%7.3f\t" % (rrms) +
                    "%7.3f\t" % (chi2) +
                    "%4.1f\n" % (inv_time[0]/60))  # to min
        with open(main_prot_fid,'a+') as f:
            f.write(logline)


#         # %% plot results
#         # blocky ##################################################################
#         diff, diff_rel = get_diffs(response=resp, measured=sim_data_noisy)

#         _ = plot_signal(ax[i, 0], time=times_rx_raw, signal=sim_data_raw, label='data',
#                         ls='--', marker='d', color='k')
#         _ = plot_signal(ax[i, 0], time=times_rx, signal=resp,
#                         label='response',  # , $\chi^2$={chi2:0.1f}
#                         ls=':', marker='.', color='dodgerblue')
#         # ax[0].loglog(times_rx, abs(noise_calc_rand), ':k', label='sim. noise')

#         ax[i, 0].set_xlabel('time (s)')
#         ax[i, 0].set_ylabel(r"$\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t$ (V/m²)")
#         ax[i, 0].set_xlim(3e-6, 5e-3)
#         ax[i, 0].set_ylim(1e-10, 1e-1)

#         ax[i, 0].set_title((f'numerical data, {loop_size} m loop\n' +
#                             f'$\chi^2$ = {chi2:0.2f}, rRMS = {rrms:0.1f}%'))


#         drawModel1D(ax[i, 1], np.diff(depth), res, color="black", ls='--', label="true model")
#         # zt_results[i].plot_inv_model(ax=ax[i, 1], color='magenta',ls='--',
#         #                              label='zond result')
#         drawModel1D(ax[i, 1], init_layer_thk, init_layer_res, color="green", marker='.', ls=':', label="init. model")
#         drawModel1D(ax[i, 1], thk_inv, res_inv, color="dodgerblue", marker='.', label="inv. model")
#         ax[i, 1].axhline(y=doi[0], ls='--', color='dodgerblue')

#         tag = f'DOI = {doi[0]:.1f} m'
#         at = AnchoredText(tag, prop={'color': 'k', 'fontsize': 14},
#                           frameon=True, loc='lower right')
#         at.patch.set_boxstyle("round, pad=0.0, rounding_size=0.2")
#         ax[i, 1].add_artist(at)

#         ax[i, 1].set_xlabel(r'$\rho$ ($\Omega$m)')
#         ax[i, 1].set_ylabel('z (m)')
#         ax[i, 1].set_xlim(lims_rho)
#         ax[i, 1].set_ylim(lims_depth)

#         ax[i, 1].set_title((f'init. lam {lam_mrqrdt}, cooling fac.: {cooling_factor}\n' +
#                             f'fin. lam {inv.getLambda():.1f}, noise floor = {noise_floor}%'))

#     ax[0, 0].legend(loc='upper right')
#     ax[0, 1].legend(loc='upper right')

# # %% add labels and save fig
# all_axs = fig.get_axes()
# tags = ['(a)', '(b)', '(c)', '(d)']

# for idx, tag in enumerate(tags):
#     at = AnchoredText(tag,
#                       prop={'color': 'k', 'fontsize': 14}, frameon=True,
#                       loc='lower left')
#     at.patch.set_boxstyle("round, pad=0.0, rounding_size=0.2")
#     all_axs[idx].add_artist(at)

# if save_fig:
#     plt.savefig(savepath + f'numerical_12-50_{version}.png', dpi=300)



# %% reload for plotting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if show_results:
    invrun = '000'

    survey_n = Survey()
    survey_n.parse_temfast_data(filename=filename + '.tem', path=savepath_temdata)
    survey_n.parse_inv_results(result_folder=savepath_result, invrun=invrun)
    survey_n.select_soundings_by(prop='tx_side1', vals=[12.5, 50.0])

    snd12_n = survey_n.soundings[survey_n.sounding_names[1]]
    filter_12 = np.r_[np.asarray(snd12_n.time_f)[0]*0.99, np.asarray(snd12_n.time_f)[-1]*1.1]
    
    log_12 = survey_n.invlog_info[survey_n.invlog_info.name == snd12_n.name.strip()]
    rrms_12 = log_12.rRMS.values[0]
    chi2_12 = log_12.chi2.values[0]
    lam_12 = log_12.lam.values[0]
    lamfin_12 = log_12.lam_fin.values[0]
    cf_12 = log_12.cf.values[0]
    noifl_12 = log_12.noifl.values[0]
    niter_12 = log_12.n_iter.values[0]
    
    fig, ax = plt.subplots(2, 2, figsize=(8, 8), constrained_layout=True)
    
    model_inv = mtrxMDL2vec(snd12_n.inv_model)
    res_inv, thk_inv = model_inv[nlayer-1:nlayer*2-1], model_inv[0:nlayer-1]
    mdl4doi = np.column_stack((np.r_[0, thk_inv], res_inv))
    doi12 = calc_doi(current=snd12_n.current,
                    tx_area=snd12_n.tx_loop**2,
                    eta=snd12_n.sgnl_c.iloc[-1],
                    mdl_rz=mdl4doi, x0=30,
                    verbose=True)
    
    
    snd12_n.plot_dBzdt(which='observed', ax=ax[0, 0], xlimits=(3e-6, 5e-3),
                          ylimits=(1e-10, 1e-1), label='data', color='k', show_sub0_label=False)
    snd12_n.plot_dBzdt(which='calculated', ax=ax[0, 0], color='dodgerblue',
                         marker='x', ls=':', label='response', show_sub0_label=False)
    ax[0, 0].set_title(f'field data 12.5 m loop\n$\chi^2$ = {chi2_12:0.1f}, rRMS = {rrms_12:0.1f}%')
    ax[0, 0].legend()

    tm_thks = np.r_[thk[:-1], 50]
    tm_res = np.r_[res, res[-1]]
    drawModel1D(ax[0, 1], tm_thks, tm_res, color="black", ls='--', label="true model")
    snd12_n.plot_initial_model(ax=ax[0, 1], color='green', 
                                   marker='.', ls='--', label='init. model'
                                   )
    snd12_n.plot_inv_model(ax=ax[0, 1], color='dodgerblue',
                               marker='.', label='inv. model'
                               )
    ax[0, 1].axhline(y=doi12[0], ls='--', color='magenta', label='DOI')

    ax[0, 1].set_title((f'init. lam {lam_12:.1f} cooling: {cf_12}, iters {niter_12}\n' +
                        f'fin. lam {lamfin_12:.1f}, noise floor = {noifl_12}%'))
    
    ax[0, 1].invert_yaxis()
    ax[0, 1].set_xlim(lims_rho)
    ax[0, 1].set_ylim(lims_depth)
    ax[0, 1].set_xlabel(r'$\rho$ ($\Omega$m)')
    ax[0, 1].set_ylabel('z (m)')
    ax[0, 1].legend(loc='upper right')
    
    tag = f'DOI = {doi12[0]:.1f} m'
    at = AnchoredText(tag, prop={'color': 'k', 'fontsize': 14},
                        frameon=True, loc='lower right')
    at.patch.set_boxstyle("round, pad=0.0, rounding_size=0.2")
    ax[0, 1].add_artist(at)

    snd50_n = survey_n.soundings[survey_n.sounding_names[0]]
    filter_50 = np.r_[np.asarray(snd50_n.time_f)[0]*0.99, np.asarray(snd50_n.time_f)[-1]*1.1]
    
    model_inv = mtrxMDL2vec(snd50_n.inv_model)
    res_inv, thk_inv = model_inv[nlayer-1:nlayer*2-1], model_inv[0:nlayer-1]
    mdl4doi = np.column_stack((np.r_[0, thk_inv], res_inv))
    doi50 = calc_doi(current=snd50_n.current,
                    tx_area=snd50_n.tx_loop**2,
                    eta=snd50_n.sgnl_c.iloc[-1],
                    mdl_rz=mdl4doi, x0=30,
                    verbose=True)
    
    
    log_50 = survey_n.invlog_info[survey_n.invlog_info.name == snd50_n.name.strip()]
    rrms_50 = log_50.rRMS.values[0]
    chi2_50 = log_50.chi2.values[0]
    lam_50 = log_50.lam.values[0]
    lamfin_50 = log_50.lam_fin.values[0]
    cf_50 = log_50.cf.values[0]
    noifl_50 = log_50.noifl.values[0]
    niter_50 = log_50.n_iter.values[0]
    
    snd50_n.plot_dBzdt(which='observed', ax=ax[1, 0], xlimits=(3e-6, 5e-3),
                          ylimits=(1e-10, 1e-1), label='field data 50.0 m loop',
                          color='k', show_sub0_label=False
                          )
    snd50_n.plot_dBzdt(which='calculated', ax=ax[1, 0],
                           color='dodgerblue', marker='x', ls=':',
                           show_sub0_label=False
                           )
    ax[1, 0].set_title(f'field data 50.0 m loop\n$\chi^2$ = {chi2_50:0.1f}, rRMS = {rrms_50:0.1f}%')
    
    drawModel1D(ax[1, 1], tm_thks, tm_res, color="black", ls='--', label="true model")
    snd50_n.plot_initial_model(ax=ax[1, 1], color='green', marker='.', ls='--')
    snd50_n.plot_inv_model(ax=ax[1, 1], color='dodgerblue', marker='.')
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
    ax[1, 1].set_xlim(lims_rho)
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
    fig.savefig(savepath + f'numerical_12-50_{version}.png', dpi=300)
