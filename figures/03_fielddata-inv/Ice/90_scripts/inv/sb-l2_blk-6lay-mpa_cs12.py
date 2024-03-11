# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 16:21:10 2022

script for blocky batch inversion of TEM-IP data with empymod and pygimli
empymod for forward solution and pygimli for inversion framework

adds the time range to the version folder name; this allows for running versions with different time ranges
--> it is possible to read the same data file multiple times when providing different time ranges

intended for inverting a full dataset with similar parameters
(maybe for slight changes to the number of layers or lambda)

Not yet available: 
TODO!!
    [] for a full parameter test use pg-smo_test-params_tXX.py
    [] batch TEM-IP inversion, using blocky pg inversion
    [] and for the L-Curve plot use: pg-smo_lcurve_vXX.py
        [] add auto L-Curve plotting
            use log file - if there are more than 5 different lambdas for similar setting
            --> create auti L-Curve plot. (later)
    [] add MPA model for inversion
        [] add model type switch

TODO:
    [Done] finish saving result - use structure as for simpeg results
        [Done] test!!
    [Done] reread results
    [Done] prepare plots
    [Done] add DOI to plots
        check jacobian also!! - later for sensitivity analysis
    [Done] fix log files - overwrites for each result
        [Done] add snd name
        [Done] fixed rerreading of logs for the plots
    [Done] add time range to folder version
    [] save results also as .xls
        or better before plotting as section?? - YES

any other ideas?
    maybe create a batch_inv class framework that handles all the looping


~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Sonnblick 6 soundings on ice glacier, only even numbered soundings
constrained version - to prior knowledge from gpr


@author: lukas
"""


# %% import modules
import os
import sys
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from matplotlib.colors import LogNorm, SymLogNorm
from matplotlib.backends.backend_pdf import PdfPages

import pygimli as pg
from pygimli.viewer.mpl import drawModel1D

# import empymod

from TEM_frwrd.empymod_frwrd_ip import empymod_frwrd

from TEM_frwrd.TEM_inv_tools import plot_signal
from TEM_frwrd.TEM_inv_tools import plot_rhoa
from TEM_frwrd.TEM_inv_tools import calc_rhoa
from TEM_frwrd.TEM_inv_tools import calc_doi
from TEM_frwrd.TEM_inv_tools import parse_TEMfastFile
# from TEM_frwrd.TEM_inv_tools import parse_zondxls
# from TEM_frwrd.TEM_inv_tools import reArr_zondMdl
from TEM_frwrd.TEM_inv_tools import get_float_from_string
from TEM_frwrd.TEM_inv_tools import round_up
from TEM_frwrd.TEM_inv_tools import query_yes_no
from TEM_frwrd.TEM_inv_tools import vectorMDL2mtrx
from TEM_frwrd.TEM_inv_tools import plot_diffs

from TEM_inv import tem_inv_smooth1D
from TEM_inv import temip_block1D_fwd
from TEM_inv import LSQRInversion

from timer import Timer

logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)
# logging.getLogger().setLevel(logging.DEBUG)
# logging.getLogger().setLevel(logging.INFO)

# %% functions
def prep_mdl_para_names(param_names, n_layers):
    mdl_para_names = []
    for pname in param_names:
        for n in range(0, n_layers):
            if 'thk' in pname and n == n_layers-1:
                break
            mdl_para_names.append(f'{pname}_{n:02d}')
    return mdl_para_names


# %% plot style
plt.style.use('ggplot')

# plot font sizes
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 16

min_time = 3e-6
limits_sign = [1e-11, 1e-2]
limits_rhoa = [1e2, 5e4]

limits_dpt = (50, 0)
limits_rho = [1e2, 5e4]
limits_m = [-0.1, 1]
limits_tau = [1e-6, 1e-2]
limits_c = [-0.1, 1]


# %% booolean switches
start_inv = True
# start_inv = False
show_results = True
# show_results = False
save_dataplot = True
# save_dataplot = False
save_resultplot = True
# save_resultplot = False
save_data = True

scriptname = os.path.basename(sys.argv[0])
print(f'running {scriptname} ...')
version = scriptname.split('.')[0].split('_')[-1]
batch_type = scriptname.split('.')[0].split('_')[-2]
inv_type = scriptname.split('.')[0].split('_')[-3]


# %% directions data
main = '../../../'
path_data = main + '01-data/selected/'
fnames = ['sb-line-02', 'sb-line-02']


# %% filtering and limits
filter_times = [(12, 120), (12, 100)]  # in us # filter_times = [[9, 200]], # filter_times = [[10, 200]]


# %% inversion settings
lambdas = [1500]  # in blocky case: initial lambdas
cooling_factor = [0.8]
max_iter = 25
mys = [1e2]  # regularization for parameter constraint

# noise_floors = np.arange(0.03, 0.3, 0.005)  # lowest relative noise accepted in the error (%/100)
noise_floors = np.arange(0.05, 0.15, 0.03)


# %% setup start model
# ip_modeltype = 'pelton'
ip_modeltype = 'mpa'  # mpa

init_layer_thk = np.r_[3, 3, 3, 3, 6]
constr_thk = np.r_[0, 0, 0, 0, 0]  # set to 1 if parameter should be fixed
init_layer_res = np.r_[300, 3000, 3000, 3000, 3000, 1000]
constr_res = np.r_[0, 0, 0, 0, 0, 0]

# init_layer_m = np.array([0.3, 0.3, 0.3, 0.3])
init_layer_mpa = np.array([0.0, 0.6, 0.6, 0.6, 0.6, 0.0])
constr_charg = np.r_[1, 0, 0, 0, 0, 1]

# init_layer_tau = np.array([1e-5, 1e-5, 1e-5, 1e-5])
init_layer_taup = np.array([1e-6, 1e-4, 1e-4, 1e-4, 1e-4, 1e-6])
constr_tau = np.r_[1, 0, 0, 0, 0, 1]

init_layer_c = np.array([0.1, 0.7, 0.7, 0.7, 0.7, 0.1])
constr_c = np.r_[1, 0, 0, 0, 0, 1]

if ip_modeltype == 'pelton':
    constr_1d = np.hstack((constr_thk, constr_res, constr_charg, constr_tau, constr_c))
    start_model = pg.Vector(np.hstack((init_layer_thk, init_layer_res, init_layer_m, init_layer_tau, init_layer_c)))
    param_names = ['thk', 'rho0','m', 'tau', 'c']
elif ip_modeltype == 'mpa':
    constr_1d = np.hstack((constr_thk, constr_res, constr_charg, constr_tau, constr_c))
    start_model = pg.Vector(np.hstack((init_layer_thk, init_layer_res, init_layer_mpa, init_layer_taup, init_layer_c)))
    param_names = ['thk', 'rho0','max_pha', 'tau_phi', 'c']
else:
    raise ValueError('this ip modeltype is not implemented here ...')
mdl_para_names = prep_mdl_para_names(param_names, n_layers=len(init_layer_res))

start_model = pg.Vector(np.hstack((init_layer_thk, init_layer_res, init_layer_mpa,
                                   init_layer_taup, init_layer_c)))
start_model_2D = np.column_stack((np.r_[init_layer_thk, 0], init_layer_res,
                                  init_layer_mpa, init_layer_taup, init_layer_c))
nlayers = start_model_2D.shape[0]
nparams = start_model_2D.shape[1]

transThk = pg.trans.TransLogLU(1, 5)  # log-transform ensures thk>0
transRho = pg.trans.TransLogLU(300, 5000)  # lower and upper bound
transM = pg.trans.TransLogLU(0.01, 1.0)  # lower and upper bound
transTau = pg.trans.TransLogLU(1e-6, 1e-3)  # lower and upper bound
transC = pg.trans.TransLogLU(0.01, 1.0)  # lower and upper bound

transData = pg.trans.TransLin()  # lin transformation for data


# %% prepare constraints
constrain_mdl_params = False
constr_1d = np.hstack((constr_thk, constr_res, constr_charg, constr_tau, constr_c))
if any(constr_1d == 1):
    constrain_mdl_params = True
    print('any constrain switch==1 --> using constraints')

ones_pos = np.where(constr_1d==1)[0]
Gi = pg.core.RMatrix(rows=len(ones_pos), cols=len(start_model))

for idx, pos in enumerate(ones_pos):
    Gi.setVal(idx, pos, 1)
# cstr_vals = Gi * true_model_1d  # use true model values
cstr_vals = None  # if None it will use the start model values


# %% loop over files in fnames list
t = Timer()

full_time = Timer()
full_time.start()
for idx, fname_data in enumerate(fnames):
    print('\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f' - starting preparation for inversion of file: {fname_data}\n')

    time_range = filter_times[idx]
    tr0 = time_range[0]
    trN = time_range[1]
    print('-----------------------------------------------------------------')
    print(f'starting with file: {fname_data}.tem')
    print('-----------------------------------------------------------------\n')
    print('selected time range (us): ', tr0, '-', trN)
    print('-----------------------------------------------------------------\n')

    fID_savename = fname_data[:-4] if fname_data[-4:] == '_txt' else fname_data
    # pdf_filename = '{:s}_{:s}_{:02d}.pdf'.format(logID_savename,
                                                 # name_snd, logID)
    pdf_filename = '{:s}_batch.pdf'.format(fID_savename)
    
    pre = f'{inv_type}_{batch_type}'
    savepath_data = main + f'03-inv_results/{pre}/{fID_savename}/{version}_tr{tr0}-{trN}us/'
    if not os.path.exists(savepath_data):
        os.makedirs(savepath_data)

    # %% read data
    print('-----------------------------------------------------------------')
    print('reading file with following number of logs:')
    rawData, nLogs, indices_hdr, indices_dat = parse_TEMfastFile(fname_data,
                                                                 path_data)
    snd_names = []
    positions = []


    # %% plot appearance
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 16


    # prepare inversion protokoll
    main_prot_fid = savepath_data + f'{fID_savename}.log'
    invprot_hdr = ('name\tri\tminT(us)\tmaxT(us)\tlam\tmy\tcf\tnoifl\t' +
                   'max_iter\tn_iter\taRMS\trRMS\tchi2\truntime(min)')
    if start_inv:
        with open(main_prot_fid, 'w') as main_prot:
            main_prot.write(invprot_hdr + '\n')
    
    # if query_yes_no(f'proceed with iteration over n={nLogs} soundings', default='no'):
    with PdfPages(savepath_data + pdf_filename) as pdf:
        for logID in range(0,nLogs):
        # for logID in [0]:  # for testing first souning only
        # for logID in range(154, 155):  # for testing last sounding
            # %% start by reading a single .tem data file
            datFrame = rawData.loc[indices_dat.start[logID]:indices_dat.end[logID]-1]
            dmeas = datFrame.drop(['c6','c7','c8'], axis=1)
            dmeas.columns = ['channel', 'time', 'signal', 'err', 'rhoa']
            dmeas = dmeas.apply(pd.to_numeric)

            # create sounding params from header:
            header = rawData.loc[indices_hdr.start[logID]:indices_hdr.end[logID]-1]
            name_snd = header.iloc[2,1].strip()
            time_key = int(header.iloc[3,1])
            current_inj = get_float_from_string(header.iloc[3,5])
            Tx_loop = float(header.iloc[4, 1])
            Rx_loop = float(header.iloc[4, 3])
            turns = float(header.iloc[4, 5])

            posX = float(header.iloc[6,1])
            posY = float(header.iloc[6,3])
            posZ = float(header.iloc[6,5])

            snd_names.append(name_snd)
            positions.append((posX, posY, posZ))
            
            print('\n\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print(f' - starting preparation for inversion of sounding: {name_snd}\n')

            # # read zond result for comp
            # if fname_ztinv is not None:
            #     mdl_zt = dat_zt.loc[indices_mdl_zt.start[logID]:indices_mdl_zt.end[logID],
            #                           ['c1','c2','c3','c4','c5','c6','c7','c8']]
            #     mdl_zt.columns = ['ctr','Rho','Pol','Tconst','Cexpo','MagP','h','z']
            #     mdl_zt.drop(['Pol','Tconst','Cexpo'], axis=1, inplace=True)
            #     ra_mdl_zt = reArr_zondMdl(mdl_zt)
            # if fname_spinv is not None:
            #     mdl_sp = dat_sp.loc[indices_mdl_sp.start[logID]:indices_mdl_sp.end[logID],
            #                           ['c1','c2','c3','c4','c5','c6','c7','c8']]
            #     mdl_sp.columns = ['ctr','Rho','Pol','Tconst','Cexpo','MagP','h','z']
            #     mdl_sp.drop(['Pol','Tconst','Cexpo'], axis=1, inplace=True)
            #     ra_mdl_sp = reArr_zondMdl(mdl_sp)

            # prepare savename
            # fID_savename = fname_data[:-4] if fname_data[-4:] == '_txt' else fname_data
            # pdf_filename = '{:s}_{:s}_{:02d}.pdf'.format(logID_savename,
                                                         # name_snd, logID)
            savepath_autoplot = savepath_data + f'{name_snd}/autoplot/'
            if not os.path.exists(savepath_autoplot):
                os.makedirs(savepath_autoplot)
            savepath_csv = savepath_data + f'{name_snd}/csv/'
            if not os.path.exists(savepath_csv):
                os.makedirs(savepath_csv)


            # %% normalize data
            Rx_area = Rx_loop**2
            Tx_area = Tx_loop**2
            M = current_inj * Tx_loop**2 * turns
            currentkey = np.round(current_inj)

            dmeas_norm = dmeas.copy()
            dmeas_norm.iloc[:, 2:4] = dmeas_norm.iloc[:, 2:4] * current_inj / Rx_area
            obs_dat = dmeas_norm.signal.values
            obs_err = dmeas_norm.err.values

            times_all = dmeas_norm.time.values * 1e-6
            rhoa = dmeas_norm.rhoa.values


            # %% filtering the data - # select subset according to time range
            dmeas_sub = dmeas_norm[(dmeas.time>tr0) & (dmeas.time<trN)]
            rx_times_sub = dmeas_sub.time.values * 1e-6
            obs_dat_sub = dmeas_sub.signal.values
            obs_err_sub = dmeas_sub.err.values

            rel_err_sub = abs(obs_err_sub) / obs_dat_sub

            rhoa_median = np.round(np.median(dmeas_sub.rhoa.values), 2)


            # %% setup system and forward solver
            device = 'TEMfast'

            setup_device = {"timekey": time_key,
                            "currentkey": currentkey,
                            "txloop": Tx_loop,
                            "rxloop": Rx_loop,
                            "current_inj": current_inj,
                            "filter_powerline": 50}

            # 'ftarg': 'key_81_CosSin_2009', 'key_201_CosSin_2012', 'ftarg': 'key_601_CosSin_2009'
            setup_solver = {'ft': 'dlf',                     # type of fourier trafo
                              'ftarg': 'key_601_CosSin_2009',  # ft-argument; filter type # https://empymod.emsig.xyz/en/stable/api/filters.html#module-empymod.filters -- for filter names
                              'verbose': 0,                    # level of verbosity (0-4) - larger, more info
                              'srcpts': 3,                     # Approx. the finite dip. with x points. Number of integration points for bipole source/receiver, default is 1:, srcpts/recpts < 3 : bipole, but calculated as dipole at centre
                              'recpts': 3,                     # Approx. the finite dip. with x points. srcpts/recpts >= 3 : bipole
                              'ht': 'dlf',                     # type of fourier trafo
                              'htarg': 'key_401_2009',         # hankel transform filter type, 'key_401_2009', 'key_101_2009'
                              'nquad': 3,                      # Number of Gauss-Legendre points for the integration. Default is 3.
                              'cutoff_f': 5e6,                 # TODO add automatisation for diff loops;  cut-off freq of butterworthtype filter - None: No filter applied, WalkTEM 4.5e5
                              'delay_rst': 0,                  # ?? unknown para for walktem - keep at 0 for fasttem
                              'rxloop': 'vert. dipole'}        # or 'same as txloop' - not yet operational

            empy_frwrd = empymod_frwrd(setup_device=setup_device,
                                       setup_solver=setup_solver,
                                       filter_times=time_range, device='TEMfast',
                                       nlayer=nlayers, nparam=nparams)


            # %% inversion setup and test startmodel using first entry in mesh related lists
            fop = temip_block1D_fwd(empy_frwrd, ip_mdltype=ip_modeltype,
                                    nPara=nparams-1, nLayers=nlayers,
                                    shift=None, verbose=True)

            fop.region(0).setTransModel(transThk)  # 0=thickness
            fop.region(1).setTransModel(transRho)  # 1=resistivity
            fop.region(2).setTransModel(transM)    # 2=m
            fop.region(3).setTransModel(transTau)  # 3=tau
            fop.region(4).setTransModel(transC)    # 4=c

            fop.setMultiThreadJacobian(1)

            t.start()
            simdata = fop.response(start_model)  # simulate start model response
            frwrd_time = t.stop(prefix='forward-')


            # %% visualize rawdata and filtering plus error!!
            fg_raw, ax_raw = plt.subplots(1, 2, figsize=(12,6))

            # rawdata first
            _ = plot_signal(ax_raw[0], times_all, obs_dat,
                            marker='o', ls=':', color='grey', sub0color='orange',
                            label='data raw')
            ax_raw[0].loglog(times_all, obs_err,  # noise
                             'd', ms=4, ls=':',
                             color='grey', alpha=0.5,
                             label='noise raw')

            _ = plot_rhoa(ax_raw[1], times_all, rhoa,
                          marker='o', ls=':', color='grey',
                          label='rhoa raw')

            # filtered data
            _ = plot_signal(ax_raw[0], rx_times_sub, obs_dat_sub,
                            marker='d', ls=':', color='k', sub0color='orange',
                            label='data subset')

            _ = plot_rhoa(ax_raw[1], rx_times_sub, dmeas_sub.rhoa,
                          marker='d', ls=':', color='k', sub0color='orange',
                          label='rhoa subset')

            # and comparing it to the measured (observed) subset of the data
            _ = plot_signal(ax_raw[0], rx_times_sub, simdata,
                            marker='x', color='crimson', ls='None', sub0color='orange',
                            label='data sim')
            sim_rhoa = calc_rhoa(empy_frwrd, simdata, rx_times_sub)
            _ = plot_rhoa(ax_raw[1], rx_times_sub, sim_rhoa,
                          marker='x', color='crimson', ls='None', sub0color='orange',
                          label='rhoa sim')

            max_time = round_up(max(times_all)*1e6, 100) / 1e6
            ax_raw[0].set_xlabel('time (s)')
            ax_raw[0].set_ylabel(r'$\frac{\delta B}{\delta t}$ (V/m²)')
            ax_raw[0].set_ylim((limits_sign[0], limits_sign[1]))
            ax_raw[0].set_xlim((min_time, max_time))
            ax_raw[0].grid(True, which='major', color='white', linestyle='-')
            ax_raw[0].grid(True, which='minor', color='white',  linestyle=':')
            ax_raw[0].legend()
            ax_raw[0].set_title(f'{name_snd}')

            ax_raw[1].set_xlabel('time (s)')
            ax_raw[1].set_ylabel(r'$\rho_a$ ($\Omega$m)')
            ax_raw[1].set_ylim((limits_rhoa[0], limits_rhoa[1]))
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

            fig_savefid = (savepath_autoplot +
                           'data_{:s}_{:02d}.png'.format(name_snd, logID))
            if save_dataplot == True:
                print('saving rawdata plot to:\n', fig_savefid)
                fg_raw.savefig(fig_savefid, dpi=150)
                print('saving rawdata to auto pdf at:\n', pdf)
                fg_raw.savefig(pdf, format='pdf')
                plt.close('all')
            else:
                print('rawdata plot not saved...')
                plt.show()

            # prepare inversion protokoll for individual sounding
            snd_prot_fid = savepath_csv.replace('csv/', '') + f'{fID_savename}_snd{name_snd}.log'


            # %% start the inversion
            total_runs = len(lambdas) * len(mys) * len(cooling_factor) * len(noise_floors)
            message = (f'proceed with inversion using n={total_runs:.0f} different settings\n' + 
                       '"no" proceeds with plotting - only if inversion was done already ...')
            # %% start the inversion
            total_runs = len(lambdas) * len(mys) * len(cooling_factor) * len(noise_floors)
            message = (f'proceed with inversion using n={total_runs:.0f} different settings\n' + 
                       '"no" proceeds with plotting - only if inversion was done already ...')
            # if start_inv and query_yes_no(message, default='no'):
            if start_inv:
                with open(snd_prot_fid, 'w') as snd_prot:
                    snd_prot.write(invprot_hdr + '\n')
                inv_run = 0

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
                                tem_inv.setModel(start_model)  # set start model
                                tem_inv.setData(obs_dat_sub)

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
                                print('and initial model:\n', start_model)

                                rel_err = np.copy(rel_err_sub)
                                if any(rel_err < noise_floor):
                                    logging.warning(f'Encountered rel. error below {noise_floor*100}% - setting those to {noise_floor*100}%')
                                    rel_err[rel_err < noise_floor] = noise_floor
                                    abs_err = abs(obs_dat_sub * rel_err)

                                tem_inv.setAbsoluteError(abs(abs_err))
                                if constrain_mdl_params:
                                    tem_inv.setParameterConstraints(G=Gi, c=cstr_vals, my=my)
                                else:
                                    print('no constraints used ...')

                                t.start()
                                model_inv = tem_inv.run()
                                inv_time = t.stop(prefix='inv-')
                                
                                inv_res, inv_thk = model_inv[nlayers-1:nlayers*2-1], model_inv[0:nlayers-1]
                                chi2 = tem_inv.chi2()
                                relrms = tem_inv.relrms()

                                model_inv_mtrx = vectorMDL2mtrx(model_inv, nlayers, nparams)
                                inv_m = model_inv_mtrx[:, 2]
                                inv_tau = model_inv_mtrx[:, 3]
                                inv_c = model_inv_mtrx[:, 4]
                                
                                fop = tem_inv.fop()
                                col_names = mdl_para_names
                                row_names = [f'dB/dt tg{i:02d}' for i in range(1, len(obs_dat_sub)+1)]
                                jac_df = pd.DataFrame(np.array(fop.jacobian()), columns=col_names, index=row_names)
                                

                                print('\ninversion runtime: {:.1f} min.'.format(inv_time/60))
                                print('--------------------   INV finished   ---------------------')

                                # %% save result and fit
                                chi2 = tem_inv.chi2()
                                rrms = tem_inv.relrms()
                                arms = tem_inv.absrms()
                                n_iter = tem_inv.n_iters  # get number of iterations

                                # TODO fix export - add cole cole params
                                result_arr = np.column_stack((np.r_[inv_thk, 0], inv_res,
                                                           inv_m, inv_tau, inv_c))

                                pred_data = np.asarray(tem_inv.response())
                                pred_rhoa = calc_rhoa(empy_frwrd, pred_data,
                                                      rx_times_sub)

                                if ip_modeltype == 'pelton':
                                    header_result = 'X,Y,Z,depth(m),rho(Ohmm),m(),tau(s),c()'
                                    labels_CC = ['chargeability m ()', r'rel. time $\tau$ (s)']
                                elif ip_modeltype == 'mpa':
                                    header_result = 'X,Y,Z,depth(m),rho(Ohmm),mpa(rad),tau_p(s),c()'
                                    labels_CC = ['mpa (rad)', r'rel. time $\tau_{\phi}$ (s)']
                                else:
                                    raise ValueError('this ip modeltype is not implemented here ...')
                                export_array = np.column_stack((np.full((len(result_arr),), posX),
                                                                np.full((len(result_arr),), posY),
                                                                np.full((len(result_arr),), posZ),
                                                                result_arr))

                                header_fit = ('time(s),signal_pred(V/m2),' +
                                              'signal_obs(V/m2),err_obs(V/m2),err_scl(V/m2),' +
                                              'rhoa_pred(V/m2),rhoa_obs(V/m2)')
                                export_fit = np.column_stack((rx_times_sub,
                                                              pred_data, obs_dat_sub,
                                                              obs_err_sub, abs_err,
                                                              pred_rhoa, dmeas_sub.rhoa))

                                header_startmdl = 'X,Y,Z,thk(m),rho(Ohmm),m(),tau(s),c()'
                                exportSM_array = np.column_stack((np.full((len(result_arr),), posX),
                                                                  np.full((len(result_arr),), posY),
                                                                  np.full((len(result_arr),), posZ),
                                                                  start_model_2D))

                                savename = ('invrun{:03d}_{:s}'.format(inv_run, name_snd))
                                if save_data == True:
                                    print(f'saving data from inversion run: {inv_run}')
                                    np.savetxt(savepath_csv + savename +'.csv',
                                               export_array, comments='',
                                               header=header_result,
                                               fmt='%.3f,%.3f,%.3f,%.3f,%.3f,%.1f,%.1e,%.3f')
                                    np.savetxt(savepath_csv + savename +'_startmodel.csv',
                                               exportSM_array, comments='',
                                               header=header_result,
                                               fmt='%.3f,%.3f,%.3f,%.3f,%.3f,%.1f,%.1e,%.3f')
                                    np.savetxt(savepath_csv + savename +'_fit.csv',
                                               export_fit,
                                               comments='',
                                               header=header_fit,
                                               fmt='%.6e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e')
                                    jac_df.to_csv(savepath_csv + savename + '_jac.csv')

                                # %% save main log
                                logline = ("%s\t" % (name_snd) +
                                           "r%03d\t" % (inv_run) +
                                           "%.1f\t" % (tr0) +
                                           "%.1f\t" % (trN) +
                                           "%8.1f\t" % (lam) +
                                           "%.1e\t" % (my) +
                                           "%.1e\t" % (cf) +  # cooling factor
                                           "%.2f\t" % (noise_floor) +
                                           "%d\t" % (max_iter) +
                                           "%d\t" % (n_iter) +
                                           "%.2e\t" % (arms) +
                                           "%7.3f\t" % (rrms) +
                                           "%7.3f\t" % (chi2) +
                                           "%4.1f\n" % (inv_time/60))  # to min
                                with open(main_prot_fid,'a+') as f:
                                    f.write(logline)
                                with open(snd_prot_fid,'a+') as f:
                                    f.write(logline)

                                inv_run += 1



    # %% reload results and plot
            if show_results:
                inv_run = 0
                # read log files:
                read_log = np.genfromtxt(snd_prot_fid,
                                         skip_header=1, delimiter='\t')
                
                if ip_modeltype == 'pelton':
                    labels_CC = ['chargeability m ()', r'rel. time $\tau$ (s)']
                elif ip_modeltype == 'mpa':
                    labels_CC = ['mpa (rad)', r'rel. time $\tau_{\phi}$ (s)']
                else:
                    raise ValueError('this ip modeltype is not implemented here ...')

                for lam in lambdas:
                    for my in mys:
                        for cf in cooling_factor:
                            for noise_floor in noise_floors:
                                savename = ('invrun{:03d}_{:s}'.format(inv_run, name_snd))
                                fid_results = savepath_csv + savename + '.csv'
                                fid_sm = savepath_csv + savename + '_startmodel.csv'
                                fid_fit = savepath_csv + savename + '_fit.csv'
                                fid_jac = savepath_csv + savename + '_jac.csv'

                                read_result = np.genfromtxt(fid_results,
                                                            skip_header=1, delimiter=',')

                                read_startmdl = np.genfromtxt(fid_sm,
                                                              skip_header=1, delimiter=',')

                                read_fit = np.genfromtxt(fid_fit,
                                                         skip_header=1, delimiter=',')
                                
                                jac_df = pd.read_csv(fid_jac, index_col=0)

                                # create variables for plotting
                                if len(read_log.shape) == 1:
                                    log_i = read_log
                                else:
                                    log_i = read_log[inv_run]

                                # sndname = log_i[0]  not working - non numeric!!!
                                lam = log_i[4]
                                my = log_i[5]
                                cf = log_i[6]
                                nf = log_i[7]
                                max_iter = log_i[8]
                                n_iter = log_i[9]
                                arms = log_i[10]
                                rrms = log_i[11]
                                chi2 = log_i[12]
                                runtime = log_i[13]

                                inv_model = read_result[:, 3:]
                                inv_thk = read_result[:, 3]
                                inv_res = read_result[:, 4]
                                inv_m = read_result[:, 5]
                                inv_tau = read_result[:, 6]
                                inv_c = read_result[:, 7]
                                pos = read_result[0, 0:3]

                                init_thk = read_startmdl[:, 3]
                                init_res = read_startmdl[:, 4]
                                init_m = read_startmdl[:, 5]
                                init_tau = read_startmdl[:, 6]
                                init_c = read_startmdl[:, 7]

                                rx_times_sub = read_fit[:,0]
                                pred_data = read_fit[:,1]
                                obs_dat_sub = read_fit[:,2]
                                obs_error = read_fit[:,3]
                                est_error = read_fit[:,4]

                                pred_rhoa = read_fit[:,5]
                                obs_rhoa = read_fit[:,6]

                                # %% plot the data fit
                                fg_fit, ax_fit = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

                                _ = plot_signal(ax_fit[0], time=rx_times_sub, signal=obs_dat_sub, label='measured',
                                                marker='o', color='k', sub0color='gray')
                                _ = plot_signal(ax_fit[0], time=rx_times_sub, signal=pred_data, label='response',
                                                marker='.', ls='--', color='crimson', sub0color='orange')
                                ax_fit[0].loglog(rx_times_sub, obs_error, label='noise measured',
                                                 marker='d', ms=2, ls=':', color='grey', alpha=0.5)
                                ax_fit[0].loglog(rx_times_sub, est_error, marker='d', label='noise estimated',
                                                 ms=2, ls='--', color='crimson', alpha=0.5)
                                axt = plot_diffs(ax=ax_fit[0], times=rx_times_sub,
                                                 response=pred_data, measured=obs_dat_sub,
                                                 relative=True, max_diff=50)
                                # ask matplotlib for the plotted objects and their labels
                                lines, labels = ax_fit[0].get_legend_handles_labels()
                                lines2, labels2 = axt.get_legend_handles_labels()
                                ax_fit[0].legend(lines + lines2, labels + labels2, loc=0)
                                ax_fit[0].legend()
                                ax_fit[0].set_xlabel('Time (s)')
                                ax_fit[0].set_ylabel(r"$\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t$ (V/m²)")
                                ax_fit[0].set_ylim((limits_sign[0], limits_sign[1]))
                                ttl = (f'inv-run: {inv_run:03d}, sndID: {name_snd}, n_iter: {n_iter}' +
                                       f'\nchi2 = {chi2:0.2f}, rrms = {rrms:0.1f}%')
                                ax_fit[0].set_title(ttl)

                                _ = plot_rhoa(ax_fit[1], time=rx_times_sub, rhoa=obs_rhoa, label='measured $\\rho_a$',
                                              marker='o', color='k', sub0color='gray')
                                _ = plot_rhoa(ax_fit[1], time=rx_times_sub, rhoa=pred_rhoa, label='response $\\rho_a$',
                                              marker='.', ls='--', color='crimson', sub0color='orange')
                                axt = plot_diffs(ax=ax_fit[1], times=rx_times_sub,
                                                 response=pred_rhoa, measured=obs_rhoa,
                                                 relative=True, max_diff=50)
                                # ask matplotlib for the plotted objects and their labels
                                lines, labels = ax_fit[1].get_legend_handles_labels()
                                lines2, labels2 = axt.get_legend_handles_labels()
                                ax_fit[1].legend()
                                ax_fit[1].set_xlabel('Time (s)')
                                ax_fit[1].set_ylabel(r'$\rho_a$ $(\Omega m)$')
                                ax_fit[1].set_ylim((limits_rhoa[0], limits_rhoa[1]))
                                ax_fit[1].set_title((f'lambda={lam:.1f}, cooling_fac={cf:.1f}\n' +
                                                     f'noise_floor={nf*100:.1f}%, runtime={runtime:.2f} min'))

                                # %% plot the models
                                fg_mdl, ax_mdl = plt.subplots(2, 2, figsize=(9, 9), constrained_layout=True)
                                
                                np.set_printoptions(formatter={'float': '{: 0.0f}'.format})
                                drawModel1D(ax_mdl[0, 0], init_thk, init_res,
                                            color="green", marker='.', ls='--', label="initial model")
                                drawModel1D(ax_mdl[0, 0], inv_thk, inv_res, color="crimson", label="inverted model")
                                # ax_mdl[0, 0].set_title(f'inv. $\\rho_0$: {inv_res}')
                                ax_mdl[0, 0].legend()
                                ax_mdl[0, 0].set_xscale('log')
                                ax_mdl[0, 0].set_ylim(limits_dpt)
                                ax_mdl[0, 0].set_xlim(limits_rho)
                                ax_mdl[0, 0].set_xlabel(r'$\rho_0 (\Omega m)$')
                                ax_mdl[0, 0].set_ylabel('z (m)')
                                
                                np.set_printoptions(formatter={'float': '{: 0.1f}'.format})
                                drawModel1D(ax_mdl[0, 1], init_layer_thk, init_layer_mpa,
                                            color="green", marker='.', ls='--', label="initial model")
                                drawModel1D(ax_mdl[0, 1], inv_thk, inv_m, color="crimson", label="inverted model")
                                # ax_mdl[0, 1].set_title(f'inv. m: {inv_m}')
                                ax_mdl[0, 1].legend()
                                ax_mdl[0, 1].set_ylim(limits_dpt)
                                ax_mdl[0, 1].set_xlim(limits_m)
                                ax_mdl[0, 1].set_xlabel(labels_CC[0])
                                ax_mdl[0, 1].set_ylabel('z (m)')
                                
                                np.set_printoptions(formatter={'float': '{: 0.1e}'.format})
                                drawModel1D(ax_mdl[1, 0], init_layer_thk, init_layer_taup,
                                            color="green", marker='.', ls='--', label="initial model")
                                drawModel1D(ax_mdl[1, 0], inv_thk, inv_tau, color="crimson", label=r"inverted model")
                                # ax_mdl[1, 0].set_title(f'inv. $\\tau$: {inv_tau}')
                                ax_mdl[1, 0].legend()
                                ax_mdl[1, 0].set_xscale('log')
                                ax_mdl[1, 0].set_ylim(limits_dpt)
                                ax_mdl[1, 0].set_xlim(limits_tau)
                                ax_mdl[1, 0].set_xlabel(labels_CC[1])
                                ax_mdl[1, 0].set_ylabel('z (m)')

                                np.set_printoptions(formatter={'float': '{: 0.1f}'.format})
                                drawModel1D(ax_mdl[1, 1], init_layer_thk, init_layer_c,
                                            color="green", marker='.', ls='--', label="initial model")
                                drawModel1D(ax_mdl[1, 1], inv_thk, inv_c, color="crimson", label="inverted model")
                                # ax_mdl[1, 1].set_title(f'inv. c: {inv_c}')
                                ax_mdl[1, 1].legend()
                                ax_mdl[1, 1].set_ylim(limits_dpt)
                                ax_mdl[1, 1].set_xlim(limits_c)
                                ax_mdl[1, 1].set_xlabel(r'disp. coefficient c ()')
                                ax_mdl[1, 1].set_ylabel('z (m)')

                                fig_savefid = (savepath_autoplot +
                                                'invrun{:03d}_{:s}_model.png'.format(inv_run, name_snd))
                                figfit_savefid = (savepath_autoplot +
                                                'invrun{:03d}_{:s}_fit.png'.format(inv_run, name_snd))
                                figjac_savefid = (savepath_autoplot +
                                                'invrun{:03d}_{:s}_jac.png'.format(inv_run, name_snd))


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

                                if save_resultplot == True:
                                    print('saving result plot to:\n', fig_savefid)
                                    fg_fit.savefig(figfit_savefid, dpi=150)
                                    fg_mdl.savefig(fig_savefid, dpi=150)
                                    figj.savefig(figjac_savefid, dpi=150)

                                    print('saving result to auto pdf at:\n', pdf)
                                    fg_fit.savefig(pdf, format='pdf')
                                    fg_mdl.savefig(pdf, format='pdf')
                                    figj.savefig(pdf, format='pdf')
                                    plt.close('all')
                                else:
                                    print('plot not saved...')
                                    plt.show()

                                # sys.exit('exit for testing ...')
                                inv_run += 1

total_time = full_time.stop(prefix='total runtime-')
