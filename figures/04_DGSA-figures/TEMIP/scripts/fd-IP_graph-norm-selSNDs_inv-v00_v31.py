# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 18:01:20 2022

script to run pyDGSA for sensitivity analysis of TEMIP graphite model obtained 
from inversion of profile P10
https://pypi.org/project/pyDGSA/

examples at:
https://github.com/zperzan/pyDGSA/blob/master/tutorial_detailed.ipynb
https://github.com/zperzan/pyDGSA/blob/master/tutorial_short.ipynb

new plotting approach, separate subplots for each cluster

Testing:
    [done] abs data (only possible for non-IP cases)     v0x
        I believe this is not enough, cluster is still all over the place
    [done] norm to 1                                                              v1x
    [done] abs_data, norm to 1                                                    v2x
    [done] abs_data, log10, norm to 1                                             v3x
    [] new test, add min(data) + 1 -> smallest val should be 1, log trafo, run!   v40x
    [] 2nd dev

Further ideas:
    [done] plot all sampled models with the prior range in a new plot
        to see whether the prior space is covered entirely
    [done] plot silhouette scores and calculate mean score
    [done] iterate over multiple cluster values, plot all!!
    [] merge silhouette score and cluster histogram

    dynamic time warping, see https://towardsdatascience.com/how-to-apply-k-means-clusthods-for-curve-fitting-parameters
    clustering algos: https://scikit-learn.org/dev/modules/clustering.html#overview-of-clustering-methods
ering-to-time-series-data-28d04a8f7da3
    normalization: https://stackoverflow.com/questions/17046397/scikits-learn-clusterization-met


TODO:
    [done] update with latest cluster number test
    [] add possibility to plot the sensitivity parameter type wise into multiple columns



@author: laigner
"""

# %% import modules
import os
import sys

# abs_path_to_libs = '/shares/laigner/home/nextcloud/09-python_coding/'
# if not abs_path_to_libs in sys.path:
#     sys.path.append(abs_path_to_libs)  # add realtive path to folder that contains all custom modules

rel_path_to_libs = '../../../../'
if not rel_path_to_libs in sys.path:
    sys.path.append('../../../../')  # add realtive path to folder that contains all custom modules

import matplotlib
import json

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib.offsetbox import AnchoredText
# from matplotlib.ticker import FuncFormatter
# from matplotlib.ticker import ScalarFormatter

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans  # alternative to k-medoids clustering
from sklearn.metrics import silhouette_score
# from inspect import currentframe, getframeinfo

from pyDGSA.cluster import KMedoids
from pyDGSA.dgsa import dgsa
# from pyDGSA.plot import vert_pareto_plot

from SALib.sample import saltelli

from pathos import multiprocessing as mp
from pathos import pools as pp

from library.TEM_frwrd.empymod_frwrd_ip import empymod_frwrd
# from TEM_frwrd.empymod_frwrd_ip import empymod_frwrd
# from library.TEM_frwrd.utils import arsinh
# from library.TEM_frwrd.utils import kth_root


from library.utils.TEM_inv_tools import mtrxMDL2vec
from library.utils.TEM_inv_tools import vecMDL2mtrx
from library.utils.universal_tools import plot_signal

from library.utils.TEM_sean_tools import evaluate_models_TEM
from library.utils.TEM_sean_tools import vert_pareto_plot
from library.utils.TEM_sean_tools import plot_prior_space
from library.utils.TEM_sean_tools import mdl2steps
from library.utils.TEM_sean_tools import plot_silhouette_scores
from library.utils.TEM_sean_tools import MODELSET
from library.utils.TEM_sean_tools import get_min
from library.utils.TEM_sean_tools import get_max
from library.utils.TEM_sean_tools import query_yes_no

# from library.tem_tools.sounding import Sounding
from library.tem_tools.survey import Survey


# %% plot appearance
dpi = 200
plt.style.use('ggplot')

fs_shift = -4
plt.rcParams['axes.labelsize'] = 18 + fs_shift
plt.rcParams['axes.titlesize'] = 18 + fs_shift
plt.rcParams['xtick.labelsize'] = 16 + fs_shift
plt.rcParams['ytick.labelsize'] = 16 + fs_shift
plt.rcParams['legend.fontsize'] = 18 + fs_shift


# %% boolean switches
show_plots = True
# show_plots = False

save_figs = True
# save_figs = False

sens_logscale_x = True
# sens_logscale_x = False
n_ticks = 3

# resp_abs = True
resp_abs = False


# units and scaling
unit = 'res (ohmm)'
# unit = 'con (mS/m)'

# parallel = False
parallel = True
# pool = None
pool = pp.ProcessPool(mp.cpu_count())  # pp.ProcessPool(mp.cpu_count())


# %% data normalization
# resp_trafo = 'oddroot_7'    # directly alter the response of forward solver, number at end refers to the root
# resp_trafo = 'areasinhyp'
resp_trafo = None
return_rhoa = False

# log_data = True
log_data = False
# log_trafo_model = [False, False, False,
#                    True, True, True, True]  # list with booleans for each model para, here only resistivity is transformed, not the thickness
log_model = False  # no log trafo for model
logTransform = [log_model, log_data]

min_to_1 = True
# min_to_1 = False

norm_to1 = True
# norm_to1 = False

# abs_data = True
abs_data = False


# %% main settings, clusters, method, loops, etc
loop_size = 12.5
timekey =  4

N_base2 = 256
# N_base2 = 4

n_clusters_s = np.arange(3, 6, 1)
# n_clusters_s = [3]

cluster_method = 'kmedoids'  # kmedoids, kmeans
# cluster_method = 'kmeans'  # kmedoids, kmeans
cmap = matplotlib.cm.get_cmap('viridis')  # hsv, spectral, jet


# %% directories
scriptname = os.path.basename(sys.argv[0])
print(f'running {scriptname} ...')
version = scriptname.split('.')[0].split('_')[-1]
mdl_invversion = scriptname.split('.')[0].split('_')[-2].split('-')[-1]
typ = scriptname.split('.')[0].split('_')[-3]
site = typ.split('-')[0]

main = '../'
savepath_figs = main + f'plots_dgsa/{typ}/{version}/'
if not os.path.exists(savepath_figs):
    os.makedirs(savepath_figs)

savepath_results = main + f'results_dgsa/{typ}/{version}/'
if not os.path.exists(savepath_results):
    os.makedirs(savepath_results)


# read results  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
line_id = 'p10'
path2hb = '../../../03_fielddata-inv/Graphite/'
name_file_raw = f'hb-{line_id}-sel'
name_file_result = f'hb-{line_id}-sel'
version = 'v00_tr5-200us'  # v01_tr20-280us
time_range = np.r_[5, 200] * 1e-6
invrun = '001'

snd_ids = ['L03', 'L12']

# rawdata
rawdata_path = path2hb + '00_data/selected/'
rawdata_fname = f'{name_file_raw}.tem'

# results
result_path = path2hb + f'30_TEM-results/hb-{line_id}_6lay-mpa//{version}/'

for kdx, snd_id in enumerate(snd_ids):
    # then det inv models
    survey = Survey()
    survey.parse_temfast_data(filename=rawdata_fname, path=rawdata_path)
    survey.parse_inv_results(result_folder=result_path, invrun=invrun)

    snd = survey.soundings[snd_id]
    inv_model = np.asarray(snd.inv_model)


    # %% define models ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ip_modeltype = 'pelton'  # mpa
    ip_modeltype = 'mpa'  # mpa
    tau_val = 5e-4  # 5e-4


    # %% define true model
    thks = inv_model[:, 0]
    rho_0 = inv_model[:, 1]
    phi_max = inv_model[:, 2]
    tau_phi = inv_model[:, 3]
    cs = inv_model[:, 4]

    modelTrue = np.r_[thks[:-1], rho_0, phi_max, tau_phi, cs]
    nLayer_true = len(rho_0)


    fctr = 1.25
    priorTEM = np.column_stack((get_min(thks, fctr), get_max(thks, fctr),
                                get_min(rho_0, fctr), get_max(rho_0, fctr),
                                get_min(phi_max, fctr), get_max(phi_max, fctr),
                                get_min(tau_phi, fctr), get_max(tau_phi, fctr),
                                get_min(cs, fctr), get_max(cs, fctr)))
    priorTEM[0, 4:6] = np.r_[0.01, 0.011]  # replace same prior min and max
    
    priorTEM[priorTEM[:, 5] > 1, 5] = 1  # set phi_max upper boundary to 1 if it is > 1
    priorTEM[priorTEM[:, 9] > 1, 9] = 1  # set c upper boundary to 1 if it is > 1

    prior_elems = priorTEM.shape[0]* priorTEM.shape[1]
    nLayer_prior = priorTEM.shape[0]
    nParams = int(priorTEM.shape[1] / 2)

    prior_min = np.column_stack((priorTEM[:,0], priorTEM[:,2], priorTEM[:,4], priorTEM[:,6], priorTEM[:,8]))
    prior_max = np.column_stack((priorTEM[:,1], priorTEM[:,3], priorTEM[:,5], priorTEM[:,7], priorTEM[:,9]))
    max_depth = prior_max.sum(axis=0)[0]

    prior_mean = np.mean(np.array([prior_min, prior_max]), axis=0)
    prior_mean1D = mtrxMDL2vec(prior_mean)
    n_all_params = nLayer_prior

    prior_bounds = np.vstack((priorTEM.reshape((nLayer_true*nParams, 2), order='C')[::nParams][:-1,:],
                              priorTEM.reshape((nLayer_true*nParams, 2), order='C')[1::nParams],
                              priorTEM.reshape((nLayer_true*nParams, 2), order='C')[2::nParams],
                              priorTEM.reshape((nLayer_true*nParams, 2), order='C')[3::nParams],
                              priorTEM.reshape((nLayer_true*nParams, 2), order='C')[4::nParams]))

    prior_mea_cum_all, mean2plot = mdl2steps(prior_mean,
                                          extend_bot=25, cum_thk=True)


    # %% define constraints; 0 -> no constraint, 1 -> constrained
    cstr_thks = np.r_[0, 0, 0, 0, 0, 0]
    cstr_rho0 = np.r_[0, 0, 0, 0, 0, 0]
    cstr_phim = np.r_[1, 0, 0, 0, 0, 0]
    cstr_taup = np.r_[1, 0, 0, 0, 0, 0]
    cstr_cs = np.r_[1, 0, 0, 0, 0, 0]

    cstr_1d = np.hstack((cstr_thks[:-1], cstr_rho0, cstr_phim, cstr_taup, cstr_cs))
    cstr_bools = np.hstack((cstr_thks[:-1], cstr_rho0, cstr_phim, cstr_taup, cstr_cs)).astype(bool)
    skip_rows_from_col = np.stack((cstr_thks, cstr_rho0, cstr_phim, cstr_taup, cstr_cs)).astype(bool)
    skip_rows_from_col[0, -1] = True
    cstr_vals = modelTrue[cstr_bools]


    params_per_col = np.r_[5, 6, 5, 5, 5]
    param_names_long = ['thickness (thk)', r'DC-resistivity ($\rho_0$)',
                        'max. phase angle ($\phi_{\mathrm{max}}$)', 'relaxation time ($\\tau_{\phi}$)', 'dispersion coefficient ($c$)']
    n_cols = len(params_per_col)


    # %% plot prior space, true model and collection of det inv results
    # plot prior space first
    fig_pr, _ = plot_prior_space(priorTEM, show_patchcorners=False)
    axes = fig_pr.axes

    idx = 0
    for ax in axes:
        if not modelTrue is None:
            true_mdl = vecMDL2mtrx(modelTrue, nLayer_prior, nParams)
            true_cum, true2plt = mdl2steps(true_mdl, extend_bot=25, cum_thk=True)
            ax.plot(mean2plot[:, idx+1], mean2plot[:, 0],
                        'k.--', label='prior mean', zorder=5)
            ax.plot(true2plt[:,idx+1], true2plt[:,0], 'crimson', ls='-',
                    marker='.')
            lines = [ax.lines[-2], ax.lines[-1]]
            leg_labels = ['prior mean', 'true model', 'prior space']
        else:
            lines = [ax.lines[-1]]
            leg_labels = ['prior mean', 'prior space']

        ax.grid(which='major', color='lightgray', linestyle='-')
        ax.grid(which='minor', color='lightgray', linestyle=':')

        # ax.set_xlabel(f'{shrtname[idx+1]} {Units[idx+1]}')
        if idx == 0:
            ax.set_ylabel('depth (m)')
            # legend control
            lines.append(Line2D([0], [0], linestyle="none",
                                marker="s", alpha=1, ms=10,
                                mfc="lightgrey", mec='lightgrey'))
            ax.legend(lines, leg_labels,
                      loc='best', fancybox=True, framealpha=1,
                      facecolor='white', frameon=True, edgecolor='k')
        idx += 1


    if save_figs:
        fig_pr.savefig(savepath_figs + f'00a-prior_TEM_{version}.png', dpi=dpi,
                    bbox_inches='tight')

    # frameinfo = getframeinfo(currentframe())
    # sys.exit(f'for dev ... - exiting in line: {frameinfo.lineno + 1}')


    # %% loop preparations
    if len(os.listdir(savepath_results)) == 0:
        question = f'\n    Folder "{savepath_results}" is empty, continue with modeling?'
        default_answer = 'yes'
        message = 'starting the model sampling and forward calculations...'
    else:
        question = f'\n    Folder "{savepath_results}" contains already results, overwrite?'
        default_answer = 'no'
        message = 'reusing existing results...'
    print(message)


    if query_yes_no(question=question,
                    default=default_answer):

        cstr_sampled_models = []

        # %% prepare forward
        setup_device = {"timekey": timekey,
                        "txloop": loop_size,
                        "rxloop": loop_size,
                        "currentkey": 4,
                        "current_inj": 4.1,
                        "filter_powerline": 50,
                        "ramp_data": 'donauinsel'}

        device = 'TEMfast'
        setup_empymod = {'ft': 'dlf',                     # type of fourier trafo
                         'ftarg': 'key_201_CosSin_2012',  # ft-argument; filter type # https://empymod.emsig.xyz/en/stable/api/filters.html#module-empymod.filters -- for filter names
                         'verbose': 0,                    # level of verbosity (0-4) - larger, more info
                         'srcpts': 3,                     # Approx. the finite dip. with x points. Number of integration points for bipole source/receiver, default is 1:, srcpts/recpts < 3 : bipole, but calculated as dipole at centre
                         'recpts': 3,                     # Approx. the finite dip. with x points. srcpts/recpts >= 3 : bipole
                         'ht': 'dlf',                     # type of fourier trafo
                         'htarg': 'key_401_2009',         # hankel transform filter type
                         'nquad': 3,                      # Number of Gauss-Legendre points for the integration. Default is 3.
                         'cutoff_f': 1e8,                # cut-off freq of butterworthtype filter - None: No filter applied, WalkTEM 4.5e5
                         'delay_rst': 0}                  # ?? unknown para for walktem - keep at 0 for fasttem

        forward = empymod_frwrd(setup_device=setup_device,
                                setup_solver=setup_empymod,
                                time_range=time_range, device='TEMfast',
                                nlayer=nLayer_prior, nparam=nParams)
        times_rx = forward.times_rx

        response_test = forward.calc_response(model=prior_mean,
                                            ip_modeltype=ip_modeltype,
                                            show_wf=False, return_rhoa=return_rhoa,
                                            resp_trafo=resp_trafo)

        tem_mdlset = MODELSET().TEMIP(prior=priorTEM, ip_modeltype=ip_modeltype,
                                      time_range=time_range, resp_abs=resp_abs,
                                      device_sttngs=setup_device, solver_sttngs=setup_empymod,
                                      solver=empymod_frwrd, return_rhoa=return_rhoa,
                                      resp_trafo=resp_trafo, logTransform=logTransform)


        # %% sample with saltelli, makes actual sense here? any other sampler, better suited?
        model_dims = len(tem_mdlset.paramNames['NamesS'])

        problem = {
            'num_vars': model_dims,
            'names': [f'${name}$' for name in tem_mdlset.paramNames['NamesS']],
            'bounds': prior_bounds  # boundaries for each model variable
        }

        param_names = problem['names']
        param_vals = saltelli.sample(problem, N_base2)
        n_mdls_pre = param_vals.shape[0]

        if any(cstr_1d == 1):
            constrain_mdl_params = True
            print('any constrain switch==1 --> using constraints')

            # set to initial model values
            param_vals[:, cstr_bools] = cstr_vals
            # param_names = param_names[cstr_1d]
            param_names = [name for i, name in enumerate(param_names) if not cstr_bools[i]]

            # remove duplicates
            param_vals = np.unique(param_vals, axis=0)
            n_mdls_pos = param_vals.shape[0]
            print(f'removed {n_mdls_pre - n_mdls_pos} duplicates from {n_mdls_pre} sampled models')

            responses = evaluate_models_TEM(models=param_vals, tem_mdlset=tem_mdlset,
                                            Parallelization=[parallel, pool])
            if responses.dtype != 'float64':
                responses = responses.astype('float64')

        else:
            responses = evaluate_models_TEM(models=param_vals, tem_mdlset=tem_mdlset,
                                            Parallelization=[parallel, pool])
            if responses.dtype != 'float64':
                responses = responses.astype('float64')


        # %% save models, times and responses
        header_rsp = [f'tg_{i:02d}' for i in range(len(times_rx))]
        header_mdl = tem_mdlset.paramNames['NamesPlain']

        extend = f'_tk-{timekey}_snd-{snd_id}.csv'
        np.savetxt(fname=savepath_results + f'models{extend}',
                X=param_vals, delimiter=',', header=','.join(header_mdl),
                fmt='%.5e', comments='')
        np.savetxt(fname=savepath_results + f'models-cstr{extend}',
                X=param_vals[:, ~cstr_bools], delimiter=',', header=','.join(header_mdl),
                fmt='%.5e', comments='')
        np.savetxt(fname=savepath_results + f'responses{extend}',
                X=responses, delimiter=',', header=','.join(header_rsp),
                fmt='%.5e', comments='')
        np.savetxt(fname=savepath_results + f'times{extend}',
                   X=times_rx, delimiter=',', fmt='%.5e', comments='')
        np.savetxt(fname=savepath_results + f'cstr_bool{extend}',
                   X=cstr_bools, delimiter=',', comments='')
        np.savetxt(fname=savepath_results + f'priorTEM{extend}',
                   X=priorTEM, delimiter=',', fmt='%.5e', comments='')

        with open(savepath_results + 'param_names.txt', 'w') as filehandle:
            json.dump(param_names, filehandle)

    # frameinfo = getframeinfo(currentframe())
    # sys.exit(f'for dev ... - exiting in line: {frameinfo.lineno + 1}')




    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# %% plot sampled models as transparent lines on top of prior space
# (to check if entire space was sampled sufficiently)
if show_plots:
    # %% reread results of modeling (models and data)
    respos_s = []
    tem_times = []
    sampled_models = []
    sampled_models_cstr = []
    cstr_bool_s = []

    with open(savepath_results + 'param_names.txt', 'r') as filehandle:
        param_names = json.load(filehandle)

    for idx, snd_id in enumerate(snd_ids):
        extend = f'_tk-{timekey}_snd-{snd_id}.csv'
        param_vals = np.loadtxt(fname=savepath_results + f'models{extend}',
                                skiprows=1, delimiter=',')
        param_vals_cstr = np.loadtxt(fname=savepath_results + f'models-cstr{extend}',
                                skiprows=1, delimiter=',')
        responses = np.loadtxt(fname=savepath_results + f'responses{extend}',
                               skiprows=1, delimiter=',')
        times_rx = np.loadtxt(fname=savepath_results + f'times{extend}',
                              delimiter=',')
        cstr_bool = np.loadtxt(fname=savepath_results + f'cstr_bool{extend}',
                              delimiter=',')
        priorTEM = np.loadtxt(fname=savepath_results + f'priorTEM{extend}',
                              delimiter=',')

        sampled_models.append(param_vals)
        sampled_models_cstr.append(param_vals_cstr)
        tem_times.append(times_rx)
        respos_s.append(responses)
        cstr_bool_s.append(cstr_bool)


    for j, sampled_mdl in enumerate(sampled_models):
        snd_id = snd_ids[j]

        fig_pr_mdls, _ = plot_prior_space(priorTEM, show_patchcorners=False)
        axes_mdls = fig_pr_mdls.axes

        for i, ax in enumerate(axes_mdls):  # plot all sampled models
            for mdl_1d in sampled_mdl:
                mdl = vecMDL2mtrx(mdl_1d, nLayer=nLayer_prior, nParam=nParams)
                mdl_cum_all, mdl2plot = mdl2steps(mdl, extend_bot=25, cum_thk=True)

                ax.plot(mdl2plot[:, i+1], mdl2plot[:, 0], label='sampled mdls',
                        color='deeppink', alpha=0.5, ls='-', zorder=5)

            ax.plot(mean2plot[:, i+1], mean2plot[:, 0],
                       'k.--', label='prior mean', zorder=5)
            lines = [ax.lines[-1]]
            leg_labels = ['prior mean', 'prior space', 'sampled mdls']

            ax.grid(which='major', color='lightgray', linestyle='-')
            ax.grid(which='minor', color='lightgray', linestyle=':')

            # ax.set_xlabel(f'{shrtname[i+1]} {Units[i+1]}')
            if i == 0:
                ax.set_ylabel('depth (m)')
                ax.set_xscale('log')

                # legend control
                lines.append(Line2D([0], [0], linestyle="none",
                            marker="s", alpha=1, ms=10,
                            mfc="lightgrey", mec='lightgrey'))
                lines.append(Line2D([0], [0], linestyle="-",
                            marker=None, alpha=0.5, ms=10,
                            color="deeppink"))
                ax.legend(lines, leg_labels,
                          loc='best', fancybox=True, framealpha=1,
                          facecolor='white', frameon=True, edgecolor='k')

        if save_figs:
            fig_pr_mdls.savefig(savepath_figs + f'00a-prior_TEM-{snd_id}-models_{version}.png', dpi=dpi,
                        bbox_inches='tight')


    # %% normalization, distances, clustering and dgsa
    fg_silscrs, ax_silscrs = plt.subplots(1, len(respos_s), figsize=(7*len(respos_s), 5),
                                          sharex=True, sharey=True, constrained_layout=True, squeeze=False)
    sil_scores = np.zeros((len(n_clusters_s), len(respos_s)))

    for clidx, n_clusters in enumerate(n_clusters_s):
        bins_custom = np.linspace(-0.5, n_clusters-0.5, n_clusters+1)
        cluster_colors = cmap(np.linspace(0.05, 0.95, n_clusters))

        fg_hist, ax_hist = plt.subplots(1, len(respos_s), figsize=(7*len(respos_s), 5),
                                        sharex=True, constrained_layout=True, squeeze=False)

        fg_rsp, ax_rsp = plt.subplots(len(respos_s), n_clusters, figsize=(3*n_clusters, 3.8*len(respos_s)),
                                      sharex=True, squeeze=False, sharey=True, constrained_layout=True)

        if log_data or abs_data or norm_to1 or min_to_1:
            fg_rsp1, ax_rsp1 = plt.subplots(len(respos_s), n_clusters, figsize=(3*n_clusters, 3.8*len(respos_s)),
                                          sharex=True, squeeze=False, sharey=True, constrained_layout=True)
        else:
            pass


        sens_s = []
        respos_preproc = []

        for k, resp_raw in enumerate(respos_s):
            if abs_data and not norm_to1 and not min_to_1:
                respos = np.abs(resp_raw)

            elif norm_to1 and not abs_data and not min_to_1:
                respos = resp_raw / np.nanmax(resp_raw)
            elif abs_data and norm_to1 and not min_to_1:
                respos = np.abs(resp_raw)
                respos = respos / np.nanmax(respos)
            elif abs_data and log_data and norm_to1 and not min_to_1:
                respos = np.abs(resp_raw)
                respos = np.log10(respos)
                respos = respos / np.nanmax(respos)

            elif min_to_1 and not log_data and not norm_to1:
                respos = resp_raw + np.nanmin(resp_raw) + 1
            elif min_to_1 and log_data and not norm_to1:
                respos = resp_raw + np.nanmin(resp_raw) + 1
                respos = np.log10(respos)
            elif min_to_1 and log_data and norm_to1:
                respos = resp_raw + np.nanmin(resp_raw) + 1
                respos = np.log10(respos)
                respos = respos / np.nanmax(respos)
            elif min_to_1 and not log_data and norm_to1:
                respos = resp_raw + np.nanmin(resp_raw) + 1
                respos = respos / np.nanmax(respos)

            if abs_data or norm_to1 or min_to_1:
                respos_preproc.append(responses[k])
            else:
                print('no preprocessing done, raw responses used for classification\n')
                respos = resp_raw

            # Now, calculate the euclidean distances between model responses
            distances1 = pdist(respos, metric='euclidean')
            distances = squareform(distances1)
            n_models = len(responses)

            # Cluster the responses using KMedoids
            if cluster_method == 'kmedoids':
                clusterer_kmed = KMedoids(n_clusters=n_clusters, max_iter=3000, tol=1e-4)
                labels = clusterer_kmed.fit_predict(distances)
            elif cluster_method == 'kmeans':
                clusterer_kmean = KMeans(n_clusters=n_clusters, max_iter=3000, tol=1e-4)
                labels = clusterer_kmean.fit_predict(distances)
            else:
                raise ValueError('cluster_method not available')
            label_colors = cluster_colors[labels]
            sil_scr_avg = silhouette_score(respos, labels)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # plot histogram with labels
            n, bins, patches = ax_hist[0, k].hist(labels, bins=bins_custom,
                                                  edgecolor='k', align='mid')
            ax_hist[0, k].set_xlabel('cluster id ()')
            ax_hist[0, k].set_ylabel('count ()')
            ax_hist[0, k].set_title(f'{snd_id}, {n_models} models, n_clusters: {n_clusters}, avr. silhouette score: {sil_scr_avg:.5f}')
            ax_hist[0, k].xaxis.set_major_locator(MaxNLocator(integer=True))
            for i in range(0, n_clusters):
                print(i)
                print(cluster_colors[i])
                patches[i].set_facecolor(cluster_colors[i])
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # plot raw responses colored by cluster id!
            for l, resp in enumerate(resp_raw):
                # ax_rsp[k, labels[l]].loglog(tem_times[k], resp, color=label_colors[l])
                plot_signal(axis=ax_rsp[k, labels[l]], time=tem_times[k], signal=resp,
                            color=label_colors[l], sub0color='k')

            for j in range(0, n_clusters):
                ax_rsp[k, j].xaxis.set_tick_params(labelbottom=True)
                ax_rsp[k, j].yaxis.set_tick_params(labelleft=True)
            # ax_rsp[k, 2].set_title((f'raw responses --- {loop_sizes[k]} m loop, {n_models} models, n_clusters: {n_clusters} ' +
            #                         f'abs-data: {int(abs_data)},  log10 data:{int(log_data)},  norm to 1: {int(norm_to1)},  min to 1: {int(min_to_1)}'))
            fg_rsp.suptitle((f'raw responses --- {snd_id}, {n_models} models, n_clusters: {n_clusters}\n' +
                             f'abs-data: {int(abs_data)},  log10 data:{int(log_data)},  norm to 1: {int(norm_to1)},  min to 1: {int(min_to_1)}'))

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # plot preproc responses colored by cluster id!
            if log_data or abs_data or norm_to1 or min_to_1:
                log10_y = True
                if log_data:
                    log10_y = False
                for l, resp in enumerate(respos):
                    # ax_rsp1[k, labels[l]].semilogx(tem_times[k], resp, color=label_colors[l])
                    plot_signal(axis=ax_rsp1[k, labels[l]], time=tem_times[k], signal=resp,
                                color=label_colors[l], sub0color='k', log10_y=log10_y)

                for j in range(0, n_clusters):
                    ax_rsp1[k, j].xaxis.set_tick_params(labelbottom=True)
                    ax_rsp1[k, j].yaxis.set_tick_params(labelleft=True)
                    if log_data != True:
                        ax_rsp1[k, j].set_yscale('log')

                # ax_rsp1[k, 2].set_title((f'pre-proc responses -- {loop_sizes[k]} m loop, {n_models} models, n_clusters: {n_clusters} ' +
                #                         f'abs-data: {int(abs_data)},  log10 data:{int(log_data)},  norm to 1: {int(norm_to1)},  min to 1: {int(min_to_1)}'))
                fg_rsp1.suptitle((f'pre-proc responses --- {snd_id}, {n_models} models, n_clusters: {n_clusters}\n' +
                                 f'abs-data: {int(abs_data)},  log10 data:{int(log_data)},  norm to 1: {int(norm_to1)},  min to 1: {int(min_to_1)}'))
            else:
                pass

            # plot silhouette scores
            (fig_sil, ax_sil, silhouette_avg,
             sample_silhouette_values) = plot_silhouette_scores(n_clusters, respos,
                                                                labels, cluster_colors=cluster_colors,
                                                                cluster_counts=n)
            ax_sil.set_title(f'Silhouette scores:\n{snd_id}, {n_models} models, n_clusters: {n_clusters}, avr. silhouette score: {sil_scr_avg:.5f}')

            if save_figs:
                fig_sil.savefig(savepath_figs + f'03-silhouette-plot_{snd_id}_nclus{n_clusters}_{version}.png', dpi=dpi,
                            bbox_inches='tight')

            sil_scores[clidx, k] = sil_scr_avg

            # calc sens, remove constrained values from param set
            cstr_bools = cstr_bool_s[k]
            if any(cstr_bools == True):
                param_vals = sampled_models_cstr[k]
            else:
                param_vals = sampled_models[k]

            sens = dgsa(param_vals, labels, parameter_names=param_names, quantile=0.95,
                        confidence=True)

            sens.to_csv(savepath_results + f'sens_{snd_id}_{n_clusters}clustr.csv')

            sens_s.append(sens)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if save_figs:
            fg_hist.savefig(savepath_figs + f'01-clusterHist_nclus{n_clusters}_{version}.png', dpi=dpi,
                        bbox_inches='tight')
            fg_rsp.savefig(savepath_figs + f'02-raw-resps-Cluster_nclus{n_clusters}_{version}.png', dpi=dpi,
                        bbox_inches='tight')

            if abs_data or norm_to1:
                fg_rsp1.savefig(savepath_figs + f'02-preproc-resps-Cluster_nclus{n_clusters}_{version}.png',
                                dpi=dpi, bbox_inches='tight')


        # %% plotting
        n_all_params = prior_bounds.shape[0]
        fig_height = int(max(params_per_col)/2) * 1.25

        figsize = (3.0*len(params_per_col), fig_height)
        fig, axes = plt.subplots(len(sens_s), len(params_per_col), figsize=figsize,  # fig_height+0.25
                                 sharex=True, constrained_layout=True, squeeze=False)

        for jdx, sens_i in enumerate(sens_s):
            snd_id = snd_ids[jdx]
            vert_pareto_plot(sens_i, ax=axes[jdx, :], np_plot='all', fmt=None,
                            colors=None, confidence=True, sort_by_sens=False,
                            n_cols=n_cols, params_per_col=params_per_col,
                            add_empty_to_col=skip_rows_from_col)

            axes[jdx, 0].set_ylabel(f'{snd_id}')
            for kdx, name in enumerate(param_names_long):
                axes[0, kdx].set_title(f'{name}')

                axes[jdx, kdx].grid(which='minor', axis='x', visible=True,
                                color='white', linestyle=':', linewidth=1.5)
                axes[jdx, kdx].grid(which='major', axis='x', visible=True,
                                color='white', linestyle='--', linewidth=1.5)
                axes[1, kdx].set_xlabel('Sensitivity ()')

                if sens_logscale_x == True:
                    print('setting xaxis to log_10 scale...')
                    axes[jdx, kdx].set_xscale('log')
    
                    axes[jdx, kdx].xaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
                    axes[jdx, kdx].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
                    axes[jdx, kdx].xaxis.set_minor_locator(plt.MaxNLocator(n_ticks))

        info = ', '.join(typ.split('-')[-2:]).replace('lay', '-layer')
        fig.suptitle(f'{site}: {info}, cluster: {n_clusters}', fontsize=16)

        all_axs = fig.get_axes()
        tags = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)']
        for idx, tag in enumerate(tags):
            at = AnchoredText(tag,
                              prop={'color': 'k', 'fontsize': 12}, frameon=True,
                              loc='upper right')
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            all_axs[idx].add_artist(at)

        if save_figs:
            fig.savefig(savepath_figs + f'04-dgsa_nclus{n_clusters}_{site}_{version}.png',
                        dpi=dpi, bbox_inches='tight')
            plt.close('all')


    # %%
    for j in range(0, len(snd_ids)):
        ax_silscrs[0, j].plot(n_clusters_s, sil_scores[:, j], '--xk')
        ax_silscrs[0, j].set_xlabel('number of clusters ()')
        ax_silscrs[0, j].set_title(f'{snd_ids[j]}')
    ax_silscrs[0, 0].set_ylabel('average silhouette score ()')

    if save_figs:
        fg_silscrs.savefig(savepath_figs + f'05-sil-scores_{site}_{version}.png', dpi=dpi,
                    bbox_inches='tight')
