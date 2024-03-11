#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  7 18:01:20 2022

script to run pyDGSA for sensitivity analysis of TEM soda lake model (5-layer)
https://pypi.org/project/pyDGSA/

examples at:
https://github.com/zperzan/pyDGSA/blob/master/tutorial_detailed.ipynb
https://github.com/zperzan/pyDGSA/blob/master/tutorial_short.ipynb


new plotting approach, separate subplots for each cluster

Testing:
    [done] log10 data (only possible for non-IP cases) v0x
    [done] norm to 1                                       v1x
    [] 2nd dev; propably not necessary!!
    [done] different prior, which follows the model        v2x
    [done] plot silhouette scores as curve at the end!!
    [done] auto prior range (narrower than 36!!)
    [done] test 2 - 10 clusters, smaller number of models
    [done] log10 scaling and normalization to 1
    [done] update param names, remove BEL1D from scripts


Further ideas:
    dynamic time warping, see https://towardsdatascience.com/how-to-apply-k-means-clustering-to-time-series-data-28d04a8f7da3
    normalization: https://stackoverflow.com/questions/17046397/scikits-learn-clusterization-methods-for-curve-fitting-parameters
    clustering algos: https://scikit-learn.org/dev/modules/clustering.html#overview-of-clustering-methods


v00, inv_nd74:
    [done] inv version in scriptname, corresponds to the used inv result!
    [done] multicolumn plot
        [] add depth axis to plot
    [] clustering automatically, save and reload!
    [] save prior range


@author: laigner
"""

# %% import modules
import os
import sys

rel_path_to_libs = '../../../'
if not rel_path_to_libs in sys.path:
    sys.path.append('../../../')  # add relative path to folder that contains all custom mudules

import matplotlib
import json

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans  # alternative to k-medoids clustering
from sklearn.metrics import silhouette_score
from inspect import currentframe, getframeinfo

from pyDGSA.cluster import KMedoids
from pyDGSA.dgsa import dgsa

from SALib.sample import saltelli

from pathos import multiprocessing as mp
from pathos import pools as pp

from library.TEM_frwrd.empymod_frwrd import empymod_frwrd

from library.utils.TEM_bel1d_tools import mtrxMDL2vec
from library.utils.TEM_bel1d_tools import vecMDL2mtrx
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

from library.tem_tools.survey import Sounding


# %% plot appearance
dpi = 200
plt.style.use('ggplot')

fs_shift = -4
plt.rcParams['axes.labelsize'] = 18 + fs_shift
plt.rcParams['axes.titlesize'] = 18 + fs_shift
plt.rcParams['xtick.labelsize'] = 16 + fs_shift
plt.rcParams['ytick.labelsize'] = 16 + fs_shift
plt.rcParams['legend.fontsize'] = 18 + fs_shift


# %% directories
scriptname = os.path.basename(sys.argv[0])
print(f'running {scriptname} ...')
version = scriptname.split('.')[0].split('_')[-1]
inv_version = scriptname.split('_')[-2].split('-')[-1]
typ = scriptname.split('_')[0]
site = scriptname.split('_')[-3]

main = './'
savepath_figs = main + f'plots_dgsa/{typ}/{version}/'
if not os.path.exists(savepath_figs):
    os.makedirs(savepath_figs)

savepath_results = main + f'results_dgsa/{typ}/{version}/'
if not os.path.exists(savepath_results):
    os.makedirs(savepath_results)


# %% boolean switches
show_plots = True
# show_plots = False

save_figs = True
# save_figs = False

sens_logscale_x = True
# sens_logscale_x = False
n_ticks = 3

# units and scaling
unit = 'res (ohmm)'
# unit = 'con (mS/m)'

# parallel = False
parallel = True
# pool = None
pool = pp.ProcessPool(mp.cpu_count())  # pp.ProcessPool(mp.cpu_count())



# %% data normalization
# log_data = True
log_data = True
# log_trafo_model = [False, False, False,
#                    True, True, True, True]  # list with booleans for each model para, here only resistivity is transformed, not the thickness
log_model = False  # no log trafo for model
logTransform = [log_model, log_data]

# min_to_1 = True
min_to_1 = False

# norm_to1 = True
norm_to1 = True

# abs_data = True
abs_data = False


# %% main settings, clusters, method, loops, etc
loop_sizes = [12.5, 50.0]
# loop_sizes = [12.5]
timekeys =  [3, 5]

N_base2 = 256
# N_base2 = 2

# n_clusters_s = np.arange(2, 10, 1)
# n_clusters_s = [3]
n_clusters_s = [2, 3, 4, 5]

cluster_method = 'kmedoids'  # kmedoids, kmeans
# cluster_method = 'kmeans'  # kmedoids, kmeans
cmap = matplotlib.cm.get_cmap('viridis')  # hsv, spectral, jet


# %% define models ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# DEFINE model that will be investigated ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
thk = np.r_[4, 10, 15, 25, 0]
res = np.r_[30, 120, 35, 100, 55]

modelTrue = np.r_[thk[:-1], res]
nLayer_true = int(len(modelTrue)/2)+1


# read model from result:
invrun = '000'

path_main = f'../../01_frwrd-figures/01_noIP_sodalakes/'
path_invresults = path_main + f'03-inv_results/nd-loopcomp-12-50/{inv_version}/'


snd_12 = Sounding()
snd_12.add_inv_result(folder=path_invresults + 'snd-12/', invrun=invrun, snd_id='snd-12')

snd_50 = Sounding()
snd_50.add_inv_result(folder=path_invresults + 'snd-50/', invrun=invrun, snd_id='snd-50')

snds = [snd_12, snd_50]
fctr = 1.25  # to get prior space centered on mean of true or given model

params_per_col=np.r_[7, 8]  # thk, rho, all mpa params
param_names_long = ['thickness (thk)', r'DC-resistivity ($\rho$)']

n_cols = len(params_per_col)


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
    for idx, loop in enumerate(loop_sizes):
        snd = snds[idx]

        result_array = snd.inv_model
        thk = result_array[:, 0]
        res = result_array[:, 1]

        priorTEM = np.column_stack((get_min(thk, fctr), get_max(thk, fctr),
                                    get_min(res, fctr), get_max(res, fctr)))

        nLayer_prior = priorTEM.shape[0]
        nParams = int(priorTEM.shape[1] / 2)

        prior_min = np.column_stack((priorTEM[:,0], priorTEM[:,2]))
        prior_max = np.column_stack((priorTEM[:,1], priorTEM[:,3]))
        max_depth = prior_max.sum(axis=0)[0]

        prior_mean = np.mean(np.array([prior_min, prior_max]), axis=0)
        prior_mean1D = mtrxMDL2vec(prior_mean)

        prior_bounds = np.vstack((priorTEM.reshape((nLayer_prior*2, 2), order='C')[::2][:-1,:],
                                  priorTEM.reshape((nLayer_prior*2, 2), order='C')[1::2]))

        prior_mea_cum_all, mean2plot = mdl2steps(prior_mean,
                                              extend_bot=25, cum_thk=True)

        # frameinfo = getframeinfo(currentframe())
        # sys.exit(f'for dev ... - exiting in line: {frameinfo.lineno + 1}')


        # %% prepare forward
        setup_device = {"timekey": timekeys[idx],
                        "txloop": loop,
                        "rxloop": loop,
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
                                time_range=None, device='TEMfast',
                                nlayer=nLayer_prior, nparam=nParams)
        times_rx = forward.times_rx

        response_test = forward.calc_response(model=prior_mean,
                                              show_wf=False)

        tem_mdlset = MODELSET().TEM(prior=priorTEM, time_range=None,
                                    device_sttngs=setup_device, solver_sttngs=setup_empymod,
                                    solver=empymod_frwrd, logTransform=logTransform,
                                    return_rhoa=False, resp_trafo=None)


        # %% sample with saltelli sampler and model the data
        model_dims = len(tem_mdlset.paramNames['NamesS'])

        problem = {
            'num_vars': model_dims,
            'names': [f'${name}$' for name in tem_mdlset.paramNames['NamesS']],
            'bounds': prior_bounds  # boundaries for each model variable
        }

        param_names = problem['names']
        param_vals = saltelli.sample(problem, N_base2)

        responses = evaluate_models_TEM(models=param_vals, tem_mdlset=tem_mdlset,
                                        Parallelization=[parallel, pool])

        if responses.dtype != 'float64':
            responses = responses.astype('float64')

        # %% save models, times and responses
        header_rsp = [f'tg_{i:02d}' for i in range(len(times_rx))]
        header_mdl = tem_mdlset.paramNames['NamesPlain']

        extend = f'_tk-{timekeys[idx]}_loop-{int(loop)}.csv'
        np.savetxt(fname=savepath_results + f'models{extend}',
                   X=param_vals, delimiter=',', header=','.join(header_mdl),
                   fmt='%.5e', comments='')
        np.savetxt(fname=savepath_results + f'responses{extend}',
                   X=responses, delimiter=',', header=','.join(header_rsp),
                   fmt='%.5e', comments='')
        np.savetxt(fname=savepath_results + f'times{extend}',
                   X=times_rx, delimiter=',', fmt='%.5e', comments='')
        np.savetxt(fname=savepath_results + f'prior{extend}',
                   X=priorTEM, delimiter=',', fmt='%.5e', comments='')

    with open(savepath_results + 'param_names.txt', 'w') as filehandle:
        json.dump(param_names, filehandle)




# %% ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ plotting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if show_plots:
    # %% reread results of modeling (models and data)
    respos_s = []
    tem_times = []
    sampled_models = []
    priors = []

    with open(savepath_results + 'param_names.txt', 'r') as filehandle:
        param_names = json.load(filehandle)

    for idx, loop in enumerate(loop_sizes):
        extend = f'_tk-{timekeys[idx]}_loop-{int(loop)}.csv'
        param_vals = np.loadtxt(fname=savepath_results + f'models{extend}',
                                skiprows=1, delimiter=',')
        responses = np.loadtxt(fname=savepath_results + f'responses{extend}',
                               skiprows=1, delimiter=',')
        times_rx = np.loadtxt(fname=savepath_results + f'times{extend}',
                              delimiter=',')
        priorTEM = np.loadtxt(fname=savepath_results + f'prior{extend}',
                              delimiter=',')

        sampled_models.append(param_vals)
        tem_times.append(times_rx)
        respos_s.append(responses)
        priors.append(priorTEM)


    # %% plot sampled models as transparent lines on top of prior space
    # (to check if entire space was sampled sufficiently)

    for j, sampled_mdl in enumerate(sampled_models):
        loop_size = loop_sizes[j]
        priorTEM = priors[j]
        nLayer_prior = priorTEM.shape[0]
        nParams = int(priorTEM.shape[1] / 2)

        prior_min = np.column_stack((priorTEM[:,0], priorTEM[:,2]))
        prior_max = np.column_stack((priorTEM[:,1], priorTEM[:,3]))
        max_depth = prior_max.sum(axis=0)[0]

        prior_mean = np.mean(np.array([prior_min, prior_max]), axis=0)
        prior_mean1D = mtrxMDL2vec(prior_mean)

        prior_bounds = np.vstack((priorTEM.reshape((nLayer_prior*2, 2), order='C')[::2][:-1,:],
                                  priorTEM.reshape((nLayer_prior*2, 2), order='C')[1::2]))
        prior_mea_cum_all, mean2plot = mdl2steps(prior_mean,
                                              extend_bot=25, cum_thk=True)

        fig_pr_mdls, _ = plot_prior_space(priorTEM, show_patchcorners=False)
        axes_mdls = fig_pr_mdls.axes

        idx = 0
        for ax in axes_mdls:  # plot all sampled models
            for mdl_1d in sampled_mdl:
                mdl = vecMDL2mtrx(mdl_1d, nLayer=nLayer_prior, nParam=nParams)
                mdl_cum_all, mdl2plot = mdl2steps(mdl, extend_bot=25, cum_thk=True)

                ax.plot(mdl2plot[:, idx+1], mdl2plot[:, 0], label='sampled mdls',
                        color='deeppink', alpha=0.5, ls='-', zorder=5)

            ax.plot(mean2plot[:, idx+1], mean2plot[:, 0],
                        'k.--', label='prior mean', zorder=5)
            lines = [ax.lines[-1]]
            leg_labels = ['prior mean', 'prior space', 'sampled mdls']

            ax.grid(which='major', color='lightgray', linestyle='-')
            ax.grid(which='minor', color='lightgray', linestyle=':')

            # ax.set_xlabel(f'{shrtname[idx+1]} {Units[idx+1]}')
            if idx == 0:
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
            idx += 1

        if save_figs:
            fig_pr_mdls.savefig(savepath_figs + f'00a-prior_TEM-{loop_size}-models_{version}.png', dpi=dpi,
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
                respos = resp_raw / np.max(resp_raw)
            elif abs_data and norm_to1 and not min_to_1:
                respos = np.abs(resp_raw)
                respos = respos / np.max(respos)
            elif abs_data and log_data and norm_to1 and not min_to_1:
                respos = np.abs(resp_raw)
                respos = np.log10(respos)
                respos = respos / np.max(respos)

            elif min_to_1 and not log_data and not norm_to1:
                respos = resp_raw + np.min(resp_raw) + 1
            elif min_to_1 and log_data and not norm_to1:
                respos = resp_raw + np.min(resp_raw) + 1
                respos = np.log10(respos)
            elif min_to_1 and log_data and norm_to1:
                respos = resp_raw + np.min(resp_raw) + 1
                respos = np.log10(respos)
                respos = respos / np.max(respos)
            elif min_to_1 and not log_data and norm_to1:
                respos = resp_raw + np.min(resp_raw) + 1
                respos = respos / np.max(respos)

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
            ax_hist[0, k].set_title(f'{loop_sizes[k]} m loop, {n_models} models, n_clusters: {n_clusters}, avr. silhouette score: {sil_scr_avg:.5f}')
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
            fg_rsp.suptitle((f'raw responses --- {loop_sizes[k]} m loop, {n_models} models, n_clusters: {n_clusters}\n' +
                              f'abs-data: {int(abs_data)},  log10 data:{int(log_data)},  norm to 1: {int(norm_to1)},  min to 1: {int(min_to_1)}'))

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # plot preproc responses colored by cluster id!
            if log_data or abs_data or norm_to1 or min_to_1:
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
                fg_rsp1.suptitle((f'pre-proc responses --- {loop_sizes[k]} m loop, {n_models} models, n_clusters: {n_clusters}\n' +
                                  f'abs-data: {int(abs_data)},  log10 data:{int(log_data)},  norm to 1: {int(norm_to1)},  min to 1: {int(min_to_1)}'))
            else:
                pass

            # plot silhouette scores
            (fig_sil, ax_sil, silhouette_avg,
              sample_silhouette_values) = plot_silhouette_scores(n_clusters, respos,
                                                                labels, cluster_colors=cluster_colors,
                                                                cluster_counts=n)
            ax_sil.set_title(f'Silhouette scores:\n{loop_sizes[k]} m loop, {n_models} models, n_clusters: {n_clusters}, avr. silhouette score: {sil_scr_avg:.5f}')

            if save_figs:
                fig_sil.savefig(savepath_figs + f'03-silhouette-plot_loop{int(loop_sizes[k])}_nclus{n_clusters}_{version}.png', dpi=dpi,
                            bbox_inches='tight')

            sil_scores[clidx, k] = sil_scr_avg

            # calc sens
            sens = dgsa(param_vals, labels, parameter_names=param_names, quantile=0.95,
                        confidence=True)
            sens.to_csv(savepath_results + f'sens_{int(loop_sizes[k])}mloop_{n_clusters}clustr.csv')

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


        # %% plotting ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        fig_height = int(max(params_per_col)/2)
        fig_width = 3.5*len(params_per_col)
        # figsize = (fig_width, fig_height)

        figsize = (3.0*len(params_per_col), 8)
        fig, axes = plt.subplots(len(sens_s), len(params_per_col), figsize=figsize,  # fig_height+0.25
                                 sharex=True, constrained_layout=True, squeeze=False)

        for jdx, sens_i in enumerate(sens_s):
            vert_pareto_plot(sens_i, ax=axes[jdx, :], np_plot='all', fmt=None,
                            colors=None, confidence=True, sort_by_sens=False,
                            n_cols=n_cols, params_per_col=params_per_col,
                            add_empty_to_thk=True)
            axes[jdx, 0].set_ylabel(f'{loop_sizes[jdx]} m loop')

            for kdx, name in enumerate(param_names_long):

                if jdx == 0:
                    axes[jdx, kdx].set_title(f'{name}')

                axes[jdx, kdx].grid(which='minor', axis='x', visible=True,
                                color='white', linestyle=':', linewidth=1.5)
                axes[jdx, kdx].grid(which='major', axis='x', visible=True,
                                color='white', linestyle='--', linewidth=1.5)

                if jdx == 1:
                    axes[jdx, kdx].set_xlabel('Sensitivity ()')

            if sens_logscale_x == True:
                print('setting xaxis to log_10 scale...')
                axes[jdx, kdx].set_xscale('log')

                axes[jdx, kdx].xaxis.set_minor_formatter(FormatStrFormatter("%.0f"))
                axes[jdx, kdx].xaxis.set_major_formatter(FormatStrFormatter("%.0f"))

                axes[jdx, kdx].xaxis.set_minor_locator(plt.MaxNLocator(n_ticks))

        # axes[0, 0].set_ylabel('12.5 m loop')
        # axes[1, 0].set_ylabel('50.0 m loop')
        # info = ', '.join(typ.split('-')[-2:]).replace('lay', '-layer')
        # fig.suptitle(f'{site}: {info}, cluster: {n_clusters}', fontsize=16)
        # info = ', '.join(typ.split('-')[-2:]).replace('lay', '-layer')
        fig.suptitle(f'cluster: {n_clusters}', fontsize=16)

        if save_figs:
            # plt.tight_layout()
            fig.savefig(savepath_figs + f'04-dgsa_nclus{n_clusters}_{site}_{version}.png', dpi=dpi,
                        bbox_inches='tight')
            plt.close('all')

    # %%
    for j in range(0, len(loop_sizes)):
        ax_silscrs[0, j].plot(n_clusters_s, sil_scores[:, j], '--xk')
        ax_silscrs[0, j].set_xlabel('number of clusters ()')
        ax_silscrs[0, j].set_title(f'{loop_sizes[jdx]} m loop')
    ax_silscrs[0, 0].set_ylabel('average silhouette score ()')

    if save_figs:
        fg_silscrs.savefig(savepath_figs + f'05-sil-scores_sodalakes_{version}.png', dpi=dpi,
                    bbox_inches='tight')
