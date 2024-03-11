#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 11:03:32 2022

function library for sensitivity analysis of TEM method

@author: laigner
"""

# %% modules
import sys

# abs_path_to_libs = '/shares/laigner/home/nextcloud/09-python_coding/'
# if not abs_path_to_libs in sys.path:
#     sys.path.append(abs_path_to_libs)  # add realtive path to folder that contains all custom modules

# rel_path_to_libs = '../../'
# if not rel_path_to_libs in sys.path:
#     sys.path.append('../../')  # add realtive path to folder that contains all custom modules

import logging
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm


from scipy import stats                     # For the statistical distributions


from sklearn.metrics import silhouette_samples, silhouette_score

from .timer import Timer



# %% logging
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.WARNING)
# logger.setLevel(logging.DEBUG)


# %% function_lib
class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    # https://stackoverflow.com/questions/42656139/set-scientific-notation-with-fixed-exponent-and-significant-digits-for-multiple
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r'$\mathdefault{%s}$' % self.format


def prepare_gaussian_kde(x, y):
    """
    estimate the probability density function of a 2D random variable
    based upon: https://stackoverflow.com/questions/36957149/density-map-heatmaps-in-matplotlib

    Parameters
    ----------
    x : array like
        x component of 2D random variable.
    y : array like
        y component of 2D random variable.

    Returns
    -------
    xi, yi, zi.

    """
    from scipy.stats import gaussian_kde

    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    return xi, yi, zi


def evaluate_models_TEM(models, tem_mdlset,
                        Parallelization=[False, None]):
    """
    function to forward model the TEM response for the provided models

    Parameters
    ----------
    models : np.ndarray
        set of models.
    tem_mdlset : MODELSET class
        from BEL1D module - see: .
    Parallelization : list, optional
        contains a boolean to enable parallelization (first list entry).
        The second entry in the list is a pp.ProcessPool(n_cores_used) object,
        where n_cores_used contains the number of cores.
        For pp see pathos module: from pathos import pools as pp
        The default is no parallelization, i.e., [False, None].

    Returns
    -------
    responses : np.ndarray
        TEM model responses.

    """

    from pathos import multiprocessing as mp
    from pathos import pools as pp

    from functools import partial               # For building parallelizable functions
    from tqdm import tqdm                       # For progress bar
    from . import tqdm_pathos

    nbModels = models.shape[0]
    response_test = tem_mdlset.forwardFun["Fun"](models[0, :])
    prebel_timer = Timer()
    prebel_timer.start()

    if Parallelization[0]:
        functionParallel = partial(ForwardParallelFun,
                                   function=tem_mdlset.forwardFun["Fun"],
                                   nbVal=len(response_test))
        inputs = [models[i,:] for i in range(nbModels)]
        if Parallelization[1] is not None:
            pool = Parallelization[1]
        else:
            pool = pp.ProcessPool(mp.cpu_count()) # Create the pool for parallelization
            Parallelization[1] = pool

        # outputs = pool.map(functionParallel, inputs)
        # outputs = tqdm_pathos.map(functionParallel, inputs, pool=pool)
        outputs = tqdm_pathos.map(functionParallel, inputs, n_cpus=pool.ncpus)
        responses = np.vstack(outputs) #ForwardParallel
        prebel_td, time_prebel = prebel_timer.stop('parallel evaluation-')
    else:
        responses = np.zeros((nbModels, len(response_test)))
        for i in tqdm(range(nbModels)):
            responses[i,:] = tem_mdlset.forwardFun["Fun"](models[i,:])
        prebel_td, time_prebel = prebel_timer.stop('non-parallel evaluation-')

    return responses


def get_min(mean, factor):
    min_val = mean - mean/factor
    return min_val


def get_max(mean, factor):
    max_val = mean + mean/factor
    return max_val


def query_yes_no(question, default='no'):
    """
    yes no query for terminal usage
    from: https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input

    Parameters
    ----------
    question : string
        query to ask the user.
    default : string, optional
        default answer. The default is 'no'.

    Raises
    ------
    ValueError
        if the expected variations of yes/no are not in the answer...

    Returns
    -------
    none

    """
    from distutils.util import strtobool
    if default is None:
        prompt = " (y/n)? "
    elif default == 'yes':
        prompt = " ([y]/n)? "
    elif default == 'no':
        prompt = " (y/[n])? "
    else:
        raise ValueError(f"Unknown setting '{default}' for default.")

    while True:
        try:
            resp = input(question + prompt).strip().lower()
            if default is not None and resp == '':
                return default == 'yes'
            else:
                return strtobool(resp)
        except ValueError:
            print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


# %% modified plotting function:
def vert_pareto_plot(df, ax=None, np_plot='+5', fmt=None, colors=None,
                     confidence=False, sort_by_sens=True, n_cols=1,
                     params_per_col=None, add_empty_to_col=False):
    """Generate a vertical Pareto plot of sensitivity analysis results.
    adapted from : https://github.com/zperzan/pyDGSA/blob/master/pyDGSA/plot.py

    params:
        df [DataFrame]: pandas dataframe containing the sensitivity analysis
                results. If fmt == 'max' or 'mean', df contains a single column
                with indices correspond to the parameters. If fmt == 'cluster_avg',
                the columns of df correspond to the number of clusters and
                the rows correspond to the parameters.
        np_plot [str|int]: number of parameters to plot. Default: '+5'
                -'all': plot all parameters
                -n: plot n parameters, where n is an int
                -'+n': plot all parameters with sensitivity >1, plus
                       the n next most sensitive parameters
        fmt [str]: format of df. Optional, will interpret fmt based on df shape
                -'mean': single sensitivity was passed per param, therefore
                         display a single bar per param
                -'max': same as 'mean'
                -'cluster_avg': cluster-specific sensitivities were passed, so
                            display cluster's sensitivity separately
                -'bin_avg': bin-specific sensitivities were passed, so display
                            each bin's sensitivity separately
                -'indiv': plots the sensitivity for each bin/cluster combination
                        separately
        confidence [bool]: whether or not to plot confidence bars. Default is False,
                but must be included in df if confidence == True.
        colors [list(str|int)]: list of clusters colors to use when plotting,
                either specified as rgba tuples or strings of matplotlib named
                colors. Only used when fmt='cluster_avg' or 'indiv'
        sort_by_sens [bool]: whether or not to sort the sensitivities ascending.
                Default is True.
        n_cols [int]: how many columns should be used for the plot.
                Default is 1.
        params_per_col [None, or list]: how many parameters should be plotted per column.
                Default is None, if n_cols > 1, list length have to equal number of columns
        add_empty_to_col [single bool, or bool matrix]: which positions in column plots 
                should be left empty (intended for layered parameters). 
                The default is False, which will distribute the parameters 
                equally along the axes.

    returns:
        fig: matplotlib figure handle
        ax: matplotlib axis handle
    """

    # Total np (number of parameters)
    np_total = df.shape[0]

    # Figure out fmt if not explicitly provided
    if fmt is None:
        if isinstance(df.columns, pd.MultiIndex):
            fmt = 'indiv'
        elif df.shape[1] > 1 and 'confidence' not in df.columns:
            # Could be either 'cluster_avg' or 'bin_avg'
            if 'Cluster' in df.columns[0]:
                fmt = 'cluster_avg'
            elif 'Bin' in df.columns[0]:
                fmt = 'bin_avg'
            else:
                raise ValueError("Could not determine fmt. Please pass explicitly.")
        else:
            # Note that 'mean' also includes 'max' format from the analysis
            fmt = 'mean'
            if 'confidence' in df.columns:
                cdf = df['confidence'].copy()
                # copy to avoid altering input df
                df = df.drop('confidence', axis=1).copy()

    if fmt == 'indiv' and colors is not None:
        if isinstance(colors[0], str):
            # Convert named colors to rgba
            named_colors = colors
            colors=[]
            for color in named_colors:
                colors.append(matplotlib.colors.to_rgba(color))

    if fmt == 'cluster_avg':
        # Check if confidence bounds were provided by counting
        # columns that end with "_conf"
        conf_cols = [col for col in df.columns if col[-5:] == '_conf']
        if len(conf_cols) > 0:
            cdf = df[conf_cols].copy()
            df = df.drop(conf_cols, axis=1).copy()

    # Get number of parameters with sensitivity >= 1
    np_sensitive = np.any((df.fillna(0).values >= 1), axis=1).sum()

    # Figure out how many parameters to plot
    if isinstance(np_plot, str):
        if np_plot == 'all':
            np_max_plot = np_total
        elif np_plot[0] == '+':
            np_max_plot = np_sensitive + int(np_plot[1:])
        else:
            raise ValueError("np_plot must be 'all', 'n', or '+n', where n is an int")
    elif isinstance(np_plot, int):
        np_max_plot = np_plot

    # Ensure that requested # of params to plot is not larger than total # of params
    if np_max_plot > np_total:
        np_max_plot = np_total

    # Y-position of bars
    y_pos = np.arange(np_max_plot)

    if fmt == 'mean' or fmt == 'max':
        # Sort so most sensitive params are on top
        if sort_by_sens:
            df.sort_values(by=df.columns[0], ascending=False, inplace=True)
        data = df.values[:np_max_plot, :].squeeze()
        params = df.index.tolist() # Get list of params after sorting
        yticks = y_pos

        # Error bars (confidence interval); by default these are plotted, but of
        # length 0 if confidence == False
        if confidence:
            xerr = cdf[df.index[:np_max_plot]].values / 2
        else:
            xerr = 0

        # Values are color-coded. If confidence intervals are provided
        if confidence:
            # colors = np.asarray([[1, 1, 1, 0.8]]*np_max_plot)
            colors = np.asarray([[0, 0, 0, 0.25]]*np_max_plot)  # lightgray - inconclusive sensitivity
            # colors[data - xerr > 1] = [1, 0, 0, 0.8]   # > confidence interval red
            colors[data - xerr > 1] = [0, 0, 0, 0.75]  # > confidence interval dark-gray

            # colors[data + xerr < 1] = [0, 0, 1, 0.8]   # < confidence interval blue
            colors[data + xerr < 1] = [1, 1, 1, 0.75]   # < confidence interval white
        else:
            if np_sensitive > 0:
                # colors = np.asarray([[1, 0, 0, 0.8]]*np_max_plot)  # red
                colors = np.asarray([[0, 0, 0, 0.75]]*np_max_plot)  # dark-gray
            else:
                # colors = np.asarray([[0, 0, 1, 0.8]]*np_max_plot)  # blue
                colors = np.asarray([[0, 0, 0, 0.75]]*np_max_plot)  # dark-gray

            if np_max_plot > np_sensitive:
                if np_sensitive > 0:
                    mask1 = data < 1               # only those white, where sensitivity is below 1
                    colors[mask1] = [1, 1, 1, 0.75]   # white

                    mask2 = (data < 1) & (data > 0.95)
                    colors[mask2] = [0, 0, 0, 0.25]   # lightgray
                # colors[np_sensitive+1:] = [0, 0, 1, 0.8]  # blue

        # Create figure
        if n_cols == 1:
            if ax is None:
                fig_height = int(np_max_plot/2*1.5)
                fig, ax = plt.subplots(figsize=(5, fig_height))
            ax.barh(y_pos, data, color=colors, edgecolor='k', xerr=xerr)

        else:
            if ax is None:
                fig_height = int(np_max_plot/(2*n_cols)*1.5)
                fig, ax = plt.subplots(nrows=1, ncols=n_cols,
                                       figsize=(5, fig_height))

            cum_param_per = np.cumsum(params_per_col)
            bool1d = add_empty_to_col.flatten()

            if np.sum(add_empty_to_col) > 0:
                y_pos = np.arange(np_max_plot + np.sum(add_empty_to_col))
                yticks_subset = y_pos[~bool1d]
                
                y_pos_2d = y_pos.reshape((len(params_per_col), np.max(params_per_col)))
                datx = np.zeros_like(y_pos_2d, dtype=float)
                errx = np.zeros_like(y_pos_2d, dtype=float)
                colors_all = np.zeros((y_pos.shape[0], 4))
                
                datx[~add_empty_to_col] = data
                errx[~add_empty_to_col] = xerr
                colors_all[~bool1d] = colors
                colors_all.reshape((len(params_per_col), np.max(params_per_col), 4))
            else:
                y_pos_2d = y_pos.reshape((len(params_per_col), np.max(params_per_col)))
                datx = data.reshape((len(params_per_col), np.max(params_per_col)))
                errx = xerr.reshape((len(params_per_col), np.max(params_per_col)))
                colors_all = colors.reshape((len(params_per_col), np.max(params_per_col), 4))

            for idx, par_per in enumerate(params_per_col):
                y_loc = y_pos_2d[idx, :]
                dat = datx[idx, :]
                err = errx[idx, :]
                clrs = colors_all[y_loc]

                ax[idx].barh(y_loc, dat, xerr=err,
                             color=clrs, edgecolor='k')

    elif fmt == 'cluster_avg':
        n_clusters = df.shape[1]

        # Sort by mean sensitivity across clusters
        sort_df = df.mean(axis=1).sort_values(ascending=False)
        df = df.reindex(sort_df.index)
        params = df.index.tolist() # Get list of params after sorting

        # Add error bars if confidence=True, otherwise set length to 0
        if confidence:
            xerr = cdf.loc[df.index, :].values
        else:
            xerr = cdf.loc[df.index, :].values*0

        height = 1/(n_clusters+1)
        yticks = y_pos - (height*(n_clusters-1)/2)

        if colors is None:
            colors = []
            cmap = matplotlib.cm.get_cmap('Set1')
            for i in range(n_clusters):
                colors.append(cmap(i))

        # Create figure
        if ax is None:
            fig_height = int(np_max_plot/2*1.5)
            fig, ax = plt.subplots(figsize=(5, fig_height))

        # Add bars for each cluster
        for i in range(n_clusters):
            ax.barh(y_pos - height*i, df.iloc[:np_max_plot, i], height=height,
                    color=colors[i], edgecolor='k', label=df.columns.tolist()[i],
                    xerr=xerr[:np_max_plot, i])
        ax.legend()

    elif fmt == 'bin_avg':
        n_bins = df.shape[1]

        # Sort by mean sensitivity across bins
        sort_df = df.mean(axis=1).sort_values(ascending=False)
        df = df.reindex(sort_df.index)
        params = df.index.tolist()

        yticks = y_pos

        if colors is None:
            cmap = matplotlib.cm.get_cmap('Set1')
            colors = cmap(1)

        # Create color array by decreasing alpha channel for each bin
        color_array = np.tile(colors, (n_bins, 1))
        for i in range(n_bins):
            color_array[i, 3] = (i+1)/(n_bins)

        # Create figure
        if ax is None:
            fig_height = int(np_max_plot/2*1.5)
            fig, ax = plt.subplots(figsize=(5, fig_height))

        for i in range(n_bins):
            width = df.iloc[:np_max_plot, i]
            left = df.iloc[:np_max_plot, :i].sum(axis=1)
            b = ax.barh(y_pos, width=width, left=left, color=color_array[i],
                        edgecolor='k')

            # Increase linewidth for parameters that are sensitive
            for w in enumerate(width.tolist()):
                if w[1] > 1:
                    b[w[0]].set_linewidth(2.5)

    elif fmt == 'indiv':
        n_clusters, n_bins = df.columns.levshape

        # Split df into sensitive and non-sensitive df's, sort each, then re-combine
        # Can't just sort on mean sensitivity, otherwise a sensitive parameter
        # could get left out because its mean might not be within the top
        # np_max_plot most sensitive, even though a single bin is >= 1
        mask = np.any((df.fillna(0).values >= 1), axis=1)
        sens = df[mask].copy()
        nsens = df[~mask].copy()
        sort_sens = sens.mean(axis=1).sort_values(ascending=False)
        sens = sens.reindex(sort_sens.index)
        sort_nsens = nsens.mean(axis=1).sort_values(ascending=False)
        nsens = nsens.reindex(sort_nsens.index)
        df = sens.append(nsens)

        params = df.index.tolist()
        df.fillna(0, inplace=True)
        height = 1/(n_clusters+1)
        yticks = y_pos - (height*(n_clusters-1)/2)

        if colors is None:
            cmap = matplotlib.cm.get_cmap('Set1')
            colors = []
            for i in range(n_clusters):
                colors.append(cmap(i))

        idx = pd.IndexSlice

        # Create color array by decreasing alpha channel for each bin
        color_array = np.zeros((n_clusters, n_bins, 4), dtype='float64')
        for i in range(n_clusters):
            for j in range(n_bins):
                color_array[i, j, :] = colors[i]
                color_array[i, j, 3] = (j+1)/(n_bins)

        # Create figure
        if ax is None:
            fig_height = int(np_max_plot/2*1.5)
            fig, ax = plt.subplots(figsize=(5, fig_height))

        for i in range(n_clusters):
            for j in range(n_bins):
                col_idx = i*n_bins + j
                width = df.iloc[:np_max_plot, col_idx]
                left = df.iloc[:np_max_plot, n_bins*i:col_idx+1].sum(axis=1) - width
                y = y_pos - height*i
                if j == n_bins - 1:
                    # Add label to last bin
                    b = ax.barh(y, width=width, height=height, left=left,
                                color=color_array[i, j], edgecolor='k',
                                label=df.columns.tolist()[i*n_bins][0])
                else:
                    b = ax.barh(y, width=width, height=height, left=left,
                                color=color_array[i, j], edgecolor='k')

                # Increase linewidth for parameters that are sensitive
                for w in enumerate(width.tolist()):
                    if w[1] > 1:
                        b[w[0]].set_linewidth(2.5)

        leg = ax.legend()
        # Ensure that linewidths in the legend are all 1.0 pt
        for legobj in leg.legendHandles:
            legobj.set_linewidth(1)

    # Add vertical line and tick labels
    if n_cols == 1:
        if fmt not in ['indiv', 'bin_avg']:
            ax.axvline(1, color='k', linestyle='--')
        # ax.set(yticks=yticks, yticklabels=params[:np_max_plot], xlabel='Sensitivity ()')
        ax.set(yticks=yticks, yticklabels=params[:np_max_plot])
        ax.invert_yaxis()
    else:
        if fmt not in ['indiv', 'bin_avg']:
            for i in range(0, n_cols):
                ax[i].axvline(1, color='k', linestyle='--')

        cum_param_per = np.cumsum(params_per_col)
        for idx, par_per in enumerate(params_per_col):
            if idx == 0:
                ax[idx].set(yticks=yticks_subset[0:par_per],
                            yticklabels=params[0:par_per])
            else:
                start = cum_param_per[idx-1]
                end = cum_param_per[idx]
                ax[idx].set(yticks=yticks_subset[start:end],
                            yticklabels=params[start:end])
            ax[idx].invert_yaxis()

    if ax is None:
        return fig, ax
    else:
        return ax


# %% silhouette plot:
def plot_silhouette_scores(n_clusters, X, cluster_labels, cluster_colors,
                           cluster_counts):
    """
    function to plot the silhouette scores for different clusters
    example from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py

    Parameters
    ----------
    n_clusters : int
        number of clusters.
    X : np.ndarray
        clustered data samples/response (for TEM those are the data curves).
    cluster_labels : np.ndarray
        labels for each response (sample) to mark to which cluster it belongs.
    cluster_colors : list
        of cluster colors, e.g. from a color map:
            cmap = matplotlib.cm.get_cmap('viridis')  # hsv, spectral, jet.
        using: cluster_colors = cmap(np.linspace(0.05, 0.95, n_clusters))
    cluster_counts : list
        contains the number of samples in each cluster.

    Returns
    -------
    fig : mpl Figure object
        contains the axes.
    ax1 : mpl axes object
        contains the actual plots.
    silhouette_avg : float
        average silhouette score.
    sample_silhouette_values : np.ndarray
        silhouette score for each sample (response) for all clusters.

    """

    # Create a subplot with 1 row and 2 columns
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(15, 8)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        # color = cm.nipy_spectral(float(i) / n_clusters)
        # color = cm.viridis(float(i) / n_clusters)
        color = cluster_colors[i]
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i,
                 str(i) + f'   ---   mdl-count: {int(cluster_counts[i])}', fontsize=12)

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    # ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("silhouette coefficient values ()")
    ax1.set_ylabel("Cluster label ()")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    return fig, ax1, silhouette_avg, sample_silhouette_values



# %% prior space visualization:
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


def mdl2steps(mdl, extend_bot=30, cum_thk=True):
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
    (fig), ax
        (figure if ax is None) and axis object.

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


def plot_prior_space(prior, show_patchcorners=False):
    """
    function to visualize the prior model space
    updated version to plot multiple parameters to subplots
    could be also used to visualize the solved means and stds

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

    fig_pr, axes = plt.subplots(nrows=1, ncols=ncols,
                                figsize=(3.4*ncols, 8), squeeze=False)
    axes = axes.flat

    xcols = np.arange(1,ncols+1)
    xcolid = 1
    for idx, ax_pr in enumerate(axes):
        k = 0
        logger.info('\n\n--------------------------')
        logger.info('preparing patch creation:\n')

        prior_min_cum = prior_min_cum_all[:,(0,xcolid)]
        prior_max_cum = prior_max_cum_all[:,(0,xcolid)]
        logger.info('prior_min_cum', prior_min_cum)
        logger.info('prior_max_cum', prior_max_cum)
        logger.info('--------------------------\n')


        patches_min = []
        for prior_cum in [prior_min_cum, prior_max_cum]:
            logger.debug(prior_cum)
            for i, thk in enumerate(prior_cum[:,0]):
                logger.debug('layerID', i)
                logger.debug('curr_thickness', thk)
                r_rli_min = prior_min_cum[i,1]
                r_rli_max = prior_max_cum[i,1]
                logger.debug('minmax_rho', r_rli_min, r_rli_max)

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
                logger.debug(corners_xy)
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

        logger.debug('\n\n--------------------------')
        logger.debug('plotting mean model:\n')
        logger.debug(mean2plot)
        logger.debug(xcolid)
        logger.debug(mean2plot[:,xcolid])
        ax_pr.invert_yaxis()
        ax_pr.grid(which='major', color='white', linestyle='-', zorder=10)

        if idx > 0:
            ax_pr.tick_params(axis='y', which='both', labelleft=False)

        xcolid += 1

    return fig_pr, axes


# %% modelsset class
class MODELSET:
    '''
    based on the MODELSET class idea from BEL1D: https://github.com/hadrienmichel/pyBEL1D
    
    MODELSET is an object class that can be initialized using:
        - the dedicated class methods (DC and SNMR) - see dedicated help
        - the __init__ method
        based upon pyBEL1D routines by Hadrien Michel (...)

    To initialize with the init method, the different arguments are:
        - prior (list of scipy stats objects): a list describing the statistical
                                                distributions for the prior model space
        - cond (callable lambda function): a function that returns True or False if the
                                            model given in argument respects (True) the
                                            conditions or not (False)
        - method (string): name of the method (e.g. "sNMR")
        - forwardFun (dictionary): a dictionary with two entries
                - "Fun" (callable lambda function): the forward model function for a given
                                                    model
                - "Axis" (np.array): the X axis along which the computation is done
        - paramNames (dictionary): a dictionary with multiple entries
                - "NamesFU" (list): Full names of all the parameters with units
                - "NamesSU" (list): Short names of all the parameters with units
                - "NamesS" (list): Short names of all the parameters without units
                - "NamesGlobal" (list): Full names of the global parameters (not layered)
                - "NamesGlobalS" (list): Short names of the global parameters (not layered)
                - "DataUnits" (string): Units for the dataset,
                - "DataName" (string): Name of the Y-axis of the dataset (result from the
                                       forward model)
                - "DataAxis" (string): Name of the X-axis of the dataset
        - nbLayer (int): the number of layers for the model (None if not layered)
        - logTransform ([bool, bool]): Applying a log transform to the models parameters
                                       (first value) and/or the datasets (second value).
                                       The first boolean can also be a list of booleans
                                       with the length of the prior which will mean that
                                       the log transform can be applied parameter by parameter.
    '''

    def __init__(self, prior=None, cond=None, method=None, forwardFun=None, paramNames=None, nbLayer=None, logTransform=[False, False]):
        if (prior is None) or (method is None) or (forwardFun is None) or (paramNames is None):
            self.prior = []
            self.method = []
            self.forwardFun = []
            self.paramNames = []
            self.nbLayer = nbLayer # If None -> Model with parameters and no layers (not geophy?)
            self.cond = cond
            self.logTransform = logTransform
        else:
            self.prior = prior
            self.method = method
            self.forwardFun = forwardFun
            self.cond = cond
            self.paramNames = paramNames
            self.nbLayer = nbLayer
            self.logTransform = logTransform



    @classmethod
    def TEM(cls, prior=None, device_sttngs=None, time_range=None,
            solver_sttngs=None, solver=None, unit='res (ohmm)', resp_abs=False,
            return_rhoa=False, resp_trafo=None, logTransform=[False, False]):
        """
        TEM is a class method that generates a MODELSET class object for TEM.

        The class method takes as arguments:
            - prior (ndarray): a 2D numpy array containing the prior model space
                               decsription. The array is structured as follow:
                               [[e_1_min, e_1_max, rho_1_min, rho_1_max],
                               [e_2_min, ...    ...           rho_2_max],
                               [:        ...    ...                   :],
                               [e_nLay-1_min,   ...           rho_nLay-1_max],
                               [0, 0,           ...           rho_2_nLay_max]]

                               It has 2 columns[thk_min, thk_max, rho_min, rho_max]
                               and nLay lines, nLay layers in the model.

            - Timing (array): a numpy array containing the timings for the dataset simulation.
            - device_props: dictionary with properties necessary for the initialization of TEM frwrd sol

            By default, all inputs are None and this generates the TEM case
            using the forward solution from SimPEG

            Units for the prior are:
                - Thickness (e) in m
                - Resistivity (rho) in Ohm

        """

        method = "TEM"
        if solver == None:
            msg = 'please provide a TEM forward solver object (e.g. empymod or simpeg version!)'
            raise ValueError(msg)

        nLayer, nParam = prior.shape
        nParam /= 2  # from min/max
        nParam = int(nParam)

        # prior = np.multiply(prior,np.matlib.repmat(np.array([1, 1, 1/100, 1/100, 1/1000, 1/1000]),nLayer,1))
        ListPrior = [None] * ((nLayer*nParam)-1)# Half space at bottom
        NamesFullUnits = [None] * ((nLayer*nParam)-1)# Half space at bottom
        NamesShort = [None] * ((nLayer*nParam)-1)# Half space at bottom
        NamesShortPlain = [None] * ((nLayer*nParam)-1)# Half space at bottom
        NamesShortUnits = [None] * ((nLayer*nParam)-1)# Half space at bottom

        Mins = np.zeros(((nLayer*nParam)-1,))
        Maxs = np.zeros(((nLayer*nParam)-1,))

        if unit == 'res (ohmm)':
            Units = ["\,(m)", "\,(\Omega\,m)"]
            NFull = ["Thickness", "Resistivity_{"]
            NShort = ["\mathrm{thk}_{", r"\rho" + "_{"]
            NShrt_plain = ["thk_", "res_"]

            NGlobal = ["\mathrm{Depth}\,(m)", "\mathrm{Resistivity}\,(\Omega m)"]
        elif unit == 'con (mS/m)':
            Units = ["\,(m)", "\,(mS/m)"]
            NFull = ["Thickness", "Conductivity_{"]
            NShort = ["\mathrm{thk}_{", "\sigma_{"]
            NShrt_plain = ["thk_", "con_"]
            NGlobal = ["\mathrm{Depth}\,(m)", "\mathrm{Conductivity}\,(mS/m)"]
        else:
            raise ValueError('unknown unit for the electrical property\n - currently available: res (ohmm), con (mS/m)')

        ident = 0
        for j in range(nParam):  # nested loop to fill the lists
            for i in range(nLayer):
                if not((i == nLayer-1) and (j == 0)):# Not the half-space thickness
                    ListPrior[ident] = stats.uniform(loc=prior[i,j*2],scale=prior[i,j*2+1]-prior[i,j*2])
                    Mins[ident] = prior[i,j*2]
                    Maxs[ident] = prior[i,j*2+1]
                    NamesFullUnits[ident] = NFull[j] + str(i+1) + Units[j]
                    NamesShortUnits[ident] = NShort[j] + str(i+1) + "}" + Units[j]
                    NamesShort[ident] = NShort[j] + str(i+1) + "}"
                    NamesShortPlain[ident] = NShrt_plain[j] + f'{i+1:02d}'
                    ident += 1

        data_units = '(V/m^2)'
        data_axis = "\mathrm{Time}\,(s)"
        data_name = r"\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t"

        paramNames = {"NamesFU":NamesFullUnits, "NamesSU":NamesShortUnits,
                      "NamesS":NamesShort, "NamesPlain": NamesShortPlain, "NamesGlobal":NFull,
                      "NamesGlobalS":NGlobal,"DataUnits":data_units,
                      "DataAxis":data_axis, "DataName":data_name}
        device = 'TEMfast'

        print('\n- initializing forward solver ...')
        frwrd_solver = solver(setup_device=device_sttngs,
                              setup_solver=solver_sttngs,
                              time_range=time_range, device=device,
                              nlayer=nLayer, nparam=nParam)
        print('done setting forward solver ...')
        timing = frwrd_solver.times_rx

        forwardFun = lambda model: frwrd_solver.calc_response(model, mdl_unit=unit, resp_abs=resp_abs,
                                                              return_rhoa=return_rhoa, resp_trafo=resp_trafo) # forwardFun with model as input
        forward = {"Fun":forwardFun, "Axis":timing}

        use = Mins != Maxs
        cond = lambda model: (np.logical_and(np.greater_equal(model[use], Mins[use]), np.less_equal(model[use], Maxs[use]))).all()
        return cls(prior=ListPrior, cond=cond, method=method, forwardFun=forward,
                   paramNames=paramNames, nbLayer=nLayer, logTransform=logTransform)


    @classmethod
    def TEMIP(cls, prior=None, time_range=None, device_sttngs=None, solver_sttngs=None,
              solver=None, resp_abs=False, ip_modeltype='pelton',
              return_rhoa=False, resp_trafo=None,  logTransform=[False, False]):
        """TEMIP is a class method that generates a MODELSET class object for TEM method affected by IP.

        The class method takes as arguments:
            - prior (ndarray): a 2D numpy array containing the prior model space
                               decsription. The array is structured as follow:
                               [[e_1_min, e_1_max, rho_1_min, rho_1_max],
                               [e_2_min, ...    ...        rho_2_max],
                               [:        ...    ...                :],
                               [e_nLay-1_min,   ...   rho_nLay-1_max],
                               [0, 0,           ...   rho_2_nLay_max]]

                               It has 2 columns[thk_min, thk_max, rho_min, rho_max]
                               and nLay lines, nLay layers in the model.

            - Timing (array): a numpy array containing the timings for the dataset simulation.
            - device_props: dictionary with properties necessary for the initialization of TEM frwrd sol

            By default, all inputs are None and this generates the TEM case
            using the forward solution from SimPEG

            Units for the prior are:
                - Thickness (e) in m
                - Resistivity (rho) in Ohm
                - Chargeability ()
                - Relaxation time (s)
                - Dispersion coefficient ()

        """

        method = "TEM"
        if solver == None:
            msg = 'please provide a TEM forward solver object (e.g. empymod or simpeg version!)'
            raise ValueError(msg)

        nLayer, nParam = prior.shape
        nParam /= 2  # from min/max
        nParam = int(nParam)

        # prior = np.multiply(prior,np.matlib.repmat(np.array([1, 1, 1/100, 1/100, 1/1000, 1/1000]),nLayer,1))
        ListPrior = [None] * ((nLayer*nParam)-1)# Half space at bottom
        NamesFullUnits = [None] * ((nLayer*nParam)-1)# Half space at bottom
        NamesShort = [None] * ((nLayer*nParam)-1)# Half space at bottom
        NamesShortPlain = [None] * ((nLayer*nParam)-1)# Half space at bottom
        NamesShortUnits = [None] * ((nLayer*nParam)-1)# Half space at bottom

        Mins = np.zeros(((nLayer*nParam)-1,))
        Maxs = np.zeros(((nLayer*nParam)-1,))
        print('initial Mins and Maxs:\n', Mins, '\n', Maxs)

        if ip_modeltype == 'pelton':
            Units = ["\,(m)", "\,(\Omega m)", '()', '(s)', '()']
            NFull = ["Thickness","Resistivity","Chargeability","Relaxation\,time","Dispersion\,coefficient"]
            NShort = ["\mathrm{thk}_{", r"\rho_{0,", "\mathrm{m}_{", "\\tau_{", "c_{"]
            NShrt_plain = ["thk_", "res0_", "charg_", "tau_", "c_"]
            NGlobal = ["\mathrm{Depth}\,(m)", "\mathrm{Resistivity}\,(\Omega m)",
                       "\mathrm{Chargeability}\,()", "\mathrm{Relaxation\,time}\,(s)", "\mathrm{Dispersion\,coefficient}\,()"]
        elif ip_modeltype == 'mpa':
            Units = ["\,(m)", "\,(\Omega m)", '(rad)', '(s)', '()']
            NFull = ["Thickness","Resistivity","Maximum\,phase\,angle","Relaxation\,time","Dispersion\,coefficient"]
            NShort = ["\mathrm{thk}_{", r"\rho_{0,", "\phi_{\mathrm{max}, ", "\\tau_{\phi, ", "c_{"]
            NShrt_plain = ["thk_", "res0_", "phim_", "taup_", "c_"]
            NGlobal = ["\mathrm{Depth}\,(m)", "\mathrm{Resistivity}\,(\Omega m)",
                       "\mathrm{Maximum\,phase\,angle}\,(rad)", "\mathrm{Relaxation\,time}\,(s)", "\mathrm{Dispersion\,coefficient}\,()"]
        else:
            raise ValueError('this "ip_modeltype" is not available here...')

        ident = 0
        for j in range(nParam):  # nested to loop to fill the lists
            for i in range(nLayer):
                if not((i == nLayer-1) and (j == 0)):# Not the half-space thickness
                    ListPrior[ident] = stats.uniform(loc=prior[i,j*2],scale=prior[i,j*2+1]-prior[i,j*2])
                    Mins[ident] = prior[i,j*2]
                    Maxs[ident] = prior[i,j*2+1]
                    NamesFullUnits[ident] = NFull[j] + str(i+1) + Units[j]
                    NamesShortUnits[ident] = NShort[j] + str(i+1) + "}" + Units[j]
                    NamesShort[ident] = NShort[j] + str(i+1) + "}"
                    NamesShortPlain[ident] = NShrt_plain[j] + f'{i+1:02d}'
                    ident += 1

        print('updated Mins and Maxs:\n', Mins, '\n', Maxs)

        data_units = '(V/m^2)'
        data_axis = "\mathrm{Time}\,(s)"
        data_name = r"\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t"

        paramNames = {"NamesFU":NamesFullUnits, "NamesSU":NamesShortUnits,
                      "NamesS":NamesShort, "NamesPlain": NamesShortPlain, "NamesGlobal":NFull,
                      "NamesGlobalS":NGlobal,"DataUnits":data_units,
                      "DataAxis":data_axis, "DataName":data_name}
        device = 'TEMfast'

        print('\n- initializing forward solver ...')
        frwrd_solver = solver(setup_device=device_sttngs,
                              setup_solver=solver_sttngs,
                              time_range=time_range, device=device,
                              nlayer=nLayer, nparam=nParam)
        print(frwrd_solver.properties_snd)
        print('done setting forward solver ...')
        timing = frwrd_solver.times_rx

        # forwardFun with model as input
        forwardFun = lambda model: frwrd_solver.calc_response(model, return_rhoa=return_rhoa,
                                                              resp_trafo=resp_trafo)
        forward = {"Fun":forwardFun, "Axis":timing}

        use = Mins != Maxs
        cond = lambda model: (np.logical_and(np.greater_equal(model[use], Mins[use]), np.less_equal(model[use], Maxs[use]))).all()
        return cls(prior=ListPrior, cond=cond, method=method, forwardFun=forward,
                   paramNames=paramNames, nbLayer=nLayer, logTransform=logTransform)


# %% Parallelization functions:
def ForwardParallelFun(Model, function, nbVal):
    '''
    from BEL1D: https://github.com/hadrienmichel/pyBEL1D
    
    This function enables the use of any function to be parralelized.
    In order for parallelization to work efficiently for different type of forward models,
    some functions are requiered:

        - ForwardParallelFun: Enables an output to the function even if the function fails
        - ForwardSNMR: Function that defines the forward model directly instead of directly
                       calling a class method (not pickable).

    In order for other forward model to run properly, please, use a similar method! The
    forward model MUST be a callable function DIRECTLY, not a class method.

    WARNING: The function MUST be pickable by dill.

    Inputs: - Model (np.ndarray): the model for which the forward must be run
            - function (lambda function): the function that, when given a model,
                                          returns the corresponding dataset
            - nbVal (int): the number of values that the forward function
                           is supposed to output. Used only in case of error.
    Output: The computed forward model or a None array of the same size.
    '''
    try:
        ForwardComputed = function(Model)
    except:
        ForwardComputed = [None]*nbVal
    return ForwardComputed
