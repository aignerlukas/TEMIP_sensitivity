#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 18:13:08 2022

@author: laigner
"""

# %% import modules
import os
import re
import sys
import math
import logging
import matplotlib

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox

from glob import glob
from matplotlib import cm
from scipy.constants import mu_0
from collections import OrderedDict

from .sounding import Sounding

from .utils import average_non_zeros
from .utils import create_xls_header
from .utils import create_xls_data
from .utils import create_xls_model


# %% logging
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.WARNING)
# logger.setLevel(logging.DEBUG)


# %% classes
class Survey():
    """
    Sounding class to handle and plot TEM data
    
    currently tailored to TEM-FAST data, but can be extended to other devices 
    in the future. uses the zond TEM xls format for saving the inversion results
    
    """
    def __init__(self):
        """
        Constructor of the survey class.
        Initializes instance of object with None for all properties

        Parameters
        ----------

        Returns
        -------
        None.

        """

        self.device = None
        self.invlog_info = None
        self.metainfos = None
        self.soundings = None
        self.sounding_names = None
        self.sounding_positions = None

        self.rawd_nsnds = None
        self.invr_nsnds = None
        self.invr_nlays = None

        self.mean_result = None
        self.thk_results = None
        self.rho_results = None
        self.result_dict = None

        self._rawd_fname = None
        self._rawd_path = None
        self._rawd_fid = None

        self._invres_path = None
        self._invres_fids = None

        self._inv_algorithm = None
        self._inv_type = None
        self._invrun = None

        self._rawd_dframe = None
        self._invres_dframe = None
        self._invres_xls_dat = None
        self._coord_df = None
        self._has_result = False


    def parse_temfast_data(self, filename, path):
        """
        method to read a tem-fast data file. this has to be converted first 
        from the proprieatary binary file format.
        Sets all necessary data directly to the corresponding properties of 
        the survey class

        Parameters
        ----------
        filename : string
            name of TEM-FAST file, needs .tem ending.
        path : string
            path to raw data .tem file.

        Returns
        -------
        None.

        """
        self._rawd_fname = filename
        self._rawd_path = path
        self._rawd_fid = path + os.sep + filename

        logging.info('reading TEM fast data')
        logging.info('from file: ', self._rawd_fname)
        self._device = 'tem_fast'

        headerLines = 8

        # Start of file reading
        myCols = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
        raw = pd.read_csv(self._rawd_fid,
                          names=myCols,
                          sep='\\t',
                          engine="python")
        self._rawd_dframe = raw[~pd.isnull(raw).all(1)].fillna('')
        length_data = len(self._rawd_dframe.c1)

        # create start and end indices of header and data lines
        start_hdr = np.asarray(np.where(raw.loc[:]['c1'] ==
                               'TEM-FAST 48 HPC/S2  Date:'))
        self.rawd_n_snds = np.size(start_hdr)
        logging.info(f'encountered {self.rawd_n_snds} soundings ...')

        start_hdr = np.reshape(start_hdr, (np.size(start_hdr),))
        end_hdr = start_hdr + headerLines

        start_dat = end_hdr
        end_dat = np.copy(start_hdr)
        end_dat = np.delete(end_dat, 0)
        end_dat = np.append(end_dat, length_data)

        # create new dataframe which contains all indices
        indices_hdr = pd.DataFrame({'start': start_hdr, 'end': end_hdr},
                                   columns=['start', 'end'])
        indices_dat = pd.DataFrame({'start': start_dat, 'end': end_dat},
                                   columns=['start', 'end'])

        self.soundings = OrderedDict()
        self.sounding_positions = OrderedDict()
        metainfo_list = []
        sounding_names = []

        for snd_id in range(0, self.rawd_n_snds):
            sounding = Sounding()
            sounding.add_rawdata(device=self._device, raw_df=raw, ind_hdr=indices_hdr,
                                 ind_dat=indices_dat, snd_id=snd_id)

            header = raw.loc[indices_hdr.start[snd_id]:indices_hdr.end[snd_id]-1]
            snd_name = header.iloc[2,1].strip()
            sounding_names.append(snd_name)

            self.soundings[snd_name] = sounding
            self.sounding_positions[snd_name] = (sounding.metainfo['posX'],
                                                 sounding.metainfo['posY'],
                                                 sounding.metainfo['posZ'])
            metainfo_list.append(sounding.metainfo)

        self.sounding_names = sounding_names
        self.metainfos = pd.DataFrame(metainfo_list)


    def parse_invlogfile(self, result_folder, invrun=0):
        """
        method to read the log file written by the inversion routines

        Parameters
        ----------
        result_folder : string
            path to the main directory containing the inv results.
        invrun : int, optional
            id of the inversion run that will be read into the survey class.
            The default is 0, which corresponds to the inversion parameters 
            of the first run.

        Returns
        -------
        None.

        """

        fid = glob(result_folder + '*.log')[0]  # expecting only one entry
        self._raw_log = pd.read_csv(fid, sep='\t', dtype=str)
        self._raw_log.iloc[:, 2:] = self._raw_log.iloc[:, 2:].apply(pd.to_numeric)

        self._invruns = self._raw_log.ri.unique()
        self._invlog_info_all = {}

        for run in self._invruns:
            log_ri = self.select_invlog(self._raw_log, run)
            self._invlog_info_all[run] = log_ri

        self.invlog_info = self._invlog_info_all[f'r{invrun}']


    def select_invlog(self, log_raw, invrun):
        log_ir = log_raw[log_raw.iloc[:, 1] == invrun]
        return log_ir


    def parse_inv_results(self, result_folder, invrun='000'):
        """
        method to parse inversion results from folder structure

        Parameters
        ----------
        result_folder : string
            path to the main directory containing the inv results.
        invrun : str, optional
            id of the inversion run that will be read into the survey class.
            format is 000, 001, 002, ..., 999
            The default is '000', which corresponds to the inversion parameters 
            of the first run.

        Returns
        -------
        None.

        """

        self._invres_path = result_folder
        self.parse_invlogfile(result_folder, invrun=invrun)

        self._result_names = [f'{i:s}' for i in self.invlog_info.name.values]
        self.invr_nsnds = len(self._result_names)
        self.invr_nlays = []
        self._invres_fids = []

        for i, snd_name in enumerate(self._result_names):
            fid = result_folder + snd_name + '/'
            self.soundings[snd_name].add_inv_result(folder=fid,
                                                    invrun=invrun,
                                                    snd_id=snd_name)
            self.invr_nlays.append(self.soundings[snd_name].n_layers)

        # remove those that are not in the results
        snds_not_in_results = list(set(self.sounding_names) - set(self._result_names))
        if len(snds_not_in_results) != 0:
            logger.warning(('removing the following soundings from the current ' +
                            f'instance of Survey, as they are not in the results: {snds_not_in_results}'))

        for snd in snds_not_in_results:
            self.soundings.pop(snd)
        self.sounding_names = list(self.soundings.keys())

        bool_indxs = []   # remove metainfos
        for snd_name in self.sounding_names:
            bool_indxs.append(self.metainfos.loc[:, 'name'] == snd_name)
        bools = np.bitwise_or.reduce(np.asarray(bool_indxs), 0)
        self.metainfos = self.metainfos[bools]

        self._invrun = invrun
        self._has_result = True



    def re_calc_rhoa(self, which='observed'):
        """
        method to re calculate the apparent resistivity

        Parameters
        ----------
        which : string, optional
            'observed' or 'calculated' apparent resistivity. The default is 'observed'.

        Returns
        -------
        None.

        """

        for i, (name, snd) in enumerate(self.soundings.items()):
            snd.calc_rhoa(which=which)


    def get_minmax_invresult(self):
        # TODO check if used in the library for paper 01?

        nlay = self.soundings[self.sounding_names[0]].n_layers
        # TODO generalize for IP models!!, and for different number of layers

        if (self.rho_results == None) and (self.rho_results == None):
            thks = np.zeros((self.invr_n_snds, nlay))
            rhos = np.zeros((self.invr_n_snds, nlay))
            for i, (name, snd) in enumerate(self.soundings.items()):
                curr_model = snd.inv_model
                if not isinstance(curr_model, np.ndarray):
                    logger.warning('sounding: {name} has no inv-result!')
                    continue
                thks[i, :-1] = curr_model[:-1, 0]
                rhos[i, :] = curr_model[:, 1]

            self.thk_results = thks
            self.rho_results = rhos

        thk_mins = np.min(self.thk_results, 0)
        thk_maxs = np.max(self.thk_results, 0)

        rho_mins = np.min(self.rho_results, 0)
        rho_maxs = np.max(self.rho_results, 0)

        mdl_range = np.column_stack((thk_mins, thk_maxs, rho_mins, rho_maxs))

        return mdl_range


    def get_mean_result(self):
        # TODO check if used in the library for paper 01?

        nlay = self.soundings[self.sounding_names[0]].n_layers
        # TODO generalize for IP models!!, and for different number of layers

        if (self.rho_results == None) and (self.rho_results == None):
            thks = np.zeros((self.invr_n_snds, nlay))
            rhos = np.zeros((self.invr_n_snds, nlay))
            for i, (name, snd) in enumerate(self.soundings.items()):
                curr_model = snd.inv_model
                if not isinstance(curr_model, np.ndarray):
                    logger.warning('sounding: {name} has no inv-result!')
                    continue
                thks[i, :-1] = curr_model[:-1, 0]
                rhos[i, :] = curr_model[:, 1]

            self.thk_results = thks
            self.rho_results = rhos

        thk_mean = np.mean(self.thk_results, 0)
        rho_mean = np.mean(self.rho_results, 0)

        self.mean_result = np.column_stack((thk_mean, rho_mean))

        return self.mean_result


    def select_soundings_by(self, prop='tx_side1', vals=[12.5]):
        """
        method to select soundings by a property in the metainfos DataFrame

        Parameters
        ----------
        prop : string, optional
            property on which the selection will be based.
            The default is 'tx_side1'.
        vals : list, optional
            data types in the lits depend on the selected property. It is 
            possible to add multiple list entries.
            The default is [12.5] -> selects all soundings with a 12.5 m loop.

        Raises
        ------
        ValueError
            if the value given in the vals list is not in the property column.

        Returns
        -------
        None.

        """

        if not prop in self.metainfos.columns:
            raise ValueError(f'{prop} is not in the column names of the metainfos ...')
        else:
            bool_indxs = []
            props = self.metainfos.loc[:, prop]
            for val in vals:
                if not any(props == val):
                    raise ValueError(f'{val} is not part of the selected column ({prop}) ...')
                else:
                    print('adding val: ', val)
                    bool_indxs.append(self.metainfos.loc[:, prop] == val)
            bools = np.bitwise_or.reduce(np.asarray(bool_indxs), 0)


            if not len(bools) == 0:
                names2be_del = [i for (i, v) in zip(self.sounding_names, bools) if not v]

                for name in names2be_del:
                    logger.warning(f'removing sounding with the ID: {name}')
                    self.soundings.pop(name)
                    self.sounding_positions.pop(name)

                    if self._has_result == True:
                        self.invlog_info = self.invlog_info[self.invlog_info.name != name]

            self.sounding_names = list(self.soundings.keys())

            bool_indxs = []
            for snd_name in self.sounding_names:
                bool_indxs.append(self.metainfos.loc[:, 'name'] == snd_name)
            bools = np.bitwise_or.reduce(np.asarray(bool_indxs), 0)
            self.metainfos = self.metainfos[bools]



    def plot_rawdata_single_fig(self, ax=None, savefid=None, dpi=200,
                                show_sub0_label=False,
                                colormap=cm.viridis, cmap_range=(0, 1)):
        """
        method to plot all rawdata of this survey into a single figure

        Parameters
        ----------
        ax : mpl axis object, optional
            axis into which the plot will be drawn. The default is None and 
            will lead to the creation of a new axis and Figure.
        savefid : string, optional
            path and filename to a diskspace where the Figure should be stored.
            The default is None -> Figure won't be stored.
        dpi : int, optional
            dots per inch for saving the Figure. The default is 200.
        show_sub0_label : boolean, optional
            Whether to show the label for readings that show negative voltage.
            The default is False.
        colormap : mpl colormap, optional
            colormap which will be used for the individual soundings.
            The default is cm.viridis.
        cmap_range : tuple, optional
            min and max fraction of the colormap in %/100.
            The default is (0, 1) -> full range will be used.

        Returns
        -------
        fig, ax or only ax
            figure and axis object, only ax will be returned if you decided to
            provide an axis to the method.

        """
        # TODO check if this is really used in paper 01

        cm_subsection = np.linspace(cmap_range[0], cmap_range[1],
                                    self.rawd_n_snds)
        colors = [colormap(x) for x in cm_subsection]

        if ax is None:
            logger.info('no instance of axis class provided - creating one...')
            fig, ax = plt.subplots(nrows=1, ncols=2,
                                   figsize=(14, 7))


        for i, (name, snd) in enumerate(self.soundings.items()):
            label = f'Tx-Loop: {snd.tx_loop} m'
            if i == len(self.soundings)-1:
                show_sub0_label = True
            snd.plot(which='observed', ax=ax,
                     color=colors[i], label=label,
                     show_sub0_label=show_sub0_label)

        ax[1].legend()

        if savefid is not None:
            logger.info('saving figure to:\n', savefid, '\n')
            plt.savefig(savefid, dpi=dpi)

        if ax is None:
            return fig, ax
        else:
            return ax


    def plot_rawdata_multi_fig(self, ax=None, savefid=None, dpi=200,
                                nrows_to_plot=1, show_sub0_label=False,
                                xlimits=None, ylimits=None):
        """
        method to plot all rawdata of this survey into a separate subplots

        Parameters
        ----------
        ax : mpl axis object, optional
            axis into which the plot will be drawn. The default is None and 
            will lead to the creation of a new axis and Figure.
        savefid : string, optional
            path and filename to a diskspace where the Figure should be stored.
            The default is None -> Figure won't be stored.
        dpi : int, optional
            dots per inch for saving the Figure. The default is 200.
        nrows_to_plot : int
            Number of rows in the Figure. TODO: auto decision based on number of soundings
            The default is 1.
        show_sub0_label : boolean, optional
            Whether to show the label for readings that show negative voltage.
            The default is False.
        xlimits : tuple
            limits for the x axis. The default is None.
        ylimits : tuple
            limits for the y axis. The default is None.

        Returns
        -------
        fig, ax or only ax
            figure and axis object, only ax will be returned if you decided to
            provide an axis to the method.

        """
        # TODO check if this is really used in paper 01
        
        n_snds = len(self.soundings)
        if ax is None:
            logger.info('no instance of axis class provided - creating one...')
            if (n_snds/nrows_to_plot).is_integer():
                ncols = n_snds/nrows_to_plot
            else:
                ncols = math.ceil((n_snds/nrows_to_plot) / 2) * 2
            fig, ax = plt.subplots(nrows=nrows_to_plot, ncols=int(ncols),
                                     figsize=(ncols*2.0, nrows_to_plot*3.5),
                                     sharex=False, sharey=False,
                                     constrained_layout=True)

        axfl = ax.flatten()

        for i, (name, snd) in enumerate(self.soundings.items()):
            snd.plot_dBzdt(which='observed', ax=axfl[i],
                     color='gray', marker='d', ls='-',
                     label='data', show_sub0_label=show_sub0_label,
                     show_xaxis_label=True, show_yaxis_label=True)
            # snd.plot_dBzdt(which='calculated', ax=axfl[i],
            #          color='crimson', marker='.', ls=':',
            #          label='response', show_sub0_label=True,
            #          show_xaxis_label=True, show_yaxis_label=True)

        axfl[0].legend()
        if xlimits != None:
            axfl[0].set_xlim(xlimits)
        if ylimits != None:
            axfl[0].set_ylim(ylimits)


        if savefid is not None:
            logger.info('saving figure to:\n', savefid, '\n')
            plt.savefig(savefid, dpi=dpi)

        if ax is None:
            return fig, ax
        else:
            return ax


    def plot_datafit_multi_fig(self, ax=None, savefid=None, dpi=200,
                                nrows_to_plot=1, width_factor=2.0,
                                xlimits=None, ylimits=None):
        """
        method to plot the data fit of all results in this survey into a
        separate subplots

        Parameters
        ----------
        ax : mpl axis object, optional
            axis into which the plot will be drawn. The default is None and 
            will lead to the creation of a new axis and Figure.
        savefid : string, optional
            path and filename to a diskspace where the Figure should be stored.
            The default is None -> Figure won't be stored.
        dpi : int, optional
            dots per inch for saving the Figure. The default is 200.
        nrows_to_plot : int
            Number of rows in the Figure. TODO: auto decision based on number of soundings
            The default is 1.
        width_factor : float, optional
            factor to enlargen the width of the automaticall created Figure.
            The default is 2.0.
        xlimits : tuple
            limits for the x axis. The default is None.
        ylimits : tuple
            limits for the y axis. The default is None.

        Returns
        -------
        fig, ax or only ax
            figure and axis object, only ax will be returned if you decided to
            provide an axis to the method.

        """

        n_snds = len(self.soundings)
        if ax is None:
            logger.info('no instance of axis class provided - creating one...')
            if (n_snds/nrows_to_plot).is_integer():
                ncols = n_snds/nrows_to_plot
            else:
                ncols = math.ceil((n_snds/nrows_to_plot) / 2 * 2)
            fig, ax = plt.subplots(nrows=nrows_to_plot, ncols=int(ncols),
                                     figsize=(ncols*width_factor, nrows_to_plot*3.5),
                                     sharex=True, sharey=True,
                                     constrained_layout=True)

        axfl = ax.flatten()

        for i, (name, snd) in enumerate(self.soundings.items()):

            snd.plot_dBzdt(which='observed', ax=axfl[i],
                     color='gray', marker='d', ls='-',
                     label='data', show_sub0_label=False,
                     show_xaxis_label=False, show_yaxis_label=False)

            snd.plot_dBzdt(which='calculated', ax=axfl[i],
                     color='crimson', marker='.', ls=':',
                     label='response', show_sub0_label=True,
                     show_xaxis_label=False, show_yaxis_label=False)

            chi2 = np.asarray(self.invlog_info.chi2)
            rRMS = np.asarray(self.invlog_info.rRMS)
            axfl[i].set_title(f'{snd.name}\n$\chi^2$={chi2[i]:.2f}, rRMSE={rRMS[i]:.1f}')
            # axfl[i].set_title(f'{snd.name}')

        axfl[0].legend()
        if xlimits != None:
            axfl[0].set_xlim(xlimits)
        if ylimits != None:
            axfl[0].set_ylim(ylimits)

        if nrows_to_plot > 1:
            for axis in ax[:, 0]:
                axis.set_ylabel(r"$\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t$ (V/m²)")
            for axis in ax[-1, :]:
                axis.set_xlabel(r'time (s)')
        else:
            ax[0].set_ylabel(r"$\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t$ (V/m²)")
            for axis in ax:
                axis.set_xlabel(r'time (s)')

        if savefid is not None:
            logger.info('saving figure to:\n', savefid, '\n')
            plt.savefig(savefid, dpi=dpi)

        if ax is None:
            return fig, ax
        else:
            return ax


    def plot_invresult_single_fig(self, ax=None, savefid=None, dpi=200,
                                  colormap='viridis', cmap_range=(0,1),
                                  show_mean_model=False, **kwargs):
        """
        method to plot all inversion results from this survey into one Figure

        Parameters
        ----------
        ax : mpl axis object, optional
            axis into which the plot will be drawn. The default is None and 
            will lead to the creation of a new axis and Figure.
        savefid : string, optional
            path and filename to a diskspace where the Figure should be stored.
            The default is None -> Figure won't be stored.
        dpi : int, optional
            dots per inch for saving the Figure. The default is 200.
        colormap : mpl colormap, optional
            colormap which will be used for the individual soundings.
            The default is cm.viridis.
        cmap_range : tuple, optional
            min and max fraction of the colormap in %/100.
            The default is (0, 1) -> full range will be used.
        show_mean_model : boolean, optional
            Switch to show the the mean of all models. The default is False.
        **kwargs : key-word arguments
            key-word arguments for the plt.plot method.

        Returns
        -------
        fig, ax or only ax
            figure and axis object, only ax will be returned if you decided to
            provide an axis to the method.

        """
        

        cmap = matplotlib.cm.get_cmap(colormap)  # hsv, spectral, jet
        cm_subsection = np.linspace(cmap_range[0], cmap_range[1],
                                    self.rawd_n_snds)
        colors = [cmap(x) for x in cm_subsection]

        if ax is None:
            logger.info('no instance of axis class provided - creating one...')
            fig, ax = plt.subplots(nrows=1, ncols=1,
                                   figsize=(4, 8))

        for i, (name, snd) in enumerate(self.soundings.items()):
            label = f'{snd.tx_loop} m'
            snd.plot_inv_model(ax=ax, color=colors[i], label=label,
                               **kwargs)

        # ax.invert_yaxis()

        ax.set_xlabel(r'$\rho$ ' + '($\Omega$m)')
        ax.set_ylabel('Depth (m)')


        if savefid is not None:
            logger.info('saving figure to:\n', savefid, '\n')
            plt.savefig(savefid, dpi=dpi, bbox_inches='tight')

        if ax is None:
            return fig, ax
        else:
            return ax



    def save_inv_as_zondxls(self, save_filename, fid_coordinates=None):
        """
        save all inversion results from this survey into a zond type xls model.

        Parameters
        ----------
        save_filename : string
            name for the xls file. Will be saved to the main directory of the 
            results folder
        fid_coordinates : string, optional
            path and filename to a coord file that has to contain two columns,
            if there is 'xz' in the filename (for profile data) namely:
                'cum_sd' for the slant distance between the soundings
                'Z' for the height of each sounding
            The default is None.

        Raises
        ------
        ValueError
            if the coord filename does not contain xz and csv.

        Returns
        -------
        None.

        """

        if fid_coordinates == None:
            logger.warning('no coord file provided -> using info from saved files')
            use_coord = False
            use_xz = False

        elif ('xz' in fid_coordinates) and ('.csv' in fid_coordinates):
            use_coord = True
            use_xz = True

            df_coord = pd.read_csv(fid_coordinates)
            xz = np.asarray(df_coord.loc[:, ['cum_sd', 'Z']])
            ids_coord = [f'{name}' for name in df_coord.name]

        else:
            use_coord = False
            raise ValueError('fid_coordinates not useful - please check!')


        # fids_results = glob(path_results + 'L*')[::-1]  # reverse order
        fids_results = glob(self._invres_path + '*')
        fids_results = [fid for fid in fids_results if '.' not in fid.split(os.sep)[-1]]  # remove non folder fids
        n_snds = len(fids_results)


        xls_set_coll = pd.DataFrame()

        if not use_coord:
            logger.info('using coord-info from saved files, no separate coordinate file available!!')
            fname_export = save_filename

            for i, fid in enumerate(fids_results):
                snd_name = fid.split(os.sep)[-1]
                logger.info('using sounding ID: ', snd_name)

                mdl = self.soundings[snd_name]._mdl
                fit = self.soundings[snd_name]._fit

                position = np.asarray(mdl.iloc[0,:3])
                dist = position[0]  # use the x coordinate as the distance (only for local coordinate sys)
                # TODO more general coordinate reading, read xyz and dist...!!

                # prepare data fit
                n_chnnls = len(fit)

                xls_data, rRMSs = create_xls_data(datafit_df=fit)

                xls_header = create_xls_header(idx=i, snd_id=snd_name,
                                               distance=dist, position=position,
                                               rRMSs=rRMSs)

                xls_model = create_xls_model(model_df=mdl)

                xls_full = xls_header.append([xls_data, xls_model]).reset_index(drop=True)

                xls_set_coll = xls_set_coll.append(xls_full)

        else:
            for j, id_coord in enumerate(ids_coord):
                for i, fid in enumerate(fids_results):
                    snd_name = fid.split(os.sep)[-1]
                    if id_coord == snd_name:
                        logger.info('using sounding ID: ', snd_name)
                        logger.info('with coord ID: ', id_coord)

                        mdl = self.soundings[snd_name]._mdl
                        fit = self.soundings[snd_name]._fit

                        export_mdl_csv = mdl.copy()
                        export_mdl_csv.drop(axis=0, columns='Y', inplace=True)

                        position = np.asarray(mdl.iloc[0,:3])
                        dist = position[0]  # use the x coordinate as the distance (only for local coordinate sys)
                        # TODO more general coordinate reading, read xyz and dist...!!

                        if use_xz:
                            dist = xz[j,0]  # replace distance with x value from xz file
                            position[2] = xz[j,1]  # replace z coordinate with info from xz file
                        elif use_coord:
                            dist = xz[j,0]  # replace distance with x value from xz file
                            position[2] = xz[j,1]  # replace z coordinate with info from xz file
                            position[0] = df_coord.iloc[j, 1]  # use easting
                            position[1] = df_coord.iloc[j, 2]  # use northing
                        else:
                            pass

                        # prepare data fit
                        n_chnnls = len(fit)
                        xls_data, rRMSs = create_xls_data(datafit_df=fit)

                        # prep header
                        xls_header = create_xls_header(idx=j, snd_id=snd_name,
                                                       distance=dist, position=position,
                                                       rRMSs=rRMSs)

                        # prep model
                        xls_model = create_xls_model(model_df=mdl)

                        # merge
                        xls_full = xls_header.append([xls_data, xls_model]).reset_index(drop=True)
                        xls_set_coll = xls_set_coll.append(xls_full)


                        # export model with xz info
                        export_mdl_csv.iloc[:, 0:2] = xz[j, :]
                        export_mdl_csv.rename(columns={'depth(m)':'thk(m)'}, inplace=True)
                        fid_mdl_new = fid + f'/csv/invrun{self._invrun}_{snd_name}_mdlXZ.csv'

                        export_mdl_csv.to_csv(fid_mdl_new, index=False)

        # concatenate and export
        new_col = pd.DataFrame(columns=['col0'])
        xls_set_coll = pd.concat([new_col, xls_set_coll]).reset_index(drop=True)
        xls_set_coll.loc[-1] = np.full([xls_set_coll.shape[1], ], np.nan)
        xls_set_coll = xls_set_coll.sort_index().reset_index(drop=True)

        xls_set_coll.to_excel(self._invres_path + f'{save_filename}.xls',
                              header=False, index=False)