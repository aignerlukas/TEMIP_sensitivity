# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 16:52:49 2021
sounding class that holds raw data and inversion results

TODO:
    [] handle IP, noIP results, decide on format (thk, cum_thk, etc) all in one array, or separate
    [] add read csv inversion result
    [] add logging
        [] write to log file

@author: lukas
"""

# %% modules
import os
import re
import sys
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.constants import mu_0
from matplotlib.colors import LogNorm, SymLogNorm

from .utils import get_float_from_string
from .utils import rearr_zond_mdl
from .utils import get_zt_inv_model
from .utils import get_zt_response
from .utils import get_zt_header


# %% logging
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.WARNING)
# logger.setLevel(logging.DEBUG)


# %% Sounding class
class Sounding():
    """
    Sounding class to handle and plot TEM data
    
    """
    def __init__(self):
        """
        Constructor of the sounding class.
        Initializes instance of object with nones for the properties

        Parameters
        ----------

        Returns
        -------
        None.

        """
        self.name = None
        self.current = None
        self.currentkey = None
        self.timekey = None
        self.tx_loop = None
        self.rx_loop = None
        self.magnetic_moment = None

        self.metainfo = None
        self.savepath = None
        self.savefid = None

        self.time_o = None  # observed time steps
        self.time_f = None  # filtered time range

        self.sgnl_o = None  # observed signal in V/m², measured data
        self.sgnl_of = None  # observed signal in V/m², filtered to time range used in the inversion
        self.sgnl_c = None  # calculated signal in V/m², from inversion response

        self.error_o = None  # observed error in V/m²
        self.error_of = None  # filtered error in V/m²

        self.rhoa_o = None  # observed apparent resistivity
        self.rhoa_of = None  # observed apparent resistivity, filtered
        self.rhoa_c = None  # calculated apparent resistivity, from inv. response

        self.inv_model = None  # step format

        self._raw_df = None
        self._header = None
        self._data = None
        self._has_result = False
        self._mdl = None
        self._fit = None
        self._jacobian = None
        # self.parse_sndng(header_type)  # data reading together with init ... makes sense?


    def add_rawdata(self, device, raw_df, ind_hdr, ind_dat, snd_id):
        """
        method to load a single raw data sounding into this class structure

        Parameters
        ----------
        device : string
            name of TEM device.
        raw_df : pd.DataFrame
            raw data as read from the file.
        ind_hdr : pd.DataFrame
            indices of the header lines within the survey file.
        ind_dat : pd.DataFrame
            indices of the data lines within the survey file.
        snd_id : int
            index for a single sounding of the survey, starting at 0.

        Returns
        -------
        None.

        """
        self._raw_df = raw_df

        # select part of raw DataFrame which contains the meta info
        self._header = raw_df.loc[ind_hdr.start[snd_id]:ind_hdr.end[snd_id]-1]

        # create variables for legend Informations:
        self.name = self._header.iloc[2,1].strip()
        self.timekey = int(self._header.iloc[3,1])
        self.current = get_float_from_string(self._header.iloc[3,5])
        self.currentkey = 4 if self.current>2.5 else 1
        self.filter_pwrline = get_float_from_string(self._header.iloc[3, 6])
        self.tx_loop = float(self._header.iloc[4,1])
        self.rx_loop = float(self._header.iloc[4,3])
        self.tx_area = self.tx_loop**2
        self.rx_area = self.rx_loop**2

        self.metainfo = {
            'device': device,
            'name': self._header.iloc[2,1].strip(),
            'tx_side1': self.tx_loop,
            'tx_side2': self.tx_loop,
            'tx_area': self.tx_loop**2,
            'rx_loop': self.rx_loop,
            'rx_area': self.rx_loop**2,
            'turns': int(self._header.iloc[4,5]),
            'current': self.current,  # in A
            'timekey': int(self._header.iloc[3,1]),
            'currentkey': 4 if self.current>2.5 else 1,
            'filter_powerline': int(self.filter_pwrline),
            'stacks': int(self._header.iloc[3,3]),
            'comments': self._header.iloc[5, 1],
            'location': self._header.iloc[1,1],
            'posX': float(self._header.iloc[6,1]),
            'posY': float(self._header.iloc[6,3]),
            'posZ': float(self._header.iloc[6,5]),
            }

        comments = self.metainfo['comments']
        if ',' in comments and (len(comments.split(',')) == 1):
            cableLenTx = comments.split('-')[0]
            cableLenRx = cableLenTx
        elif ',' in comments and (len(comments.split(',')) == 2):
            cableLenTx = comments.split(',')[0].split('-')[0]
            cableLenRx = comments.split(',')[1].split('-')[0]
        elif '-' in comments and (len(comments.split('-')) == 2):
            cableLenTx = comments.split('-')[1]
            cableLenRx = cableLenTx
        elif 'M' in comments:
            cableLenTx = comments.split('M')[0]
            cableLenRx = cableLenTx
        elif  'm' in comments:
            cableLenTx = comments.split('m')[0]
            cableLenRx = cableLenTx
        else:
            logging.warning('Warning! comments not useful')
            logging.warning('setting to 4 * TxRx: ...')
            cableLenTx, cableLenRx = 4*self.tx_loop, 4*self.rx_loop
        self.metainfo['tx_cable'] = cableLenTx
        self.metainfo['rx_cable'] = cableLenRx

        #select part of dataframe which contains the actual data
        data = raw_df.loc[ind_dat.start[snd_id]:ind_dat.end[snd_id]-1]
        data = data.drop(['c6','c7','c8'], axis=1)
        data.columns = ['channel', 'time', 'signal', 'error', 'rhoa']
        data = data.apply(pd.to_numeric)
        data.replace(0, np.nan, inplace=True)
        data.replace(99999.99, np.nan, inplace=True)

        self._data = data

        self.time_o = np.asarray(data.time * 1e-6)  # from us to s
        self.sgnl_o = np.asarray(data.signal * self.current / self.rx_area)  # from V/A to V/m²
        self.error_o = np.asarray(data.error * self.current / self.rx_area)  # from V/A to V/m²
        self.rhoa_o = np.asarray(data.rhoa)

        self.magnetic_moment = (self.current *
                                self.tx_area *
                                self.metainfo['turns'])


    def add_inv_result(self, folder, snd_id, invrun='000'):
        """
        method to load a single inversion result into this class structure

        Parameters
        ----------
        folder : string
            main path to the folder holding all inversion results.
        snd_id : string
            name or identifier of the sounding that should be added.
        invrun : str, optional
            id of the inversion run that will be read into the survey class.
            format is 000, 001, 002, ..., 999
            The default is '000', which corresponds to the inversion parameters 
            of the first run.

        Returns
        -------
        None.

        """
        fid_mdl = folder + f'/csv/invrun{invrun}_{snd_id}.csv'
        fid_init_mdl = folder + f'/csv/invrun{invrun}_{snd_id}_startmodel.csv'
        fid_fit = folder + f'/csv/invrun{invrun}_{snd_id}_fit.csv'
        fid_jac = folder + f'/csv/invrun{invrun}_{snd_id}_jac.csv'

        mdl = pd.read_csv(fid_mdl)
        init_mdl = pd.read_csv(fid_init_mdl)
        fit = pd.read_csv(fid_fit)
        jacobian = pd.read_csv(fid_jac, index_col=0)

        self._mdl = mdl
        self._init_mdl = init_mdl
        self._fit = fit
        self._jacobian = jacobian

        self.n_params = mdl.shape[1]
        self.n_layers = mdl.shape[0]

        self.time_f = fit.iloc[:, 0]
        self.sgnl_c = fit.iloc[:, 1]
        self.sgnl_of = fit.iloc[:, 2]

        self.err_of = fit.iloc[:, 3]
        self.err_est = fit.iloc[:, 4]

        self.rhoa_c = fit.iloc[:, 5]
        self.rhoa_of = fit.iloc[:, 6]

        self.inv_model = np.asarray(mdl.iloc[:, 3:])
        self.init_model = np.asarray(init_mdl.iloc[:, 3:])
        self.inv_position = mdl.iloc[:, 0:3]

        self._has_result = True


    def add_bel1d_result(self, folder, bel_type, snd_id):
        """
        method to load a bel1d result to this class structure

        Parameters
        ----------
        folder : string
            main path to the folder holding all inversion results.
        bel_type : TYPE
            DESCRIPTION.
        snd_id : string
            name or identifier of the sounding that should be added.

        Returns
        -------
        None.

        """
        # TODO needed for paper 01?

        fid_means = folder + f'/csv/{snd_id}{bel_type}_means.csv'
        fid_stds = folder + f'/csv/{snd_id}{bel_type}_stds.csv'
        fid_fit = folder + f'/csv/{snd_id}{bel_type}_fit.csv'

        means = pd.read_csv(fid_means)
        stds = pd.read_csv(fid_stds)
        fit = pd.read_csv(fid_fit)


        self._means = means
        self._stds = stds
        self._fit = fit

        self.n_params = means.shape[1]
        self.n_layers = stds.shape[0]

        self.time_f = fit.iloc[:, 0]
        self.sgnl_c = fit.iloc[:, 1]
        self.rhoa_c = fit.iloc[:, 5]

        self.means = np.asarray(means.iloc[:, 3:])
        self.stds = np.asarray(stds.iloc[:, 3:])
        self.inv_position = means.iloc[:, 0:3]

        self._has_result = True


    def parse_zond_result(self, file_id, current, rx_area,
                          snd_id=0, remove_IP=True):
        """
        method to read a result from the zond TEM .xls format


        Parameters
        ----------
        file_id : string
            path and filename of the file to be read.
        current : float
            injected current in transmitter loop (A).
        rx_area : float
            area covered by the receiver loop (m²).
        remove_IP : boolean
            DESCRIPTION.

        Raises
        ------
        ValueError
            if the file extension is not supported by the reading routine.

        Returns
        -------
        None.

        """
        self._filename = file_id.split(f'{os.sep}')[-1]
        self._file_id = file_id
        self._file_ext = self._file_id.split('.')[-1]
        self.name = self._filename.split('.')[0]

        self._has_result = True

        # assume the zondVersion is IP and read all needed columns
        try:
            if self._file_ext == 'xls':
                raw = pd.read_excel(file_id,
                                    usecols=range(1, 9),
                                    names=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8'],
                                    header=None)
            elif self._file_ext == 'csv':
                raw = pd.read_csv(file_id,
                                  usecols=range(1, 9),
                                  names=['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8'],
                                  header=None,
                                  engine='python')
            else:
                raise ValueError('no valid file ending in filename... exiting!')

            version_zond = 'IP'

        # catch the ValueError exception (if noIP) and read only with 6 columns
        except ValueError:
            if self._file_ext == 'xls':
                raw = pd.read_excel(file_id,
                                    usecols=range(1, 7),
                                    names=['c1', 'c2', 'c3', 'c4', 'c5', 'c6'],
                                    header=None)
            elif self._file_ext == 'csv':
                raw = pd.read_csv(file_id,
                                  usecols=range(1, 7),
                                  names=['c1', 'c2', 'c3', 'c4', 'c5', 'c6'],
                                  header=None,
                                  engine='python')
            else:
                raise ValueError('no valid file ending in filename... exiting!')
            version_zond = 'noIP'

        if self._file_ext == 'xls':
            # dat = raw.drop(labels='empty', axis=1)
            dat = raw.drop(labels=0, axis=0)
            dat.dropna(axis=0, how='all', inplace=True)
        elif self._file_ext == 'csv':
            dat = raw.drop(labels=0, axis=0)
            dat.dropna(axis=0, how='all', inplace=True)
        else:
            raise ValueError('file extension not available')

        indices_labels = np.flatnonzero(dat.c1 == '#') + 1
        ev2nd0 = indices_labels[::2]; ev2nd1 = indices_labels[1::2]
        end = len(dat)
        endMdl = np.append(ev2nd0[1:]-5, [end])

        idcs_hdr = pd.DataFrame({'start':ev2nd0-4, 'end':ev2nd0},
                                   columns=['start', 'end'])
        idcs_mresp = pd.DataFrame({'start':ev2nd0+1, 'end':ev2nd1-1},
                                   columns=['start', 'end'])
        idcs_mdl = pd.DataFrame({'start':ev2nd1+1, 'end':endMdl},
                                   columns=['start', 'end'])

        n_snds = len(idcs_hdr)
        sndNames = []
        for logID in range(0, n_snds):
            hdr = dat.loc[idcs_hdr.start[logID]:idcs_hdr.end[logID]]
            sndNames.append(hdr.iloc[0][1])
        idcs_hdr.insert(0, "sndID", sndNames)
        idcs_mresp.insert(0, "sndID", sndNames)
        idcs_mdl.insert(0, "sndID", sndNames)

        # read data into class attributes, only single snd
        self._header = get_zt_header(dat, idcs_hdr, snd_id=snd_id)

        data = get_zt_response(dat, idcs_mresp, snd_id=snd_id)
        self._data = data    #['ctr','time','rho_O','rho_C','U_O','U_C']

        self.time = np.asarray(data.time)  # already in s
        factor = 1e-6 / rx_area * current  # from uV/A to V/m²
        self.sgnl_c = np.asarray(data.U_C) * factor  # TODO no scaling yet
        self.sgnl_o = np.asarray(data.U_O) * factor  # TODO sanity check!! ISSUE - no current and rx area available in zondTEM excel format
        self.rhoa_c = np.asarray(data.rho_C)
        self.rhoa_o = np.asarray(data.rho_O)

        self._model_df = get_zt_inv_model(dat, idcs_mdl, snd_id, remove_IP=remove_IP)

        self.inv_model = np.abs(rearr_zond_mdl(self._model_df))


    def calc_rhoa(self, which='observed'):
        """
        method to calculate the apparent resistivity

        Parameters
        ----------
        which : string, optional
            observed (measured data) or calculated (inversion response).
            The default is 'observed'.

        Raises
        ------
        ValueError
            if which is neither 'observed' or 'calculated'.

        Returns
        -------
        rhoa : np.ndarray
            apparent resistivity (Ohmm).

        """

        if which == 'observed':
            signal = self.sgnl_o
            time = self.time_o

            mask = signal < 0
            self._sgnl_sub0 = self.sgnl_o[mask]
            self._time_sub0 = self.time_o[mask]

        elif which == 'calculated':
            signal = self.sgnl_c
            time = self.time_f

            mask = signal < 0
            self._sgnl_sub0 = self.sgnl_c[mask]
            self._time_sub0 = self.time_f[mask]

        else:
            raise ValueError('please use either observed or calculated')

        M = self.magnetic_moment
        rhoa = ((1 / np.pi) *
                  (M / (20 * (abs(signal))))**(2/3) *
                  (mu_0 / (time))**(5/3)
                  )
        rhoa[mask] = rhoa[mask] * -1


        if which == 'observed':
            self.rhoa_o = rhoa

        elif which == 'calculated':
            self.rhoa_c = rhoa

        else:
            raise ValueError('please use either observed or calculated')

        return rhoa


    def plot_dBzdt(self, which='observed', ax=None,
                   xlimits=(2e-6, 1e-2), ylimits=(2e-10, 1e-1),
                   show_xaxis_label=True, show_yaxis_label=True,
                   sub0col='k', show_sub0_label=True,
                   save_fid=None, dpi=150,
                   **kwargs):
        """
        Method to plot the TX voltages of a sounding with a predefined style.

        Parameters
        ----------
        which : str, optional
            select which signal to plot (observed or calculated).
            The default is 'observed' (alternative: 'calculated').
        ax : axis object, optional
            use this if you want to plot to an existing axis outside of this method.
            The default is None.
        xlimits : tuple or None, optional
            x-axis limits (s). The default is (2e-6, 1e-3).
        ylimits : tuple or None, optional
            y-axis limits (V/m²). The default is (2e-10, 1e-1).
        sub0col : string, optional
            color for the sub0 voltage markers. The default is 'k' (black).
        show_sub0_label : boolean, optional
            decide whether a label for the sub0 markers should be added 
            to the legend. The default is True.
        save_fid : str, optional
            full path + filename with extension to save file.
            If none it won't be saved. The default is None.
        dpi : int, optional
            dots-per-inch for the saved file. The default is 150.
        **kwargs : key-word arguments
            for the plt.plot method.

        Raises
        ------
        ValueError
            if which is neither 'observed' or 'calculated'.

        Returns
        -------
        fig, ax or only ax
            figure and axis object, only ax will be returned if you decided to
            provide an axis to the method.

        """
        if ax is None:
            logger.info('no instance of axis class provided - creating one...')
            fig, ax = plt.subplots(nrows=1, ncols=1,
                                   figsize=(7, 7))

        if which == 'observed':
            signal = self.sgnl_o
            time = self.time_o
        elif which == 'calculated':
            signal = self.sgnl_c
            time = self.time_f
        else:
            raise ValueError('please use either observed or calculated')

        # select neg values, to mark them explicitly within the plot
        time_sub0 = time[signal < 0]
        sgnl_sub0 = signal[signal < 0]

        ax.loglog(time, abs(signal), 'd--',
                  lw=1, ms=5, **kwargs)
        if show_sub0_label:
            ax.loglog(time_sub0, abs(sgnl_sub0), marker='s', ls='none',
                      markerfacecolor='none', markersize=5,
                      markeredgewidth=1, markeredgecolor=sub0col, label='negative vals')
        else:
            ax.loglog(time_sub0, abs(sgnl_sub0), marker='s', ls='none',
                      markerfacecolor='none', markersize=5,
                      markeredgewidth=1, markeredgecolor=sub0col)
        if show_xaxis_label:
            ax.set_xlabel(r'time (s)')
        if show_yaxis_label:
            ax.set_ylabel(r"$\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t$ (V/m²)")

        if xlimits != None:
            ax.set_xlim(xlimits)
        if ylimits != None:
            ax.set_ylim(ylimits)

        if save_fid is not None:
            plt.savefig(save_fid, dpi=dpi)

        if ax is None:
            return fig, ax
        else:
            return ax


    def plot_rhoa(self, which='observed', ax=None, log_rhoa=False,
                  xlimits=(2e-6, 1e-2), ylimits=(1e0, 1e4),
                  show_xaxis_label=True, show_yaxis_label=True,
                  sub0col='k', show_sub0_label=True,
                  save_fid=None, dpi=150,
                   **kwargs):
        """
        Method to plot the apparent resistivity with a predefined style.

        Parameters
        ----------
        which : str, optional
            select which signal to plot (observed or calculated).
            The default is 'observed' (alternative: 'calculated').
        ax : axis object, optional
            use this if you want to plot to an existing axis outside of this method.
            The default is None.
        xlimits : tuple or None, optional
            x-axis limits (s). The default is (2e-6, 1e-3).
        ylimits : tuple or None, optional
            y-axis limits (Ohmm). The default is (1e0, 1e4).
        sub0col : string, optional
            color for the sub0 rhoa markers. The default is 'k' (black).
        show_sub0_label : boolean, optional
            decide whether a label for the sub0 markers should be added 
            to the legend. The default is True.
        save_fid : str, optional
            full path + filename with extension to save file.
            If none it won't be saved. The default is None.
        dpi : int, optional
            dots-per-inch for the saved file. The default is 150.
        **kwargs : key-word arguments
            for the plt.plot method.

        Raises
        ------
        ValueError
            if which is neither 'observed' or 'calculated'.

        Returns
        -------
        fig, ax or only ax
            figure and axis object, only ax will be returned if you decided to
            provide an axis to the method.

        """

        if ax is None:
            logger.info('no instance of axis class provided - creating one...')
            fig, ax = plt.subplots(nrows=1, ncols=1,
                                   figsize=(7, 7))

        if which == 'observed':
            rhoa = self.rhoa_o
            time = self.time_o
        elif which == 'calculated':
            rhoa = self.rhoa_c
            time = self.time_f
        else:
            raise ValueError('please use either observed or calculated')

        if rhoa is None:
            rhoa = self.calc_rhoa()

        time_sub0 = time[rhoa < 0]
        rhoa_sub0 = rhoa[rhoa < 0]

        ax.semilogx(time, abs(rhoa), 'd--',  # marker='d', ls=':',
                    lw=1, ms=5, **kwargs)
        if show_sub0_label:
            ax.semilogx(time_sub0, abs(rhoa_sub0), marker='s', ls='none',
                        markerfacecolor='none', markersize=5,
                        markeredgewidth=1, markeredgecolor=sub0col, label='negative vals')
        else:
            ax.semilogx(time_sub0, abs(rhoa_sub0), marker='s', ls='none',
                        markerfacecolor='none', markersize=5,
                        markeredgewidth=1, markeredgecolor=sub0col)
        if show_xaxis_label:
            ax.set_xlabel(r'time (s)')
        if show_yaxis_label:
            ax.set_ylabel(r'$\rho_a$ ($\Omega$m)')

        if xlimits != None:
            ax.set_xlim(xlimits)
        if ylimits != None:
            ax.set_ylim(ylimits)

        if log_rhoa is True:
            ax.set_yscale('log')

        if save_fid is not None:
            plt.savefig(save_fid, dpi=dpi)

        if ax is None:
            return fig, ax
        else:
            return ax


    def plot(self, which='observed', ax=None,
             xlimits=(2e-6, 1e-2), ylimits_dBzdt=(2e-10, 1e-1),
             ylimits_rhoa=(1e0, 1e4), log_rhoa=True,
             show_sub0_label=False,
             sub0_color_sig='k', sub0_color_roa='k',
             **kwargs):
        """
        method to plot both the signal and the apparent resistivit in two
        subplots.

        Parameters
        ----------
        which : str, optional
            select which signal to plot (observed or calculated).
            The default is 'observed' (alternative: 'calculated').
        ax : axis object, optional
            use this if you want to plot to an existing axis outside of this method.
            The default is None.
        xlimits : tuple or None, optional
            x-axis limits. The default is (2e-6, 1e-2).
        ylimits_dBzdt : TYPE, optional
            y-axis limits (V/m²). The default is (2e-10, 1e-1).
        ylimits_rhoa : tuple or None, optional
            y-axis limits (Ohmm). The default is (1e0, 1e4).
        log_rhoa : boolean, optional
            switch to log-scale the y-axis of the app. resistivity.
            The default is True.
        show_sub0_label : boolean, optional
            decide whether a label for the sub0 markers should be added 
            to the legend. The default is True.
        sub0_color_sig : string, optional
            color for the sub0 signal markers. The default is 'k' (black).
        sub0_color_roa : string, optional
            color for the sub0 rhoa markers. The default is 'k' (black).
        **kwargs : key-word arguments
            for the plt.plot method.

        Returns
        -------
        fig, ax or only ax
            figure and axis object, only ax will be returned if you decided to
            provide an axis to the method.

        """

        if ax is None:
            logger.info('no instance of axis class provided - creating one...')
            fig, ax = plt.subplots(nrows=1, ncols=2,
                                   figsize=(14, 7), constrained_layout=True)

        ax1 = self.plot_dBzdt(which, ax=ax[0],
                              xlimits=xlimits, ylimits=ylimits_dBzdt,
                              show_sub0_label=show_sub0_label, sub0col=sub0_color_sig,
                              **kwargs)

        ax2 = self.plot_rhoa(which, ax=ax[1], log_rhoa=log_rhoa,
                             xlimits=xlimits, ylimits=ylimits_rhoa,
                             show_sub0_label=show_sub0_label, sub0col=sub0_color_roa,
                             **kwargs)
        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")

        if ax is None:
            return fig, ax
        else:
            return ax


    def plot_inv_model(self, ax=None, add_bottom=50, **kwargs):
        """
        method to plot the inverted model of a TEM sounding (thk, res) as a
        step model.

        Parameters
        ----------
        ax : axis object, optional
            use this if you want to plot to an existing axis outside of this method.
            The default is None.
        add_bottom : float, optional
            add thickness to the bottom layer (m). The default is 50 m.
        **kwargs : key-word arguments
            for the plt.plot method.

        Returns
        -------
        fig, ax or only ax
            figure and axis object, only ax will be returned if you decided to
            provide an axis to the method.

        """

        if ax is None:
            logger.info('no instance of axis class provided - creating one...')
            fig, ax = plt.subplots(nrows=1, ncols=1,
                                   figsize=(3, 8), constrained_layout=True)

        if self.inv_model[0, 0] != 0.0:
            # print(self.inv_model)
            # print(self.inv_model)
            logger.warning('found thk model, converting to step model')
            dpths = np.r_[0, np.cumsum(self.inv_model[:-1, 0])]
            res = self.inv_model[:, 1]
            self.inv_model = np.column_stack((dpths.repeat(2, 0)[1:],
                                              res.repeat(2, 0)[:-1]))

            # print(self.inv_model)


        if add_bottom is not None:
            res = np.r_[self.inv_model[:, 1], self.inv_model[-1, 1]]
            dpth = np.r_[self.inv_model[:, 0], self.inv_model[-1, 0]+add_bottom]

            ax.plot(res, dpth, **kwargs)
        else:
            ax.plot(self.inv_model[:, 1], self.inv_model[:, 0], **kwargs)

        if ax is None:
            return fig, ax
        else:
            return ax


    def plot_initial_model(self, ax=None, add_bottom=50, **kwargs):
        """
        method to plot the initial model for the inversion of a TEM model
        (thk, res) as a step model.

        Parameters
        ----------
        ax : axis object, optional
            use this if you want to plot to an existing axis outside of this method.
            The default is None.
        add_bottom : float, optional
            add thickness to the bottom layer (m). The default is 50 m.
        **kwargs : key-word arguments
            for the plt.plot method.

        Returns
        -------
        fig, ax or only ax
            figure and axis object, only ax will be returned if you decided to
            provide an axis to the method.

        """

        if ax is None:
            logger.info('no instance of axis class provided - creating one...')
            fig, ax = plt.subplots(nrows=1, ncols=1,
                                   figsize=(3, 8), constrained_layout=True)

        if self.init_model[0, 0] != 0.0:
            logger.warning('found thk model, converting to step model')
            dpths = np.r_[0, np.cumsum(self.init_model[:-1, 0])]
            res = self.init_model[:, 1]
            self.init_model = np.column_stack((dpths.repeat(2, 0)[1:],
                                               res.repeat(2, 0)[:-1]))

        if add_bottom is not None:
            res = np.r_[self.init_model[:, 1], self.init_model[-1, 1]]
            dpth = np.r_[self.init_model[:, 0], self.init_model[-1, 0]+add_bottom]

            ax.plot(res, dpth, **kwargs)
        else:
            ax.plot(self.init_model[:, 1], self.init_model[:, 0], **kwargs)

        if ax is None:
            return fig, ax
        else:
            return ax


    def plot_model(self, model, ax=None, add_bottom=50, **kwargs):
        """
        method to plot a model (thk, res) as a step model.

        Parameters
        ----------
        model : np.ndarray
            the model that should be plotted
        ax : axis object, optional
            use this if you want to plot to an existing axis outside of this method.
            The default is None.
        add_bottom : float, optional
            add thickness to the bottom layer (m). The default is 50 m.
        **kwargs : key-word arguments
            for the plt.plot method.

        Returns
        -------
        fig, ax or only ax
            figure and axis object, only ax will be returned if you decided to
            provide an axis to the method.

        """
        if ax is None:
            logger.info('no instance of axis class provided - creating one...')
            fig, ax = plt.subplots(nrows=1, ncols=1,
                                   figsize=(3, 8), constrained_layout=True)

        if model[0, 0] != 0.0:
            logger.warning('found thk model, converting to step model')
            dpths = np.r_[0, np.cumsum(model[:-1, 0])]
            res = model[:, 1]
            model = np.column_stack((dpths.repeat(2, 0)[1:],
                                               res.repeat(2, 0)[:-1]))

        if add_bottom is not None:
            res = np.r_[model[:, 1], model[-1, 1]]
            dpth = np.r_[model[:, 0], model[-1, 0]+add_bottom]

            ax.plot(res, dpth, **kwargs)
        else:
            ax.plot(model[:, 1], model[:, 0], **kwargs)

        if ax is None:
            return fig, ax
        else:
            return ax


    def plot_jacobian(self, ax=None):
        """
        

        Parameters
        ----------
        ax : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        fig, ax or only ax
            figure and axis object, only ax will be returned if you decided to
            provide an axis to the method.

        """

        # TODO improve!! calculate sensitivity from auken etal
        vmin = -1e-5  # todo automatize
        vmax = abs(vmin)
        norm = SymLogNorm(linthresh=3, linscale=3,
                          vmin=vmin, vmax=vmax, base=10)
        # norm = SymLogNorm(linthresh=0.3, linscale=0.3, base=10)

        plt.figure(figsize=(12, 8))
        axj = sns.heatmap(self._jacobian, cmap="BrBG", annot=True,
                          fmt='.2g', robust=True, center=0,
                          vmin=vmin, vmax=vmax, norm=norm)  #
        axj.set_title('jacobian last iteration')
        axj.set_xlabel('model parameters')
        axj.set_ylabel('data parameters')
        plt.tight_layout()
        figj = axj.get_figure()

        if ax is None:
            return figj, axj
        else:
            return axj


    def get_obsdata_dataframe(self):
        """
        Method that returns the observed data as a pd.DataFrame

        Returns
        -------
        pd.DataFrame
            columns: ['channel()', 'time(s)', 'signal(V/m²)', 'err(V/m²)', 'rhoa(Ohmm)'].

        """
        channels = np.arange(1, len(self.time_o)+1)
        data = np.column_stack((channels, self.time_o, self.sgnl_o,
                                self.error_o, self.rhoa_o))

        columns = ['channel', 'time', 'signal', 'err', 'rhoa']
        self.data_obs = pd.DataFrame(data, columns=columns)

        return self.data_obs


    def get_device_settings(self):
        """
        Method that returns the device settings in a dict

        Returns
        -------
        dictionary
            contains: timekey, currentkey, tx_loop, rx_loop
                      current_inj, filter_powerline.
        """
        self.setup_device = {"timekey": self.timekey,
                             "currentkey": self.currentkey,
                             "txloop": self.tx_loop,
                             "rxloop": self.rx_loop,
                             "current_inj": self.current,
                             "filter_powerline": self.filter_pwrline}
        return self.setup_device


    def save_csv(self, save_root, num_idx):
        """
        method to save a predefined csv

        Parameters
        ----------
        save_root : str
            path to a new main directory.
        num_idx : int
            numerical index; will be added to the beginning of the file.

        Returns
        -------
        None.

        """
        self.savepath = (f"stat_{self.metainfo['station_id']}/" +
                         f"rx_{self.metainfo['rx_id']}/")
        filename = ('{:05d}'.format(num_idx+1) +
                    f'_{self.name}.csv')

        if not os.path.exists(save_root + self.savepath):
            os.makedirs(save_root + self.savepath)
        self.savefid = save_root + self.savepath + filename

        self._header.to_csv(self.savefid, header=None, index=None, sep=';', mode='w')
        self.data.to_csv(self.savefid, index=None, sep=';', mode='a')


# TODO double check, is this needed?
    # def plot_model(self, model, ax=None, add_bottom=50, **kwargs):

    #     if ax is None:
    #         logger.info('no instance of axis class provided - creating one...')
    #         fig, ax = plt.subplots(nrows=1, ncols=1,
    #                                figsize=(3, 8), constrained_layout=True)

    #     if model[0, 0] != 0.0:
    #         logger.warning('found thk model, converting to step model')
    #         dpths = np.r_[0, np.cumsum(model[:-1, 0])]
    #         res = model[:, 1]
    #         model = np.column_stack((dpths.repeat(2, 0)[1:],
    #                                            res.repeat(2, 0)[:-1]))

    #     if add_bottom is not None:
    #         res = np.r_[model[:, 1], model[-1, 1]]
    #         dpth = np.r_[model[:, 0], model[-1, 0]+add_bottom]

    #         ax.plot(res, dpth, **kwargs)
    #     else:
    #         ax.plot(model[:, 1], model[:, 0], **kwargs)

    #     if ax is None:
    #         return fig, ax
    #     else:
    #         return ax
