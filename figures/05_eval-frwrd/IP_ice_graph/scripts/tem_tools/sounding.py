# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 16:52:49 2021

TODO:
    [] handle IP, noIP results, decide on format (thk, cum_thk, etc) all in one array, or separate


@author: lukas
"""

# %% modules
import os
import re
import sys
import logging

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox

from glob import glob
from matplotlib import cm
from scipy.constants import mu_0


# %% plot appearance
# mpl.rcParams['axes.labelsize'] = 28
# mpl.rcParams['axes.titlesize'] = 28
# mpl.rcParams['xtick.labelsize'] = 12
# mpl.rcParams['ytick.labelsize'] = 14
# mpl.rcParams['legend.fontsize'] = 16

# mpl.style.use('ggplot')

# plt.rcParams['axes.labelsize'] = 28
# plt.rcParams['axes.titlesize'] = 28
# plt.rcParams['xtick.labelsize'] = 14
# plt.rcParams['ytick.labelsize'] = 14
# plt.rcParams['legend.fontsize'] = 16

# plt.style.use('ggplot')



# %% Sounding class
class Sounding():
    """
    Sounding class to handle and plot TEM data
    """
    def __init__(self):
        """
        Constructor of the sounding class. Initializes instance of object with nones for the properties

        Parameters
        ----------

        Returns
        -------
        None.

        """
        self.current = None
        self.currentkey = None
        self.tx_loop = None
        self.rx_loop = None
        self.tx_area = None
        self.rx_area = None
        self.magnetic_moment = None

        self.time = None
        self.sgnl_o = None  # observed signal
        self.rhoa_o = None
        self.error_o = None
        self.metainfo = None
        self.savepath = None
        self.savefid = None

        self.sgnl_c = None  # calculated signal
        self.rhoa_c = None
        self.inv_model = None  # TODO decision on format, for now step format

        self._header = None
        self._data = None
        self._has_result = False

        # self.parse_sndng(header_type)  # data reading together with init ... makes sense?


    @staticmethod
    def get_float_from_string(string):
        import re
        numeric_const_pattern = r"""
               [-+]? # optional sign
               (?:
                  (?: \d* \. \d+ ) # .1 .12 .123 etc 9.1 etc 98.1 etc
                  |
                  (?: \d+ \.? ) # 1. 12. 123. etc 1 12 123 etc
               )
               # followed by optional exponent part if desired
               (?: [Ee] [+-]? \d+ ) ?
               """

        rx = re.compile(numeric_const_pattern, re.VERBOSE)
        numeric_list = rx.findall(string)
        if len(numeric_list) > 1:
            return np.float_(numeric_list)
        elif len(numeric_list) == 1:
            return np.float_(numeric_list)[0]
        else:
            print('error - no numerics found in string')



    @staticmethod
    def reArr_zondMdl(Mdl2_reArr):
        """
        function to rearrange model value structure in order to
        plot a step-model.
        """
        Mdl2_reArr = np.asarray(Mdl2_reArr, dtype='float')
        FileLen=len(Mdl2_reArr)
        MdlElev = np.zeros((FileLen*2,2)); r=1;k=0
        for i in range(0,FileLen*2):
            if i == FileLen:
                MdlElev[-1,1] = MdlElev[-2,1]
                break
            if i == 0:
                MdlElev[i,0] = Mdl2_reArr[i,4]
                MdlElev[i:i+2,1] = Mdl2_reArr[i,1]
            else:
                MdlElev[k+i:k+i+2,0] = -Mdl2_reArr[i,4]    # height
                MdlElev[r+i:r+i+2,1] = Mdl2_reArr[i,1]     # res
                k+=1; r+=1

        MdlElev = np.delete(MdlElev,-1,0) # delete last row!!
        return MdlElev


    @staticmethod
    def get_response(dat, idcs_resp, snd_id):
        response =  dat.loc[idcs_resp.start[snd_id]:idcs_resp.end[snd_id],
                            ['c1','c2','c3','c4','c5','c6']]
        response.columns = ['ctr','time','rho_O','rho_C','U_O','U_C']
        return response.apply(pd.to_numeric)


    @staticmethod
    def get_header(dat, idcs_hdr, snd_id):
        header = dat.loc[idcs_hdr.start[snd_id]:idcs_hdr.end[snd_id],
                         ['c1','c2','c3','c4']]
        return header


    @staticmethod
    def get_inv_model(self, dat, idcs_mdl, snd_id):
        model_df = dat.loc[idcs_mdl.start[snd_id]:idcs_mdl.end[snd_id],
                              ['c1','c2','c3','c4','c5','c6','c7','c8']]
        model_df.columns = ['ctr','Rho','Pol','Tconst','Cexpo','MagP','h','z']
        model_df.Pol /= 100  # remove scale from zondTEM chargeability

        if (model_df.loc[:, 'Pol'] == model_df.iloc[0, 2]).all():
            logging.info('IP params are all the same, removing them from df')
            model_df.drop(['Pol','Tconst','Cexpo'], axis=1, inplace=True)
            return model_df
        else:
            return model_df



    def parse_sndng(self, file_id, header_type='basic'):
        """
        method to read the data file and add sounding info to class properties

        Returns
        -------
        None.

        """

        # sep = '\\' if '\\' in file_id else '/'
        self._filename = file_id.split(f'{os.sep}')[-1]
        self._file_id = file_id
        self._file_ext = self._file_id.split('.')[-1]
        self.name = self._filename.split('.')[0]

        # ################ FAST SNAP ##########################################
        if self._file_ext == 'txt':
            logging.info('reading FASTsnap data')
            logging.info('from file: ', self.name)
            self._device = 'fast_snap'

            if header_type == 'basic':
                logging.info('reading basic header structure')
                cols = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5']
                self._header = pd.read_csv(self._file_id, sep='\s+', names=cols,
                                          engine='python', nrows=4)

                self.metainfo = {
                    'station_id':self.name.split('_')[0],
                    'rx_id':self.name.split('_')[1],
                    'snd_id':self.name.split('_')[2],
                    'tx_loop':self._header.iloc[1,0],
                    'rx_loop':self._header.iloc[2,0],
                    'offset':self._header.iloc[3,0],
                    }

                self._data = pd.read_csv(self._file_id, sep='\s+',
                                         skiprows=4)

                self.time = np.asarray(self._data.iloc[:,1]) * 1e-6  # to s
                self.sgnl = np.asarray(self._data.iloc[:,2]) * 1e-6  # to V

            elif header_type == 'extensive':
                logging.info('reading extensive header structure')
                cols = ['c0', 'c1']
                self._header = pd.read_csv(self._file_id, sep='=', names=cols,
                                           engine='python', nrows=12)

                self.metainfo = {
                    'device': self._device,
                    'station_id':int(self.name.split('_')[-4]),
                    'tx_id':int(self.name.split('_')[-3]),
                    'rx_id':int(self.name.split('_')[-2]),
                    'snd_id':int(self.name.split('_')[-1]),
                    'tx_side1':float(self._header.iloc[1,1]),
                    'tx_side2':float(self._header.iloc[2,1]),
                    'rx_loop':float(self._header.iloc[3,1]),
                    'discretization':float(self._header.iloc[4,1]),
                    'adc_quant':float(self._header.iloc[5,1]),  # in mcV
                    'gain':float(self._header.iloc[6,1]),
                    'turns':int(self._header.iloc[7,1]),
                    'current':float(self._header.iloc[8,1]),  # in A
                    'samples':int(self._header.iloc[9,1]),
                    'pulses':int(self._header.iloc[10,1]),
                    'processed':self._header.iloc[11,1]
                    }
                self.current = self.metainfo['current']
                self.rx_area = self.metainfo['rx_loop']**2
                self.tx_area = self.metainfo['tx_side1']*self.metainfo['tx_side2']
                self.magnetic_moment = (self.current *
                                        self.tx_area *
                                        self.metainfo['turns'])

                self._data = pd.read_csv(self._file_id, sep='\s+',
                                        skiprows=12)

                self.time = np.asarray(self._data.iloc[:,0]) * 1e-6  # to s
                self.sgnl = np.asarray(self._data.iloc[:,1]) * 1e-6  # to V

        # ################ TEM FAST ###########################################
        elif self._file_ext == 'tem': # TODO finish import and adding to properties
            header_type = None  # temfast hsa only one type of header
            logging.info('reading TEM fast data')
            logging.info('from file: ', self.name)
            self._device = 'tem_fast'

            headerLines = 8
            # properties = generate_metainfo('TEMfast')

            # Start of file reading
            myCols = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
            raw = pd.read_csv(self._file_id,
                              names=myCols,
                              sep='\\t',
                              engine="python")
            self.raw = raw[~pd.isnull(raw).all(1)].fillna('')
            length_data = len(self.raw.c1)

            # create start and end indices of header and data lines
            start_hdr = np.asarray(np.where(raw.loc[:]['c1'] ==
                                   'TEM-FAST 48 HPC/S2  Date:'))
            n_snds = np.size(start_hdr)
            logging.info(f'encountered {n_snds} soundings ...')

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

            if n_snds == 1:
                logging.info('reading a single sounding')
                snd_id = 0

                # select part of raw DataFrame which contains the meta info
                self._header = raw.loc[indices_hdr.start[snd_id]:indices_hdr.end[snd_id]-1]

                # create variables for legend Informations:
                self.tx_loop = float(self._header.iloc[4,1])
                self.rx_loop = float(self._header.iloc[4,3])
                self.current = self.get_float_from_string(self._header.iloc[3,5])
                self.tx_area = self.tx_loop**2
                self.rx_area = self.rx_loop**2

                self.metainfo = {
                    'device': self._device,
                    'snd_id': self._header.iloc[2,1],
                    'tx_side1': self.tx_loop,
                    'tx_side2': self.tx_loop,
                    'tx_area': self.tx_loop**2,
                    'rx_loop': self.rx_loop,
                    'rx_area': self.rx_loop**2,
                    'turns': int(self._header.iloc[4,5]),
                    'current': self.current,  # in A
                    'timekey': int(self._header.iloc[3,1]),
                    'currentkey': 4 if self.current>2.5 else 1,
                    'stacks': int(self._header.iloc[3,3]),  # in mcV
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
                elif 'M' in comments:
                    cableLenTx = comments.split('M')[0]
                    cableLenRx = cableLenTx
                elif  'm' in comments:
                    cableLenTx = comments.split('m')[0]
                    cableLenRx = cableLenTx
                else:
                    print('Warning! comments not useful')
                    print('setting to 4 * TxRx: ...')
                    cableLenTx, cableLenRx = 4*self.tx_loop, 4*self.rx_loop
                self.metainfo['tx_cable'] = cableLenTx
                self.metainfo['rx_cable'] = cableLenRx

                #select part of dataframe which contains the actual data
                data = raw.loc[indices_dat.start[snd_id]:indices_dat.end[snd_id]-1]
                data = data.drop(['c6','c7','c8'], axis=1)
                data.columns = ['channel', 'time', 'signal', 'error', 'rhoa']
                data = data.apply(pd.to_numeric)
                data.replace(0, np.nan, inplace=True)
                data.replace(99999.99, np.nan, inplace=True)

                self._data = data

                self.time = np.asarray(data.time * 1e-6)  # from us to s
                self.sgnl_o = np.asarray(data.signal * self.current / self.rx_area)  # from V/A to V/m²
                self.error_o = np.asarray(data.error * self.current / self.rx_area)  # from V/A to V/m²
                self.rhoa_o = np.asarray(data.rhoa)

        else:
            raise ValueError('not yet available, or unknown instrument')



    def parse_zond_result(self, file_id):
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

        # print(raw)
        if self._file_ext == 'xls':
            # dat = raw.drop(labels='empty', axis=1)
            dat = raw.drop(labels=0, axis=0)
        elif self._file_ext == 'csv':
            dat = raw.drop(labels=0, axis=0)
        else:
            print('no valid file ending in filename... exiting!')
            sys.exit(0)

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

        if n_snds == 1:  # if there is only one sounding in the file directly extract info and data
            self._header = self.get_header(dat, idcs_hdr, snd_id=0)

            data = self.get_response(dat, idcs_mresp, snd_id=0)
            self._data = data    #['ctr','time','rho_O','rho_C','U_O','U_C']

            self.time = np.asarray(data.time)  # already in s
            factor = 1e-6 / self.rx_area * self.current  # from uV/A to V/m²
            self.sgnl_c = np.asarray(data.U_C) * factor  # TODO no sclaing yet
            self.sgnl_o = np.asarray(data.U_O) * factor  # TODO sanity check!! ISSUE - no current and rx area available in zondTEM excel format
            self.rhoa_c = np.asarray(data.rho_C)
            self.rhoa_o = np.asarray(data.rho_O)

            self._model_df = self.get_inv_model(self, dat, idcs_mdl, snd_id=0)


    def calc_rhoa(self):

        mask = self.sgnl < 0
        time_sub0 = self.time[mask]
        sgnl_sub0 = self.sgnl[mask]


        M = self.magnetic_moment
        rhoa = ((1 / np.pi) *
                  (M / (20 * (abs(self.sgnl))))**(2/3) *
                  (mu_0 / (self.time))**(5/3)
                  )

        rhoa[mask] = rhoa[mask] * -1

        self.rhoa = rhoa
        return rhoa


    def plot_dBzdt(self, which='observed', ax=None,
                   xlimits=(1e-6, 1e-3), ylimits=(2e-10, 1e-1), sub0col='k',
                   save_fid=None, dpi=150,
                   **kwargs):
        """

        Method to plot a sounding with a predefined style.

        Parameters
        ----------
        ax : axis object, optional
            use this if you want to plot to an existing axis outside of this method.
            The default is None.
        save_fid : str, optional
            full path + filename with extension to save file.
            If none it won't be saved. The default is None.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if ax is None:
            # print('no instance of axis class provided - creating one...')
            # ncols = 1 if showRhoa is False else 2
            fig, ax = plt.subplots(nrows=1, ncols=1,
                                   figsize=(7, 7))
            # ax = ax.flat

        if which == 'observed':
            signal = self.sgnl_o
        elif which == 'calculated':
            signal = self.sgnl_c
        else:
            raise ValueError('please use either observed or calculated')

        # select neg values, to mark them explicitly within the plot
        time_sub0 = self.time[signal < 0]
        sgnl_sub0 = signal[signal < 0]

        ax.loglog(self.time, abs(signal), marker='d', ls=':',
                  lw=1, ms=5, **kwargs)
        ax.loglog(time_sub0, abs(sgnl_sub0), marker='s', ls='none',
                    markerfacecolor='none', markersize=5,
                    markeredgewidth=1, markeredgecolor=sub0col, label='negative vals')
        # ax.set_xlabel(r'time (s)')
        ax.set_ylabel(r"$\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t$ (V)")
        ax.set_xlim(xlimits[0], xlimits[1])
        ax.set_ylim(ylimits[0], ylimits[1])

        if save_fid is not None:
            plt.savefig(save_fid, dpi=dpi)

        if ax is None:
            return fig, ax
        else:
            return ax


    def plot_rhoa(self, which='observed', ax=None, log_rhoa=False,
                  xlimits=(1e-6, 1e-3), ylimits=(1e0, 1e4), sub0col='k',
                  save_fid=None, dpi=150,
                   **kwargs):
        """
        Method to plot rhoa with a predefined style.

        Parameters
        ----------
        ax : axis object, optional
            use this if you want to plot to an existing axis outside of this method.
            The default is None.
        log_rhoa : boolean
            decide to log-scale the rhoa (y) axis. The default is False
        save_fid : str, optional
            full path + filename with extension to save file.
            If none it won't be saved. The default is None.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        fig, ax or ax : figure and axis object
            figure and axis where the plot is created.

        """
        if ax is None:
            print('no instance of axis class provided - creating one...')
            # ncols = 1 if showRhoa is False else 2
            fig, ax = plt.subplots(nrows=1, ncols=1,
                                   figsize=(7, 7))
            # ax = ax.flat

        if which == 'observed':
            rhoa = self.rhoa_o
        elif which == 'calculated':
            rhoa = self.rhoa_c
        else:
            raise ValueError('please use either observed or calculated')

        if rhoa is None:
            rhoa = self.calc_rhoa()

        time_sub0 = self.time[rhoa < 0]
        rhoa_sub0 = rhoa[rhoa < 0]

        ax.semilogx(self.time, abs(rhoa), marker='d', ls=':',
                    lw=1, ms=5, **kwargs)
        ax.semilogx(time_sub0, abs(rhoa_sub0), marker='s', ls='none',
                    markerfacecolor='none', markersize=5,
                    markeredgewidth=1, markeredgecolor=sub0col, label='negative vals')
        ax.set_xlabel(r'time (s)')
        ax.set_ylabel(r'$\rho_a$ ($\Omega$m)')
        ax.set_xlim(xlimits[0], xlimits[1])
        ax.set_ylim(ylimits[0], ylimits[1])

        if log_rhoa is True:
            ax.set_yscale('log')

        if save_fid is not None:
            plt.savefig(save_fid, dpi=dpi)

        if ax is None:
            return fig, ax
        else:
            return ax


    def plot(self, which='observed', log_rhoa=True,
             xlimits=(2e-6, 2e-3),
             ylimits_dBzdt=(2e-10, 1e-1),
             ylimits_rhoa=(1e-1, 1e4),
             **kwargs):

        fig, ax = plt.subplots(nrows=1, ncols=2,
                               figsize=(14, 7))

        ax1 = self.plot_dBzdt(which, ax=ax[0],
                              xlimits=xlimits, ylimits=ylimits_dBzdt,
                              **kwargs)


        ax2 = self.plot_rhoa(which, ax=ax[1], log_rhoa=log_rhoa,
                             xlimits=xlimits, ylimits=ylimits_rhoa,
                             **kwargs)

        #         if ax is None:
        #     print('no instance of axis class provided - creating one...')
        #     # ncols = 1 if showRhoa is False else 2
        #     fig, ax = plt.subplots(nrows=1, ncols=1,
        #                            figsize=(7, 7))

        # ax = self.plot_dBzdt(ax=ax, xlimits=xlimits, ylimits=ylimits_dBzdt)

        # axR = ax.twinx()
        # axR = self.plot_rhoa(ax=axR, log_rhoa=log_rhoa,
        #                      xlimits=xlimits, ylimits=ylimits_rhoa)

        ax2.yaxis.tick_right()
        ax2.yaxis.set_label_position("right")
        plt.tight_layout()

        return fig, ax


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

