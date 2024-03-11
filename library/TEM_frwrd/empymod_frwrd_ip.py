# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 16:46:17 2020

class to create a TEMfast frwrd solution from empymod including the IP-effect

TODOs:
    TODO: merge with noIP version - one for both cases
        [] mapping function update, case ip_modeltype == None


@author: lukas
"""
# %% modules
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.constants import mu_0
from scipy.interpolate import interp1d

import empymod

from .utils import reshape_model
from .utils import simulate_error
from .utils import get_time
from .utils import waveform
from .utils import get_TEMFAST_timegates
from .utils import get_TEMFAST_rampdata
from .utils import arsinh
from .utils import kth_root

from library.utils.TEM_ip_tools import cole_cole
from library.utils.TEM_ip_tools import pelton_res
from library.utils.TEM_ip_tools import mpa_model
from library.utils.TEM_ip_tools import cc_con_koz
from library.utils.TEM_ip_tools import cc_eps

from library.utils.universal_tools import plot_signal



# %% logging
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
logger.setLevel(logging.WARNING)
# logger.setLevel(logging.DEBUG)


# %% classes
class empymod_frwrd(object):
    """
    """
    def __init__(self, setup_device, setup_solver,
                 time_range=None, device='TEMfast',
                 relerr=1e-6, abserr=1e-28,
                 nlayer=None, nparam=4):
        """
        Constructor for the frwrd solution with empymod including IP effects.
        based upon:
            https://empymod.emsig.xyz/en/stable/gallery/tdomain/tem_walktem.html#sphx-glr-gallery-tdomain-tem-walktem-py
            https://empymod.emsig.xyz/en/stable/gallery/tdomain/cole_cole_ip.html#sphx-glr-gallery-tdomain-cole-cole-ip-py

        Parameters
        ----------
        setup_device : dictionary
            sounding setup_device of TEM device. have to fit to device.
            timekey, txloop, rxloop, current, 
            TODO more details!
        setup_solver : dictionary
            TODO details, empymod setup_device
        device : string, optional
            Name of tem device. The default is 'TEMfast'.
        time_range : tuple (minT, maxT) in (s), optional
            if not None it will be used to filter the times at which to
            compute the frwrd sol (minT, maxT)
        relerr : float, optional
            modelled relative error. The default is 1e-6 (%/100).
            the algorithm will return per default the clean data.
            model response with errors will be stored to self.response_noise.
        abserr : float, optional
            modelled absolute error. The default is 1e-28 (V/m²).
            the algorithm will return per default the clean data.
            model response with errors will be stored to self.response_noise.
        nlayer : int, optional
            number of layers in the model. The default is None.
            relevant if the model is given as 1D vector including all params.
            required to reshape to matrix.
        nparam : int, optional
            number of parameters in the model. The default is 4. (thk, rho, m, tau, c)
            relevant if the model is given as 1D vector including all params.
            required to reshape to matrix.

        Returns
        -------
        None.

        """

        self.setup_device = setup_device
        self.setup_solver = setup_solver
        self.relerr = relerr
        self.abserr = abserr
        self.device = device
        self.nlayer = nlayer
        self.nparam = nparam
        self.time_range = time_range

        self.model = None
        self.depth = None
        self.res = None
        self.response = None
        self.response_noise = None
        self.rhoa = None
        self.rhoa_noise = None
        self.mesh = None
        self.properties_snd = None
        self.times_rx = None
        self.ip_modeltype = None

        if self.device == 'TEMfast':
            logger.info('Initializing TEM forward solver')
            self.create_props_temfast(show_rampInt=False)
        else:
            message = ('device/name not available ' +
                       'currently available: TEMfast' +
                       '!! forward solver not initialized !!')
            raise ValueError(message)


    def calc_rhoa(self, response, turns=1):
        """
        Method that calculates the apparent resistivity of a TEM sounding
        using equation from Christiansen et al (2006)

        Parameters
        ----------
        response : array-like
            model response, TEM data in V/m².
        turns : int, optional
            number of turns in the transmitter loop. The default is 1.

        Returns
        -------
        rhoa : array-like
            apparent resistivity for response in Ohmm.

        """
        sub0 = (response <= 0)
        M = (self.setup_device['current_inj'] *
             self.setup_device['txloop']**2 * turns)
        self.rhoa = ((1 / np.pi) *
                     (M / (20 * (abs(response))))**(2/3) *
                     (mu_0 / (self.times_rx))**(5/3))
        self.rhoa[sub0] = self.rhoa[sub0]*-1
        return self.rhoa


    def create_props_temfast(self, show_rampInt=False):
        """
        This method creates the device properties of the TEMfast device.
        Necessary to calculate the forward solution. It sets the class attributes
        times_rx and properties_snd according to the selected device setup_device.

        Parameters
        ----------
        show_rampInt : boolean, optional
            To decide wether to show the ramp interpolation. The default is False.

        Returns
        -------
        properties_snd : dictionary
            {"radiustx": radius of transmitter (equal area as the square),
             "timesrx": times in (s) at which the signal should be sampled,
             "pulsel": length of the dc pulse (s),
             "rampoff": length of the turn-off ramp in (s),
             "current": injected current in (A)}.

        """

        time_key = self.setup_device["timekey"]
        tx_loop = self.setup_device["txloop"]
        current = self.setup_device["currentkey"]
        current_inj = self.setup_device["current_inj"]
        filter_powerline = self.setup_device["filter_powerline"]
        ramp_data = self.setup_device["ramp_data"]

        # input check
        available_timekeys = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        available_currents = [1, 4]
        if not time_key in available_timekeys:
            message = ("You chose a time key that is not available for the TEMfast instrument." +
                       "Please select a time key between 1 and 9." +
                       "Creation of properties not succesful!!!")
            raise ValueError(message)

        if not current in available_currents:
            message = ("You chose a current that is not available for the TEMfast instrument." +
                       "Please select either 1 or 4 A of current." +
                       "Creation of properties not succesful!!!")
            raise ValueError(message)

        if filter_powerline == 50:                                                     # in Hz
            times_onoff = np.r_[0.31,0.63,1.25,2.50,5.00,10.00,30.00,50.00,90.00]      # in milliseconds
            times_on = times_onoff / 4 * 3                                             # aka pulse lengths
            times_off = times_onoff / 4
        elif filter_powerline == 60:
            times_onoff = np.r_[0.26,0.52,1.04,2.08,4.17,8.33,25.00,41.67,75.00]       # in milliseconds
            times_on = times_onoff / 4 * 3
            times_off = times_onoff / 4
        else:
            message = ('Choose either 50 or 60 as Frequency (Hz) for the powergrid filter!' +
                       "Creation of properties not succesful!!!")
            raise ValueError(message)

        # generate properties of transmitter
        # necessary to chose adequate reciever params from gates
        timekeys = np.arange(1, 10)
        all_gates = np.arange(16, 52, 4)
        maxTime = 2**(timekeys + 5)
        analogStack = 2**(timekeys[::-1] + 1)

        propDict = {"timeKeys": timekeys,
                    "maxTimes": maxTime,
                    "allGates": all_gates,
                    "analogStack": analogStack,
                    "ONOFFtimes": times_onoff,
                    "ONtimes": times_on,
                    "OFFtimes": times_off}
        properties_device = pd.DataFrame(propDict)

        # ~~~~~~~~~~~~~~~~ read gates ~~~~~~~~~~~~~~~~~~~~~~~~
        timegates = get_TEMFAST_timegates()

        # get timegates of tem fast device and combine with the adequate key
        gates = properties_device.allGates[properties_device.timeKeys == time_key].values[0]
        times_rx = np.asarray(timegates.centerT[0:gates] * 1e-6) # from mus to s

        # create parameters of source and waveform:
        pulse_length = properties_device.ONtimes[properties_device.timeKeys == time_key].values[0]
        pulse_length = pulse_length * 1e-3 # from ms to s

        # ~~~~~~~~~~~~~~~~ ramp data ~~~~~~~~~~~~~~~~~~~~~~~~
        if isinstance(ramp_data, str) or isinstance(ramp_data, np.ndarray):
            if isinstance(ramp_data, str):
                self._rampdata_df = get_TEMFAST_rampdata(location=ramp_data,
                                                         current_key=f'{int(current)}A')
                rampdata = np.column_stack((self._rampdata_df.side, self._rampdata_df.ramp_off))

            elif isinstance(ramp_data, np.ndarray):
                rampdata = ramp_data

            rampf = interp1d(rampdata[:, 0], rampdata[:, 1],
                             kind='linear',
                             fill_value='extrapolate')
            ramp_off = rampf(tx_loop) * 1e-6 # from mus to s

            if show_rampInt:
                plt.plot(rampdata[:, 0], rampdata[:, 1], '--dk')
                plt.plot(tx_loop, ramp_off * 1e6, 'dr')
                plt.xlabel('square loop sizes (m)')
                plt.ylabel('turn-off ramp ($\mu$s)')
            else:
                logger.info('You did not want to see the interpolation of the turn-off ramp.')

        elif isinstance(ramp_data, float):
            ramp_off = ramp_data

        else:
            raise ValueError(('ramp_data needs to be either: a string (site name), ' +
                              'an array with side lengths and ramp times ' +
                              'or a float with the time for the chosen tx loop side length'))

        # ~~~~~~~~~~~~~~~~ params to dictionary ~~~~~~~~~~~~~~~~~~~~~~~~
        if self.time_range is not None:
            logger.debug(f'about to apply given filter {self.time_range}')
            logger.debug(f'times before filtering: {self.times_rx}')
            mask = ((times_rx > self.time_range[0]) &
                    (times_rx < self.time_range[1]))

            times_rx = times_rx[mask]
            self.times_rx = times_rx
            logger.debug(f'times after filtering: {self.times_rx}')
        else:
            self.times_rx = times_rx
            logger.info(f'times without filtering: {self.times_rx}')

        self.properties_snd = {"timesrx": times_rx,
                               "pulsel": pulse_length,
                               "rampoff": ramp_off,
                               "current_inj": current_inj}
        logger.info('DONE')

        return self.properties_snd


    def prep_waveform_temfast(self, show_wf=False):
        """
        method to prepare the TEM-FAST waveform, i.e., the shape of the current
        pulse.

        Parameters
        ----------
        show_wf : boolean, optional
            to decide whether or not to show a plot of the waveform. The default is False.

        Returns
        -------
        waveform_times : np.ndarray
            contains the time points in s of the waveform.
        waveform_current : np.ndarray
            contains the current in A of the waveform at the time points.

        """
        current_inj = self.properties_snd['current_inj']
        pulse_length = self.properties_snd['pulsel']
        rampOff_len = self.properties_snd['rampoff']
        off_times = self.times_rx

        rampOn_len = 3e-6
        time_cutoff = pulse_length + rampOn_len
        ramp_on = np.r_[0., rampOn_len]
        ramp_off = np.r_[time_cutoff, time_cutoff+rampOff_len]

        logger.info('current pulse:')
        logger.info('[ramp_on_min, ramp_on_max], [ramp_off_min, ramp_off_max]:\n%s, %s',
                     np.array2string(ramp_on), np.array2string(ramp_off))

        waveform_times = np.r_[-time_cutoff, -pulse_length,
                               0.000E+00, rampOff_len]
        waveform_current = np.r_[0.0, 1.0,
                                 1.0, 0.0] * current_inj

        if show_wf:
            plt.figure(figsize=(10,5))

            ax1 = plt.subplot(121)
            ax1.set_title('Waveforms - Tem-Fast system')
            ax1.plot(np.r_[-2*pulse_length, waveform_times, 1e-3]*1e3, np.r_[0, waveform_current, 0],
                     label='current pulse')
            ax1.plot(off_times*1e3, np.full_like(off_times, 0), '|',
                     label='off time sampling')
            # ax1.xscale('symlog')
            # ax1.xlabel('Time ($\mu$s)')
            ax1.set_xlabel('Time (ms)')
            ax1.set_xlim([(-1.1*pulse_length)*1e3, 1])

            ax2 = plt.subplot(122)
            ax2.set_title('Waveforms - zoom')
            ax2.plot(np.r_[-2*pulse_length, waveform_times, 1e-3]*1e6, np.r_[0, waveform_current, 0],
                     label=f'current pulse - t_r: {rampOff_len*1e6:.1e} us')
            ax2.plot(off_times*1e6, np.full_like(off_times, 0), '|',
                     label='off time sampling')
            ax2.set_xlim([-0.005*1e3, 0.03*1e3])
            ax2.set_xlabel('Time ($\mu$s)')
            ax2.legend(loc='upper right')
            plt.show()

        return waveform_times, waveform_current


    def parse_TEM_model(self, add_air=True, ip_modeltype='pelton'):
        """
        method to map an IP model to a dictionary as required for empymod

        Parameters
        ----------
        add_air : boolean, optional
            switch to decide whether to add air layer. The default is True.
        ip_modeltype : string, optional
            which ip model to use. The default is 'pelton'.

        Raises
        ------
        TypeError
            in case the requested IP model is not implemented.

        Returns
        -------
        None.

        """
        if self.model.ndim == 1:
            logger.info('found a one dimensional model:\n%s', str(self.model))
            logger.info('reshaping to [thk, res] assuming ...')
            logger.info(('thk_l_0, thk_l_1, ..., thk_l_n-1, ' +
                          'res_l_0, res_l_1, ..., res_l_n'))
            logger.info(' structure, bottom thk will be set to 0\n')

            mdlrshp = reshape_model(self.model, self.nlayer, self.nparam)
            self.model = mdlrshp
            self.depth = self.model[:,0]

            logger.info('model property:\n%s', str(self.model))

        else:
            logger.info('directly using input model')

        if self.model[-1, 0] == 0:
            logger.info('encountered thickness model - converting to layer depth ...')
            self.depth = np.cumsum(self.model[:-1, 0])  # ignore bottom thk 0, assume inf
        elif self.model[0, 0] == 0:
            logger.info('encoutered depth model - keeping layer depth ...')
            self.depth = self.model[1:, 0]
        else:
            raise ValueError('unknown geometry of model - make sure you provide either layer depth or thicknesses')

        if add_air:
            logger.info('adding air interface.')
            if ip_modeltype == 'cole_cole':
                logger.info('using the CCM.')
                eta_func = cole_cole
                self.depth = np.r_[0, self.depth]
                self.res = {'res': np.r_[2e14, self.model[:, 1]],
                            'cond_0': np.r_[2e-14, self.model[:, 2]],
                            'cond_8': np.r_[2e-14, self.model[:, 3]],
                            'tau': np.r_[1e-7, self.model[:, 4]],
                            'c': np.r_[0.01, self.model[:, 5]],
                            'func_eta': eta_func}

            elif ip_modeltype == 'pelton':
                logger.info('using the PEM.')
                eta_func = pelton_res
                self.depth = np.r_[0, self.depth]
                self.res = {'res': np.r_[2e14, self.model[:, 1]],
                            'rho_0': np.r_[2e14, self.model[:, 1]],
                            'm': np.r_[0, self.model[:, 2]],
                            'tau': np.r_[1e-7, self.model[:, 3]],
                            'c': np.r_[0.01, self.model[:, 4]],
                            'func_eta': eta_func}

            elif ip_modeltype == 'mpa':
                logger.info('using the max phase model.')
                eta_func = mpa_model
                self.depth = np.r_[0, self.depth]
                self.res = {'res': np.r_[2e14, self.model[:, 1]],
                            'rho_0': np.r_[2e14, self.model[:, 1]],
                            'phi_max': np.r_[0, self.model[:, 2]],
                            'tau_phi': np.r_[1e-7, self.model[:, 3]],
                            'c': np.r_[0.01, self.model[:, 4]],
                            'func_eta': eta_func}

            elif ip_modeltype == 'cc_kozhe':
                logger.info('using the CC_Kozhevnikov model.')
                eta_func = cc_con_koz
                self.depth = np.r_[0, self.depth]
                self.res = {'res': np.r_[2e14, self.model[:, 1]],
                            'sigma_0': np.r_[2e-14, self.model[:, 2]],
                            'epsilon_s': np.r_[1.0006, self.model[:, 3]],  # add permittivity of air
                            'epsilon_8': np.r_[1.0006, self.model[:, 4]],
                            'tau': np.r_[1e-7, self.model[:, 5]],
                            'c': np.r_[0.01, self.model[:, 6]],
                            'func_eta': eta_func}

            elif ip_modeltype == 'dielperm':
                logger.info('using the dielectric permittivity model after Mulder et al. based on Zorin and Ageev, 2017.')
                eta_func = cc_eps
                self.depth = np.r_[0, self.depth]
                self.res = {'res': np.r_[2e14, self.model[:, 1]],
                            'rho_DC': np.r_[2e14, self.model[:, 1]],
                            'epsilon_DC': np.r_[1.0006, self.model[:, 2]],
                            'epsilon_HF': np.r_[1.0006, self.model[:, 3]],
                            'tau': np.r_[1e-7, self.model[:, 4]],
                            'c': np.r_[0.01, self.model[:, 5]],
                            'func_eta': eta_func}

            elif ip_modeltype is None:
                logger.info('solving without IP effect')
                self.depth = np.r_[0, self.depth]
                self.res = np.r_[2e14, self.model[:, 1]]

            else:
                raise TypeError('Requested IP model is not implemented.')

        else:
            logger.info('no air interface.')
            if ip_modeltype == 'cole_cole':
                logger.info('using the CCM.')
                eta_func = cole_cole
                self.res = {'res': self.model[:, 1],
                            'cond_0': self.model[:, 2],
                            'cond_8': self.model[:, 3],
                            'tau': self.model[:, 4],
                            'c': self.model[:, 5],
                            'func_eta': eta_func}

            elif ip_modeltype == 'pelton':
                logger.info('using the PEM.')
                eta_func = pelton_res
                self.res = {'res': self.model[:, 1],
                            'rho_0': self.model[:, 1],
                            'm': self.model[:, 2],
                            'tau': self.model[:, 3],
                            'c': self.model[:, 4],
                            'func_eta': eta_func}

            elif ip_modeltype == 'mpa':
                logger.info('using the max phase model.')
                eta_func = self.mpa_model
                self.res = {'res': np.r_[self.model[:, 1]],
                            'rho_0': np.r_[self.model[:, 1]],
                            'phi_max': np.r_[self.model[:, 2]],
                            'tau_phi': np.r_[self.model[:, 3]],
                            'c': np.r_[self.model[:, 4]],
                            'func_eta': eta_func}

            elif ip_modeltype == 'cc_kozhe':
                logger.info('using the CC_Kozhevnikov model.')
                eta_func = cc_con_koz
                self.res = {'res': np.r_[self.model[:, 1]],
                            'sigma_0': np.r_[self.model[:, 2]],
                            'epsilon_s': np.r_[self.model[:, 3]],  # add permittivity of air
                            'epsilon_8': np.r_[self.model[:, 4]],
                            'tau': np.r_[self.model[:, 5]],
                            'c': np.r_[self.model[:, 6]],
                            'func_eta': eta_func}

            elif ip_modeltype == 'dielperm':
                logger.info('using the dielectric permittivity model after Mulder et al. based on Zorin and Ageev, 2017.')
                eta_func = cc_eps
                self.res = {'res': np.r_[self.model[:, 1]],
                            'rho_DC': np.r_[self.model[:, 1]],
                            'epsilon_DC': np.r_[self.model[:, 2]],  # add permittivity of air
                            'epsilon_HF': np.r_[self.model[:, 3]],
                            'tau': np.r_[self.model[:, 4]],
                            'c': np.r_[self.model[:, 5]],
                            'func_eta': eta_func}

            elif ip_modeltype is None:
                logger.info('solving without IP effect')
                self.res = self.model[:, 1]

            else:
                raise TypeError('Requested IP model is not implemented.')

        self.ip_modeltype = ip_modeltype


    def calc_response(self, model, add_air=True, eperm_to0=True,
                      ip_modeltype='pelton', resp_trafo=None, return_rhoa=False,
                      show_wf=False, show_em_trafos=False):
        """Custom method wrapper of empymod.model.bipole.

        Here, we compute TEM data using the ``empymod.model.bipole`` routine as
        an example. We could achieve the same using ``empymod.model.dipole`` or
        ``empymod.model.loop``.

        We model the big source square loop by computing only half of one side of
        the electric square loop and approximating the finite length dipole with 3
        point dipole sources. The result is then multiplied by 8, to account for
        all eight half-sides of the square loop.

        The implementation here assumes a central loop configuration, where the
        receiver (1 m2 area) is at the origin, and the source is a
        2*half_sl_side by 2*half_sl_side m electric loop, centered around the origin.

        Note: This approximation of only using half of one of the four sides
              obviously only works for central, horizontal square loops. If your
              loop is arbitrary rotated, then you have to model all four sides of
              the loop and sum it up.

        Parameters
        ----------
        model : np.ndarray or dict
            resistivity/thickness/depth model
            will be converted internally to empymod requirements
            (see ``empymod.model.bipole`` for more info.)
            or dictionary with IP model including the rho(w) function
            (see empymod.emsig.xyz/en/stable/examples/time_domain/cole_cole_ip.html#sphx-glr-examples-time-domain-cole-cole-ip-py)
        add_air : boolean
            switch to remove the air layer from the model space.
            The default is True
        eperm_to0 : boolean
            switch to set the electrical permittivity of the air layer to 0 
            to avoid a singularity in the numerical calculations.
            (see https://empymod.emsig.xyz/en/stable/gallery/tdomain/note_for_land_csem.html)
            The default is True.
        ip_modeltype : string
            which cole-cole like model to use for the calculation of the complex resistivity
            The default is 'pelton'.
            see https://empymod.emsig.xyz/en/stable/gallery/tdomain/cole_cole_ip.html#sphx-glr-gallery-tdomain-cole-cole-ip-py
        resp_trafo : string
            scale of the returned response (None, or log10). The default is None
        return_rhoa : boolean
            switch to decide whether the frwrd solver should return rhoa 
            (in Ohmm) instead of the voltage decay.
        show_wf : boolean
            switch to decide whether to show the current waveform.
            The default is False.
        show_em_trafos : boolean
            switch to decide whether to show the FD to TD transformation done by empymod.
            The default is False.

        Returns
        -------
        self.response : np.array
            TEM fast response (dB/dt) - in V/m².

        """

        rx_times = self.times_rx
        half_sl_side = self.setup_device['txloop'] / 2
        wf_times, wf_current = self.prep_waveform_temfast(show_wf)
        prms = self.setup_solver  # parameters for empymod

        self.model = model
        self.parse_TEM_model(add_air=add_air, ip_modeltype=ip_modeltype)

        logger.info('about to calculate frwrd of model (depth, res):\n(%s,\n%s) \n\n',
                     str(self.depth), str(self.res))

        # === GET REQUIRED TIMES ===
        # adds additional times to calculate
        time = get_time(rx_times, wf_times)
        fourier_trafo = {'time_in': time}

        # === GET REQUIRED FREQUENCIES ===
        time, freq, ft, ftarg = empymod.utils.check_time(
            time=time,                          # Required times
            signal=-1,                          # Switch-on response (1); why not switch-off in example walk tem???
            ft=prms['ft'],                      # Use DLF, if prms['ft'] = 'dlf'
            ftarg={prms['ft']: prms['ftarg']},  # filter type for fourier transformation
            verb=prms['verbose'],               # need higher accuracy choose a longer filter.
        ) # https://empymod.emsig.xyz/en/stable/api/filters.html#module-empymod.filters  -- for filter names
        fourier_trafo.update(time_out=time, freq=freq, ft=ft, ftarg=ftarg)
        self._fourier_trafo = fourier_trafo

        # === COMPUTE FREQUENCY-DOMAIN RESPONSE ===
        src = [half_sl_side, half_sl_side,      # x0, x1
               0, half_sl_side,                 # y0, y1
               0, 0]                            # z0, z1

        if eperm_to0:
            if add_air:
                epermH = np.ones_like(self.res['res'])
                epermH[0] = 0  # air layer to zero
            else:
                epermH = np.zeros_like(self.res['res'])
        else:
            epermH = np.ones_like(self.res['res'])

        # We only define a few parameters here. You could extend this for any
        # parameter possible to provide to empymod.model.bipole.
        EM = empymod.model.bipole(
            src=src,                            # El. bipole source; half of one side.
            rec=[0, 0, 0, 0, 90],               # Receiver at the origin, vertical.
            depth=self.depth,                   # Depth-model
            res=self.res,                       # Provided resistivity model (either numpy array, or dictionary with function to calculate complex resistivity)
            epermH=epermH,
            # aniso=aniso,                      # Here you could implement anisotropy...
            #                                   # ...or any parameter accepted by bipole.
            freqtime=freq,                      # Required frequencies.
            mrec=True,                          # It is an el. source, but a magn. rec.
            strength=8,                         # To account for 8 half-sides of square loop.
            srcpts=prms['srcpts'],              # Approx. the finite dip. of the source with x points.
            recpts=prms['recpts'],              # Approx. the finite dip. of the receiver with x points.
            htarg={prms['ht']: prms['htarg']},  # filter type for hankel transformation
            verb=prms['verbose'])
        self._hankel_trafo = EM

        # Multiply the frequency-domain result with
        # \mu for H->B, and i\omega for B->dB/dt.
        EM *= 2j*np.pi*freq*4e-7*np.pi
        self._EM_to_dBdt = np.copy(EM)

        # === Butterworth-type filter (implemented from simpegEM1D.Waveforms.py)===
        cutofffreq = prms['cutoff_f']       # As stated in the WalkTEM manual
        if cutofffreq is not None:
            begin_coff = cutofffreq / 1.5
            h = (1+1j*freq/cutofffreq)**-1      # First order type
            self._bw_filter1 = h
            h *= (1+1j*freq/begin_coff)**-1
            self._bw_filter2 = h
            EM *= h
        self._EM_after_bw = np.copy(EM)

        # === CONVERT TO TIME DOMAIN ===
        delay_rst = prms['delay_rst']
        EM, _ = empymod.model.tem(EM[:, None],
                                  np.array([1]),
                                  freq,
                                  time+delay_rst,
                                  1,
                                  ft,
                                  ftarg)
        EM = np.squeeze(EM)

        if show_em_trafos:
            fig, ax = plt.subplots(nrows=1, ncols=4, constrained_layout=True,
                                   figsize=(18, 6))
            ymin, ymax = 1e-20, 1e7
            # ax[0].loglog(self.times_rx, self.response, 'xk', label='response at TEM-FAST times')
            # ax[0].loglog(time, self._EM_after_tdconv, '.r-', label='times for calculation')
            # ax[0].set_xlabel("time (s)")
            # ax[0].set_ylabel(r"$\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t$ (V/m²)")
            # ax[0].set_title('TD responses')
            # ax[0].legend()

            f = self._fourier_trafo['freq']
            # ax[0].loglog(self._fourier_trafo['freq'], np.real(self._EM_to_dBdt), '.r-', label='before cutoff-filter')
            # ax[0].loglog(self._fourier_trafo['freq'], np.real(self._EM_after_bw), '+k', label='after cutoff-filter')
            plot_signal(axis=ax[0], time=f, signal=np.real(self._EM_to_dBdt), sub0color='aqua',
                        marker='.', color='r', ls='-', label='before cutoff-filter', sub0label='negative vals')
            plot_signal(axis=ax[0], time=f, signal=np.real(self._EM_after_bw), sub0color='aqua',
                        marker='+', color='k', ls='none', label='after cutoff-filter', sub0label=None)

            ax[0].set_ylabel("fd-EM' (V/m²)")
            ax[0].set_title('real part of FD response')
            ax[0].set_ylim((ymin, ymax))

            # ax[1].loglog(self._fourier_trafo['freq'], abs(np.imag(self._EM_to_dBdt)), '.r-')
            # ax[1].loglog(self._fourier_trafo['freq'], abs(np.imag(self._EM_after_bw)), '+k')
            plot_signal(axis=ax[1], time=f, signal=np.imag(self._EM_to_dBdt), sub0color='aqua',
                        marker='.', color='r', ls='-', label='before cutoff-filter', sub0label='negative vals')
            plot_signal(axis=ax[1], time=f, signal=np.imag(self._EM_after_bw), sub0color='aqua',
                        marker='+', color='k', ls='none', label='after cutoff-filter', sub0label=None)
            ax[1].set_ylabel('fd-EM" (V/m²)')
            ax[1].set_title('imag part of FD response')
            ax[1].set_ylim((ymin, ymax))

            # ax[2].loglog(self._fourier_trafo['freq'], np.abs(self._EM_to_dBdt), '.r-')
            # ax[2].loglog(self._fourier_trafo['freq'], np.abs(self._EM_after_bw), '+k')
            plot_signal(axis=ax[2], time=f, signal=np.abs(self._EM_to_dBdt), sub0color='aqua',
                        marker='.', color='r', ls='-', label='before cutoff-filter', sub0label='negative vals')
            plot_signal(axis=ax[2], time=f, signal=np.abs(self._EM_after_bw), sub0color='aqua',
                        marker='+', color='k', ls='none', label='after cutoff-filter', sub0label=None)
            ax[2].set_ylabel("|fd-EM| (V/m²)")
            ax[2].set_title('magnitude of FD response')
            ax[2].set_ylim((ymin, ymax))

            # ax[3].loglog(self._fourier_trafo['freq'], abs(np.angle(self._EM_to_dBdt)), '.r-')
            # ax[3].loglog(self._fourier_trafo['freq'], abs(np.angle(self._EM_after_bw)), '+k')
            plot_signal(axis=ax[3], time=f, signal=np.angle(self._EM_to_dBdt), sub0color='aqua',
                        marker='.', color='r', ls='-', label='before cutoff-filter', sub0label='negative vals')
            plot_signal(axis=ax[3], time=f, signal=np.angle(self._EM_after_bw), sub0color='aqua',
                        marker='+', color='k', ls='none', label='after cutoff-filter', sub0label=None)
            ax[3].set_ylabel("$\phi$ (rad)")
            ax[3].set_title('phase of FD response')
            # ax[3].set_ylim((ymin, ymax))
            ax[3].yaxis.tick_right()
            ax[3].yaxis.set_label_position("right")

            for axis in ax:
                if cutofffreq is not None:
                    axis.axvline(begin_coff, color='k', ls=':', alpha=0.5, label='bw-filter start')
                    axis.axvline(cutofffreq, color='k', ls=':', label='bw-filter end')
                axis.set_xlabel('frequency (Hz)')
            ax[0].legend()

            f2plot = f'{cutofffreq:.1e} Hz' if cutofffreq is not None else 'None'
            suptitle = (f'{self.setup_device["txloop"]:.2f} m loop, bw-filter cut-off freq: {f2plot}' +
                        f', ft: {self.setup_solver["ftarg"]}, ht: {self.setup_solver["htarg"]}')
            fig.suptitle(suptitle)

        # === APPLY WAVEFORM ===
        self.response = waveform(time, EM, rx_times,
                                 wf_times, wf_current, nquad=prms['nquad'])
        self.response_noise = self.response + simulate_error(self.relerr, self.abserr, self.response)

        self.rhoa = self.calc_rhoa(self.response)
        self.rhoa_noise = self.calc_rhoa(self.response_noise)

        if return_rhoa:
            logger.info('\nabout to rertun the apparent resistivity')
            if resp_trafo == None:
                return self.rhoa
            elif resp_trafo == 'min_to_1':
                self._raw_rhoa = self.rhoa
                self.rhoa = self.rhoa + np.min(self.rhoa) + 1
                return self.rhoa
            elif resp_trafo == 'areasinhyp':
                self._raw_rhoa = self.rhoa
                self.rhoa = arsinh(self.rhoa)
                return self.rhoa
            elif 'oddroot' in resp_trafo:
                root = int(resp_trafo.split('_')[1])
                self._raw_rhoa = self.rhoa
                self.rhoa = kth_root(self.rhoa, root)
                return self.rhoa
            else:
                raise ValueError('This response transformation is not available!')
        else:
            logger.info('\nabout to rertun the impulse response')
            if resp_trafo == None:
                return self.response
            elif resp_trafo == 'min_to_1':
                self._raw_response = self.response
                self.response = self.response + np.min(self.response) + 1
                return self.response
            elif resp_trafo == 'areasinhyp':
                self._raw_response = self.response
                self.response = arsinh(self.response)
                return self.response
            elif 'oddroot' in resp_trafo:
                root = int(resp_trafo.split('_')[1])
                self._raw_response = self.response
                self.response = kth_root(self.response, root)
                return self.response
            else:
                raise ValueError('This response transformation is not available!')



