# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 16:46:17 2020

class to create a TEMfast frwrd solution from empymod including the IP-effect

TODOs:
    TODO [DONE] finish the parser for IP models only - handle both matrix and 1D vectors (which typically come from bel1d)
    TODO [] add possibility to use different IP Models!!!
    TODO [] error handling - add error to clean synthetic data - or outside of this function?

@author: lukas
"""
# %% modules
import os
import logging

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

import empymod
from scipy.special import roots_legendre
from scipy.constants import mu_0
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline

from .utils import get_m_taur_MPA
from .utils import plot_signal
from .utils import reshape_model
from .utils import getR_fromSquare
from .utils import simulate_error

from .utils import cole_cole
from .utils import pelton_et_al
from .utils import mpa_model
from .utils import cc_con_koz

logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)
# logging.getLogger().setLevel(logging.DEBUG)
# logging.getLogger().setLevel(logging.INFO)


# %% classes
class empymod_frwrd(object):
    """
    """
    def __init__(self, setup_device, setup_solver,
                 filter_times=None, device='TEMfast',
                 relerr=1e-6, abserr=1e-28,
                 nlayer=None, nparam=2):
        """
        Constructor for the frwrd solution with simpeg.
        Parameters
        ----------
        setup_device : dictionary
            sounding setup_device of TEM device. have to fit to device.
            timekey, txloop, rxloop, current, filter_powerline
        setup_solver : dictionary
            TODO details, empymod setup_device
        device : string, optional
            Name of tem device. The default is 'TEMfast'.
        filter_times : tuple (minT, maxT) in (s), optional
            if not None it will be used to filter the times at which to
            compute the frwrd sol (minT, maxT)
        relerr : float, optional
            relative error of measurements. The default is 1e-6 (%/100).
        abserr : float, optional
            absolute error level of measurements. The default is 1e-28 (V/m²).
        nlayer : int, optional
            number of layers in the model.
        nparam : int, optional
            number of parameters in the model.

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
        self.model = None
        self.depth = None
        self.res = None
        self.response = None
        self.response_noise = None
        self.mesh = None
        self.properties_snd = None
        self.times_rx = None
        self.filter_times = None
        self.ip_modeltype = None

        if self.device == 'TEMfast':
            logging.info('Initializing TEM forward solver')
            self.create_props_temfast(show_rampInt=False)
        else:
            message = ('device/name not available ' +
                       'currently available: TEMfast' +
                       '!! forward solver not initialized !!')
            raise ValueError(message)


    def get_time(self, times, wf_times):  # move to utils!!
        """Additional time for ramp.

        Because of the arbitrary waveform, we need to compute some times before and
        after the actually wanted times for interpolation of the waveform.

        Some implementation details: The actual times here don't really matter. We
        create a vector of time.size+2, so it is similar to the input times and
        accounts that it will require a bit earlier and a bit later times. Really
        important are only the minimum and maximum times. The Fourier DLF, with
        `pts_per_dec=-1`, computes times from minimum to at least the maximum,
        where the actual spacing is defined by the filter spacing. It subsequently
        interpolates to the wanted times. Afterwards, we interpolate those again to
        compute the actual waveform response.

        Note: We could first call `waveform`, and get the actually required times
              from there. This would make this function obsolete. It would also
              avoid the double interpolation, first in `empymod.model.time` for the
              Fourier DLF with `pts_per_dec=-1`, and second in `waveform`. Doable.
              Probably not or marginally faster. And the code would become much
              less readable.

        Parameters
        ----------
        times : ndarray
            Desired times

        wf_times : ndarray
            Waveform times

        Returns
        -------
        time_req : ndarray
            Required times
        """
        tmin = np.log10(max(times.min()-wf_times.max(), 1e-10))
        tmax = np.log10(times.max()-wf_times.min())
        return np.logspace(tmin, tmax, times.size+2)


    def waveform(self, times, resp, times_wanted, wave_time, wave_amp, nquad=3):
        """Apply a source waveform to the signal.

        Parameters
        ----------
        times : ndarray
            Times of computed input response; should start before and end after
            `times_wanted`.

        resp : ndarray
            EM-response corresponding to `times`.

        times_wanted : ndarray
            Wanted times. Rx-times at which the decay is observed

        wave_time : ndarray
            Time steps of the wave. i.e current pulse

        wave_amp : ndarray
            Amplitudes of the wave corresponding to `wave_time`, usually
            in the range of [0, 1].

        nquad : int
            Number of Gauss-Legendre points for the integration. Default is 3.

        Returns
        -------
        resp_wanted : ndarray
            EM field for `times_wanted`.

        """

        # Interpolate on log.
        PP = iuSpline(np.log10(times), resp)

        # Wave time steps.
        dt = np.diff(wave_time)
        dI = np.diff(wave_amp)
        dIdt = dI/dt

        # Gauss-Legendre Quadrature; 3 is generally good enough.
        # (Roots/weights could be cached.)
        g_x, g_w = roots_legendre(nquad)

        # Pre-allocate output.
        resp_wanted = np.zeros_like(times_wanted)

        # Loop over wave segments.
        for i, cdIdt in enumerate(dIdt):

            # We only have to consider segments with a change of current.
            if cdIdt == 0.0:
                continue

            # If wanted time is before a wave element, ignore it.
            ind_a = wave_time[i] < times_wanted
            if ind_a.sum() == 0:
                continue

            # If wanted time is within a wave element, we cut the element.
            ind_b = wave_time[i+1] > times_wanted[ind_a]

            # Start and end for this wave-segment for all times.
            ta = times_wanted[ind_a]-wave_time[i]
            tb = times_wanted[ind_a]-wave_time[i+1]
            tb[ind_b] = 0.0  # Cut elements

            # Gauss-Legendre for this wave segment. See
            # https://en.wikipedia.org/wiki/Gaussian_quadrature#Change_of_interval
            # for the change of interval, which makes this a bit more complex.
            logt = np.log10(np.outer((tb-ta)/2, g_x)+(ta+tb)[:, None]/2)
            fact = (tb-ta)/2*cdIdt
            resp_wanted[ind_a] += fact*np.sum(np.array(PP(logt)*g_w), axis=1)

        return resp_wanted


    def calc_rhoa(self):
        """

        Function that calculates the apparent resistivity of a TEM sounding
        using equation from Christiansen et al (2006)

        Parameters
        ----------
        forward : instance of forward solver class
            instance of wrapping class for TEM inductive loop measurements.
        signal : np.array
            signal in V/m².

        Returns
        -------
        rhoa : np.array
            apparent resistivity.

        """
        sub0 = (self.response <= 0)
        turns = 1

        M = (self.setup_device['current_inj'] *
             self.setup_device['txloop']**2 * turns)
        self.rhoa = ((1 / np.pi) *
                     (M / (20 * (abs(self.response))))**(2/3) *
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
        timekeys = np.arange(1,10)
        all_gates = np.arange(16,52,4);
        maxTime = 2**(timekeys+5);
        analogStack = 2**(timekeys[::-1]+1)

        propDict = {"timeKeys": timekeys,
                    "maxTimes": maxTime,
                    "allGates": all_gates,
                    "analogStack": analogStack,
                    "ONOFFtimes": times_onoff,
                    "ONtimes": times_on,
                    "OFFtimes": times_off}
        properties_device = pd.DataFrame(propDict)

        # path2csv = './TEM_frwrd/TEMfast_props/'

        logging.debug('current working directory')
        logging.debug(os.getcwd())
        
        if ramp_data == 'donauinsel':
            subfolder = 'DI'
        elif ramp_data == 'salzlacken':
            subfolder = 'SL'
        else:
            raise ValueError('ramp_data not available, please choose either donauinsel or salzlacken')
        
        if not 'TEM_frwrd' in os.getcwd():
            path2csv = f'./TEM_frwrd/TEMfast_props/ramp_times_{subfolder}/'
        else:
            path2csv = f'./TEMfast_props/ramp_times_{subfolder}/'
        # path2csv = os.path.dirname(inspect.getfile(self)) + '/TEMfast_props/'

        timegates = pd.read_csv(path2csv + 'TEMfast_timegates.csv',
                                delimiter=',', skiprows=1)
        rampdata = pd.read_csv(path2csv + f'TEMfast_rampOFF_{int(current)}A.csv',
                               delimiter=',', skiprows=1)

        # convert square side length to radius of a ring loop with the same area
        r_Tx = getR_fromSquare(tx_loop) # in m

        # get timegates of tem fast device and combine with the adequate key
        gates = properties_device.allGates[properties_device.timeKeys == time_key].values[0]
        times_rx = np.asarray(timegates.centerT[0:gates] * 1e-6) # from mus to s

        # create parameters of source and waveform:
        pulse_length = properties_device.ONtimes[properties_device.timeKeys == time_key].values[0]
        pulse_length = pulse_length * 1e-3 # from ms to s
        rampf = interp1d(rampdata.side, rampdata.rampOFF,
                         kind='linear',
                         fill_value='extrapolate')
        ramp_off = rampf(tx_loop) * 1e-6 # from mus to s

        if show_rampInt:
            plt.plot(rampdata.side, rampdata.rampOFF, '--dk')
            plt.plot(tx_loop, ramp_off * 1e6, 'dr')
            plt.xlabel('square loop sizes (m)')
            plt.ylabel('turn-off ramp ($\mu$s)')
        else:
            logging.debug('You did not want to see the interp. of the turn-off ramp.')

        if self.filter_times is not None:
            logging.debug(f'about to apply given filter {self.filter_times}')
            logging.debug(f'times before filtering: {self.times_rx}')
            mask = ((times_rx > self.filter_times[0] * 1e-6) &
                    (times_rx < self.filter_times[1] * 1e-6))
            
            times_rx = times_rx[mask]
            self.times_rx = times_rx
            logging.debug(f'times after filtering: {self.times_rx}')
        else:
            self.times_rx = times_rx
            logging.info(f'times without filtering: {self.times_rx}')

        properties_snd = {"radiustx": r_Tx,
                          "timesrx": times_rx,
                          "pulsel":pulse_length,
                          "rampoff": ramp_off,
                          "current_inj": current_inj}

        self.properties_snd = properties_snd
        logging.info('DONE')

        return properties_snd


    def prep_waveform(self, show_wf=False):  # to utils also??
        ###############################################################################
        # TEM-fast Waveform and other characteristics (get from frwrd class)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        current_inj = self.properties_snd['current_inj']
        pulse_length = self.properties_snd['pulsel']
        rampOff_len = self.properties_snd['rampoff']
        off_times = self.times_rx

        rampOn_len = 3e-6
        time_cutoff = pulse_length + rampOn_len
        ramp_on = np.r_[0., rampOn_len]
        ramp_off = np.r_[time_cutoff, time_cutoff+rampOff_len]

        logging.info('current pulse:')
        logging.info('[ramp_on_min, ramp_on_max], [ramp_off_min, ramp_off_max]:\n%s, %s',
                     np.array2string(ramp_on), np.array2string(ramp_off))

        waveform_times = np.r_[-time_cutoff, -pulse_length,
                               0.000E+00, rampOff_len]
        waveform_current = np.r_[0.0, 1.0,
                                 1.0, 0.0] * current_inj
        if show_wf:
            plt.figure(figsize=(10,5))

            ax1 = plt.subplot(121)
            ax1.set_title('Waveforms - Tem-Fast system')
            ax1.plot(np.r_[-5e-3, waveform_times, 1e-3]*1e3, np.r_[0, waveform_current, 0],
                     label='current pulse')
            ax1.plot(off_times*1e3, np.full_like(off_times, 0), '|',
                     label='off time sampling')
            # ax1.xscale('symlog')
            # ax1.xlabel('Time ($\mu$s)')
            ax1.set_xlabel('Time (ms)')
            # ax1.xlim([-9, 0.5])
            ax1.legend()

            ax2 = plt.subplot(122)
            ax2.set_title('Waveforms - zoom')
            ax2.plot(np.r_[-5e-3, waveform_times, 1e-3]*1e3, np.r_[0, waveform_current, 0],
                     label='current pulse')
            ax2.plot(off_times*1e3, np.full_like(off_times, 0), '|',
                     label='off time sampling')
            ax2.set_xlim([-0.005, 0.03])
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
            in case the requested IP model is not implemented or unavailable.

        Returns
        -------
        None.

        """
        if self.model.ndim == 1:
            logging.info('found a one dimensional model:\n%s', str(self.model))
            logging.info('reshaping to [thk, res] assuming ...')
            logging.info(('thk_l_0, thk_l_1, ..., thk_l_n-1, ' +
                          'res_l_0, res_l_1, ..., res_l_n'))
            logging.info(' structure, bottom thk will be set to 0\n')

            mdlrshp = reshape_model(self.model, self.nlayer, self.nparam)
            self.model = mdlrshp
            self.depth = self.model[:,0]

            logging.info('model property:\n%s', str(self.model))

        else:
            logging.info('directly using input model')

        if self.model[-1, 0] == 0:
            logging.info('encountered thickness model - converting to layer depth ...')
            self.depth = np.cumsum(self.model[:-1, 0])  # ignore bottom thk 0, assume inf
        elif self.model[0, 0] == 0:
            logging.info('encoutered depth model - keeping layer depth ...')
            self.depth = self.model[1:, 0]
        else:
            raise ValueError('unknown geometry of model - make sure you provide either layer depth or thicknesses')

        if add_air:
            logging.info('adding air interface.')
            if ip_modeltype == 'cole_cole':
                logging.info('using the CCM.')
                eta_func = cole_cole
                self.depth = np.r_[0, self.depth]
                self.res = {'res': np.r_[2e14, self.model[:, 1]],
                            'cond_0': np.r_[2e-14, self.model[:, 2]],
                            'cond_8': np.r_[2e-14, self.model[:, 3]],
                            'tau': np.r_[0, self.model[:, 4]],
                            'c': np.r_[0, self.model[:, 5]],
                            'func_eta': eta_func}

            elif ip_modeltype == 'pelton':
                logging.info('using the PEM.')
                eta_func = pelton_et_al
                self.depth = np.r_[0, self.depth]
                self.res = {'res': np.r_[2e14, self.model[:, 1]],
                            'rho_0': np.r_[2e14, self.model[:, 1]],
                            'm': np.r_[0, self.model[:, 2]],
                            'tau': np.r_[0, self.model[:, 3]],
                            'c': np.r_[0, self.model[:, 4]],
                            'func_eta': eta_func}

            elif ip_modeltype == 'mpa':
                logging.info('using the max phase model.')
                eta_func = mpa_model
                self.depth = np.r_[0, self.depth]
                self.res = {'res': np.r_[2e14, self.model[:, 1]],
                            'rho_0': np.r_[2e14, self.model[:, 1]],
                            'phi_max': np.r_[0, self.model[:, 2]],
                            'tau_phi': np.r_[0, self.model[:, 3]],
                            'c': np.r_[0, self.model[:, 4]],
                            'func_eta': eta_func}

            elif ip_modeltype == 'cc_kozhe':
                logging.info('using the CC_Kozhevnikov model.')
                eta_func = cc_con_koz
                self.depth = np.r_[0, self.depth]
                self.res = {'res': np.r_[2e14, self.model[:, 1]],
                            'sigma_0': np.r_[2e-14, self.model[:, 2]],
                            'epsilon_s': np.r_[1.0006, self.model[:, 3]],  # add permittivity of air
                            'epsilon_8': np.r_[1.0006, self.model[:, 4]],
                            'tau': np.r_[0, self.model[:, 5]],
                            'c': np.r_[0, self.model[:, 6]],
                            'func_eta': eta_func}

            elif ip_modeltype is None:
                logging.info('solving without IP effect')
                self.depth = np.r_[0, self.depth]
                self.res = np.r_[2e14, self.model[:, 1]]

            else:
                raise TypeError('Requested IP model is not implemented.')

        else:
            logging.info('no air interface.')
            if ip_modeltype == 'cole_cole':
                logging.info('using the CCM.')
                eta_func = cole_cole
                self.res = {'res': self.model[:, 1],
                            'cond_0': self.model[:, 2],
                            'cond_8': self.model[:, 3],
                            'tau': self.model[:, 4],
                            'c': self.model[:, 5],
                            'func_eta': eta_func}

            elif ip_modeltype == 'pelton':
                logging.info('using the PEM.')
                eta_func = pelton_et_al
                self.res = {'res': self.model[:, 1],
                            'rho_0': self.model[:, 1],
                            'm': self.model[:, 2],
                            'tau': self.model[:, 3],
                            'c': self.model[:, 4],
                            'func_eta': eta_func}

            elif ip_modeltype == 'mpa':
                logging.info('using the max phase model.')
                eta_func = self.mpa_model
                self.res = {'res': np.r_[self.model[:, 1]],
                            'rho_0': np.r_[self.model[:, 1]],
                            'phi_max': np.r_[self.model[:, 2]],
                            'tau_phi': np.r_[self.model[:, 3]],
                            'c': np.r_[self.model[:, 4]],
                            'func_eta': eta_func}

            elif ip_modeltype == 'cc_kozhe':
                logging.info('using the CC_Kozhevnikov model.')
                eta_func = cc_con_koz
                self.res = {'res': np.r_[self.model[:, 1]],
                            'sigma_0': np.r_[self.model[:, 2]],
                            'epsilon_s': np.r_[self.model[:, 3]],  # add permittivity of air
                            'epsilon_8': np.r_[self.model[:, 4]],
                            'tau': np.r_[self.model[:, 5]],
                            'c': np.r_[self.model[:, 6]],
                            'func_eta': eta_func}

            elif ip_modeltype is None:
                logging.info('solving without IP effect')
                self.res = self.model[:, 1]

            else:
                raise TypeError('Requested IP model is not implemented.')

        self.ip_modeltype = ip_modeltype


    def calc_response(self, model, ip_modeltype='pelton', show_wf=False, show_em_trafos=False):
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
        depth :


        resistivity : ndarray or dict
            Resistivities of the resistivity model (see ``empymod.model.bipole``
            for more info.) or dictionary with ip model (see https://empymod.emsig.xyz/en/stable/examples/time_domain/cole_cole_ip.html#sphx-glr-examples-time-domain-cole-cole-ip-py)

        show_wf : boolean
            switch for showing the waveform of the current pulse

        Returns
        -------
        self.response : np.array
            TEM fast response (dB/dt) - in V/m².

        """

        rx_times = self.times_rx
        half_sl_side = self.setup_device['txloop'] / 2
        wf_times, wf_current = self.prep_waveform(show_wf)
        prms = self.setup_solver  # parameters for empymod

        self.model = model
        self.parse_TEM_model(add_air=True, ip_modeltype=ip_modeltype)

        logging.info('about to calculate frwrd of model (depth, res):\n(%s,\n%s) \n\n',
                     str(self.depth), str(self.res))

        # === GET REQUIRED TIMES ===
        # adds additional times to calculate
        time = self.get_time(rx_times, wf_times)
        fourier_trafo = {'time_in': time}

        # === GET REQUIRED FREQUENCIES ===
        time, freq, ft, ftarg = empymod.utils.check_time(
            time=time,                          # Required times
            signal=-1,                          # Switch-on response (1); why not switch-off in example walk tem???
            ft=prms['ft'],                      # Use DLF, if prms['ft'] = 'dlf'
            ftarg={prms['ft']: prms['ftarg']},  # fourier trafo and filter arg
            verb=prms['verbose'],               # need higher accuracy choose a longer filter.
        ) # https://empymod.emsig.xyz/en/stable/api/filters.html#module-empymod.filters  -- for filter names
        fourier_trafo.update(time_out=time, freq=freq, ft=ft, ftarg=ftarg)
        self._fourier_trafo = fourier_trafo

        # === COMPUTE FREQUENCY-DOMAIN RESPONSE ===
        src = [half_sl_side, half_sl_side,      # x0, x1
               0, half_sl_side,                 # y0, y1
               0, 0]                            # z0, z1
        # We only define a few parameters here. You could extend this for any
        # parameter possible to provide to empymod.model.bipole.
        resis_0 = self.res['res']
        EM = empymod.model.bipole(
            src=src,  # El. bipole source; half of one side.
            rec=[0, 0, 0, 0, 90],               # Receiver at the origin, vertical.
            depth=self.depth,                  # Depth-model including air layer.
            res=self.res,                       # Provided resistivity model including air.
            epermH=np.zeros_like(resis_0), epermV=np.zeros_like(resis_0),  # TODO fix all!! set electrical permittivity to zero
            # aniso=aniso,                      # Here you could implement anisotropy...
            #                                   # ...or any parameter accepted by bipole.
            freqtime=freq,                      # Required frequencies.
            mrec=True,                          # It is an el. source, but a magn. rec.
            strength=8,                         # To account for 4 sides of square loop.
            srcpts=prms['srcpts'],              # Approx. the finite dip. of the source with x points.
            recpts=prms['recpts'],              # Approx. the finite dip. of the receiver with x points.
            htarg={prms['ht']: prms['htarg']},  # filter type
            verb=prms['verbose'])
        self._hankel_trafo = EM

        # Multiply the frequency-domain result with
        # \mu for H->B, and i\omega for B->dB/dt.
        EM *= 2j*np.pi*freq*4e-7*np.pi
        self._EM_to_dBdt = np.copy(EM)

        # TODO - check this part in detail to adjust for TEM-fast!
        # === Butterworth-type filter (implemented from simpegEM1D.Waveforms.py)===
        # Note: Here we just apply one filter. But it seems that WalkTEM can apply
        #       two filters, one before and one after the so-called front gate
        #       (which might be related to ``delay_rst``, I am not sure about that
        #       part.)
        cutofffreq = prms['cutoff_f']       # As stated in the WalkTEM manual
        # begin_coff = 3e5  # original walk tem
        # begin_coff = 5e7
        if cutofffreq is not None:
            begin_coff = cutofffreq / 1.5
            h = (1+1j*freq/cutofffreq)**-1      # First order type
            self._bw_filter1 = h
            h *= (1+1j*freq/begin_coff)**-1
            self._bw_filter2 = h
            EM *= h
        self._EM_after_bw = np.copy(EM)

        # === CONVERT TO TIME DOMAIN === ?? how to do that for tem-fast data
        # delay_rst = 1.8e-7               # As stated in the WalkTEM manual
        delay_rst = prms['delay_rst']      # TODO check if 0 makes sense, some kind of offset?
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
        self.response = self.waveform(time, EM, rx_times,
                                      wf_times, wf_current, nquad=prms['nquad'])

        self.response_noise = self.response + simulate_error(self.relerr, self.abserr, self.response)

        return self.response

