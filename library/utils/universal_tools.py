# %% import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from scipy.constants import mu_0


# %% general_stuff
def round_up(x, level):
    x = int(x)
    shift = x % level
    return x if not shift else x + level - shift


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


def multipage(filename, figs=None, dpi=200):
    """
    function to save all plots to multipage pdf
    https://stackoverflow.com/questions/26368876/saving-all-open-matplotlib-figures-in-one-file-at-once

    Parameters
    ----------
    filename : string
        name of the the desired pdf file including the path and file extension.
    figs : list, optional
        list of instances of figure objects. If none automatically retrieves currently opened figs.
        The default is None.
    dpi : int, optional
        dpi of saved figures. The default is 200.

    Returns
    -------
    None.

    """
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf', dpi=dpi)
    pp.close()


def save_all(savepath, filenames,
             file_ext='.png', figs=None, dpi=150):
    """
    function to save all open figures to individual files

    Parameters
    ----------
    savepath : string
        path where the figures should be saved.
    filename : string
        name of the the desired pdf file including the file extension.
    file_ext : string, optional
        extension of the saved files. The default is '.png'.
    figs : list, optional
        list of instances of figure objects. If none automatically retrieves currently opened figs.
        The default is None.
    dpi : int, optional
        dpi of saved figures. The default is 200.

    Returns
    -------
    None.

    """
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
        print('Number of opened figs:', len(figs))
    for id, fig in enumerate(figs):
        try:
            fig.savefig(savepath + filenames[id] + file_ext, dpi=dpi)
        except IndexError:
            fig.savefig(savepath + f'unknownfig_{id}' + file_ext, dpi=dpi)


def get_temfast_date():
    """
    get current date and time and return temfast date string.
    eg. Thu Dec 30 09:34:11 2021

    Returns
    -------
    temf_datestr : str
        current date including name of day, month and adding year at the end.

    """
    import datetime
    tdy = datetime.datetime.today()
    time_fmt = ('{:02d}:{:02d}:{:02d}'.format(tdy.hour,
                                              tdy.minute,
                                              tdy.second))
    temf_datestr = ('{:s} '.format(tdy.strftime('%a')) +  # uppercase for long name of day
                    '{:s} '.format(tdy.strftime('%b')) +  # uppercase for long name of month
                    '{:d} '.format(tdy.day) +
                    '{:s} '.format(time_fmt) +
                    '{:d}'.format(tdy.year))
    return temf_datestr


def save_as_tem(savepath, template_fid,
                filename, metadata, setup_device, properties_snd,
                times, signal, error, rhoa,
                save_rhoa=True, append_to_existing=False):
    """
    function to save tem data as .tem file (TEM-FAST style)

    Parameters
    ----------
    savepath : str
        path where the .tem file should be saved.
    template_fid : str
        path to template file.
    filename : str
        name of file .
    metadata : dict
        dictionary containing all metainfos for .tem header.
    setup_device : dict
        dictionary containing all settings of the TEM-FAST device for the header.
    properties_snd : dict
        dictionary containing all sounding settings.
    times : array-like (us)
        times at which the signal was measured.
    signal : array-like (V/A)
        measured voltages normalized by the injected current.
    error : array-like (V/A)
        observed/modeled (abs.) data error .
    rhoa : array-like (Ohmm)
        apparent resisiitvity.
    save_rhoa : boolean, optional
        whether to save rhoa also to the .tem file. The default is True.
    append_to_existing : boolean, optional
        whether to append to an already existing file. The default is False.

    Returns
    -------
    None.

    """


    myCols = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"]
    template = pd.read_csv(template_fid, names=myCols,
                           sep='\\t', engine="python")

    tf_date = get_temfast_date()
    template.iat[0,1] = tf_date                          # set date
    template.iat[1,1] = f'{metadata["location"]}'             # set location
    template.iat[2,1] = metadata["snd_name"]

    template.iat[3,1] = f'{setup_device["timekey"]}'
    template.iat[3,4] = 'ramp={:.2f} us'.format(properties_snd['rampoff']*1e6)
    template.iat[3,5] = 'I={:.1f} A'.format(properties_snd['current_inj'])
    template.iat[3,6] = 'FILTR={:d} Hz'.format(setup_device['filter_powerline'])

    template.iat[4,1] = '{:.3f}'.format(setup_device['txloop'])
    template.iat[4,3] = '{:.3f}'.format(setup_device['rxloop'])

    template.iat[5,1] = metadata["comments"]

    template.iat[6,1] = '{:+.3f}'.format(metadata["x"])  # x
    template.iat[6,3] = '{:+.3f}'.format(metadata["y"])       # y
    template.iat[6,5] = '{:+.3f}'.format(metadata["z"])       # z

    template.iat[7,1] = 'Time[us]'

    chnls_act = np.arange(1, len(times)+1)
    data_norm = signal * (setup_device['txloop']**2) / setup_device['current_inj']
    err_norm = error * (setup_device['txloop']**2) / setup_device['current_inj']

    # clear data first:
    chnls_id = len(times) + 8
    template.iloc[8:, :] = np.nan

    # add new data
    template.iloc[8:chnls_id, 0] = chnls_act
    template.iloc[8:chnls_id, 1] = times*1e6  # to us
    template.iloc[8:chnls_id, 2] = data_norm
    template.iloc[8:chnls_id, 3] = abs(err_norm)

    if save_rhoa:
        template.iloc[8:chnls_id, 4] = rhoa
        exp_fmt = '%d\t%.2f\t%.5e\t%.5e\t%.2f'
        data_fid = savepath + f'{filename}.tem'
    else:
        exp_fmt = '%d\t%.2f\t%.5e\t%.5e'
        data_fid = savepath + f'{filename}.tem'

    # write to file
    data4exp = np.asarray(template.iloc[8:, :], dtype=np.float64)
    data4exp = data4exp[~np.isnan(data4exp).all(axis=1)]
    data4exp = data4exp[:, ~np.isnan(data4exp).all(axis=0)]

    if append_to_existing:
        print('saving data to: ', data_fid)
        with open(data_fid, 'a') as fid:
            fid.write('\n')  # new line
        header = template.iloc[:8, :]
        header.to_csv(data_fid, header=None,
                        index=None, sep='\t', mode='a')

        with open(data_fid, 'a') as fid:
            np.savetxt(fid, X=data4exp,
                       header='', comments='',
                       delimiter='\t', fmt=exp_fmt)

    else:
        print('saving data to: ', data_fid)
        header = template.iloc[:8, :]
        header.to_csv(data_fid, header=None,
                        index=None, sep='\t', mode='w')

        with open(data_fid, 'a') as fid:
            np.savetxt(fid, X=data4exp,
                       header='', comments='',
                       delimiter='\t', fmt=exp_fmt)

    with open(data_fid) as file: # remove trailing spaces of each line
        lines = file.readlines()
        lines_clean = [l.strip() for l in lines if l.strip()]
    with open(data_fid, "w") as f:
        f.writelines('\n'.join(lines_clean))

    return template


# %% calculations
def calc_rhoa(setup_device, signal, times):
    """

    Function that calculates the apparent resistivity of a TEM sounding
    using equation from Christiansen et al (2006)

    Parameters
    ----------
    forward : instance of forward class
        instance of wrapping class for TEM inductive loop measurements.
    signal : np.array
        signal in V/m².

    Returns
    -------
    rhoa : np.array
        apparent resistivity.

    """
    sub0 = (signal <= 0)
    turns = 1
    M = (setup_device['current_inj'] *
         setup_device['txloop']**2 * turns)
    rhoa = ((1 / np.pi) *
            (M / (20 * (abs(signal))))**(2/3) *
            (mu_0 / (times))**(5/3))
    rhoa[sub0] = rhoa[sub0]*-1
    return rhoa


def simulate_error(relerr, abserr, data):
    np.random.seed(42)
    rndm = np.random.randn(len(data))

    rand_error_abs = (relerr * np.abs(data) +
                  abserr) * rndm

    return rand_error_abs



# %% plotting
def plot_signal(axis, time, signal, log10_y=True,
                sub0color='aqua', sub0marker='s', sub0ms=6,
                sub0mfc='none', sub0label=None, **kwargs):
    sub0 = (signal <= 0)

    line, = axis.semilogx(time, abs(signal), **kwargs)

    if any(sub0):
        sub0_sig = signal[sub0]
        sub0_time = time[sub0]
        if sub0label is not None:
            line_sub0, = axis.semilogx(sub0_time, abs(sub0_sig),
                                     marker=sub0marker, ls='none',
                                     mfc=sub0mfc, ms=sub0ms,
                                     mew=1.2, mec=sub0color,
                                     label=sub0label)
        else:
            line_sub0, = axis.semilogx(sub0_time, abs(sub0_sig),
                                     marker=sub0marker, ls='none',
                                     mfc=sub0mfc, ms=sub0ms,
                                     mew=1.2, mec=sub0color)
        if log10_y:
            axis.set_yscale('log')
        return axis, line, line_sub0
    else:
        if log10_y:
            axis.set_yscale('log')
        return axis, line


def plot_rhoa(axis, time, rhoa, log10_y=True,
                sub0color='aqua', sub0marker='s', sub0label=None, **kwargs):
    sub0 = (rhoa <= 0)

    line, = axis.loglog(time, abs(rhoa), **kwargs)
    if any(sub0):
        sub0_rhoa = rhoa[sub0]
        sub0_time = time[sub0]
        if sub0label is not None:
            line_sub0, = axis.loglog(sub0_time, abs(sub0_rhoa), 's',
                                     marker=sub0marker, ls='none',
                                     mfc='none', ms=6,
                                     mew=1.2, mec=sub0color,
                                     label=sub0label)
        else:
            line_sub0, = axis.loglog(sub0_time, abs(sub0_rhoa), 's',
                                     marker=sub0marker, ls='none',
                                     mfc='none', ms=6,
                                     mew=1.2, mec=sub0color)
        if log10_y:
            axis.set_yscale('log')
        return axis, line, line_sub0
    else:
        if log10_y:
            axis.set_yscale('log')
        return axis, line
