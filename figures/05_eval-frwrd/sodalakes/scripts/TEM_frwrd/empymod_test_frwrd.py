# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 18:03:31 2020

script to test empymod frwrd class

goals:
    input as lin or log
    input res or con

[] TODO streamline and format output
[] TODO check slight shift at late times - issue with filter?
[DONE] TODO res scaling and conv
[Done] TODO improve ramp output

@author: lukas
"""
import numpy as np
from empymod_frwrd import empymod_frwrd
from empymod_frwrd import empymod_frwrd
import matplotlib.pyplot as plt

plt.style.use('ggplot')

shift_sizes = 5
plt.rcParams['axes.labelsize'] = 18 - shift_sizes
plt.rcParams['axes.titlesize'] = 18 - shift_sizes
plt.rcParams['xtick.labelsize'] = 16 - shift_sizes
plt.rcParams['ytick.labelsize'] = 16 - shift_sizes
plt.rcParams['legend.fontsize'] = 18 - shift_sizes


# %% setup
device = 'TEMfast'

settings = {"timekey": 5,
            "txloop": 12.5,
            "rxloop": 12.5,
            "currentkey": 4,
            "current_inj": 4.1,
            "filter_powerline": 50}

setup_empymod = {'ft': 'dlf',                     # type of fourier trafo
                 'ftarg': 'key_201_CosSin_2012',  # ft-argument; filter type # https://empymod.readthedocs.io/en/stable/code-other.html#id12  -- for filter names      
                 'verbose': 2,                    # level of verbosity (0-4) - larger, more info
                 'srcpts': 3,                     # Approx. the finite dip. with 3 points.
                 'htarg': 'key_401_2009',         # hankel transform filter type
                 'nquad': 3,                      # Number of Gauss-Legendre points for the integration. Default is 3.
                 'cutoff_f': None,                # cut-off freq of butterworthtype filter - None: No filter applied
                 'delay_rst': 0}                  # ?? unknown para for walktem - keep at 0 for fasttem


# %% forward calc
thk = np.r_[5., 10., 0]
res = np.r_[20., 5., 50.]

# TODO test log scaling and conductivity input --> convert both to linear res for forward modeling


model1 = np.column_stack((thk, res))

forward = empymod_frwrd(setup_device=settings,
                        setup_empymod=setup_empymod,
                        filter_times=None, device='TEMfast',
                        relerr=0.0001, abserr=1e-16,
                        nlayer=3, nparam=2)
forward.calc_response(model=model1, show_wf=True)



# %% plot model1
fig, ax = plt.subplots(1, 1, figsize=(6,6))
ax.loglog(forward.times_rx, forward.response, '-ok',
          label=f'mdl-res-lin: {res}')
ax.set_xlabel("Time (s)")
ax.set_ylabel(r"$\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t$")


model2 = model1.copy()
model2[:,1] = np.log10(model2[:,1])

frwrd_log = forward.calc_response(model2, unit='res (ohmm)', scale='log10',
                                  show_wf=False)
ax.loglog(forward.times_rx, forward.response, '-xg',
          label=f'mdl-res-log: {res}')


model3 = model1.copy()
model3[:,1] = np.log10(1000 / model3[:,1])  # to 

frwrd_log = forward.calc_response(model3, unit='con (mS/m)', scale='log10',
                                  show_wf=False)
ax.loglog(forward.times_rx, forward.response, ':.r',
          label=f'mdl-con-log: {res}')

# res = [50, 50, 50]
# thk = [5, 10, 0]
# model2 = np.column_stack((thk, res))

# forward.calc_response(model2)

# plt.loglog(forward.times_rx, forward.response, '-or',
#            label='mdl-res: [50, 50, 50]')
# plt.legend(loc="best")
# plt.show()