#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: laigner
"""

# %% import of necessary modules
import os
import sys
from glob import glob

rel_path_to_libs = '../../../'
if not rel_path_to_libs in sys.path:
    sys.path.append(rel_path_to_libs)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as colors

from matplotlib import ticker
from matplotlib.patches import Polygon
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import LogLocator, LogFormatterSciNotation as LogFormatter
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mpl_toolkits.axes_grid1 import make_axes_locatable

from library.tem_tools.survey import Survey
from library.tem_tools.sounding import Sounding


# %% plot appearance
dpi = 200
plt.style.use('ggplot')

fs_shift = -4
plt.rcParams['axes.labelsize'] = 18 + fs_shift
plt.rcParams['axes.titlesize'] = 16 + fs_shift
plt.rcParams['xtick.labelsize'] = 16 + fs_shift
plt.rcParams['ytick.labelsize'] = 16 + fs_shift
plt.rcParams['legend.fontsize'] = 14 + fs_shift


# %% directions
scriptname = os.path.basename(sys.argv[0])
print(f'running {scriptname} ...')
vis_version = scriptname.split('.')[0].split('_')[-1]
inv_version = scriptname.split('_')[2].split('-')[1]

invrun = '002'
time_range = 'tr5-200us'
fname_result = f'p10_xz_{invrun}_{inv_version}'
lid = 'hb-p10_6lay-mpa'

path_main = '../../03_fielddata-inv/Graphite/'
path = path_main + f'30_TEM-results/{lid}/{inv_version}_{time_range}/'
fids = glob(path + fname_result + '.xls')

path_raw = path_main + "00_data/selected/"
fname_raw = 'hb-p10-sel.tem'

# path_xz_coord = path_main + '50-coord/'
# fname_coord = 'p10_xz.csv'

savefid = f'../plots/datafit_{lid}_{vis_version}_{inv_version}_{time_range}_invr{invrun}'


# %% read data
srvy = Survey()

srvy.parse_temfast_data(filename=fname_raw, path=path_raw)
srvy.parse_inv_results(result_folder=path, invrun=invrun)
srvy.select_soundings_by(prop='name', vals=srvy.sounding_names[:-1])
srvy.plot_datafit_multi_fig(ax=None, savefid=savefid, dpi=200,
                            nrows_to_plot=2, 
                            xlimits=(3e-6, 3e-4), ylimits=(1e-8, 1e-2))