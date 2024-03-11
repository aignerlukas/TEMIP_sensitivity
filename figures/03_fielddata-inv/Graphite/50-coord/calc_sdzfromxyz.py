#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:16:33 2022

script to interpolate coordinates (add a single point) and save as x and z file

@author: laigner
"""

import os
import numpy as np
import scipy as sc
import pandas as pd

import matplotlib.pyplot as plt

from scipy.interpolate import InterpolatedUnivariateSpline
# from natsort import natsorted


# %% directions
path = "./"
filename = "20221109_TEM-p20_GKeast_dem.csv"
fin = path + filename

export_folder = "./"
if not os.path.exists(export_folder):
    os.makedirs(export_folder)

export_name_topo = 'p20_xz'

# %% read coordinate file
data_raw = pd.read_csv(fin, engine='python', header=0, sep=',', index_col=False)

data_dis = data_raw.copy()
n = len(data_dis)

# calculate slant distance from coor
data_dis['sd'] = np.zeros(n) #true slant distance
data_dis['cum_sd'] = np.zeros(n)
dx = np.asarray(data_dis.iloc[1:, 1]) - np.asarray(data_dis.iloc[:-1, 1])
dy = np.asarray(data_dis.iloc[1:, 2]) - np.asarray(data_dis.iloc[:-1, 2])
dz = np.asarray(data_dis.iloc[1:, 3]) - np.asarray(data_dis.iloc[:-1, 3])
data_dis.iloc[1:, 4] = np.sqrt(dx*dx + dy*dy + dz*dz)

data_dis.iloc[:, 5] = np.cumsum(data_dis.iloc[:, 4])


# %% export
data_export = data_dis.loc[:, ['name', 'cum_sd', 'Z']].copy()
data_export.to_csv(f'{export_name_topo}.csv', index=False, float_format='%.3f')
