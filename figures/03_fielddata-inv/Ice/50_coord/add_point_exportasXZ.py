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
filename = "20210716-sb_Line-GK.txt"
fin = path + filename

export_folder = "./"
if not os.path.exists(export_folder):
    os.makedirs(export_folder)


# %% read coordinate file
data = pd.read_csv(fin, engine='python', header=0, sep=',', index_col=False)
n = len(data)

# %% distances for interpol
dist_int = np.r_[0, 10, 20, 30, 37.5, 40, 50, 60, 70, 80, 90, 100]


# %% calculate distances
data['cum_td'] = np.r_[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100] #theoretical distance



# %% interpolate
new_names = [i for i in range(1, 13)]
data_int = pd.DataFrame(data=new_names, columns=['name'])

fx_i = InterpolatedUnivariateSpline(data.cum_td, data[' Easting'], k=1, ext='extrapolate')
fy_i = InterpolatedUnivariateSpline(data.cum_td, data[' Northing'], k=1, ext='extrapolate')
fz_i = InterpolatedUnivariateSpline(data.cum_td, data[' Ortho. Hight'], k=1, ext='extrapolate')

data_int['Easting'] = np.reshape(fx_i(dist_int), (len(dist_int),1))
data_int['Northing'] = np.reshape(fy_i(dist_int), (len(dist_int),1))
data_int['OrthoHeight'] = np.reshape(fz_i(dist_int), (len(dist_int),1))


# %% calculate distances
ni = len(data_int)
data_int['cum_td'] = dist_int
data_int['sd'] = np.zeros(ni) #true slant distance
data_int['cum_sd'] = np.zeros(ni)
for i in range(0,ni):
    if i != 0:
        dx = data_int.loc[i,'Easting'] - data_int.loc[i-1,'Easting']
        dy = data_int.loc[i,'Northing'] - data_int.loc[i-1,'Northing']
        dz = data_int.loc[i,'OrthoHeight'] - data_int.loc[i-1,'OrthoHeight']
        data_int.loc[i,'sd'] = np.sqrt(dx*dx + dy*dy + dz*dz)
        #data_int.loc[i,'td'] = data_int.td[i-1] + data_int.sd[i]
        data_int.loc[i,'cum_sd'] = data_int.cum_sd[i-1] + data_int.sd[i]


# %% save
# data_int_arr = np.asarray(data_int, dtype=float)
# np.savetxt(export_folder + '20210716-sb_Line-GK-int.txt', data_int_arr, delimiter=',', comments='',
#            header='sndID,x,y,z,td_cum,sd,sd_cum',
#            fmt='%02d,%10.3f,%10.3f,%6.3f,%6.2f,%.1f,%5.2f')

xz_arr = np.column_stack((data_int['name'], data_int['cum_sd'], data_int['OrthoHeight']))
np.savetxt(export_folder + '20210716-sb_Line-xz.csv', xz_arr, delimiter=',', comments='',
           header='sndID,x,y,z,td_cum,sd,sd_cum',
           fmt='%02d,%5.2f,%6.3f')

