# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 12:37:17 2018
collection of functions to parse, plot and generate a protocol from TEMfast data.
can be adapted for other devices in the future.

deprecated:
    need to be transferred to the sounding and survey classes respectively
    TODO

@author: laigner
"""

# %%
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox
import matplotlib.gridspec as gridspec
# import scipy as sc
import os

from time import strptime
from datetime import date
from matplotlib import cm
from matplotlib.lines import Line2D


# %% function library
def generate_props(device = 'TEMfast'):
    """
    Generate gate properties of a TEM-device

    This function generates the time keys, maxTime, gates and analogStack
    properties of a TEM-device. Necessary for use in parseTEMfastFile to get
    the amount of discrete data points of a TEM-measurement(gates).

    Parameters
    ----------
    device : str, optional
        name of the TEM device - only TEM-FASt available for now.
        The default is 'TEMfast'.

    Returns
    -------
    propFrame : pd.DataFrame
        contains the max time, gates and number of analogue stacks.

    """
    if device == 'TEMfast':
        #set TEM-fast measurement mode properties:
        keys = np.arange(1,10); gates = np.arange(16,52,4);
        maxTime = 2**(keys+5); pre = keys[::-1]; analogStack = 2**(pre+1)
        #put into panda DataFrame
        properties = {"maxTime": maxTime,
                      "gates": gates,
                      "analogStack": analogStack}
        propFrame = pd.DataFrame(properties, index = keys)
    else:
        print('Please enter a valid device Name!!')
    return propFrame


def parse_TEMfastFile(filename, path):
    """
    read a .tem file and return as a pd.dataframe !!Convert first!!

    Keyword arguments:
    filename
    path
    sepFiles ... whether or not you want to separate the files at export
    matlab ... let's you decide if you want to start indexing for the separated files at 0 or 1 (for Adrian)

    Parameters
    ----------
    filename : str
        name of the file.
    path : str
        path to the file.

    Returns
    -------
    rawData : TYPE
        DESCRIPTION.
    nLogs : TYPE
        DESCRIPTION.
    indices_hdr : TYPE
        DESCRIPTION.
    indices_dat : TYPE
        DESCRIPTION.

    """
    headerLines = 8
    properties = generate_props('TEMfast')

    fName = filename[:-4] if filename[-4:] == '_txt' else filename
    
    if 'selected' in path:
        fin = path +  filename
    else:
        fin = path + 'rawdata/' +  filename
    
    if 'sub' in filename:
        fin = fin.replace('rawdata/', 'subsets/')

    #Start of file reading
    cols = ["c1","c2","c3","c4","c5","c6","c7","c8"]
    rawData = pd.read_csv(fin, names=cols, sep='\\t', engine="python")
    rawData = rawData[~pd.isnull(rawData).all(1)].fillna('')
    lengthData = len(rawData.c1)

    #create start and end indices of header and data lines
    start_hdr = np.asarray(np.where(rawData.loc[:]['c1'] == 'TEM-FAST 48 HPC/S2  Date:'))
    nLogs = np.size(start_hdr)
    print(nLogs)
    start_hdr = np.reshape(start_hdr, (np.size(start_hdr),))
    end_hdr = start_hdr + headerLines

    start_dat = end_hdr
    end_dat = np.copy(start_hdr)
    end_dat = np.delete(end_dat, 0)
    end_dat = np.append(end_dat, lengthData)

    #create new dataframe which contains all indices
    indices_hdr = pd.DataFrame({'start':start_hdr, 'end':end_hdr},
                               columns=['start', 'end'])
    indices_dat = pd.DataFrame({'start':start_dat, 'end':end_dat},
                               columns=['start', 'end'])
    
    nSnds = len(indices_hdr)
    snd_names = []
    for logID in range(0,nSnds):
        hdr = rawData.loc[indices_hdr.start[logID]:indices_hdr.end[logID]]
        snd_names.append(str(hdr.iloc[2, 1]).strip())
    indices_hdr.insert(0, "sndID", snd_names)
    indices_dat.insert(0, "sndID", snd_names)
    
    indices_hdr = indices_hdr.astype(dtype={'sndID': 'string', 'start': int, 'end': int})
    indices_dat = indices_dat.astype(dtype={'sndID': 'string', 'start': int, 'end': int})

    return rawData, nLogs, indices_hdr, indices_dat


def sep_TEMfiles(filename, path, matlab=True):
    """ separate multiple soundings and save to single files
    """
    fName = filename[:-4] if filename[-4:] == '_txt' else filename
    fName = fName.replace('.tem', '')
    rawData, nLogs, indices_hdr, indices_dat = parse_TEMfastFile(filename, path)
#    print(rawData)
    singleDirectory = path + os.sep + 'singleFiles\\' + fName + '_single\\'
    if not os.path.exists(singleDirectory):
        os.makedirs(singleDirectory)

    ctr = 1 if matlab == True else 0
    for i in range(0,nLogs):
        data = rawData.loc[indices_hdr.start[i]:indices_dat.end[i]-1]
        header = rawData.loc[indices_hdr.start[i]:indices_hdr.end[i]-1]
        name = header.iloc[2][1].lstrip()

        pathExp = singleDirectory + fName + '_' + name + '_' + str(ctr) + '.tem'
        data.to_csv(pathExp, header=None, index=None, sep='\t', mode='w')
#        print(nLogs)
        ctr += 1
    return


def select_soundings(data, ids_hdr, ids_dat, sounding_names, fid):
    
    indHdr_sld = ids_hdr[ids_hdr.sndID.isin(sounding_names)].copy()
    indHdr_sld['sort_cat'] = pd.Categorical(indHdr_sld['sndID'],
                                            categories=sounding_names,
                                            ordered=True).copy()
    indHdr_sld.sort_values('sort_cat', inplace=True)

    indHdr_sld.reset_index(inplace=True)
    indHdr_sld.drop(indHdr_sld.columns[[0,-1]], axis=1, inplace=True)

    indDat_sld = ids_dat[ids_hdr.sndID.isin(sounding_names)].copy()
    indDat_sld['sort_cat'] = pd.Categorical(indDat_sld['sndID'],
                                            categories=sounding_names,
                                            ordered=True).copy()
    indDat_sld.sort_values('sort_cat', inplace=True)
    indDat_sld.reset_index(inplace=True)
    indDat_sld.drop(indDat_sld.columns[[0,-1]], axis=1, inplace=True)

    data_selected = pd.DataFrame()
    for idx in range(0, len(indHdr_sld)):
        print(indHdr_sld.start[idx], indDat_sld.end[idx]-1)
        sel = data.loc[indHdr_sld.start[idx]:indDat_sld.end[idx]-1]
        print(data)
        data_selected = data_selected.append(sel)

    # for sel_snd in sounding_names:
    #     for idx in range(0, len(ids_hdr)):
    #         curr_id = ids_hdr[idx].sndID
    #         print('current soudning id: ', curr_id)
    #         print('selection: ', sel_snd)
    #         if sel_snd == curr_id:
    #             header = 
    
    data_selected.to_csv(fid, header=None, index=None, sep='\t', mode='w')
    
    with open(fid) as file: # remove trailing spaces of each line
        lines = file.readlines()
        lines_clean = [l.strip() for l in lines if l.strip()]
    with open(fid, "w") as f:
        f.writelines('\n'.join(lines_clean))
    
    return data_selected, indHdr_sld, indDat_sld


def add_coordinates(filename, coordfile_name, path, path2coord, show_map=True):
    """
    add Coordinates to .tem file
    
    """
    fName = filename[:-4] if filename[-4:] == '_txt' else filename
    fName = fName.replace('.tem', '')
    filename_coord = fName + '_coord'
    
    if '.csv' in coordfile_name:
        sep = ','
    elif '.txt' in coordfile_name:
        sep = '\s+'
    else:
        raise TypeError('unknown file type')
    # read coordinate file
    finCoord = path2coord + coordfile_name
    raw_coord = pd.read_csv(finCoord, sep=sep, engine='python')

    rawData, nLogs, indices_hdr, indices_dat = parse_TEMfastFile(filename, path)

    # create new dataframe with coordinates
    dataCoord = rawData.copy(deep=True)
    for i in range(0,nLogs):
        header = rawData.loc[indices_hdr.start[i]:indices_hdr.end[i]-1]
        name = header.iloc[2][1].lstrip()
        print(name)

        x = float(raw_coord.loc[(raw_coord.id == name)]['x'].values)
        dataCoord.at[indices_hdr.start[i]+6, 'c2'] = str(round(x,3))

        y = float(raw_coord.loc[(raw_coord.id == name)]['y'].values)
        dataCoord.at[indices_hdr.start[i]+6, 'c4'] = str(round(y,3))

    #    z = raw_coord.loc[(raw_coord.id == name)]['z']
        z = float(raw_coord.loc[(raw_coord.id == name)]['z'].values)
        dataCoord.at[indices_hdr.start[i]+6, 'c6'] = str(round(z,2))

    # export to .tem file
    path_coord = path + os.sep + 'rawdata' + os.sep + filename_coord + '.tem'
    dataCoord.to_csv(path_coord, header=None, index=None, sep='\t', mode='w')

    if show_map == True:
        mappos = plt.figure()
        ax2 = mappos.add_subplot(111)
        ax2.plot(raw_coord.loc[:]['x'], raw_coord.loc[:]['y'], 'k-o',
                 markeredgecolor='r', markerfacecolor='w')
        k=0
        for xy in zip(raw_coord.loc[:]['x'],raw_coord.loc[:]['y']):
            text = ax2.annotate(raw_coord.loc[k]['id'],
                                xy=xy,
                                xytext=(-15,0),
                                textcoords='offset points',
                                )
            k+=1
    else:
        print('You chose not to view the map.')

    return filename_coord

def select_subset(filename, path, start_t=10, end_t=300):
    """ select a subset by giving a time range of interest
    """
    fName = filename[:-4] if filename[-4:] == '_txt' else filename
    fName = fName.replace('.tem', '')
    filename_subset = (fName + '_sub' + str(start_t) + '-' + str(end_t) + 'us')
    rawData, nLogs, indices_hdr, indices_dat = parse_TEMfastFile(filename, path)

    ind2drop_list = []
    # create dataframe with subsets... header stays the same
    data_full = rawData.copy(deep=True)
    for i in range(0,nLogs):
        data = data_full.loc[indices_dat.start[i]:indices_dat.end[i]-1]
        data_num = data.apply(pd.to_numeric)
        data_full.loc[indices_dat.start[i]:indices_dat.end[i]-1] = data_num
        ind2drop_list.append(data_num[(data_num['c2'] < start_t) | (data_num['c2'] > end_t)].index.values)

    ind2drop = np.hstack(ind2drop_list)
    data_subset = rawData.drop(ind2drop)

    # export to .tem file
    path_subset = (path + 'subsets/')
    if not os.path.exists(path_subset):
        os.makedirs(path_subset)
    data_subset.to_csv(path_subset + filename_subset + '.tem',
                       header=None, index=None, sep='\t', mode='w')
    return filename_subset

def filterbyError(filename, path, delta=0):
    """ Filter TEM data by error level
    Enter the difference between error level and measured signal to select from
    which point onward data should be aborted.

    Take care, if no data point fulfills the comparison it leads to a breakdown
    ToDo:
        -) Improve error handling!! Especially for the above mentioned case
    """
    fName = filename[:-4] if filename[-4:] == '_txt' else filename
    fName = fName.replace('.tem', '')
    filename_filtered = (fName + '_d' + str(delta))
    rawData, nLogs, indices_hdr, indices_dat = parse_TEMfastFile(filename, path)

    ind2drop_list = []
    # create dataframe with subsets... header stays the same
    data_full = rawData.copy(deep=True)
    for i in range(0,nLogs):
        data = data_full.loc[indices_dat.start[i]:indices_dat.end[i]-1]
        data = data.drop(['c6','c7','c8'], axis=1)
        data.columns = ['channel', 'time', 'signal', 'err', 'appRes']
        data_num = data.apply(pd.to_numeric)

        # calculate difference Data - Error
        diff = data_num.signal - data_num.err
        # print('Log_nID: ', i)
        # print(diff)
        # Look for points where Data - Error > Delta
        time_FirstID = data_num.time[(data_num[diff < delta]).index.values[0]]
        # print(time_FirstID)
        # print(data_num[(data_num.time > time_FirstID)])
        ind2drop_list.append(data_num[(data_num.time > time_FirstID)].index.values)

    ind2drop = np.hstack(ind2drop_list)
    # print(ind2drop)
    data_subset = rawData.drop(ind2drop)
    # export to .tem file
    path_filtered = (path + os.sep + 'rawdata' + os.sep +
                   filename_filtered + '.tem')
    data_subset.to_csv(path_filtered, header=None, index=None, sep='\t', mode='w')
    return filename_filtered

def create_minmax(filename, path):
    """ create the overall min and max values of the time, signal and res from
        a file, which contains multiple soundings.
    """

    rawData, nLogs, indices_hdr, indices_dat = parse_TEMfastFile(filename, path)
    #create array for min and max (time, signal, res)
    minmaxAll = np.zeros((nLogs,6))
    for i in range(0,nLogs):
        datFrame = datFrame = rawData.loc[indices_dat.start[i]:indices_dat.end[i]-1]
        minmaxAll[i,0] = np.min(datFrame.iloc[:,1].astype('float'))
        minmaxAll[i,1] = np.max(datFrame.iloc[:,1].astype('float'))
        minmaxAll[i,2] = np.min(datFrame.iloc[:,2].astype('float'))
        minmaxAll[i,3] = np.max(datFrame.iloc[:,2].astype('float'))
        minmaxAll[i,4] = np.min(datFrame.iloc[:,4].astype('float'))
        minmaxAll[i,5] = np.max(datFrame.iloc[:,4].astype('float'))

    T_min = min(minmaxAll[:,0])
    T_max = max(minmaxAll[:,1])
    S_min = min(minmaxAll[:,2])
    S_max = max(minmaxAll[:,3])
    R_min = min(minmaxAll[:,4])
    R_max = max(minmaxAll[:,5])

    print('The time[ms] of all soundings ranges between %.2f and %.2f.' % (T_min, T_max))
    print('The signal[mV] of all soundings ranges between %.2f and %.2f.' % (S_min, S_max))
    print('The app. resistivity[Ohmm] of all soundings ranges between %.2f and %.2f.' % (R_min, R_max))

    return nLogs, T_min, T_max, S_min, S_max, R_min, R_max

def plot_singleTEMlog(filename, path, snd_id=0,
                      tmin=2, tmax=15500,
                      Smin=10e-7, Smax=1.5,
                      Rmin=999, Rmax=999,
                      dpi=300, label_s=12,
                      log_rhoa=False, errBars=False,
                      errLine=False):
    """ plots a single TEM sounding

    Keyword arguments:
    filename
    path
    dataIndexes ... indexes where the actual data part is located
    data ... dataframe that contains all the data
    REST ... some variables to control the plot appearance outside of this function

    ToDo: add the possibility to scale also the app.Res axis to log
    """
    fName = filename[:-4] if filename[-4:] == '_txt' else filename
    fName = fName.replace('.tem', '')
    data, nLogs, indices_hdr, indices_dat = parse_TEMfastFile(filename, path)
    header = data.loc[indices_hdr.start[snd_id]:indices_hdr.end[snd_id]-1]

    #create variables for legend Informations:
    loc = header.iloc[1][1]
    name = header.iloc[2][1]
    cableLenTx = float(header.iloc[4][1])*4 / np.sqrt(float(header.iloc[4][5]))
    cableLenRx = float(header.iloc[4][3])*4 / np.sqrt(float(header.iloc[4][5]))
    TXsize = header.iloc[4][1]
    RXsize = header.iloc[4][3]
    timeKey = header.iloc[3][1]
    Stacks = header.iloc[3][3]
    Curr = header.iloc[3][5]

    #create strings where necessary
    cabLenTX_txt = '%dm TX-Cable\n' % cableLenTx
    cabLenRX_txt = '%dm RX-Cable\n' % cableLenRx
    TXsize_txt = '%.2fm TX-Loop\n' % float(TXsize)
    RXsize_txt = '%.2fm RX-Loop\n' % float(RXsize)
    time_txt = 'Time-key: %d\n' % float(timeKey)
    Stacks_txt = 'Stacks: %d\n' % float(Stacks)
    Curr_txt = 'Current: %.1f A' % float(Curr[3:6])

    #selcet part of dataframe which contains the actual data
    datFrame = data.loc[indices_dat.start[snd_id]:indices_dat.end[snd_id]-1]
    dmeas = datFrame.drop(['c6','c7','c8'], axis=1)
    dmeas.columns = ['channel', 'time', 'signal', 'err', 'appRes']
    dmeas = dmeas.apply(pd.to_numeric)
    dmeas.replace(0, np.nan, inplace=True)
    dmeas.replace(99999.99, np.nan, inplace=True)

    #select neg values, to mark them explicitly within the plot
    dS_sub0 = dmeas.loc[dmeas.signal < 0]
    daR_sub0 = dmeas.loc[dmeas.appRes < 0]

    fig, ax1 = plt.subplots()
    markSz = 2.5

    ax1.loglog(dmeas.time, abs(dmeas.signal), 'b:d',
               linewidth=0.8, markersize=markSz)
    ax1.loglog(dS_sub0.time, abs(dS_sub0.signal), 's',
               markerfacecolor='none', markersize=5,
               markeredgewidth=0.8, markeredgecolor='aqua')

    if errBars == True:
        ax1.errorbar(dmeas.time, abs(dmeas.signal), yerr=dmeas.err,
                     fmt='none', barsabove=True,
                     capsize=1.5, capthick=0.5,
                     elinewidth=0.5, ecolor='b', alpha=0.8)
    if errLine == True:
        ax1.loglog(dmeas.time, abs(dmeas.err), 'k:d',
                   linewidth=0.5, markersize=markSz*0.6,
                   alpha=0.8, label='Error level (V/A)')
        ax1.legend(loc='lower left', framealpha=1)

    ax1.set_xlabel('time ($\mu$s)')
    ax1.set_xlim([tmin,tmax])
    ax1.set_ylim([Smin,Smax])
    ax1.set_title(name + ' at ' + loc)
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel(r'$\frac{\delta B_z}{\delta t}$ (V/A)',
                   fontsize=label_s, color='b')
    ax1.tick_params('y') #, colors='b'

    textstr = (cabLenTX_txt + cabLenRX_txt + TXsize_txt + RXsize_txt +
               time_txt + Stacks_txt + Curr_txt)
    ob = offsetbox.AnchoredText(textstr, loc=1)
    ax1.add_artist(ob)

    ax2 = ax1.twinx()
    ax2.semilogx(dmeas.time, abs(dmeas.appRes), 'r--d',
                 linewidth=0.8, markersize=2.5)
    ax2.semilogx(daR_sub0.time, abs(daR_sub0.appRes), 's',
                 markerfacecolor='none', markersize=5,
                 markeredgewidth=0.8, markeredgecolor='gold')
    ax2.set_ylabel(r'$\rho_a$ ($\Omega$m)',
                   fontsize=label_s, color='r')
    if Rmin != 999 and Rmax!= 999:
        ax2.set_ylim([Rmin,Rmax])
    if log_rhoa == True:
        ax2.set_yscale('log')
    ax2.tick_params('y') #, colors='r'

    #create new directory for every file
    newDirectory = path + os.sep + 'singlePlots\\' + fName + '\\'
    if not os.path.exists(newDirectory):
        os.makedirs(newDirectory)

    pltpath = (newDirectory + fName + '_' + name +
               '_' + str(snd_id+1))

    if errBars == True:
        pltpath = pltpath + '_errB'
    if errLine == True:
        pltpath = pltpath + '_errL'

    plt.savefig(pltpath + '.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return dmeas

def plot_multiTEMlog(filename, path, minmaxALL=False,
                     tmin=2, tmax=15500,
                     Smin=10e-7, Smax=1.5,
                     Rmin=1, Rmax=1000,
                     dpi=300, label_s=12,
                     log_rhoa=False, errBars=False,
                     errLine=False):
    """ plots multiple TEM soundings to separate windows, by looping over plot_singleTEMlog
    Keyword arguments:
    filename
    path
    REST ... some variables to control the plot appearance outside of this function
             ONLY used when minmaxALL == False
    """
    nLogs, T_min, T_max, S_min, S_max, R_min, R_max = create_minmax(filename, path)
    for i in range(0,nLogs):
        if minmaxALL == True:
            plot_singleTEMlog(filename, path, snd_id=i,
                              tmin=T_min*0.95, tmax=T_max*1.05,
                              Smin=S_min*0.95, Smax=S_max*1.05,
                              Rmin=R_min*0.95, Rmax=R_max*1.05,
                              dpi=300, label_s=label_s,
                              log_rhoa=log_rhoa, errBars=errBars,
                              errLine=errLine)
        else:
            plot_singleTEMlog(filename, path, snd_id=i,
                             tmin=tmin, tmax=tmax,
                             Smin=Smin, Smax=Smax,
                             Rmin=Rmin, Rmax=Rmax,
                             dpi=300, label_s=label_s,
                             log_rhoa=log_rhoa, errBars=errBars,
                             errLine=errLine)
    return


def plot_TEMsingleFig(filename, path, minmaxALL=True,
                      tmin=2, tmax=15500,
                      Smin=10e-7, Smax=1.5,
                      Rmin=1, Rmax=1000,
                      dpi=300, ms=2.5, lw=1.5,
                      log_rhoa=False, errBars=False,
                      errLine=False, mkLeg=False,
                      lg_cols=1):
    """ plots all soundings from one file into a single Fig
    this routine plots all soundings from a single .tem file in two separate
    figures. One for the signal and another one for the apparent Resistivity.

    Keyword arguments:
    filename
    path
    REST ... some variables to control the plot appearance outside of this function

    ToDo: add the possibility to scale also the app.Res axis to log
          add both the signal and the appRes together into two subplots of a single Fig
    """
    fName = filename[:-4] if filename[-4:] == '_txt' else filename
    fName = fName.replace('.tem', '')
    data, nLogs, indices_hdr, indices_dat = parse_TEMfastFile(filename, path)

    if minmaxALL == True:
        (nLogs,
         T_min, T_max,
         S_min, S_max,
         R_min, R_max) = create_minmax(filename, path)

        tmin, tmax = T_min*0.95, T_max*1.05
        Smin, Smax = S_min*0.95, S_max*1.05
        Rmin, Rmax = R_min*0.95, R_max*1.05

    start = 0.0; stop = 1.0
    cm_subsection = np.linspace(start, stop, nLogs)
    colors = [cm.jet(x) for x in cm_subsection]

    fig1 = plt.figure(figsize=(10, 9)) #figSize in inch
    ax1 = fig1.add_subplot()
    fig2 = plt.figure(figsize=(10, 9)) #figSize in inch
    ax2 = fig2.add_subplot()


    for snd_id in range(0,nLogs):
        datFrame = data.loc[indices_dat.start[snd_id]:indices_dat.end[snd_id]-1]
        dmeas = datFrame.drop(['c6','c7','c8'], axis=1)
        dmeas.columns = ['channel', 'time', 'signal', 'err', 'appRes']
        dmeas = dmeas.apply(pd.to_numeric)
        dmeas.replace(0, np.nan, inplace=True)
        dmeas.replace(99999.99, np.nan, inplace=True)

        header = data.loc[indices_hdr.start[snd_id]:indices_hdr.end[snd_id]-1]
        #create variables for legend Informations:
        loc = header.iloc[1,1]
        name = header.iloc[2,1]
        cableLenTx = float(header.iloc[4,1])*4 / np.sqrt(float(header.iloc[4,5]))
        cableLenRx = float(header.iloc[4,3])*4 / np.sqrt(float(header.iloc[4,5]))
        TXsize = get_float_from_string(header.iloc[4,1])
        RXsize = get_float_from_string(header.iloc[4,3])
        turns = int(header.iloc[4,5])
        timeKey = header.iloc[3,1]
        stacks = header.iloc[3,3]
        curr = header.iloc[3,5]
        posX = header.iloc[6,1]
        posZ = header.iloc[6,3]
        posY = header.iloc[6,5]

        #create strings where necessary
        cabLenTX_txt = '%dm TX-Cable\n' % cableLenTx
        cabLenRX_txt = '%dm RX-Cable\n' % cableLenRx
        TXsize_txt = '%.2f m TX-Loop\n' % float(TXsize)
        RXsize_txt = '%.2f m RX-Loop\n' % float(RXsize)
        time_txt = 'Time-key: %d\n' % float(timeKey)
        stacks_txt = 'stacks: %d\n' % float(stacks)
        curr_txt = 'Current: %.1f A' % float(curr[3:6])

        #select neg values, to mark them explicitly within the plot
        dS_sub0 = dmeas.loc[dmeas.signal < 0]
        daR_sub0 = dmeas.loc[dmeas.appRes < 0]
        
        if TXsize == RXsize:
            label_str = f'{name}, {timeKey}, {stacks}, {curr[3:8]}, {int(TXsize)} m, {turns}, {posX}'
            header_str = 'Symbol, SndID, TimeKey, Stacks, Current, Tx, turns, posX'
        else:
            label_str = f'{name}, {timeKey}, {stacks}, {curr[3:8]}, {TXsize} m, {RXsize} m, {turns}, {posX}'
            header_str = 'Symbol, SndID, TimeKey, Stacks, Current, Tx, Rx, turns, posX'

        ax1.loglog(dmeas.time, abs(dmeas.signal),
                   color=colors[snd_id], marker='d',
                   linewidth=lw, markersize=ms,
                   label=label_str)
        ax1.loglog(dS_sub0.time, abs(dS_sub0.signal), 's',
                   markerfacecolor='none', ms=ms*1.8,
                   markeredgewidth=0.8, markeredgecolor='black')

        if errBars == True:
            ax1.errorbar(dmeas.time, abs(dmeas.signal), yerr=dmeas.err,
                         fmt='none', barsabove=True,
                         capsize=1.5, capthick=0.5,
                         elinewidth=0.5, alpha=0.8)
        if errLine == True:
            ax1.loglog(dmeas.time, abs(dmeas.err), ':d',
                       linewidth=0.5, ms=ms*0.6,
                       color=colors[snd_id], alpha=0.5)
                       # label=f'Err.lvl [V/A] - {name}')

        ax1.set_xlabel('time ($\mu$s)')
        ax1.set_ylabel(r"$\mathrm{d}\mathrm{B}_\mathrm{z}\,/\,\mathrm{d}t$ (V/A)")
        ax1.set_xlim([tmin,tmax])
        ax1.set_ylim([Smin,Smax])
        ax1.tick_params(axis='both', which='major')
        ax1.grid(True, which='Major', linestyle='-')
        ax1.grid(True, which='Minor', linestyle=':')

        ax2.semilogx(dmeas.time, abs(dmeas.appRes),
                     color=colors[snd_id], marker='d',
                     linewidth=lw, markersize=ms,
                     label=label_str)
        ax2.semilogx(daR_sub0.time, abs(daR_sub0.appRes), 's',
                     markerfacecolor='none', markersize=ms*1.8,
                     markeredgewidth=1, markeredgecolor='black')
        ax2.set_xlabel('time ($\mu$s)')
        ax2.set_ylabel(r'$\rho_a$ ($\Omega$m)')
        ax2.set_xlim([tmin,tmax])
        ax2.tick_params(axis='both', which='major')
        if Rmin != Rmax:
            ax2.set_ylim((Rmin, Rmax))
        if log_rhoa == True:
            ax2.set_yscale('log')
        ax2.grid(True, which='Major', linestyle='-')
        ax2.grid(True, which='Minor', linestyle=':')

    if mkLeg == True:
        leg_a = ax1.legend(bbox_to_anchor=(1.05, 1),loc='upper left',
                           framealpha=1, title_fontsize=10, ncol=lg_cols,
                           title=header_str)
        ax2.legend(bbox_to_anchor=(1.05, 1),loc='upper left',
                   framealpha=1, title_fontsize=10, ncol=lg_cols,
                   title=header_str)
        if errLine == True:
            leg_errLines = [Line2D([0], [0], color=colors[0], lw=1, ls=':',
                                   marker='d', ms=ms*0.6, alpha=0.5),
                            Line2D([0], [0], color=colors[-1], lw=1, ls=':',
                                   marker='d', ms=ms*0.6, alpha=0.5)]
            leg_b = ax1.legend(leg_errLines,
                               ['Err.lvl (V/A) 1st sounding',
                                'Err.lvl (V/A) last sounding'],
                               loc='lower left')
            ax1.add_artist(leg_a)

    pltpath1 = path + os.sep + fName + '_allin1_sig'
    pltpath2 = path + os.sep + fName + '_allin1_appR'
    if errBars == True:
        pltpath1 = pltpath1 + '_errB'
    if errLine == True:
        pltpath1 = pltpath1 + '_errL'

    fig1.savefig(pltpath1 + '.png', dpi=dpi, bbox_inches='tight')
    fig2.savefig(pltpath2 + '.png', dpi=dpi, bbox_inches='tight')
    return



def generate_TEMprotocol(filename, path, sepFiles=True, matlab=True):
    """ Generate the protocol for a .tem file
    this routine uses the parse_TEMfastfile routine and generates a protocol of
    the measurements. It uses an array pos, which contains the positions of the
    parameters within the dataframe.

    ToDo: change to iloc notation instead of . notation?

    Keyword arguments:
    filename
    path
    sepFiles ... whether or not you want to separate the files at export
    matlab ... let's you decide if you want to start indexing for the separated files at 0 or 1 (for Adrian)
    """

    fName = filename[:-4] if filename[-4:] == '_txt' else filename
    fName = fName.replace('.tem', '')
    data, nLogs, indices_hdr, indices_dat = parse_TEMfastFile(filename, path)
    first_header = data.loc[indices_hdr.start[0]:indices_hdr.end[0]-1]

    # create fitting date format for title
    dt_raw = first_header.c2[0].split()
    d = date(int(dt_raw[4]), strptime(dt_raw[1],'%b').tm_mon, int(dt_raw[2]))
    protName = (path + os.sep + fName + '_' + d.strftime("%Y%m%d") + '_prot.txt')


    # create Protocol:
    ctr = 1 if matlab == True else 0

    with open(protName, 'w') as new_prot:
        new_prot.write('!!! Attention this is a automatically generated protocol; please check with field notes !!!'
                       + '\n!!! When you are done please remove these lines !!!'
                       + '\nField Measurements Protocol - TEM\n'
                       + '\nLocation: ' + first_header.c2[1]
                       + '\nDate: ' + d.strftime('%Y-%m-%d')
                       + '\nDevice: TEM-fast'
                       + '\n\n==========================================================='
                       + '\nFilename: ' + filename + '.tem ; ==> Profile type = single'
                       + '\nid name X Y Z cable Tx Rx turns Time Stacks Filter Current Comments\n')

    with open(protName, 'a') as app2prot:
        for log_id in range(0,nLogs):
            header = first_header = data.loc[indices_hdr.start[log_id]:indices_hdr.end[log_id]-1]
            cableLenTx = float(header.iloc[4][1])*4 / np.sqrt(float(header.iloc[4][5]))
            cableLenRx = float(header.iloc[4][3])*4 / np.sqrt(float(header.iloc[4][5]))
            app2prot.write(str(ctr) + ' '
                           + header.iloc[2][1] + ' '
                           + header.iloc[6][1] + ' '
                           + header.iloc[6][3] + ' '
                           + header.iloc[6][5] + ' '
                           + str(cableLenTx) + ' '
                           + str(cableLenRx) + ' '
                           + header.iloc[4][1] + ' ' + header.iloc[4][3] + ' '   #Tx and Rx
                           + header.iloc[4][5] + ' ' + header.iloc[3][1] + ' ' + header.iloc[3][3]
                           + ' ' + header.iloc[3][6] + ' ' + header.iloc[3][5] + '\n')
            ctr += 1
    return data, protName


def get_float_from_string(string):
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


def round_up(x, level):
    x = int(x)
    shift = x % level
    return x if not shift else x + level - shift