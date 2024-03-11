"""

"""

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import linecache as lc
import glob
import os

from decimal import Decimal
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

# import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# import matplotlib


def get_PatchCollection(elemx, elemz, colormap='jet_r', log10=False, zorder=0):
    """Script to derive polygon patches collections out of grid information and
    data values; Updated version to enable easy log10 scaling!!

    Author: Jakob Gallistl
    """
    import matplotlib.colors as colors

    # size of elemx
    a, b = np.shape(elemx)
    # patches list
    patches = []

    # loop to compute polygon patches
    for elem in np.arange(b):
        elx = np.expand_dims(elemx[:, elem], axis=1)
        elz = np.expand_dims(elemz[:, elem], axis=1)

        v = np.concatenate((elx, elz), axis=1)

        poly = Polygon(v)

        patches.append(poly)

    norm = colors.Normalize() if (log10 == False) else colors.LogNorm()

    p = PatchCollection(patches,
                        edgecolors='None',
                        linewidth=0,
                        zorder=zorder,
                        cmap=colormap, #default is reversed jetMap
                        norm=norm)     #default is linear norm
    return p


def parse_inv(path_inv,
              profile_length,
              sens_blank=None,
              plot_stats=True):
    """
    @author: jakob

    Reads CRTomo inversion files and outputs models to be plotted, as well as
    statistics and parameters of the inversion.
    """

    fig, axes = plt.subplots(nrows=2,
                             ncols=3,
                             sharex='all',
                             figsize=(7.4, 3.5))
    ax1 = axes.flat[0]
    ax2 = axes.flat[1]
    ax3 = axes.flat[2]
    ax4 = axes.flat[3]
    ax5 = axes.flat[4]
    ax6 = axes.flat[5]
    label_s = 7
    ms = 5

    # read inversion parameters
    # smoothing
    print(path_inv + os.sep + 'inv.ctr')
    print('test')
    print(lc.getline(path_inv + os.sep + 'inv.ctr', 24))
    
    
    xsmooth = lc.getline(path_inv + os.sep + 'inv.ctr', 24).split()[0]
    ysmooth = lc.getline(path_inv + os.sep + 'inv.ctr', 25).split()[0]

    # DC, robust, final phase?
    dc_inv = lc.getline(path_inv + os.sep + 'inv.ctr', 27).split()[0]
    robust_inv = lc.getline(path_inv + os.sep + 'inv.ctr', 28).split()[0]
    fpi_inv = lc.getline(path_inv + os.sep + 'inv.ctr', 29).split()[0]

    if dc_inv is 'F':
        inv_type = 'complex'
    else:
        inv_type = 'DC'

    # error parameterization
    a_res = lc.getline(path_inv + os.sep + 'inv.ctr', 30).split()[0]
    b_res = lc.getline(path_inv + os.sep + 'inv.ctr', 31).split()[0]
    a_ph = lc.getline(path_inv + os.sep + 'inv.ctr', 32).split()[0]
    b_ph = lc.getline(path_inv + os.sep + 'inv.ctr', 33).split()[0]
    c_ph = lc.getline(path_inv + os.sep + 'inv.ctr', 34).split()[0]
    d_ph = lc.getline(path_inv + os.sep + 'inv.ctr', 35).split()[0]

    if dc_inv == 'T':
        inv_label = 'DC inversion'
    else:
        inv_label = 'complex inversion'
    if robust_inv == 'T':
        rob_label = 'robust'
    else:
        rob_label = 'non-robust'
    if fpi_inv == 'T':
        fpi_label = 'FPI'
    else:
        fpi_label = 'non-FPI'

    xsmoothm = Decimal(xsmooth).normalize()
    ysmoothm = Decimal(ysmooth).normalize()

    a_resm = Decimal(a_res).normalize()
    b_resm = Decimal(b_res).normalize()
    a_phm = Decimal(a_ph).normalize()
    b_phm = Decimal(b_ph).normalize()
    c_phm = Decimal(c_ph).normalize()
    d_phm = Decimal(d_ph).normalize()

    inv_params = [xsmoothm,
                  ysmoothm,
                  inv_label,
                  rob_label,
                  fpi_label,
                  a_resm,
                  b_resm,
                  a_phm,
                  b_phm,
                  c_phm,
                  d_phm]
    
    # plot starting values
    if "mgs" in path_inv:
        print("encountered an mgs inversion...")
        l_frags = lc.getline(path_inv + os.sep + 'inv.ctr', 116).split()
        r_start = 117
        if len(l_frags) == 1:
            l_frags = lc.getline(path_inv + os.sep + 'inv.ctr', 115).split()
            r_start = 116
    else:
        print("encountered standard inversion...")
        l_frags = lc.getline(path_inv + os.sep + 'inv.ctr', 114).split()
        r_start = 115
        if len(l_frags) == 1:
            l_frags = lc.getline(path_inv + os.sep + 'inv.ctr', 113).split()
            r_start = 114

    ax4.plot(float(l_frags[1]),
             float(l_frags[2]),
             'b^',
             markersize=ms)
    ax4.set_ylabel('data RMS', fontsize=label_s)
    ax4.grid()
    ax5.plot(float(l_frags[1]),
             float(l_frags[3]),
             'g^',
             markersize=ms)
    ax5.set_ylabel('mag RMS', fontsize=label_s)
    ax5.grid()
    ax6.plot(float(l_frags[1]),
             float(l_frags[4]),
             'r^',
             markersize=ms)
    ax6.set_ylabel('pha RMS', fontsize=label_s)
    ax6.grid()

    # read iterations
    fin_it = 0
    for line in range(1000):

        l_frags = lc.getline(path_inv + os.sep +
                             'inv.ctr', r_start+line).split()

        if (len(l_frags) is not 0) and (l_frags[0] == 'IT'):

            ax1.semilogy(float(l_frags[1]),
                         float(l_frags[3]),
                         'mo',
                         markersize=ms)
            ax2.plot(float(l_frags[1]),
                     float(l_frags[4]),
                     'co',
                     markersize=ms)
            ax3.plot(float(l_frags[1]),
                     float(l_frags[5]),
                     'ko',
                     markersize=ms)
            ax4.plot(float(l_frags[1]),
                     float(l_frags[2]),
                     'bo',
                     markersize=ms)
            ax5.plot(float(l_frags[1]),
                     float(l_frags[7]),
                     'go',
                     markersize=ms)
            ax6.plot(float(l_frags[1]),
                     float(l_frags[8]),
                     'ro',
                     markersize=ms)
            fin_it = float(l_frags[1])

        elif len(l_frags) is not 0 and l_frags[0] == 'PIT':

            ax1.semilogy(float(l_frags[1]),
                         float(l_frags[3]),
                         'm^',
                         markersize=ms)
            ax2.plot(float(l_frags[1]),
                     float(l_frags[4]),
                     'c^',
                     markersize=ms)
            ax3.plot(float(l_frags[1]),
                     float(l_frags[5]),
                     'k^',
                     markersize=ms)
            ax4.plot(float(l_frags[1]),
                     float(l_frags[2]),
                     'b^',
                     markersize=ms)
            ax5.plot(float(l_frags[1]),
                     float(l_frags[7]),
                     'g^',
                     markersize=ms)
            ax6.plot(float(l_frags[1]),
                     float(l_frags[8]),
                     'r^',
                     markersize=ms)
            fin_it = float(l_frags[1])

    ax1.set_ylabel('stepsize', fontsize=label_s)
    ax1.grid()
    ax2.set_ylabel('lambda', fontsize=label_s)
    ax2.grid()
    ax3.set_ylabel('roughness', fontsize=label_s)
    ax3.grid()

    ax4.set_xlabel('Iteration', fontsize=label_s)
    ax5.set_xlabel('Iteration', fontsize=label_s)
    ax6.set_xlabel('Iteration', fontsize=label_s)

    ax1.set_xlim([-0.3, fin_it + 0.3])
    ax2.set_xlim([-0.3, fin_it + 0.3])
    ax3.set_xlim([-0.3, fin_it + 0.3])

    ax1.tick_params(axis='both', which='both', labelsize=label_s)
    ax2.tick_params(axis='both', which='both', labelsize=label_s)
    ax3.tick_params(axis='both', which='both', labelsize=label_s)
    ax4.tick_params(axis='both', which='both', labelsize=label_s)
    ax5.tick_params(axis='both', which='both', labelsize=label_s)
    ax6.tick_params(axis='both', which='both', labelsize=label_s)

    xspace = np.arange(0, fin_it + 1)
    yspace = np.ones(len(xspace))

    ax4.plot(xspace, yspace, '--k')
    ax5.plot(xspace, yspace, '--k')
    ax6.plot(xspace, yspace, '--k')

    sup_string_1 = ('%d:%d, %s, %s, %s, a_res: %s, b_res: %s, a_ph: %s' %
                    (float(xsmooth),
                     float(ysmooth),
                     inv_label,
                     rob_label,
                     fpi_label,
                     Decimal(a_res).normalize(),
                     Decimal(b_res).normalize(),
                     Decimal(a_ph).normalize()))

    sup_string_2 = (', b_ph: %s, c_ph: %s, d_ph: %s' %
                    (Decimal(b_ph).normalize(),
                     Decimal(c_ph).normalize(),
                     Decimal(d_ph).normalize()))

    fig.suptitle(sup_string_1 + sup_string_2,
                 y=1.02,
                 fontsize=label_s)

    plt.tight_layout()
    if plot_stats is True:
        inv_num = path_inv.split(os.sep)[-1]
        path_plot = os.sep.join([part for
                                 part in path_inv.split(os.sep)[:-1]])
        fig.savefig(path_plot + os.sep + inv_num + '_inv_stat.tiff',
                    bbox_inches='tight',
                    dpi=300)
        plt.close(fig)
    else:
        plt.close(fig)

    # distinguish between complex and DC inversion
    if inv_type is 'complex':
        print('Reading complex inversion results for:')
        print(path_inv)

        # number of iterations
        numit = len(glob.glob(path_inv + '/*.pha'))-1

        # read the inversion results
        if numit > 9:
            if os.path.exists(path_inv + '/rho' +
                              str(numit) + '.pha'):
                pha = np.genfromtxt(path_inv + '/rho' +
                                    str(numit) + '.pha',
                                    skip_header=1)
                rho = np.genfromtxt(path_inv + '/rho' +
                                    str(numit) + '.mag',
                                    skip_header=1)
                if os.path.exists(path_inv + '/coverage.mag'):
                    cov = np.genfromtxt(path_inv + '/coverage.mag',
                                        skip_header=1,
                                        skip_footer=1)
                else:
                    print('Sensitivity model not found:')
                    print(path_inv)
        else:
            if os.path.exists(path_inv + '/rho0' +
                              str(numit) + '.pha'):
                pha = np.genfromtxt(path_inv + '/rho0' +
                                    str(numit) + '.pha',
                                    skip_header=1)
                rho = np.genfromtxt(path_inv + '/rho0' +
                                    str(numit) + '.mag',
                                    skip_header=1)
                if os.path.exists(path_inv + '/coverage.mag'):
                    cov = np.genfromtxt(path_inv + '/coverage.mag',
                                        skip_header=1,
                                        skip_footer=1)
                else:
                    print('Sensitivity model not found:')
                    print(path_inv)

        # linear resistivity and phase of CR
        xz = rho[:, :2]
        rho = 10**rho[:, 2]
        pha = pha[:, 2]

        # linear real and imaginary components of CC
        # mS/m for real; uS/m for imag
        real = (np.abs(1/(rho))*np.cos(-pha/1000))*10**3
        imag = (np.abs(1/(rho))*np.sin(-pha/1000))*10**6

        if sens_blank is not None:
            # print('shape mask1: ', (cov[:, 2] < sens_blank).shape)
            # print('shape mask2: ', (xz[:, 0] < 10).shape)
            # print('shape mask3: ', (xz[:, 0] > 50).shape)
            cov_mask = ((cov[:, 2] < sens_blank) |
                        (xz[:, 0] < 0) |
                        (xz[:, 0] > profile_length))
            # print(cov_mask)
            # print('number of True values: ', sum(cov_mask))
            pha[cov_mask] = np.nan
            # print(pha)
            xz[cov_mask] = np.nan
            # print(xz)

            pha = np.ma.masked_where(np.isnan(pha), pha)
            rho = np.ma.masked_where(np.isnan(pha), rho)

            real = np.ma.masked_where(np.isnan(pha), real)
            imag = np.ma.masked_where(np.isnan(pha), imag)
            xz = np.ma.masked_where(np.isnan(xz), xz)

            print('Sensitivity threshold for blanking: %.1f' %
                  sens_blank)

    # distinguish between complex and DC inversion
    else:
        print('Reading DC inversion results for:')
        print(path_inv)

        # number of iterations
        numit = len(glob.glob(path_inv + '/*.modl'))-1

        # read the inversion results
        if numit > 9:
            if os.path.exists(path_inv + '/rho' +
                              str(numit) + '.mag'):
                rho = np.genfromtxt(path_inv + '/rho' +
                                    str(numit) + '.mag',
                                    skip_header=1)
                if os.path.exists(path_inv + '/coverage.mag'):
                    cov = np.genfromtxt(path_inv + '/coverage.mag',
                                        skip_header=1,
                                        skip_footer=1)
                else:
                    print('Sensitivity model not found:')
                    print(path_inv)
        else:
            if os.path.exists(path_inv + '/rho0' +
                              str(numit) + '.mag'):
                rho = np.genfromtxt(path_inv + '/rho0' +
                                    str(numit) + '.mag',
                                    skip_header=1)
                if os.path.exists(path_inv + '/coverage.mag'):
                    cov = np.genfromtxt(path_inv + '/coverage.mag',
                                        skip_header=1,
                                        skip_footer=1)
                else:
                    print('Sensitivity model not found:')
                    print(path_inv)

        # linear resistivity and phase of CR
        xz = rho[:, :2]
        rho = 10**rho[:, 2]
        pha = np.zeros(len(rho))
        # linear real and imaginary components of CC
        # mS/m for real; uS/m for imag
        real = (np.abs(1/(rho))*np.cos(-pha/1000))*10**3
        imag = np.zeros(len(rho))


        if sens_blank is not None:
            # print('shape mask1: ', (cov[:, 2] < sens_blank).shape)
            # print('shape mask2: ', (xz[:, 0] < 10).shape)
            # print('shape mask3: ', (xz[:, 0] > 50).shape)
            cov_mask = ((cov[:, 2] < sens_blank) |
                        (xz[:, 0] < 0) |
                        (xz[:, 0] > profile_length))
            # print(cov_mask)
            # print('number of True values: ', sum(cov_mask))
            pha[cov_mask] = np.nan
            # print(pha)
            xz[cov_mask] = np.nan
            # print(xz)
            
            # print('nans in pha array: ', np.sum(np.isnan(pha)))
            
            pha = np.ma.masked_where(np.isnan(pha), pha, copy=True)
            rho = np.ma.masked_where(np.isnan(pha), rho, copy=True)

            real = np.ma.masked_where(np.isnan(pha), real, copy=True)
            imag = np.ma.masked_where(np.isnan(pha), imag, copy=True)
            xz = np.ma.masked_where(np.isnan(xz), xz, copy=True)

            print('Sensitivity threshold for blanking: %.1f' %
                  sens_blank)

    return inv_params, xz, rho, pha, real, imag


def read_grid_rect(pathelec, pathelem):
    """
    Function to read finite element grids used for inversion
    """

    # read elec file
    elec = np.genfromtxt(pathelec)

    # read file in 1d array
    with open(pathelem) as f:
        data = np.array(f.read().split(), dtype=float)

    # value which defines number of header rows
    endh = 9

    if data[endh] == 11:
        nheadr = 4
    else:
        nheadr = 3

    # write header to array
    header = data[0:nheadr*3].copy()

    # number of coordinate values
    nco = int(header[0])
    # number of elements
    nelem = int(header[4])

    # delete header information out of data file (for easier usage)
    data = np.delete(data, np.arange(len(header)))

    # array of coordinates
    coord = data[0:nco*3].reshape(nco, 3)

    # array of elements
    elem = data[nco*3:nco*3+nelem*4].reshape(nelem, 4)

    # get electrode positions
    elecind = coord[elec[1:].astype(int)-1, 0]
    elec_c1 = np.expand_dims(coord[elecind.astype(int)-1, 1], axis=1)
    elec_c2 = np.expand_dims(coord[elecind.astype(int)-1, 2], axis=1)

    elec = np.concatenate((np.expand_dims(elec[1:], axis=1),
                          elec_c1,
                          elec_c2), axis=1)

    # get indices
    elemind0 = coord[elem[:, 0].astype(int)-1, 0]
    elemind1 = coord[elem[:, 1].astype(int)-1, 0]
    elemind2 = coord[elem[:, 2].astype(int)-1, 0]
    elemind3 = coord[elem[:, 3].astype(int)-1, 0]

    elemind = np.concatenate((elemind0,
                              elemind1,
                              elemind2,
                              elemind3), axis=0)

    # get x-position of grid
    elemx0 = np.expand_dims(coord[elemind0.astype(int)-1, 1], axis=1)
    elemx1 = np.expand_dims(coord[elemind1.astype(int)-1, 1], axis=1)
    elemx2 = np.expand_dims(coord[elemind2.astype(int)-1, 1], axis=1)
    elemx3 = np.expand_dims(coord[elemind3.astype(int)-1, 1], axis=1)

    elemx = np.transpose(np.concatenate((elemx0,
                                         elemx1,
                                         elemx2,
                                         elemx3), axis=1))

    # get z-position of grid
    elemz0 = np.expand_dims(coord[elemind0.astype(int)-1, 2], axis=1)
    elemz1 = np.expand_dims(coord[elemind1.astype(int)-1, 2], axis=1)
    elemz2 = np.expand_dims(coord[elemind2.astype(int)-1, 2], axis=1)
    elemz3 = np.expand_dims(coord[elemind3.astype(int)-1, 2], axis=1)

    elemz = np.transpose(np.concatenate((elemz0,
                                         elemz1,
                                         elemz2,
                                         elemz3), axis=1))

    return elec, elemx, elemz


def read_grid_triang(pathelec, pathelem):
    """
    Function to read finite element grids used for inversion
    """

    # read elec file
    elec = np.genfromtxt(pathelec)

    # read file in 1d array
    with open(pathelem) as f:
        data = np.array(f.read().split(), dtype=float)

    # value which defines number of header rows
    endh = 9

    if data[endh] == 11:
        nheadr = 4
    else:
        nheadr = 3

    # write header to array
    header = data[0:nheadr*3].copy()
    # number of coordinate values
    nco = int(header[0])
    # number of elements
    nelem = int(header[4])

    # delete header information out of data file (for easier usage)
    data = np.delete(data, np.arange(len(header)))

    # array of coordinates
    coord = data[0:nco*3].reshape(nco, 3)

    # array of elements
    elem = data[nco*3:nco*3+nelem*3].reshape(nelem, 3)

    # get electrode positions
    elecind = coord[elec[1:].astype(int)-1, 0]
    elec_c1 = np.expand_dims(coord[elecind.astype(int)-1, 1], axis=1)
    elec_c2 = np.expand_dims(coord[elecind.astype(int)-1, 2], axis=1)

    elec = np.concatenate((np.expand_dims(elec[1:], axis=1),
                          elec_c1,
                          elec_c2), axis=1)

    # get indices
    elemind0 = coord[elem[:, 0].astype(int)-1, 0]
    elemind1 = coord[elem[:, 1].astype(int)-1, 0]
    elemind2 = coord[elem[:, 2].astype(int)-1, 0]

    elemind = np.concatenate((elemind0, elemind1, elemind2), axis=0)

    # get x-position of grid
    elemx0 = np.expand_dims(coord[elemind0.astype(int)-1, 1], axis=1)
    elemx1 = np.expand_dims(coord[elemind1.astype(int)-1, 1], axis=1)
    elemx2 = np.expand_dims(coord[elemind2.astype(int)-1, 1], axis=1)

    elemx = np.transpose(np.concatenate((elemx0,
                                         elemx1,
                                         elemx2), axis=1))

    # get z-position of grid
    elemz0 = np.expand_dims(coord[elemind0.astype(int)-1, 2], axis=1)
    elemz1 = np.expand_dims(coord[elemind1.astype(int)-1, 2], axis=1)
    elemz2 = np.expand_dims(coord[elemind2.astype(int)-1, 2], axis=1)

    elemz = np.transpose(np.concatenate((elemz0,
                                         elemz1,
                                         elemz2), axis=1))

    return elec, elemx, elemz


def put_blanking(ax, elec, elemx, elemz, depth):

    maxx, minx = elec[-1, 1]+1, elec[0, 1]-1
    init_z = np.max(elec[:, 2])
    final_z = elec[-1, 2]

    ax.add_patch(Polygon([(minx, init_z-depth),
                          (maxx, final_z-depth),
                          (maxx, final_z-100),
                          (minx, init_z-100)],
                         closed=True,
                         facecolor='white'))

def extract_values(xz, model, elec, xpos, pad, zpos=None):
    """
    improved routine to extract values from IP-results model,
    needs only the xpos and finds automatically the necessary depth and zpos
    """
    if zpos == None:
        elec_dx = elec[:,1] - xpos
        elec_mask = (np.abs(elec_dx) == np.min(np.abs(elec_dx)))
        
        zpos = elec[elec_mask][0,2]
        
    
    bottom = np.min(xz[:,1])
    zdep = zpos-bottom
    
    left = xpos-pad
    right = xpos+pad
#    top = zpos
    
    mask = np.where(((xz[:, 0] > left) &
                     (xz[:, 0] < right) &
                     #(xz[:, 1] < top) &
                     (xz[:, 1] > bottom)))
    
    ex = model[mask]
    xz_ex = xz[mask]
    
    return xz_ex, ex, zpos, zdep


def indicate_pos(ax, xpos, zpos, depth, label=None, pad=1, label_s=7):

    ax.add_patch(mpatches.Rectangle((xpos-pad, zpos), pad*2, depth, fill=False))

    if label is not None:
        ax.text(xpos, zpos+4, label,  #zpos*0.01
                ha='center',
                va='center',
                rotation=0,
                size=label_s)

def compute_maverage(data, inc=0.5):

    data_s = data[np.lexsort((data[:, 2], data[:, 1]))]
    data_s = data_s[::-1]

    cdepth = data_s[0, 1]

    out1 = []
    out2 = []

    for nc in range(0, 2000):
        datac = data_s[np.where((data_s[:, 1] <= cdepth) &
                                (data_s[:, 1] > cdepth-inc))]
        cdepth = cdepth-inc
        mdep = (np.median(datac[:, 1]))
        mval = (np.median(datac[:, 2]))

        out1.append(mdep)
        out2.append(mval)

        if cdepth < data_s[-1, 1]:
            break

    out1 = np.array(out1)
    out2 = np.array(out2)

    out1 = out1[~np.isnan(out1)]
    out2 = out2[~np.isnan(out2)]

    return out1, out2


# %% plotting
def plot_ERT_con(figure, axis, fid_inv:str, fid_grids:str,
                 x_shift:float, z_shift:float, log10:bool, 
                 colMap:str, c_lims=np.r_[20,150], expand_topo=None,
                 sens_blank=-3.0, plot_stats=False):
    # TODO colorbar left or right [DONE]

    if axis is None:  # initialize fig if necessary
        fig, ax = plt.subplots(1,1,figsize=(12,6))
    else:
        fig = figure
        ax = axis

    print('reading grids from:\n', fid_grids)
    elec, elemx, elemz = read_grid_triang(fid_grids + '.elc',
                                          fid_grids + '.elm')

    print('reading inv results from:\n', fid_inv)
    inv_params, xz, ip_rho, ip_pha, ip_real, ip_imag = parse_inv(fid_inv,
                                                                 profile_length=max(elec[:,1]),
                                                                 sens_blank=sens_blank,
                                                                 plot_stats=plot_stats)

    # add offset
    elec[:,1] += x_shift
    elemx += x_shift
    
    elec[:,2] -= z_shift
    elemz -= z_shift
    
    maxx, minx = elec[-1, 1]+1, elec[0, 1]-1
    maxz, minz = np.max(xz[:,1]) + 10, np.min(xz[:,1]) - 10
    
    # alternatively plot the magnitude of complex con
    # TODO add switch or automatize
    # ip_mag = np.sqrt(ip_real**2 + (ip_imag/1000)**2)
    
    pr = get_PatchCollection(elemx, elemz, colormap=colMap, log10=log10)
    pr.set_array(ip_real)
    pr.set_clim([c_lims[0], c_lims[1]])
    ax.add_collection(pr)

    ax.plot(elec[:, 1], elec[:, 2]+0.25, '|', color='k', ms=8, markeredgewidth=2.0, zorder=5)
    if expand_topo is not None:
        ax.plot(np.r_[elec[0,1] - expand_topo, elec[:, 1], elec[-1,1] + expand_topo],
                np.r_[elec[0,2], elec[:, 2], elec[-1,2]], '-k')
        print(f'expanding topo by {expand_topo} m')
    else:
        ax.plot(elec[:, 1], elec[:, 2], '-k')

    ax.locator_params(axis='x', nbins=10)
    ax.locator_params(axis='y', nbins=6)
    ax.tick_params(axis='both', which='both')
    # ax.yaxis.set_label_coords(-0.03, 1.05)
    # ax.set_ylabel('h (m)', rotation=0)
    # ax.yaxis.set_label_coords(-0.03, 1.05)
    
    ax.set_xlim([minx, maxx])
    ax.set_ylim([minz, maxz])
    # ax.set_aspect('equal')

    # cb = fig.colorbar(pr,
    #                   ax=ax,
    #                   orientation='vertical',
    #                   aspect=10,
    #                   pad=0.01)

    # cb.ax.tick_params(labelsize=label_s)
    # tick_locator = ticker.MaxNLocator(nbins=7)
    # cb.locator = tick_locator
    # cb.update_ticks()

    if axis is None:
        return fig, ax, pr
    else:
        return ax, pr


def plot_ERT_res(figure, axis, fid_inv:str, fid_grids:str,
                 x_shift:float, z_shift:float, log10:bool, 
                 colMap:str, c_lims=np.r_[1,1000], expand_topo=None,
                 sens_blank=-3.0, plot_stats=False):

    if axis is None:  # initialize fig if necessary
        fig, ax = plt.subplots(1,1,figsize=(12,6))
    else:
        fig = figure
        ax = axis

    print('reading grids from:\n', fid_grids)
    elec, elemx, elemz = read_grid_triang(fid_grids + '.elc',
                                          fid_grids + '.elm')

    print('reading inv results from:\n', fid_inv)
    inv_params, xz, ip_rho, ip_pha, ip_real, ip_imag = parse_inv(fid_inv,
                                                                 profile_length=max(elec[:,1]),
                                                                 sens_blank=sens_blank,
                                                                 plot_stats=plot_stats)

    # add offset
    elec[:,1] += x_shift
    elemx += x_shift
    
    elec[:,2] -= z_shift
    elemz -= z_shift
    
    maxx, minx = elec[-1, 1]+1, elec[0, 1]-1
    maxz, minz = np.max(xz[:,1]) + 10, np.min(xz[:,1]) - 10
    
    pr = get_PatchCollection(elemx, elemz, colormap=colMap, log10=log10)
    pr.set_array(ip_rho)
    pr.set_clim([c_lims[0], c_lims[1]])
    ax.add_collection(pr)

    ax.plot(elec[:, 1], elec[:, 2]+0.25, '|', color='k', ms=8, markeredgewidth=2.0, zorder=5)
    if expand_topo is not None:
        ax.plot(np.r_[elec[0,1] - expand_topo, elec[:, 1], elec[-1,1] + expand_topo],
                np.r_[elec[0,2], elec[:, 2], elec[-1,2]], '-k')
        print(f'expanding topo by {expand_topo} m')
    else:
        ax.plot(elec[:, 1], elec[:, 2], '-k')

    # ax.locator_params(axis='x', nbins=10)
    # ax.locator_params(axis='y', nbins=6)
    ax.tick_params(axis='both', which='both')
    # ax.yaxis.set_label_coords(-0.03, 1.05)
    # ax.set_ylabel('h (m)', rotation=0)
    # ax.yaxis.set_label_coords(-0.03, 1.05)
    
    ax.set_xlim([minx, maxx])
    ax.set_ylim([minz, maxz])
    # ax.set_aspect('equal')

    # cb = fig.colorbar(pr,
    #                   ax=ax,
    #                   orientation='vertical',
    #                   aspect=10,
    #                   pad=0.01)

    # cb.ax.tick_params(labelsize=label_s)
    # tick_locator = ticker.MaxNLocator(nbins=7)
    # cb.locator = tick_locator
    # cb.update_ticks()

    if axis is None:
        return fig, ax, pr
    else:
        return ax, pr


def plot_IP_pha(figure, axis, fid_inv:str, fid_grids:str,
                 x_shift:float, z_shift:float, log10:bool, 
                 colMap:str, c_lims=np.r_[1, 1000], expand_topo=None,
                 sens_blank=-3.0, plot_stats=False):

    if axis is None:  # initialize fig if necessary
        fig, ax = plt.subplots(1,1,figsize=(12,6))
    else:
        fig = figure
        ax = axis

    print('reading grids from:\n', fid_grids)
    elec, elemx, elemz = read_grid_triang(fid_grids + '.elc',
                                          fid_grids + '.elm')

    print('reading inv results from:\n', fid_inv)
    inv_params, xz, ip_rho, ip_pha, ip_real, ip_imag = parse_inv(fid_inv,
                                                                 profile_length=max(elec[:,1]),
                                                                 sens_blank=sens_blank,
                                                                 plot_stats=plot_stats)

    # add offset
    elec[:,1] += x_shift
    elemx += x_shift
    
    elec[:,2] -= z_shift
    elemz -= z_shift
    
    maxx, minx = elec[-1, 1]+1, elec[0, 1]-1
    maxz, minz = np.max(xz[:,1]) + 10, np.min(xz[:,1]) - 10
    
    pr = get_PatchCollection(elemx, elemz, colormap=colMap, log10=log10)
    pr.set_array(-ip_pha)
    pr.set_clim([c_lims[0], c_lims[1]])
    ax.add_collection(pr)

    ax.plot(elec[:, 1], elec[:, 2]+0.25, '|', color='k', ms=8, markeredgewidth=2.0, zorder=5)
    if expand_topo is not None:
        ax.plot(np.r_[elec[0,1] - expand_topo, elec[:, 1], elec[-1,1] + expand_topo],
                np.r_[elec[0,2], elec[:, 2], elec[-1,2]], '-k')
        print(f'expanding topo by {expand_topo} m')
    else:
        ax.plot(elec[:, 1], elec[:, 2], '-k')

    ax.locator_params(axis='x', nbins=10)
    ax.locator_params(axis='y', nbins=6)
    ax.tick_params(axis='both', which='both')
    # ax.yaxis.set_label_coords(-0.03, 1.05)
    # ax.set_ylabel('h (m)', rotation=0)
    # ax.yaxis.set_label_coords(-0.03, 1.05)
    
    ax.set_xlim([minx, maxx])
    ax.set_ylim([minz, maxz])
    # ax.set_aspect('equal')

    # cb = fig.colorbar(pr,
    #                   ax=ax,
    #                   orientation='vertical',
    #                   aspect=10,
    #                   pad=0.01)

    # cb.ax.tick_params(labelsize=label_s)
    # tick_locator = ticker.MaxNLocator(nbins=7)
    # cb.locator = tick_locator
    # cb.update_ticks()

    if axis is None:
        return fig, ax, pr
    else:
        return ax, pr
    

def plot_IP_imag(figure, axis, fid_inv:str, fid_grids:str,
                 x_shift:float, z_shift:float, log10:bool, 
                 colMap:str, c_lims=np.r_[1, 1000], expand_topo=None,
                 sens_blank=-3.0, plot_stats=False):

    # TODO read outside of this function, provide only necessary params as dict!
    if axis is None:  # initialize fig if necessary
        fig, ax = plt.subplots(1,1,figsize=(12,6))
    else:
        fig = figure
        ax = axis

    print('reading grids from:\n', fid_grids)
    elec, elemx, elemz = read_grid_triang(fid_grids + '.elc',
                                          fid_grids + '.elm')

    print('reading inv results from:\n', fid_inv)
    inv_params, xz, ip_rho, ip_pha, ip_real, ip_imag = parse_inv(fid_inv,
                                                                 profile_length=max(elec[:,1]),
                                                                 sens_blank=sens_blank,
                                                                 plot_stats=plot_stats)

    # add offset
    elec[:,1] += x_shift
    elemx += x_shift
    
    elec[:,2] -= z_shift
    elemz -= z_shift
    
    maxx, minx = elec[-1, 1]+1, elec[0, 1]-1
    maxz, minz = np.max(xz[:,1]) + 10, np.min(xz[:,1]) - 10
    
    pr = get_PatchCollection(elemx, elemz, colormap=colMap, log10=log10)
    pr.set_array(ip_imag)
    pr.set_clim([c_lims[0], c_lims[1]])
    ax.add_collection(pr)

    ax.plot(elec[:, 1], elec[:, 2]+0.25, '|', color='k', ms=8, markeredgewidth=2.0, zorder=5)
    if expand_topo is not None:
        ax.plot(np.r_[elec[0,1] - expand_topo, elec[:, 1], elec[-1,1] + expand_topo],
                np.r_[elec[0,2], elec[:, 2], elec[-1,2]], '-k')
        print(f'expanding topo by {expand_topo} m')
    else:
        ax.plot(elec[:, 1], elec[:, 2], '-k')

    ax.locator_params(axis='x', nbins=10)
    ax.locator_params(axis='y', nbins=6)
    ax.tick_params(axis='both', which='both')
    # ax.yaxis.set_label_coords(-0.03, 1.05)
    # ax.set_ylabel('h (m)', rotation=0)
    # ax.yaxis.set_label_coords(-0.03, 1.05)
    
    ax.set_xlim([minx, maxx])
    ax.set_ylim([minz, maxz])
    # ax.set_aspect('equal')

    # cb = fig.colorbar(pr,
    #                   ax=ax,
    #                   orientation='vertical',
    #                   aspect=10,
    #                   pad=0.01)

    # cb.ax.tick_params(labelsize=label_s)
    # tick_locator = ticker.MaxNLocator(nbins=7)
    # cb.locator = tick_locator
    # cb.update_ticks()

    if axis is None:
        return fig, ax, pr
    else:
        return ax, pr


def plot_vals_to_grid(figure, axis, elemx, elemz, elec,
                      values, color_map, clims:tuple, log10:bool=False,
                      x_shift:float=0.0, y_shift:float=0.0):

    if axis is None:  # initialize fig if necessary
        fig, ax = plt.subplots(1,1,figsize=(12,6))
    else:
        fig = figure
        ax = axis

    print('reading grids from:\n', fid_grids)
    elec, elemx, elemz = read_grid_triang(fid_grids + '.elc',
                                          fid_grids + '.elm')

    # add offset
    elec[:, 1] += x_shift
    elemx += x_shift
    
    elec[:, 2] -= z_shift
    elemz -= z_shift
    
    maxx, minx = elec[-1, 1]+1, elec[0, 1]-1
    maxz, minz = np.max(xz[:,1]) + 10, np.min(xz[:,1]) - 10
    
    pr = get_PatchCollection(elemx, elemz, colormap=color_map, log10=log10)
    pr.set_array(values)
    pr.set_clim([c_lims[0], c_lims[1]])
    ax.add_collection(pr)

    ax.plot(elec[:, 1], elec[:, 2]+0.25, '|', color='k', ms=8, markeredgewidth=2.0, zorder=5)
    if expand_topo is not None:
        ax.plot(np.r_[elec[0,1] - expand_topo, elec[:, 1], elec[-1,1] + expand_topo],
                np.r_[elec[0,2], elec[:, 2], elec[-1,2]], '-k')
        print(f'expanding topo by {expand_topo} m')
    else:
        ax.plot(elec[:, 1], elec[:, 2], '-k')

    ax.locator_params(axis='x', nbins=10)
    ax.locator_params(axis='y', nbins=6)
    ax.tick_params(axis='both', which='both')

    ax.set_xlim([minx, maxx])
    ax.set_ylim([minz, maxz])
    # ax.set_aspect('equal')

    # cb = fig.colorbar(pr,
    #                   ax=ax,
    #                   orientation='vertical',
    #                   aspect=10,
    #                   pad=0.01)

    if axis is None:
        return fig, ax, pr
    else:
        return ax, pr


def plot_ERT_cbar(fig, axis, pacoll, label, cmin, cmax, 
                  cbar_pos='right', log10_switch=True, 
                  label_fmt='%.1f', round_tick_label=1):
    """
    

    Parameters
    ----------
    fig : TYPE
        DESCRIPTION.
    axis : TYPE
        DESCRIPTION.
    pacoll : TYPE
        DESCRIPTION.
    label : TYPE
        DESCRIPTION.
    cmin : TYPE
        DESCRIPTION.
    cmax : TYPE
        DESCRIPTION.
    cbar_pos : TYPE, optional
        DESCRIPTION. The default is 'right'.
    log10_switch : TYPE, optional
        DESCRIPTION. The default is True.
    label_fmt : TYPE, optional
        DESCRIPTION. The default is '%.1f'.
    round_tick_label : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    None.

    """
    if cbar_pos == 'right':
        divider = make_axes_locatable(axis)
        cax2 = divider.append_axes("right", size="2%", pad=0.3)
        cb2 = fig.colorbar(pacoll, cax=cax2, format=label_fmt)
        # cb.ax.yaxis.set_ticks([10, 20, 40, 100, 200])
        if log10_switch is True:
            ticks = np.round(np.logspace(np.log10(cmin), np.log10(cmax), 5), round_tick_label)
            cb2.ax.yaxis.set_ticks(ticks)
        else:
            tick_locator = ticker.MaxNLocator(nbins=5)
            cb2.locator = tick_locator; cb2.update_ticks()
        cb2.ax.minorticks_off()
        cb2.set_label('ERT: ' + label)

    elif cbar_pos == 'left':
        divider = make_axes_locatable(axis)
        cax2 = divider.append_axes("left", size="2%", pad=0.8)
        cb2 = fig.colorbar(pacoll, cax=cax2, format=label_fmt)
        if log10_switch is True:
            ticks = np.round(np.logspace(np.log10(cmin), np.log10(cmax), 5), round_tick_label)
            cb2.ax.yaxis.set_ticks(ticks)
        else:
            tick_locator = ticker.MaxNLocator(nbins=5)
            cb2.locator = tick_locator; cb2.update_ticks()
        cb2.ax.minorticks_off()
        cb2.ax.yaxis.set_ticks_position('left')
        cb2.ax.yaxis.set_label_position('left')
        # cb2.set_label('TEM: ' + labelRho)
        cb2.set_label('ERT: ' + label)
        axis.yaxis.set_label_coords(-0.015, 0.05)
        axis.set_ylabel('H (m)', rotation=90)
    
    return fig, axis


# def plot_ERT_con(figure, axis, fid_inv:str, fid_grids:str,
#                  x_shift:float, z_shift:float,
#                  colMap:str, c_lims:tuple, log10:bool, cbar_pos:str,
#                  sens_blank=-3.0, plot_stats=False):
    
#     # TODO colorbar left or right [DONE]
    
#     if axis is None:  # initialize fig if necessary
#         fig, ax = plt.subplots(1,1,figsize=(12,6))
#     else:
#         fig = figure
#         ax = axis
    
#     print('reading inv results from:\n', fid_inv)
#     inv_params, xz, ip_rho, ip_pha, ip_real, ip_imag = parse_inv(fid_inv,
#                                                                  sens_blank=sens_blank,
#                                                                  plot_stats=plot_stats)
    
#     print('reading grids from:\n', fid_grids)
#     elec, elemx, elemz = read_grid_triang(fid_grids + '.elc',
#                                           fid_grids + '.elm')
    
#     # add offset
#     elec[:,1] += x_shift
#     elemx += x_shift
    
#     elec[:,2] -= z_shift
#     elemz -= z_shift
    
#     maxx, minx = elec[-1, 1]+1, elec[0, 1]-1
#     maxz, minz = np.max(xz[:,1]) + 10, np.min(xz[:,1]) - 10
    
#     # alternatively plot the magnitude of complex con
#     # TODO add switch or automatize
#     # ip_mag = np.sqrt(ip_real**2 + (ip_imag/1000)**2)
    
#     pr = get_PatchCollection(elemx, elemz, colormap=colMap, log10=log10)
#     pr.set_array(ip_real)
#     pr.set_clim([c_lims[0], c_lims[1]])

#     ax.add_collection(pr)
#     ax.plot(elec[:, 1], elec[:, 2], '.', color='gray', markersize=2.5, zorder=10)
#     ax.locator_params(axis='x', nbins=10)
#     ax.locator_params(axis='y', nbins=6)
#     ax.tick_params(axis='both', which='both')
#     # ax.yaxis.set_label_coords(-0.03, 1.05)
#     ax.set_ylabel('h (m)', rotation=0)
#     ax.yaxis.set_label_coords(-0.03, 1.05)
    
#     ax.set_xlim([minx, maxx])
#     ax.set_ylim([minz, maxz])
#     # ax.set_aspect('equal')

#     if cbar_pos == 'right':
#         divider = make_axes_locatable(ax)
#         cax1 = divider.append_axes("right", size="2%", pad=0.5)
#         cb = fig.colorbar(pr, cax=cax1, format='%.0f')
#         # cb.ax.yaxis.set_ticks([10, 20, 40, 100, 200])
#         if log10 is True:
#             ticks = np.round(np.logspace(np.log10(c_lims[0]), np.log10(c_lims[1]), 6), 1)
#             cb.ax.yaxis.set_ticks(ticks)
#         else:
#             tick_locator = ticker.MaxNLocator(nbins=6)
#             cb.locator = tick_locator; cb.update_ticks()
#         cb.ax.minorticks_off()
#         cb.set_label(r"ERT: $\sigma'$" + ' (mS/m)')

#     elif cbar_pos == 'left':
#         divider = make_axes_locatable(ax)
#         cax1 = divider.append_axes("left", size="2%", pad=0.8)
#         cb = fig.colorbar(pr, cax=cax1, format='%.0f')
#         if log10 is True:
#             ticks = np.round(np.logspace(np.log10(c_lims[0]), np.log10(c_lims[1]), 6), 1)
#             cb.ax.yaxis.set_ticks(ticks)
#         else:
#             tick_locator = ticker.MaxNLocator(nbins=6)
#             cb.locator = tick_locator; cb.update_ticks()
#         cb.ax.minorticks_off()
#         cb.ax.yaxis.set_ticks_position('left')
#         cb.ax.yaxis.set_label_position('left')
#         cb.set_label(r"ERT: $\sigma'$" + ' (mS/m)')
#     else:
#         raise ValueError('choose either right or left for the cbar_pos argument...')

#     # cb = fig.colorbar(pr,
#     #                   ax=ax,
#     #                   orientation='vertical',
#     #                   aspect=10,
#     #                   pad=0.01)

#     # cb.ax.tick_params(labelsize=label_s)
#     # tick_locator = ticker.MaxNLocator(nbins=7)
#     # cb.locator = tick_locator
#     # cb.update_ticks()
    
    
    
#     if axis is None:
#         return fig, ax, cb
#     else:
#         return ax, cb
