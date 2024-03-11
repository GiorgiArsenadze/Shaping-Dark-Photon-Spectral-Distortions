import numpy  as np
import pandas as pd
from   scipy.interpolate import interp1d
from   scipy.interpolate import interp2d
from   scipy.misc import derivative
from   scipy import optimize
import astropy.units as u
#import astropy.constants as c
from   astropy.cosmology import FlatLambdaCDM, z_at_value
from   tqdm import *
from   sympy import *
from   astropy.cosmology import Planck13 as cosmo
from   astropy import constants as const
import sys
from   scipy.interpolate import interp1d
from   scipy.interpolate import interp2d
from   scipy.special import zeta
import pickle

import matplotlib.pyplot as plt
from   matplotlib import ticker
from matplotlib import gridspec
import matplotlib.pylab as pylab
#from matplotlib import colormaps
import matplotlib.ticker as mticker



import time

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

#suppress warnings
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(1, '../packages')

import units
import import_packages
sys.path.insert(1, '../notebooks')
from FIRAS import *
#importing generic constants and functions
from const_func_general import *
#importing distortion specific constants and functions
from cons_func_distortion import *
#importing functions actually calculatting distortion
from calc_distortion import *
#importing functions for maximum liklihood analysis 
from max_likelihood_analysis import *


# ----------------------------------------------------------------------
# Parameters for scanning in (mAp,x) plane
# Here we calculate P(A->Ap) for (mAp,x) plane. Result will be used later
# To interpolate probability for much faster calculations. 
# We have checked and error between interpolated and directly calcualted probability is negiligible.
# ----------------------------------------------------------------------
# Output the P(A->Ap) table on x-mAp plane
N_P_x   = 500
N_P_mAp = 300

nu_P_min = 10**(-2)  # in cm^-1
nu_P_max = 30        # in cm^-1    # Note: The highest nu_FIRAS is 21.33 cm^-1

x_P_min = 2 * np.pi * nu_P_min * cmInv_to_eV/TCMB_0   # in eV
x_P_max = 2 * np.pi * nu_P_max * cmInv_to_eV/TCMB_0   # in eV

mAp_P_min = 10**(-17)                               # in eV     # Note: The lower mAp of FIRAS bound is ~10^(-15) eV
mAp_P_max = 10**(-3)                                # in eV     # Note: mAp~10^-4eV, z_res~10^6 (Double Compton Scattering). 


print('N_P_x     = ', N_P_x )
print('N_P_mAp   = ', N_P_mAp )
print('')
print('x_P_min   = ', x_P_min )
print('x_P_max   = ', x_P_max )
print('')
print('mAp_P_min = ', mAp_P_min, 'eV')
print('mAp_P_max = ', mAp_P_max, 'eV')
print('')

# ----------------------------------------------------------------------
# Scan in (x,mAp) plane
# ----------------------------------------------------------------------

x_P_scan_ary   = np.linspace( x_P_min, x_P_max, N_P_x )
mAp_P_scan_ary = np.logspace( np.log10(mAp_P_min), np.log10(mAp_P_max), N_P_mAp )

P_over_eps2_scan_2Dary = np.zeros( (len(x_P_scan_ary),len(mAp_P_scan_ary)) )

zres_scan_2Dary        = np.zeros( (len(x_P_scan_ary),len(mAp_P_scan_ary)) )




for j in tqdm_notebook(range(0,len(mAp_P_scan_ary))):

    
    for i in range(0,len(x_P_scan_ary)):
        
        x_i   = x_P_scan_ary[i]
        mAp_j = mAp_P_scan_ary[j]   # in eV
        
        get_z_crossing_ij = get_z_crossings(mAp_j,TCMB_0*x_i)
        
        # Check whether A and Ap crosses
        # If no crossing, choose zres=0
        if (len( get_z_crossing_ij ) > 0):
            zres_scan_2Dary[i][j] = get_z_crossing_ij[0]   # Only pick the smallest z_cross
        else:
            zres_scan_2Dary[i][j] = 0  # When there is no crossing
        
        # P_over_eps2_scan_2Dary[i][j] = P_over_eps2(mAp_j, x_i)
        P_over_eps2_scan_2Dary[i][j] = P_pre_over_eps2(mAp_j, x_i, get_z_crossing_ij)
        
        
# Exporting Probability and mAp and x_p values
np.savez("../data/Probability.npz",
         x_P_scan_ary=x_P_scan_ary,
         mAp_P_scan_ary=mAp_P_scan_ary,
         P_over_eps2_scan_2Dary=P_over_eps2_scan_2Dary,
         zres_scan_2Dary=zres_scan_2Dary
         )


# x_prime_int_test = np.logspace(-4,2,1000)
# x_prime_int_test = np.logspace(-1,2,1000)
x_prime_int_test = np.logspace(-3,5,200)

# m_Ap_min = 10**(-10) # in eV
# m_Ap_max = 10**(-3)  # in eV
# eps_min  = 10**(-8)
# eps_max  = 10**(-3)
# N_m_Ap   = 50
# N_eps    = 70


m_Ap_min = 10**(-10) # in eV
m_Ap_max = 10**(-3)  # in eV
eps_min  = 10**(-8)
eps_max  = 10**2
N_m_Ap   = 200
N_eps    = 300


m_Ap_ary = np.logspace( np.log10(m_Ap_min), np.log10(m_Ap_max), N_m_Ap )
eps_ary  = np.logspace( np.log10(eps_min) , np.log10(eps_max) , N_eps  )


# 2D array: N_eps * N_mAp
TS_2Dary = np.zeros((len(eps_ary),len(m_Ap_ary)))


# TS_ij: (eps_i, mAp_j)
# j: The number of column
for j in tqdm_notebook(range(0, len(m_Ap_ary))):
   
    
    m_Ap_j   =  m_Ap_ary[j]
    
    chi_sq_minT0_mineps_value = chi_sq_minT0_mineps( x_prime_int_test, m_Ap_j )
    
    
    # i: The number of row
    for i in range(0, len(eps_ary)):
        
        # print('This is ', i, 'th eps')

        eps_i   =  eps_ary[i]
        
        TS = chi_sq_minT0( x_prime_int_test, m_Ap_j, eps_i ) - chi_sq_minT0_mineps_value
        
        TS_2Dary[i][j] = TS
        
#         print([mAp, eps, chisq])


# Exporting Probability and mAp and x_p values
np.savez("../data/transFIRAS_5p8e4_1p88.npz",
         m_Ap_ary=m_Ap_ary,
         eps_ary=eps_ary,
         TS_2Dary=TS_2Dary
         )



#importing files
file_name = "../data/transFIRAS_5p8e4_1p88.npz"
if file_name is not None:
    file = np.load(file_name)
# mAp:   1D array
mAp_1Dary_import_5p8e4_1p88 = file['m_Ap_ary']
# -------------------------------------------------------
# eps: 1D array
eps_1Dary_import_5p8e4_1p88 = file['eps_ary']

# -------------------------------------------------------
# TS: 2D array
TS_2Dary_import_5p8e4_1p88 = file['TS_2Dary']



Xucheng_FIRAS_2Dary         = np.transpose( np.array( pd.read_csv('../data/Xucheng_22_FIRAS.csv') ) )
Samuel_FIRAS_2Dary          = np.transpose( np.array( pd.read_csv('../data/Samuel_20_FIRAS.csv') ) )
Redondo_09_FIRAS_2Dary      = np.transpose( np.array( pd.read_csv('../data/Redondo_09_FIRAS.csv') ) )
Hongwan_20_FIRAS_ho_2Dary   = np.transpose( np.array( pd.read_csv('../data/Hongwan_20_FIRAS_ho.csv') ) )
Hongwan_20_FIRAS_inho_2Dary = np.transpose( np.array( pd.read_csv('../data/Hongwan_20_FIRAS_inho.csv') ) )


# ====================================================
# Plot Parameters
mAp_pltmin = 1e-15 # eV
mAp_pltmax = 3e-3  # eV

eps_pltmin = 1e-9
eps_pltmax = 1e-3

majortick_len = 7 # length of major tick
minortick_len = 4 # length of minor tick

twin_majortick_len = majortick_len  # length of twin major tick
twin_minortick_len = minortick_len  # length of twin minor tick
# ====================================================



# Boundary of mu-y transition era
z_trans_0   = 10**3
z_trans_1   = 10**4
z_trans_2   = 3 * 10**5
z_dcs       = 2 * 10**6  # Redshif of double Compton scattering
z_max       = 10**10   # We just choose some very large z_max value

mAp_trans_0 = np.sqrt( mAsq(z_trans_0, 0) ) # in eV
mAp_trans_1 = np.sqrt( mAsq(z_trans_1, 0) ) # in eV
mAp_trans_2 = np.sqrt( mAsq(z_trans_2, 0) ) # in eV
mAp_dcs     = np.sqrt( mAsq(z_dcs, 0) )     # in eV
mAp_max     = np.sqrt( mAsq(z_max, 0) )     # in eV


# Delta 95%CL, From PDG Statistics
# TS_choose = 5.99
TS_choose = 2.71

TS_Reg = 0.001


fig, ax = plt.subplots()

ax1 = ax.twiny()

fig.set_size_inches(9, 6)

ax.set_xscale('log')
ax.set_yscale('log')
ax1.set_xscale('log')    # twin_y axis
ax1.set_yscale('log')    # twin_y axis

ax.set_xlim(mAp_pltmin, mAp_pltmax)
ax.set_ylim(eps_pltmin, eps_pltmax)
ax1.set_xlim(mAp_pltmin, mAp_pltmax)    # twin_y axis
ax1.set_ylim(eps_pltmin, eps_pltmax)    # twin_y axis

# ax.plot( Xucheng_FIRAS_2Dary[0]    , Xucheng_FIRAS_2Dary[1], label = r'Gan et.al. 2022')
ax.plot( Hongwan_20_FIRAS_ho_2Dary[0] , Hongwan_20_FIRAS_ho_2Dary[1], linewidth=2, label = r'Liu et.al. 20: Homogeneous', color = 'gray', linestyle = 'dashed')
# ax.plot( Redondo_09_FIRAS_2Dary[0] , Redondo_09_FIRAS_2Dary[1], label = r'Redondo et.al. 2009', color = 'red')
ax.plot( Samuel_FIRAS_2Dary[0]     , Samuel_FIRAS_2Dary[1], linewidth=2, label = r'McDermott et.al. 20', color = 'orange', linestyle = 'dashed')

ax.plot( Xucheng_Greens_trans_5p8e4_1p88[0], Xucheng_Greens_trans_5p8e4_1p88[1], label = r'XG: $z_\mathrm{trans} = 5.8 \times 10^4$, $\mathrm{Power}=1.88$', color=color_5p8e4_1p88,  linewidth=2.0)


ax.axvspan(mAp_trans_1, mAp_trans_2, alpha=0.1, facecolor='blue', edgecolor=None)
ax.text(1.3e-7, 1e-6, r'$\mu$-y era', fontsize=17, color='blue', rotation=-90)

ax.axvspan(mAp_trans_0, mAp_trans_1, alpha=0.1, facecolor='red', edgecolor=None)
ax.text(0.9e-9, 1.3e-6, r'y era', fontsize=17, color='red', rotation=-90)

ax.axvspan(mAp_trans_2, mAp_dcs, alpha=0.1, facecolor='green', edgecolor=None)
ax.text(7e-6, 1.3e-6, r'$\mu$ era', fontsize=17, color='green', rotation=-90)

ax.axvspan(mAp_dcs, mAp_max, alpha=0.1, facecolor='orange', edgecolor=None)
ax.text(3e-4, 1.3e-6, r'T era', fontsize=17, color='orange', rotation=-90)


ax.xaxis.set_major_locator( mticker.LogLocator(numticks=999, base=100) )
ax.xaxis.set_minor_locator( mticker.LogLocator(numticks=999, base=10,subs=(.1, .2, .3, .4, .5, .6, .7, .8, .9)))

ax.tick_params(which='major', length=majortick_len, labelbottom=True)
ax.tick_params(which='minor', length=minortick_len, labelbottom=False)

ax.tick_params(axis='both', left=True, top=True, right=True, bottom=True, labelleft=True, labeltop=False, labelright=False, labelbottom=True)


plt.setp(ax.xaxis.get_ticklabels(), rotation=0)

ax.set_xlabel(r'$m_{A^\prime}$[eV]',fontsize=19)
ax.set_ylabel(r'$\epsilon$',fontsize=19, rotation=0)

ax1.xaxis.set_major_locator( mticker.LogLocator(numticks=999, base=10) )
ax1.xaxis.set_minor_locator( mticker.LogLocator(numticks=999, base=10,subs=(.1, .2, .3, .4, .5, .6, .7, .8, .9)))

ax1.tick_params(which='major', length=twin_majortick_len)
ax1.tick_params(axis='both', left=False, top=True, right=False, bottom=True, labelleft=False, labeltop=False, labelright=False, labelbottom=False)


ax.legend(bbox_to_anchor=(0.29, 0.73), loc = 'center',fontsize=8.5)
fig.suptitle(r"\bf{COBE-FIRAS Constraints: $\mathrm{TS}=-2.71$}", y=0.94, fontsize=18)

plt.savefig('../plots/mAp_eps_plt_2.pdf')