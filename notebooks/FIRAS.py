

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



# Import COBE-FIRAS data
import pandas as pd

FIRAS_data_ary = np.transpose( np.array( pd.read_csv('../data/FIRASData.csv') ) )

# nu [cm^-1]
FIRAS_nu_ary  = np.ndarray.flatten( FIRAS_data_ary[[0]] )

# Intensity [MJy * sr^-1]
FIRAS_I0_ary  = np.ndarray.flatten( FIRAS_data_ary[[1]] )

# Residual [kJy * sr^-1]
FIRAS_res_ary   = np.ndarray.flatten( FIRAS_data_ary[[2]] )

# Uncertainty [kJy * sr^-1]
FIRAS_sigma_ary = np.ndarray.flatten( FIRAS_data_ary[[3]] )


# Length of FIRAS data
len_FIRAS = len(FIRAS_nu_ary)

# Smallest frequency in FIRAS data [cm^-1]
FIRAS_nu_min = FIRAS_nu_ary[0]

# Largest frequency in FIRAS data [cm^-1]
FIRAS_nu_max= FIRAS_nu_ary[-1]


# Q array  (From astro-ph/9605054, '3.3 Variance Estimation')
Q_ary = np.array([1.000, 0.176, -0.203, 0.145, 0.077, -0.005, -0.022, 0.032,
                          0.053, 0.025, -0.003, 0.007, 0.029, 0.029, 0.003, -
                          0.002, 0.016, 0.020, 0.011, 0.002, 0.007,
                          0.011, 0.009, 0.003, -0.004, -0.001, 0.003, 0.003, -
                          0.001, -0.003, 0.000, 0.003, 0.009, 0.015,
                          0.008, 0.003, -0.002, 0.000, -0.006, -0.006, 0.000, 0.002, 0.008])



# Covariance Matrix: Initialization
Cov   = np.zeros((len_FIRAS, len_FIRAS))



# Calc: Cov[i,j] = Q(|i-j|) * sig_i * sig_j
for i in range(len_FIRAS):
    
    for j in range(len_FIRAS):
        
        # Index of Q array
        # More intuitively, this is the 'distance' between nu_i and nu_j
        idx_Q = np.abs(i-j)
        
        # Q value
        Q_value = Q_ary[idx_Q]
        
        # Cov_ij
        Cov[i,j] = FIRAS_sigma_ary[i] * FIRAS_sigma_ary[j] * Q_value
        
#         print('[i,j] index = ',[i,j])
#         print('Q index = ', idx_Q)
#         print('Q value = ', Q_value)
#         print('Cov_ij  = ', Cov[i,j])


# Inverse of Covariance Matrix 
Cov_Inv = np.linalg.inv(Cov)  # in (kJy*sr^-1)^2
