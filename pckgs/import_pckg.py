import numpy  as np
import scipy as sc
import pandas as pd
import math
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


from pckgs.units import *
import time

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

#suppress warnings
import warnings
warnings.filterwarnings('ignore')




# this function lightens the color for plotting
# ref:  https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib
def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])



# def contour_func(mAp_1Dary, eps_1Dary, TS_2Dary):
    
#     plt.figure()

#     plt.xscale('log')
#     plt.yscale('log')


#     # Delta 95%CL, From PDG Statistics
#     TS_choose = 2.71

#     TS_Reg = 0.001

#     X_plt, Y_plt = np.meshgrid( mAp_1Dary, eps_1Dary )
#     Z_plt                 = np.log10(    TS_2Dary   + TS_Reg )
#     CS_trans=  plt.contour(X_plt, Y_plt, Z_plt,  levels = [ np.log10(TS_choose) ]);
#     Greens_trans = np.transpose(CS_trans.collections[0].get_paths()[0].vertices );
#     return Greens_trans
    
    
