from pckgs.import_pckg import *
from pckgs.units import *


# Import COBE-FIRAS data from astro-ph/9605054
import pandas as pd


# PIXIE Projection Data got from 2405.20403
# (Also see lambda.gsfc.nasa.gov/product/pixie/pixie_baseline_noise_get.html)
# Stoke I Parameter

# [1]. Center Frequency (GHz)
# [2]. Sensitivity (Jy * sr^−1 * \sqrt{sec})
# [3]. Map Noise (Jy * sr^−1)

# 2 year mission
# 45% time for calibrator deployed (sensitive to spectral distortions and polarization)
# 70% of the sky
year_in_s = 365.2422*24*3600

Eff_Time_in_s = 0.7 * 0.45 * (2*year_in_s)

Jy_to_MJy = 10**(-6)

PIXIE_data_ary = np.transpose( np.array(pd.read_csv('../data/data_input_bound/PIXIE_Proj24_Data.csv') ) )

# nu [GHz]
PIXIE_nu_GHz_ary = PIXIE_data_ary[0]

# Sensitivity (Divided by sqrt(effective_integration_time) ) [Jy]
PIXIE_sen_ary = PIXIE_data_ary[1]/np.sqrt(Eff_Time_in_s)

# Length of PIXIE projection data
len_PIXIE = len(PIXIE_nu_GHz_ary)
