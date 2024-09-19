import sys
sys.path.insert(1, '../')

from pckgs.import_pckg import *
from pckgs.units import *


import pandas as pd



next_generation_nu_ary = np.linspace(
                30 * 1e9 * Hz, 6 * 1e12 * Hz, 400) / (Centimeter ** -1)

            
next_generation_sigma_ary = np.ones_like(next_generation_nu_ary) * 5 * Jy * 1e3 / (1e6 * Jy)

Cov = np.zeros((len(next_generation_nu_ary), len(next_generation_nu_ary)))
for i1 in range(len(next_generation_nu_ary)):
    for i2 in range(len(next_generation_nu_ary)):
        if i1 == i2:
            Cov[i1, i2] = next_generation_sigma_ary[i1] * next_generation_sigma_ary[i1]

Cov_Inv = np.linalg.inv(Cov)

