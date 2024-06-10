import sys
sys.path.insert(1, '../')

from pckgs.import_pckg import *
from pckgs.units import *


# Import COBE-FIRAS data from astro-ph/9605054
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


# Inverse of Covariance Matrix 
Cov_Inv = np.linalg.inv(Cov)  # in (kJy*sr^-1)^2





nu_PIXIE = np.linspace(
                30 * 1e9 * Hz, 6 * 1e12 * Hz, 400) / (Centimeter ** -1)
            self.omega_FIRAS = 2 * np.pi * nu_PIXIE * Centimeter ** -1
            self.d = self.B_CMB(self.omega_FIRAS, 2.725) / (1e6 * Jy)
            unc = np.ones_like(nu_PIXIE) * 5 * Jy / (1e6 * Jy)
            print("shevedi PIXIE")
            print(self.omega_FIRAS)
            print(self.d)
            print(unc)
            Cov = np.zeros((len(nu_PIXIE), len(nu_PIXIE)))
            for i1 in range(len(nu_PIXIE)):
                for i2 in range(len(nu_PIXIE)):
                    if i1 == i2:
                        Cov[i1, i2] = unc[i1] * unc[i1]

        self.Cinv = la.inv(Cov)

        # Null chi2
        self.chi2_null = minimize(self.chi2_FIRAS, x0=[2.725], args=(
            0, np.ones_like(self.omega_FIRAS)), method='Powell')

        # Find the critical value for 95% confidence for two-sided and one-sided chi2
        self.delta_chi2_95_level_one_sided = fsolve(lambda crit: (
            1. - chi2.cdf(crit, df=1.)) / 2. - 0.05, x0=[3.])[0]

        # Fiducial array of mixing angles to scan over
        self.eps_ary = np.logspace(-11., 0., 500)
