

import sys
sys.path.insert(1, '../')

from pckgs.import_pckg import *
from pckgs.units import *



# Fundamental Constants
h = 4.135667696*1e-15          # in eV s

hbar = h / (2*np.pi)           # in eV s

c = 2.99792458*1e10            # in cm/s

mpl = 1.2209 * 10**19 * 10**9  # in eV


# Consistent with my MMA
cm_to_m     = 10**(-2)
cm_to_km    = 10**(-5)

m_to_cm     = 1/cm_to_m
km_to_cm    = 1/cm_to_km

pc_to_cm    = 3.08567758149137* 10**18
Mpc_to_cm   = 10**6 * pc_to_cm

cmInv_to_eV = hbar*c
sInv_to_eV  = hbar
eV_to_J     = 1.602176634 * 10**(-19)
kB_to_JperK = 1.38064852 * 10**(-23)

eV_to_K = eV_to_J/kB_to_JperK

J = 1/eV_to_J  # in eV
K = 1/eV_to_K  # in eV

# Cosmological Constants

# from 1807.06209 VI Table 2 Final column (68% confidence interval), 

planck18_cosmology = {'Oc0': 0.2607,
                              'Ob0': 0.04897,
                              'Om0': 0.3111,
                              'Hubble0': 67.66,
                              'n': 0.9665,
                              'sigma8': 0.8102,
                              'tau': 0.0561,
                              'z_reion': 7.82,
                              't0': 13.787,
                              'Tcmb0': 2.7255,
                              'Neff': 3.046,
                              'm_nu': [0., 0., 0.06],
                              'z_recomb': 1089.80,
                              'reference': "Planck 2018 results. VI. Cosmological Parameters, "
                                           "A&A, submitted, Table 2 (TT, TE, EE + lowE + lensing + BAO)"
                              }


# TCMB_0 from 1807.06209
TCMB_0   = planck18_cosmology["Tcmb0"]*K # in eV

# H = h_Hubble * 100 km/s/Mpc
# Note: 'h' has already been used for Planck constant
h_Hubble = planck18_cosmology["Hubble0"]/100

H0       = h_Hubble * ( 100 * km_to_cm * sInv_to_eV/ Mpc_to_cm )   # in eV

# Faction of matter
Omega_m  = planck18_cosmology["Om0"]

# Fraction of radiation
Omega_r = (8*np.pi**3/90) * 3.38 * (TCMB_0**4/(mpl**2 * H0**2))

# Fraction of dark energy
Omega_Lambda = 1 - Omega_r - Omega_m

# Baryon-to-photon ratio, from 1912.01132
eta = 6.129 * 10**(-10)

# Helium-to-hydrogen mass fraction, from 1912.01132
Yp  = 0.247

# Electron mass (from PDG)
m_e = 0.511 * 10**6 # in eV

# From PDG
alpha = 1/137.035999084

ee = np.sqrt(4*np.pi*alpha)

# Thompson Scattering Cross Section (Double-checked with Wiki)
thomson_xsec = 6.6524587158e-25   # cm^2

# electron’s Compton wavelength (Double-checked with Wiki)
# lambda_e = 2.426e-10  cm
lambda_e = (2*np.pi/m_e)*cmInv_to_eV # in cm

# red shift for mu-y transition era [z_trans_1, z_trans_2]
z_trans_1 = 10**4
z_trans_2 = 3 * 10**5

# Hubble Parameter
# in eV
def hubble(z):
    
    return H0 * np.sqrt(Omega_Lambda + Omega_m * (1.0 + z) ** 3 + Omega_r * (1.0 + z) ** 4)

# n_p function [cm^-3]
def n_p(z):
    
    n_p_0 = (1-Yp/2) * eta * (2*zeta(3)/np.pi**2) * ( TCMB_0 / cmInv_to_eV )**3   # in cm^-3
    
    n_p_z = n_p_0 * (1+z)**3   # in cm^-3
    
    return n_p_z


# n_H function [cm^-3]
def n_H(z):
    
    n_H_0 = (1-Yp) * eta * (2*zeta(3)/np.pi**2) * ( TCMB_0 / cmInv_to_eV )**3   # in cm^-3
    
    n_H_z = n_H_0 * (1+z)**3   # in cm^-3
    
    return n_H_z



# ========================================================
# x_e function
# ========================================================

x_e_data = pickle.load(open("../data/std_soln_He.p", "rb"))

def x_e(z):
    
    return np.interp(z, np.flipud(x_e_data[0]) - 1.0, np.flipud(x_e_data[2]))


# Parameters used in Class
class_parameters = {'H0':         planck18_cosmology["Hubble0"],
                    'Omega_b':    planck18_cosmology["Ob0"],
                    'N_ur':       planck18_cosmology["Neff"],
                    'Omega_cdm':  planck18_cosmology["Oc0"],
                    'YHe':        Yp,
                    'z_reio':     planck18_cosmology["z_reion"]}

# Call class
from classy import Class

CLASS_inst = Class()
CLASS_inst.set(class_parameters)
CLASS_inst.compute()

# z array
z_ary_log     = np.logspace(-6, 8, 100000)



# Add 0(today) at the beginning of z_ary
z_ary = np.insert(z_ary_log,0,[0])

# Xe array
Xe_ary   = np.array([ CLASS_inst.ionization_fraction(z) for z in z_ary ])

# XeH array
# (If Xe>=1, replace it with 1)
XeH_ary = np.where(Xe_ary<=1,Xe_ary,1)

# Xe interpolation
Xe_interp  = interp1d(z_ary, Xe_ary, fill_value="extrapolate")

# XeH interpolation
XeH_interp = interp1d(z_ary, XeH_ary, fill_value="extrapolate")

# dXe/dz array
# (with the length of z_ary[:-1])
dXe_dz_ary = np.diff(Xe_ary)/np.diff(z_ary)

# dXeH/dz array
# (with the length of z_ary[:-1])
dXeH_dz_ary = np.diff(XeH_ary)/np.diff(z_ary)


# dXe/dz interpolation
dXe_dz_interp  = interp1d(z_ary[:-1], dXe_dz_ary, fill_value="extrapolate")

# # dXeH/dz interpolation
dXeH_dz_interp  = interp1d(z_ary[:-1],  dXeH_dz_ary, fill_value="extrapolate")



# n_e function [cm^-3]
def n_e(z):
    
    n_e_z = Xe_interp(z) * n_H(z)
    
    return n_e_z   # in cm^-3


ne0 = n_e(0)

# mA^2 at z=0
mASq0 = ee**2 * (ne0 * cmInv_to_eV**3)/m_e  # in eV^2


# m_Aprime_res for RAD universe(Approx)
def m_Aprime_res_RAD(z_res):
    
    m_Aprime_res = np.sqrt( mASq0 * (1+z_res)**3 )
    
    return m_Aprime_res   # in eV


def rho_gamma(T):
    
    return (np.pi**2/15) * T**4