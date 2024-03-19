import sys
sys.path.insert(1, '../')

from pckgs.import_pckg import *
from pckgs.units import *
import FIRAS
from const_func_general import *





#these constants and functions are specific to distortion

alpha_rho = zeta(3)/(3*zeta(4))
alpha_mu  = zeta(2)/(3*zeta(3))
x_0 = (4*zeta(4))/zeta(3)
kappa_c = (45/np.pi**4)*( 2*np.pi**6/(135*zeta(3)) - 6*zeta(3) )


#visibility function
def J_bb(z):
    
    z_mu = 1.98e6
    
    return 0.983 * np.exp(-(z/z_mu)**2.5) * (1 - 0.0381 * (z/z_mu)**2.29)


# Surviving Probability
# From 1506.06582
def survival_prob(x, z):
    
    xc_DC = 8.60e-3 * ((1.+z)/2e6)**0.5
    xc_BR = 1.23e-3 * ((1.+z)/2e6)**(-0.672)
    
    xc = np.sqrt(xc_DC**2 + xc_BR**2)
    
    return np.exp(-xc/x)


def lambda_func(x_prime, z_prime):
    
    return alpha_rho * (x_prime - (x_prime - x_0 * survival_prob(x_prime, z_prime)) * J_bb(z_prime))


# Temperature Shift
def T_shift(x):
    
    T_shift = np.where(x<=300, x*np.exp(x)/(np.exp(x) - 1), x)
    
    return T_shift




# Black Body Intensity
def I0(x, T0, units='eV_per_cmSq'):
    
    if   units == 'eV_per_cmSq': # units:    eV * cm^-2 * sr^-1
        prefac = 1
    elif units == 'SI':          # SI units: kg * s^-2 * sr^-1
        prefac = eV_to_J * (1/cm_to_m)**2
    elif units == 'MJy':         # units:    MJy * sr^-1
        prefac = eV_to_J * (1/cm_to_m)**2 * 1e26 / 1e6
        
    I0 = np.where(x<=300, prefac * 2 * T0**3 / h**2 / c**2  * x**3 / (np.exp(x)-1), 0 )
        
    return I0


# rho_bar(x):
# Normalized and Dimless Intensity, from 2306.13135
def rho_bar(x):
    
    return (15/np.pi**4) * x**3/(np.exp(x)-1)


# G-function from 1506.06582
def G(x, T0, units='eV_per_cmSq'):

    return I0(x, T0, units=units) * T_shift(x)


# M-function from 1506.06582
def M(x, T0, units='eV_per_cmSq'):
    
    return G(x, T0, units=units) * (alpha_mu - 1./x)


# Y-function from 1506.06582
def Y(x, T0, units='eV_per_cmSq'):
    
    Y = G(x, T0, units=units) * ( x/np.tanh(x/2)-4 )
    
    return Y

def f(x):
    
    return np.exp(-x) * ( 1 + x**2/2 )

# Compton-y parameter (Photon)
# Modified, ``x_e(z_to_int) * ne0 * (1.0 + z_to_int) ** 3'' with ``n_e(z)''
def y_gamma(z):
    
    z_to_int = np.logspace(-5, np.log10(z), 500)
    
    fac_1 = TCMB_0 * (1.0 + z_to_int) / m_e

    fac_2 = thomson_xsec * n_e(z_to_int) * c / ( hubble(z_to_int) / hbar * (1.0 + z_to_int))

    return np.trapz(fac_1 * fac_2, z_to_int)


# T_e: Electron Temperature
# in eV   
def T_e(z):
    
    try:
        
        _ = iter(z)
        
        T_out = np.zeros_like(z)

        T_out[z  < 2999] = np.interp(z[z < 2999], np.flipud(x_e_data[0]) - 1.0, np.flipud(x_e_data[1]))
        T_out[z >= 2999] = (1.0 + z[z >= 2999]) * TCMB_0
    
    except:
        
        if z < 2999:
            
            T_out = np.interp(z, np.flipud(x_e_data[0]) - 1.0, np.flipud(x_e_data[1]))

        else:
            
            T_out = (1.0 + z) * TCMB_0

    return T_out


# Approx gff for high x (Draine)
def approx_high_x_Draine(x, T_e):
    
    theta_e = T_e / m_e
        
    return np.log(np.exp(1.0) + np.exp(5.960 - np.sqrt(3) / np.pi * np.log(270.783 * x * theta_e ** (-0.5))))


# Approx gff for low x (Draine)
def approx_low_x_Draine(x, T_e):
    
    theta_e = T_e / m_e
    
    return 4.691 * (1.0 - 0.118 * np.log(27.0783 * x * theta_e ** (-0.5)))


# Gaunt factor: g_ff   
# x    : 1D array N_x
# g_ff : 1D array N_x
def g_ff(x, T_e):  # here we define the Gaunt factor as in https://astrojacobli.github.io/astro-ph/ISM/Draine.pdf, sec.10.2
    
    # thr = 1e-3
    
    g_ff_out = approx_high_x_Draine(x, T_e)

    return g_ff_out


# Lambda_BR          
# x    : 1D array N_x
# XeH  : 1D array N_XeH

# [[[Warning: Xe is the ionization ratio, not the dimless frequency]]]
def Lambda_BR(x, XeH, z):
    
    theta_e = T_e(z) / m_e
    
    # Number density of free proton
    # 1D array N_XeH

    n_free_p = n_H(z) * XeH  # in cm^-3

    # 2D array N_x * 1
    x_electron = x[:,None] * TCMB_0 * (1.0 + z) / T_e(z)

    # 2D array N_x * N_Xe
    Lambda_BR = alpha * lambda_e ** 3 / (2 * np.pi * np.sqrt(6 * np.pi)) * n_free_p * theta_e ** (-7.0 / 2.0) * g_ff(x_electron, T_e(z))
    
    return Lambda_BR


# tau_ff
# x    : 1D array N_x
# z    : Number
def tau_ff(x, z):
    
    N_z_to_int = 500
    
    # 1D array N_z_to_int:  small number to z
    z_to_int = np.logspace(-5, np.log10(z), N_z_to_int)
    
    # 2D array N_x * N_z_to_int
    # [:, None] make x to transfer from 1D array to N_x * 1 2D array
    x_electron = x[:,None] * TCMB_0 * (1.0 + z_to_int) / T_e(z_to_int)

    # 2D array N_x * N_z_to_int

    Lambda = Lambda_BR(x, XeH_interp(z_to_int), z_to_int)

    # 2D array N_x * N_z_to_int
    fac_1 = Lambda * (1 - np.exp(-x_electron)) / x_electron ** 3

    # 1D array N_z_to_int

    fac_2 = thomson_xsec * n_e(z_to_int) * c / (hubble(z_to_int) / hbar * (1.0 + z_to_int))
    
    # 1D array N_x
    # ( Note that fac_1 * fac_2 is a 2D array N_x * N_z_to_int )
    tau_ff = np.trapz(fac_1 * fac_2, z_to_int)
    
    return tau_ff





# Plasma mass^2 [eV^2]
# Input omega0 in eV
def mAsq(z, omega0):
    
    # from free electron
    mAsq_1 = 1.4*10**(-21) * Xe_interp(z) * n_H(z)
    
    # from neutral hydrogen
    mAsq_2 = - 8.4 * 10**(-24) * ( omega0 * (1+z) )**2 * (1-XeH_interp(z) ) * n_H(z)
    
    # total mA^2
    mAsq = mAsq_1 + mAsq_2
    
    return mAsq


# d(Plasmon mass^2)/dz [eV^2]
# Input omega0 in eV
def dmAsq_over_dz(z, omega0):
    
    # from free electron
    dmAsq_over_dz_1 = 1.4*10**(-21)*( dXe_dz_interp(z) + 3*Xe_interp(z)/(1+z) ) * n_H(z)
    
    # from neutral hydrogen
    dmAsq_over_dz_2 = - 8.4 * 10**(-24) * ( omega0 * (1+z) )**2 * ( -dXeH_dz_interp(z) + 5*(1-XeH_interp(z))/(1+z) ) * n_H(z)
    
    # total d(mA^2)/dz
    dmAsq_over_dz = dmAsq_over_dz_1 + dmAsq_over_dz_2
    
    return dmAsq_over_dz


# Get z at A-Ap crossings
# Modified from 'grf.py' file in https://github.com/smsharma/dark-photons-perturbations

# z_ary: Grids to check whether the level crossing exists

def get_z_crossings(mAp, omega0, z_ary = np.logspace(-3, 8, 10000)):
    
    # mA^2 on grids
    mAsq_ary = mAsq(z_ary, omega0)
    
    # Find the grid containing level crossing
    # np.where is used to find the position of 'True' in array

    where_ary =  np.where( np.logical_or( 
            (mAsq_ary[:-1]<mAp**2) * (mAsq_ary[1:]>mAp**2), (mAsq_ary[:-1]>mAp**2) * (mAsq_ary[1:]<mAp**2) ) )
    
    # mA^2-mAp^2 on grids
    def mAsq_minus_mApsq_ary(z):
        
        return mAsq(z, omega0) - mAp**2
    
    # We use list, because '.append' can be applied in list, but not array
    z_cross_list = []
    
    # Solve z_cross for each grid containing level crossing
    for i in range(len(where_ary[0])):
        z_cross_list.append(
            optimize.brenth(mAsq_minus_mApsq_ary, z_ary[where_ary[0][i]], z_ary[where_ary[0][i] + 1] ) )

    
    # Make list to array
    z_cross_ary = np.array(z_cross_list)
    
    return z_cross_ary

# Perturbative P(A->Ap): Needs to input z_res_ary
def P_pre_over_eps2(mAp, x, z_res_ary):
    
    # omega at z=0
    omega0 = x * TCMB_0  # in eV
    
    # omega at z_res
    omega_res_ary = omega0 * (1+z_res_ary) # in eV
    
    # P/eps^2 ary
    P_pre_over_eps2_ary = (np.pi * mAp**4)/( omega_res_ary * (1+z_res_ary) * hubble(z_res_ary) ) * ( 1 / np.abs(dmAsq_over_dz(z_res_ary, omega0)) ) 
    

    
    # P/eps^2
    P_pre_over_eps2  = np.sum(P_pre_over_eps2_ary)
    
    return P_pre_over_eps2


# Perturbative P(A->Ap):  z_res_ary is solved inside
def P_over_eps2(mAp, x):
    
    # omega at z=0
    omega0 = x * TCMB_0  # in eV
    
    # z_res array
    z_res_ary = get_z_crossings(mAp, omega0)
    
    # P_tot
    P_over_eps2 = P_pre_over_eps2(mAp, x, z_res_ary)
    
    return P_over_eps2



# Green's Function in mu-era: M-part
def greens_mu_M(x, x_prime, z_prime,  T0, units='eV_per_cmSq'):
    
    # 2D array  N_x * N_xp
    # Note that:
    # f(x)            : 1D array N_x
    # f(x)[:,None]    : 2D array N_x * N_xp
    greens_mu_ary = (   3 * alpha_rho / kappa_c
        * (x_prime - x_0 * survival_prob(x_prime, z_prime))
        * J_bb(z_prime) * M(x, T0, units = units)[:, None]   )
    
    return greens_mu_ary

# Green's Function in mu-era: T-part
def greens_mu_T(x, x_prime, z_prime,  T0, units='eV_per_cmSq'):
    
    # 2D array  N_x * N_xp
    # Note that:
    # f(x)            : 1D array N_x
    # f(x)[:,None]    : 2D array N_x * N_xp
    greens_mu_ary = ( lambda_func(x_prime, z_prime) * G(x, T0, units = units)[:, None] / 4. )
    
    return greens_mu_ary

# Green's Function in mu-era: M+T
def greens_mu_MT(x, x_prime, z_prime,  T0, units='eV_per_cmSq'):
    
    # 2D array  N_x * N_xp
    
    greens_mu_M_ary =  greens_mu_M(x, x_prime, z_prime, T0, units=units)
    
    greens_mu_T_ary =  greens_mu_T(x, x_prime, z_prime, T0, units=units)
    
    greens_mu_ary   =  greens_mu_M_ary + greens_mu_T_ary
    
    return greens_mu_ary


# alpha function from 2306.13135 and 1506.06582
def alpha_func(x_prime,z_prime):
    
    # Compton y-parameter (photon)
    y = y_gamma(z_prime)
    
    # alpha
    alpha_func = ( 3 - 2 * f(x_prime) )/np.sqrt(1+x_prime * y)
    
    return alpha_func


# beta function from 2306.13135 and 1506.06582
def beta_func(x_prime,z_prime):
    
    # Compton y-parameter (photon)
    y = y_gamma(z_prime)
    
    # beta
    beta_func = 1/( 1 + x_prime * y * (1-f(x_prime)) )
    
    return beta_func


# Green's function: y-era (Y-part)
# x   : 1D array N_x
# xp  : 1D array N_xp
# T0  : Number  # in eV
def greens_Y(x,x_prime,z_prime,T0,units="eV_per_cmSq"):
    
    if   units == 'eV_per_cmSq': # units:    eV * cm^-2 * sr^-1
        prefac = 1
    elif units == 'SI':          # SI units: kg * s^-2 * sr^-1
        prefac = eV_to_J * (1/cm_to_m)**2
    elif units == 'MJy':         # units:    MJy * sr^-1
        prefac = eV_to_J * (1/cm_to_m)**2 * 1e26 / 1e6    
    
    # Compton y-parameter (photon)
    y = y_gamma(z_prime)
    
    # photon energy density
    rho_gamma = (np.pi**2/15) * T0**4 /( hbar*c )**3  # in eV/cm^3
    
    # tau_ff
    # 1D array N_xp
    tau = tau_ff(x_prime, z_prime)
    
    # alpha
    alpha = (3-2 * f(x_prime))/np.sqrt(1+x_prime * y)
    
    # beta
    beta = 1/( 1 + x_prime * y * (1-f(x_prime)) )
    
    
    # 2D array: N_x * N_xp
    # We use [:,None] to make Y(x) to be a 2D N_x * 1 array
    term_Y = (1.0 - np.exp(y * (alpha + beta)) * np.exp(-tau) / (1.0 + x_prime * y)) * Y(x, T0, units="eV_per_cmSq")[:, None] / 4
    # in eV/cm^2
    
    # 2D array: N_x * N_xp
    green_Y = prefac * term_Y * x_prime * alpha_rho  
    
    return green_Y


# Green's function: y-era (Doppler-part)
# x   : 1D array N_x
# xp  : 1D array N_xp
# T0  : Number  # in eV
def greens_Doppler(x,x_prime,z_prime,T0,units="eV_per_cmSq"):
    
    if   units == 'eV_per_cmSq': # units:    eV * cm^-2 * sr^-1
        prefac = 1
    elif units == 'SI':          # SI units: kg * s^-2 * sr^-1
        prefac = eV_to_J * (1/cm_to_m)**2
    elif units == 'MJy':         # units:    MJy * sr^-1
        prefac = eV_to_J * (1/cm_to_m)**2 * 1e26 / 1e6    
        
    # Compton y-parameter (photon)
    y = y_gamma(z_prime)
    
    # photon energy density
    rho_gamma = (np.pi**2/15) * T0**4 /( hbar*c )**3  # in eV/cm^3
    
    # tau_ff
    # 1D array N_xp
    tau = tau_ff(x_prime, z_prime)
    
    # alpha
    alpha = (3-2 * f(x_prime))/np.sqrt(1+x_prime * y)
    
    # beta
    beta = 1/( 1 + x_prime * y * (1-f(x_prime)) )    
 
    # 2D array: N_x * N_xp
    # We use [:,None] to make x to be a 2D N_x * 1 array
    gaussian = np.exp(-((np.log(x[:, None] / x_prime) - alpha * y + np.log(1 + x_prime * y)) ** 2) / (4 * y * beta)) / (x_prime * np.sqrt(4 * np.pi * y * beta))
    
    # 2D array: N_x * N_xp
    term_Doppler = c * hbar * ( rho_gamma / (4*np.pi) )  * ( 2*np.pi/T0 ) * np.exp(-tau) * gaussian
    # in eV/cm^2

    # 2D array: N_x * N_xp
    green_Doppler = prefac * term_Doppler * x_prime * alpha_rho
    
    return green_Doppler


# Green's function: y-era (Total)
# x   : 1D array N_x
# xp  : 1D array N_xp
# T0  : Number  # in eV
def greens_y(x,x_prime,z_prime,T0,units="eV_per_cmSq"):
    
    if   units == 'eV_per_cmSq': # units:    eV * cm^-2 * sr^-1
        prefac = 1
    elif units == 'SI':          # SI units: kg * s^-2 * sr^-1
        prefac = eV_to_J * (1/cm_to_m)**2
    elif units == 'MJy':         # units:    MJy * sr^-1
        prefac = eV_to_J * (1/cm_to_m)**2 * 1e26 / 1e6

    # Compton y-parameter (photon)
    y = y_gamma(z_prime)
    
    # photon energy density
    rho_gamma = (np.pi**2/15) * T0**4 /( hbar*c )**3  # in eV/cm^3
    
    # tau_ff
    # 1D array N_xp
    tau = tau_ff(x_prime, z_prime)
    
    # alpha
    alpha = (3-2 * f(x_prime))/np.sqrt(1+x_prime * y)
    
    # beta
    beta = 1/( 1 + x_prime * y * (1-f(x_prime)) )
    
    # 2D array: N_x * N_xp
    # We use [:,None] to make x to be a 2D N_x * 1 array
    gaussian = np.exp(-((np.log(x[:, None] / x_prime) - alpha * y + np.log(1 + x_prime * y)) ** 2) / (4 * y * beta)) / (x_prime * np.sqrt(4 * np.pi * y * beta))

    # 2D array: N_x * N_xp
    # We use [:,None] to make Y(x) to be a 2D N_x * 1 array
    term_Y = (1.0 - np.exp(y * (alpha + beta)) * np.exp(-tau) / (1.0 + x_prime * y)) * Y(x, T0, units="eV_per_cmSq")[:, None] / 4
    # in eV/cm^2
    
    # 2D array: N_x * N_xp
    term_Doppler = c * hbar * ( rho_gamma / (4*np.pi) )  * ( 2*np.pi/T0 ) * np.exp(-tau) * gaussian
    # in eV/cm^2
    
    # 2D array: N_x * N_xp
    green_y = prefac * (term_Y + term_Doppler) * x_prime * alpha_rho
    
    return green_y


