import sys
sys.path.insert(1, '../')

from pckgs.import_pckg import *
from pckgs.units import *
import FIRAS
from const_func_general import *
from cons_func_distortion import *



# Importing probability

file_name = "../data/Probability.npz"
if file_name is not None:
    file = np.load(file_name)
# x:   1D array
x_1Dary_P_import = file['x_P_scan_ary']

# -------------------------------------------------------
# mAp: 1D array
mAp_1Dary_P_import = file['mAp_P_scan_ary']

# -------------------------------------------------------
# P/eps^2: 2D array
P_over_eps2_2Dary_import = file['P_over_eps2_scan_2Dary']


zres_2Dary_import=file['zres_scan_2Dary']


# Interpolation: Log10(P/eps^2), Log10(z_res)
# Input: Log10(mAp), x

Reg_trans= 10**(-100)

log10P_over_eps2_interp = interp2d( np.log10(mAp_1Dary_P_import), x_1Dary_P_import, np.log10(P_over_eps2_2Dary_import + Reg_trans) )
log10zres_interp        = interp2d( np.log10(mAp_1Dary_P_import), x_1Dary_P_import, np.log10(zres_2Dary_import + Reg_trans)  )

# P/eps^2 from 2D interpolation
# Input: Log10(mAp), x
def P_over_eps2_interp(mAp, x):
    
    P_over_eps2_interp = 10**log10P_over_eps2_interp( np.log10(mAp), x )
    
    return P_over_eps2_interp

# zres from 2D interpolation
# Input: Log10(mAp), x
# Note! Here, 'zres_interp' only picks the minimal zres when there is multi solution
def zres_interp(mAp,x):
    
    zres_interp = 10**log10zres_interp( np.log10(mAp), x )
    
    return zres_interp



# rtrans_xp1em1 = 100

rtrans_xp1em1 = 100

# Power = -2
xp_z_trans_r_trans_m2_list = np.transpose([ [0.001,2.92e5,2.97], [0.01,2.10e5,5.34], [0.1,1.97423492e5,rtrans_xp1em1], [1,1.42e5,6.05], [5, 1.00e5, 5.23], [15, 9.98e4, 2.49] ])

log10_xp_z_trans_m2_interp = interp1d(np.log10(xp_z_trans_r_trans_m2_list[0]), np.log10(xp_z_trans_r_trans_m2_list[1]), fill_value="extrapolate")
log10_xp_r_trans_m2_interp = interp1d(np.log10(xp_z_trans_r_trans_m2_list[0]), np.log10(xp_z_trans_r_trans_m2_list[2]), fill_value="extrapolate")

# Power = -1
xp_z_trans_r_trans_m1_list = np.transpose([ [0.001,3.09e5,3.28], [0.01,2.25e5,7.13], [0.1,1.93335107e5,rtrans_xp1em1], [1,1.31e5,5.29], [5,8.46e4,5.49], [15,9.99e4,2.49]])

log10_xp_z_trans_m1_interp = interp1d(np.log10(xp_z_trans_r_trans_m1_list[0]), np.log10(xp_z_trans_r_trans_m1_list[1]), fill_value="extrapolate")
log10_xp_r_trans_m1_interp = interp1d(np.log10(xp_z_trans_r_trans_m1_list[0]), np.log10(xp_z_trans_r_trans_m1_list[2]), fill_value="extrapolate")

# Power = 0
xp_z_trans_r_trans_0_list = np.transpose([ [0.001,4.1e5,0.732], [0.01,2.30e5,8.42], [0.1,1.93335107e5,rtrans_xp1em1], [1,1.17e5,3.100], [5,7.49e4,4.72], [15, 9.36e4, 2.49] ])

log10_xp_z_trans_0_interp = interp1d(np.log10(xp_z_trans_r_trans_0_list[0]), np.log10(xp_z_trans_r_trans_0_list[1]), fill_value="extrapolate")
log10_xp_r_trans_0_interp = interp1d(np.log10(xp_z_trans_r_trans_0_list[0]), np.log10(xp_z_trans_r_trans_0_list[2]), fill_value="extrapolate")

# Power = 1
xp_z_trans_r_trans_1_list = np.transpose([ [0.001,7.1e5,0.27], [0.01,2.31e5,8.88], [0.1,1.93662158e5,rtrans_xp1em1], [1,1.03e5,2.53], [5,6.65e4,3.18], [15,9.0e4,2.55] ])

log10_xp_z_trans_1_interp = interp1d(np.log10(xp_z_trans_r_trans_1_list[0]), np.log10(xp_z_trans_r_trans_1_list[1]), fill_value="extrapolate")
log10_xp_r_trans_1_interp = interp1d(np.log10(xp_z_trans_r_trans_1_list[0]), np.log10(xp_z_trans_r_trans_1_list[2]), fill_value="extrapolate")

# Power = 2
xp_z_trans_r_trans_2_list = np.transpose([ [0.001,8.62e5,0.195], [0.01,2.31e5,9.00], [0.1,1.93662158e5,rtrans_xp1em1], [1, 9.03e4, 1.94], [5, 5.12e4, 1.46], [15,8.89e4,2.55] ])


log10_xp_z_trans_2_interp = interp1d(np.log10(xp_z_trans_r_trans_2_list[0]), np.log10(xp_z_trans_r_trans_2_list[1]), fill_value="extrapolate")
log10_xp_r_trans_2_interp = interp1d(np.log10(xp_z_trans_r_trans_2_list[0]), np.log10(xp_z_trans_r_trans_2_list[2]), fill_value="extrapolate")

# ==============================================================================
# ==============================================================================

# ------------------------------------------------------------------------------
# Power = -2
def z_trans_m2_Func(xp):
    
    xp_list         = xp_z_trans_r_trans_m2_list[0]
    z_trans_list    = xp_z_trans_r_trans_m2_list[1]
    r_trans_list    = xp_z_trans_r_trans_m2_list[2]
    
    xp_z_trans_func = lambda xp: 10**log10_xp_z_trans_m2_interp(np.log10(xp))
    xp_r_trans_func = lambda xp: 10**log10_xp_r_trans_m2_interp(np.log10(xp))
    
    [xp_min, z_trans_min, r_trans_min] = [ xp_list[0]  , z_trans_list[0]  , r_trans_list[0]  ]
    [xp_max, z_trans_max, r_trans_max] = [ xp_list[-1] , z_trans_list[-1] , r_trans_list[-1] ]
    
    z_trans_ext_list = np.where( xp<=xp_min, z_trans_min, np.where( xp<=xp_max, xp_z_trans_func(xp), z_trans_max ) )

    return z_trans_ext_list


def r_trans_m2_Func(xp):
    
    xp_list         = xp_z_trans_r_trans_m2_list[0]
    z_trans_list    = xp_z_trans_r_trans_m2_list[1]
    r_trans_list    = xp_z_trans_r_trans_m2_list[2]
    
    xp_z_trans_func = lambda xp: 10**log10_xp_z_trans_m2_interp(np.log10(xp))
    xp_r_trans_func = lambda xp: 10**log10_xp_r_trans_m2_interp(np.log10(xp))
    
    [xp_min, z_trans_min, r_trans_min] = [ xp_list[0]  , z_trans_list[0]  , r_trans_list[0]  ]
    [xp_max, z_trans_max, r_trans_max] = [ xp_list[-1] , z_trans_list[-1] , r_trans_list[-1] ]
    
    r_trans_ext_list = np.where( xp<=xp_min, r_trans_min, np.where( xp<=xp_max, xp_r_trans_func(xp), r_trans_max ) )

    return r_trans_ext_list

# ------------------------------------------------------------------------------
# Power = -1
def z_trans_m1_Func(xp):
    
    xp_list         = xp_z_trans_r_trans_m1_list[0]
    z_trans_list    = xp_z_trans_r_trans_m1_list[1]
    r_trans_list    = xp_z_trans_r_trans_m1_list[2]
    
    xp_z_trans_func = lambda xp: 10**log10_xp_z_trans_m1_interp(np.log10(xp))
    xp_r_trans_func = lambda xp: 10**log10_xp_r_trans_m1_interp(np.log10(xp))
    
    [xp_min, z_trans_min, r_trans_min] = [ xp_list[0]  , z_trans_list[0]  , r_trans_list[0]  ]
    [xp_max, z_trans_max, r_trans_max] = [ xp_list[-1] , z_trans_list[-1] , r_trans_list[-1] ]
    
    z_trans_ext_list = np.where( xp<=xp_min, z_trans_min, np.where( xp<=xp_max, xp_z_trans_func(xp), z_trans_max ) )

    return z_trans_ext_list


def r_trans_m1_Func(xp):
    
    xp_list         = xp_z_trans_r_trans_m1_list[0]
    z_trans_list    = xp_z_trans_r_trans_m1_list[1]
    r_trans_list    = xp_z_trans_r_trans_m1_list[2]
    
    xp_z_trans_func = lambda xp: 10**log10_xp_z_trans_m1_interp(np.log10(xp))
    xp_r_trans_func = lambda xp: 10**log10_xp_r_trans_m1_interp(np.log10(xp))
    
    [xp_min, z_trans_min, r_trans_min] = [ xp_list[0]  , z_trans_list[0]  , r_trans_list[0]  ]
    [xp_max, z_trans_max, r_trans_max] = [ xp_list[-1] , z_trans_list[-1] , r_trans_list[-1] ]
    
    r_trans_ext_list = np.where( xp<=xp_min, r_trans_min, np.where( xp<=xp_max, xp_r_trans_func(xp), r_trans_max ) )

    return r_trans_ext_list


# ------------------------------------------------------------------------------
# Power = 0
def z_trans_0_Func(xp):
    
    xp_list         = xp_z_trans_r_trans_0_list[0]
    z_trans_list    = xp_z_trans_r_trans_0_list[1]
    r_trans_list    = xp_z_trans_r_trans_0_list[2]
    
    xp_z_trans_func = lambda xp: 10**log10_xp_z_trans_0_interp(np.log10(xp))
    xp_r_trans_func = lambda xp: 10**log10_xp_r_trans_0_interp(np.log10(xp))
    
    [xp_min, z_trans_min, r_trans_min] = [ xp_list[0]  , z_trans_list[0]  , r_trans_list[0]  ]
    [xp_max, z_trans_max, r_trans_max] = [ xp_list[-1] , z_trans_list[-1] , r_trans_list[-1] ]
    
    z_trans_ext_list = np.where( xp<=xp_min, z_trans_min, np.where( xp<=xp_max, xp_z_trans_func(xp), z_trans_max ) )

    return z_trans_ext_list


def r_trans_0_Func(xp):
    
    xp_list         = xp_z_trans_r_trans_0_list[0]
    z_trans_list    = xp_z_trans_r_trans_0_list[1]
    r_trans_list    = xp_z_trans_r_trans_0_list[2]
    
    xp_z_trans_func = lambda xp: 10**log10_xp_z_trans_0_interp(np.log10(xp))
    xp_r_trans_func = lambda xp: 10**log10_xp_r_trans_0_interp(np.log10(xp))
    
    [xp_min, z_trans_min, r_trans_min] = [ xp_list[0]  , z_trans_list[0]  , r_trans_list[0]  ]
    [xp_max, z_trans_max, r_trans_max] = [ xp_list[-1] , z_trans_list[-1] , r_trans_list[-1] ]
    
    r_trans_ext_list = np.where( xp<=xp_min, r_trans_min, np.where( xp<=xp_max, xp_r_trans_func(xp), r_trans_max ) )

    return r_trans_ext_list

# ------------------------------------------------------------------------------
# Power = 1
def z_trans_1_Func(xp):
    
    xp_list         = xp_z_trans_r_trans_1_list[0]
    z_trans_list    = xp_z_trans_r_trans_1_list[1]
    r_trans_list    = xp_z_trans_r_trans_1_list[2]
    
    xp_z_trans_func = lambda xp: 10**log10_xp_z_trans_1_interp(np.log10(xp))
    xp_r_trans_func = lambda xp: 10**log10_xp_r_trans_1_interp(np.log10(xp))
    
    [xp_min, z_trans_min, r_trans_min] = [ xp_list[0]  , z_trans_list[0]  , r_trans_list[0]  ]
    [xp_max, z_trans_max, r_trans_max] = [ xp_list[-1] , z_trans_list[-1] , r_trans_list[-1] ]
    
    z_trans_ext_list = np.where( xp<=xp_min, z_trans_min, np.where( xp<=xp_max, xp_z_trans_func(xp), z_trans_max ) )

    return z_trans_ext_list

def r_trans_1_Func(xp):
    
    xp_list         = xp_z_trans_r_trans_1_list[0]
    z_trans_list    = xp_z_trans_r_trans_1_list[1]
    r_trans_list    = xp_z_trans_r_trans_1_list[2]
    
    xp_z_trans_func = lambda xp: 10**log10_xp_z_trans_1_interp(np.log10(xp))
    xp_r_trans_func = lambda xp: 10**log10_xp_r_trans_1_interp(np.log10(xp))
    
    [xp_min, z_trans_min, r_trans_min] = [ xp_list[0]  , z_trans_list[0]  , r_trans_list[0]  ]
    [xp_max, z_trans_max, r_trans_max] = [ xp_list[-1] , z_trans_list[-1] , r_trans_list[-1] ]
    
    r_trans_ext_list = np.where( xp<=xp_min, r_trans_min, np.where( xp<=xp_max, xp_r_trans_func(xp), r_trans_max ) )

    return r_trans_ext_list

# ------------------------------------------------------------------------------
# Power = 2
def z_trans_2_Func(xp):
    
    xp_list         = xp_z_trans_r_trans_2_list[0]
    z_trans_list    = xp_z_trans_r_trans_2_list[1]
    r_trans_list    = xp_z_trans_r_trans_2_list[2]
    
    xp_z_trans_func = lambda xp: 10**log10_xp_z_trans_2_interp(np.log10(xp))
    xp_r_trans_func = lambda xp: 10**log10_xp_r_trans_2_interp(np.log10(xp))
    
    [xp_min, z_trans_min, r_trans_min] = [ xp_list[0]  , z_trans_list[0]  , r_trans_list[0]  ]
    [xp_max, z_trans_max, r_trans_max] = [ xp_list[-1] , z_trans_list[-1] , r_trans_list[-1] ]
    
    z_trans_ext_list = np.where( xp<=xp_min, z_trans_min, np.where( xp<=xp_max, xp_z_trans_func(xp), z_trans_max ) )

    return z_trans_ext_list

def r_trans_2_Func(xp):
    
    xp_list         = xp_z_trans_r_trans_2_list[0]
    z_trans_list    = xp_z_trans_r_trans_2_list[1]
    r_trans_list    = xp_z_trans_r_trans_2_list[2]
    
    xp_z_trans_func = lambda xp: 10**log10_xp_z_trans_2_interp(np.log10(xp))
    xp_r_trans_func = lambda xp: 10**log10_xp_r_trans_2_interp(np.log10(xp))
    
    [xp_min, z_trans_min, r_trans_min] = [ xp_list[0]  , z_trans_list[0]  , r_trans_list[0]  ]
    [xp_max, z_trans_max, r_trans_max] = [ xp_list[-1] , z_trans_list[-1] , r_trans_list[-1] ]
    
    r_trans_ext_list = np.where( xp<=xp_min, r_trans_min, np.where( xp<=xp_max, xp_r_trans_func(xp), r_trans_max ) )

    return r_trans_ext_list

# ==============================================================================
# ==============================================================================


def z_trans_Func(xp):
    
    # return z_trans_m2_Func(xp)
    return z_trans_m1_Func(xp)
    # return z_trans_0_Func(xp)
    # return z_trans_1_Func(xp)
    # return z_trans_2_Func(xp)

def r_trans_Func(xp):
    
    # return r_trans_m2_Func(xp)
    return r_trans_m1_Func(xp)
    # return r_trans_0_Func(xp)
    # return r_trans_1_Func(xp)
    # return r_trans_2_Func(xp)

# Trans_mu varying with xp
def Trans_mu_Vary(zp,xp):
    
    Trans_mu = 1 - np.exp( -((1+zp)/(1+z_trans_Func(xp)))**r_trans_Func(xp) )
    
    return Trans_mu

# Trans_y varying with xp
def Trans_y_Vary(zp,xp):
    
    Trans_y  = 1 - Trans_mu_Vary(zp,xp)
    
    return Trans_y



# Delta_I/eps^2 for RAD universe (Approx)
# x :      1D array N_x
# xp:      1D array N_xp  (integration variable)
# output:  1D array N_x

# ----------------------------------------------------------------------
# M-Term Only
# ----------------------------------------------------------------------
def DeltaI_over_eps2_mu_M_int(x, x_prime_int, m_Aprime, T0, units = 'eV_per_cmSq'):

    # Here I choose x_prime = 0
    z_res = zres_interp(m_Aprime, 0)[0]
    
    # (M-Term Green's Func)
    # Input x is array:
    # 2D array: N_x * N_xp  
    greens_mu_ary    =  greens_mu_M(x, x_prime_int, z_res, T0, units=units)
    
    # P(A->Ap)/eps^2
    # 1D array: N_xp
    P_over_eps2_ary  =  np.transpose(P_over_eps2_interp(m_Aprime, x_prime_int))[0]
    
    # Trans_mu Func
    # 1D array: N_xp
    Trans_mu_Vary_ary     =  Trans_mu_Vary(z_res, x_prime_int)

    # Delta_I(A->Ap)
    # int over xp
    # 1D array: N_x
    DeltaI_AToAp_over_eps2 = (-1)/(2*zeta(3)) * np.trapz(Trans_mu_Vary_ary * greens_mu_ary *  x_prime_int**2/(np.exp(x_prime_int)-1) * P_over_eps2_ary, x_prime_int)

    return DeltaI_AToAp_over_eps2


# Delta_I/eps^2
# x :      1D array N_x
# xp:      1D array N_xp  (integration variable)
# output:  1D array N_x

# DeltaI/eps^2: Y-part
def DeltaI_over_eps2_Y_int(x, x_prime_int, m_Aprime, T0, units = 'eV_per_cmSq'):
    
    # Here I choose x_prime = 0
    z_res = zres_interp(m_Aprime, 0)[0]
    
    # Green's Func at y-era: Y-part
    # 2D array: N_x * N_xp
    greens_Y_ary = greens_Y(x, x_prime_int, z_res, T0, units=units)
    
    # P(A->Ap)/eps^2
    # 1D array: N_xp
    P_over_eps2_ary = np.transpose( P_over_eps2_interp(m_Aprime, x_prime_int) )[0]
    
    # Trans_y Func
    # 1D array: N_xp
    Trans_y_Vary_ary     =  Trans_y_Vary(z_res, x_prime_int)
    
    # Delta_I(A->Ap)
    # int over xp
    # 1D array: N_x
    # DeltaI_over_eps2_Y = (-1)/(2*zeta(3)) * np.trapz( greens_Y_ary *  x_prime_int**2/(np.exp(x_prime_int)-1) * P_over_eps2_ary, x_prime_int)
    DeltaI_over_eps2_Y = (-1)/(2*zeta(3)) * np.trapz( Trans_y_Vary_ary * greens_Y_ary *  x_prime_int**2/(np.exp(x_prime_int)-1) * P_over_eps2_ary, x_prime_int)
    
    return DeltaI_over_eps2_Y


# Delta_I/eps^2: Doppler-part
def DeltaI_over_eps2_Doppler_int(x, x_prime_int, m_Aprime, T0, units = 'eV_per_cmSq'):
    
    # Here I choose x_prime = 0
    z_res = zres_interp(m_Aprime, 0)[0]
    
    # Green's Func at y-era: Y-part
    # 2D array: N_x * N_xp
    greens_Doppler_ary = greens_Doppler(x, x_prime_int, z_res, T0, units=units)
    
    
    # P(A->Ap)/eps^2
    # 1D array: N_xp
    P_over_eps2_ary = np.transpose( P_over_eps2_interp(m_Aprime, x_prime_int) )[0]
    
    # Trans_y Func
    # 1D array: N_xp
    Trans_y_Vary_ary     =  Trans_y_Vary(z_res, x_prime_int)
    
    # Delta_I(A->Ap)
    # int over xp
    # 1D array: N_x
    # DeltaI_over_eps2_Doppler = (-1)/(2*zeta(3)) * np.trapz( greens_Doppler_ary *  x_prime_int**2/(np.exp(x_prime_int)-1) * P_over_eps2_ary, x_prime_int)
    DeltaI_over_eps2_Doppler = (-1)/(2*zeta(3)) * np.trapz( Trans_y_Vary_ary * greens_Doppler_ary *  x_prime_int**2/(np.exp(x_prime_int)-1) * P_over_eps2_ary, x_prime_int)
    
    return DeltaI_over_eps2_Doppler


# Delta_I/eps^2: Doppler-part (small y_gamma limit)
def DeltaI_over_eps2_Doppler_delta(x, m_Aprime, T0, units = 'eV_per_cmSq'):
    
    if   units == 'eV_per_cmSq': # units:    eV * cm^-2 * sr^-1
        prefac = 1
    elif units == 'SI':          # SI units: kg * s^-2 * sr^-1
        prefac = eV_to_J * (1/cm_to_m)**2
    elif units == 'MJy':         # units:    MJy * sr^-1
        prefac = eV_to_J * (1/cm_to_m)**2 * 1e26 / 1e6
    
    # Here I choose x_prime = 0
    z_res = zres_interp(m_Aprime, 0)[0]
    
    # P(A->Ap)/eps^2
    # 1D array: N_x
    # Analytically, we integrate out delta(x_prime -x)
    P_over_eps2_num = np.transpose(P_over_eps2_interp(m_Aprime, x))[0]
    
    # <<<Use T0_vary>>> in 'rho_gamma(T0)'
    # photon energy density
    rho_gamma = (np.pi**2/15) * T0**4  # in eV^4
    
    # tau_ff
    # 1D array N_xp
    # Note: Here, x_prime = x, because of delta(x_prime-x)
    tau = tau_ff(x, z_res)
    
    
    # Delta_I(A->Ap): small y_gamma limit
    # int over xp
    # 1D array: N_x
    # unit: eV^3
    # DeltaI_over_eps2_Doppler_eV3 = - alpha_rho/(2*zeta(3)) * rho_gamma/(4*np.pi) * (2*np.pi/T0) * np.exp(-tau) * x**3/(np.exp(x)-1) * P_over_eps2_num
    DeltaI_over_eps2_Doppler_eV3 = - Trans_y_Vary(z_res, x) * alpha_rho/(2*zeta(3)) * rho_gamma/(4*np.pi) * (2*np.pi/T0) * np.exp(-tau) * x**3/(np.exp(x)-1) * P_over_eps2_num
    
    # units = units(input)
    DeltaI_over_eps2_Doppler = prefac * DeltaI_over_eps2_Doppler_eV3 * (1/cmInv_to_eV)**2
    
    return DeltaI_over_eps2_Doppler


# Delta_I/eps^2: Doppler-part
# When y_gamma<=10^-3, switch approximate Doppler term with delta function
def DeltaI_over_eps2_Doppler_switch(x, x_prime_int, m_Aprime, T0, units = 'eV_per_cmSq'):
    
    # Here I choose x_prime = 0
    z_res = zres_interp(m_Aprime, 0)[0]
    
    # Compton y-parameter (photon)
    y = y_gamma(z_res)
    
    # Determine whether do delta-func approximation
    Approx_Det = (y <= 10**(-3))
    
    # If y<<1   (Approx_Det = True ), we use delta func approx for Doppler,
    # Otherwise (Approx_Det = False), we do numerical integral for Doppler
    DeltaI_over_eps2_Doppler  = np.where(Approx_Det
                          ,DeltaI_over_eps2_Doppler_delta(x,m_Aprime,T0,units=units)
                          ,DeltaI_over_eps2_Doppler_int(x,x_prime_int,m_Aprime,T0,units=units))
    
    return DeltaI_over_eps2_Doppler


# DeltaI/eps^2 (Total)
# Full Integration
def DeltaI_over_eps2_y_int(x, x_prime_int, m_Aprime, T0, units = 'eV_per_cmSq'):
    
    # Delta_I(A->Ap)
    # int over xp
    # 1D array: N_x
    DeltaI_over_eps2_y_int = DeltaI_over_eps2_Y_int(x,x_prime_int,m_Aprime,T0,units=units) + DeltaI_over_eps2_Doppler_int(x,x_prime_int,m_Aprime,T0,units=units)
    
    return DeltaI_over_eps2_y_int


# Delta_I/eps^2 (Total):
# Y-contribution(int) + Doppler-contribution(switch)
def DeltaI_over_eps2_y_switch(x, x_prime_int, m_Aprime, T0, units = 'eV_per_cmSq'):
    
    DeltaI_over_eps2  = DeltaI_over_eps2_Y_int(x,x_prime_int,m_Aprime,T0,units=units) + DeltaI_over_eps2_Doppler_switch(x,x_prime_int,m_Aprime,T0,units=units)

    return DeltaI_over_eps2

def DeltaI_over_eps2_muy_trans(x, x_prime_int, m_Aprime, T0, units = 'eV_per_cmSq'):
    
    # Here I choose x_prime = 0
    z_res = zres_interp(m_Aprime, 0)[0]
    
    # Check whether in the pure-mu region
    mu_era_Check = (z_res > z_trans_2)
    
    # In the mu era, we directly choose Delta_I = Delta_I_mu
    DeltaI_over_eps2_muy_trans =  np.where(mu_era_Check,
                                        DeltaI_over_eps2_mu_M_int(x, x_prime_int, m_Aprime, T0, units = units),
                                        ( DeltaI_over_eps2_mu_M_int(x, x_prime_int, m_Aprime, T0, units = units) 
                                        + DeltaI_over_eps2_y_switch(x, x_prime_int, m_Aprime, T0, units = units) ) 
                                              )
    return DeltaI_over_eps2_muy_trans


# [[[This is what we used in chi^2 analysis!!!]]]
def I0_dist_muy_trans(x, x_prime_int, m_Aprime, eps, T0, units = 'eV_per_cmSq'):
    
    # CMB intensity after distortion
    # 1D array: N_x
    I0_dist_y = I0(x, T0, units=units) + eps**2 * DeltaI_over_eps2_muy_trans(x, x_prime_int, m_Aprime, T0, units = units)
    
    return I0_dist_y

def Ttrans_y_McDermott(z):
    
    Ttrans_y = 1/( 1 + ((1+z)/6e4)**2.58 )
    
    return Ttrans_y


def Ttrans_mu_Chluba(z):
    
    z_trans     = 5.8 * 10**4
    Power_trans = 1.88
    
    Trans_mu = Ttrans_mu_para(z,z_trans,Power_trans)
    
    return Trans_mu


def Ttrans_mu_McDermott(z):
    
    Ttrans_mu = 1 - Ttrans_y_McDermott(z)
    
    return Ttrans_mu

# def Ttrans_mu_McDermott(z):
    
#     Ttrans_mu = Ttrans_mu_Chluba(z)
    
#     return Ttrans_mu



# Exact calculation of mu from thermal energy injection
def mu_th(x_prime_int, m_Aprime, eps):
    
    # Here I choose x_prime = 0
    z_res = zres_interp(m_Aprime, 0)[0]
    
    # P(A->Ap)/eps^2
    # 1D array: N_xp
    P_over_eps2_ary = np.transpose( P_over_eps2_interp(m_Aprime, x_prime_int) )[0]
    
    mu_th = - (3/kappa_c) * J_bb(z_res) * (15/np.pi**4) * eps**2 * np.trapz( x_prime_int**3/(np.exp(x_prime_int) -1) * P_over_eps2_ary, x_prime_int)
    
    return mu_th

# Exact calculation of y from thermal energy injection
def y_th(x_prime_int, m_Aprime, eps):
    
    # Here I choose x_prime = 0
    z_res = zres_interp(m_Aprime, 0)[0]
    
    # P(A->Ap)/eps^2
    # 1D array: N_xp
    P_over_eps2_ary = np.transpose( P_over_eps2_interp(m_Aprime, x_prime_int) )[0]
    
    y_th = - (1/4) * (15/np.pi**4) * eps**2 * np.trapz( x_prime_int**3/(np.exp(x_prime_int) -1) * P_over_eps2_ary, x_prime_int)
    
    return y_th


def DeltaI_over_eps2_mu_th(x, x_prime_int, m_Aprime, T0, units = 'eV_per_cmSq'):
    
    DeltaI_over_eps2 =  mu_th(x_prime_int, m_Aprime, 1) * M(x, T0, units=units)[:, None]
    
    return DeltaI_over_eps2


def DeltaI_over_eps2_y_th(x, x_prime_int, m_Aprime, T0, units = 'eV_per_cmSq'):
    
    DeltaI_over_eps2 =  y_th(x_prime_int, m_Aprime, 1) * Y(x, T0, units=units)[:, None]
    
    return DeltaI_over_eps2