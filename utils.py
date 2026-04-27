import numpy as np
import pandas as pd

from opacity import interp_opacity

import scipy

from scipy.interpolate import RegularGridInterpolator


def calculate_energies(rho, T, X, Y):
    Z = 1-X-Y
    psi = 2 # to be adjusted based on results
    zeta = 1

    if rho is None or T is None:
        return 0.0, 0.0
    
    if not np.isfinite(rho) or not np.isfinite(T):
        return 0.0, 0.0
    
    if rho <= 0 or T <= 0:
        return 0.0, 0.0

    T7 = (T/10**7)
    T9 = (T/10**9)
    
    if T7 < 1:
        psi = 1.0
    elif T7 > 3.5:
        psi = 1.4
    elif T7 >=1 and T7 <=2:
        psi = T7
    elif T7 > 2 and T7 <= 3.5:
        psi = 1/3 * (T7 - 3.5) + 1.5
    
    f_11 = np.exp(5.92*10**(-3) * 1 * ((zeta * rho)/(T7)**3)**(1/2)) # assuming Z_1*Z_2 = 1
    g_11 = 1+3.82*(T9) + 1.51*(T9)**2 + 0.144*(T9)**3 - 0.0114*(T9)**4
    e_pp = 2.57*10**4 * psi * f_11 * g_11 * rho * X**2 * (T9)**(-2/3) * np.exp(-3.381/T9**(1/3))

    g_14 = 1 - 2.00*(T9) + 3.41 * (T9)**2 - 2.43*(T9)**3
    e_cno = 8.24 * 10**(25) * g_14 * Z * X * rho * (T9)**(-2/3) * np.exp(-15.231*(T9)**(-1/3) - (T9/0.8)**2) 
    # print(f"Energy from CNO: {e_cno}")
    # print(f"Energy from PP-chain: {e_pp}")

    return e_pp, e_cno

def P_equations(rho_values, T, M, Rstar, mu, interpolator):
    G = 6.674*10**(-8)
    K = 1.38*10**(-16)
    N_a = 6.0221408 * 10**(23)
    a = 7.56557705*10**(-15)
    
    
    rho = rho_values[0]

    R = rho/(T * 10**(-6))**3
    logR = np.log10(R)
    logT = np.log10(T)

    if not(-8.0 <= logR <= 1.0):
        print(f"logR = {logR} is invalid")
        return 10e20
    if not (3.75 <= logT <= 7.5):
        print(f"logT = {logT} is invalid")
        return 10e20
    
    kappa = interp_opacity(np.log10(rho), np.log10(T), interpolator)

    P1 = (G*M)/(Rstar**2) * (2/3) * (1/kappa)
    P2 = (N_a * K * T * rho)/mu + 1/3 * a * T**4

    return 1-(P1/P2)

def derivs(M_r, y, composition, interpolator):

    K = 1.38*10**(-16)
    N_a = 6.0221408 * 10**(23)

    a = 7.56557705*10**(-15)
    c = 2.998*10**(10)
    G = 6.674*10**(-8)
    
    L, P, T, r = y
    X, Y = composition
    mu = 4/(3+5*X)

    rho = mu/(N_a * K * T) * (P - 1/3 * a * T**4)
    
    #rho = scipy.optimize.fsolve(P_equations, x0 = [1e-10], args=(T, M_r, r, mu))[0]
    epsilon_all = calculate_energies(rho, T, X, Y)
    epsilon = epsilon_all[0] + epsilon_all[1]
    
    dPdM = -G*M_r/(4*np.pi * r**4)
    drdM = 1/(4*np.pi * r**2 * rho)
    dLdM = epsilon

    kappa = interp_opacity(np.log10(rho), np.log10(T), interpolator)

    nabla_rad = 3/(16*np.pi*a*c) * P * kappa/T**4 * L/(G * M_r)
    nabla_ad = 0.4
    if nabla_rad < nabla_ad:
        nabla = nabla_rad
    else:
        nabla = nabla_ad
    
    dTdM = (-G * M_r * T)/(4*np.pi * r**4 * P) * nabla

    # print("nan derivatives: ", np.any(np.isnan([dLdM, dPdM, dTdM, drdM])))
    # print("inf derivatives: ", np.any(np.isinf([dLdM, dPdM, dTdM, drdM])))

    # print(f"dLdM? {np.isnan(dLdM)}")
    # print(f"dPdM? {np.isnan(dPdM)}")
    # print(f"dTdM? {np.isnan(dTdM)}")
    # print(f"drdM? {np.isnan(drdM)}")

    #return [dPdM, drdM, dLdM, dTdM]
    return [dLdM, dPdM, dTdM, drdM]
    
    
    
def calculate_rho(P, T, mu):
    K = 1.38*10**(-16)
    N_a = 6.0221408 * 10**(23)
    a = 7.56557705*10**(-15)

    # Assuming an ideal gas + radiation equation of state
    rho = (P - 1/3 * a * T**4)*mu/(N_a * K * T)
    return rho

def load1(M_r, composition, Pc, Tc, interpolator):
    # Gives the value of T, L, r, and P just slightly off from the center of the star
    #print("got here")

    #Constants
    K = 1.38*10**(-16)
    N_a = 6.0221408 * 10**(23)

    a = 7.56557705*10**(-15)
    c = 2.998*10**(10)
    G = 6.674*10**(-8)

    
    X, Y = composition
    mu = 4/(3+5*X)

    nabla_adc = 0.4

    # Equation of state
    rho_c = calculate_rho(Pc, Tc, mu)
    # print(f"Central density: {rho_c}")

    #Boundary condition equations
    r_c = (3/(4*np.pi*rho_c))**(1/3) * M_r**(1/3)
    Pr = Pc - 3*G/(8*np.pi)*((4*np.pi*rho_c))**(4/3) * M_r **(2/3)
    # print(f"Pr = {Pr}")
    # print(f"R = {r_c}")

    kappa_c = interp_opacity(np.log10(rho_c), np.log10(Tc), interpolator)
    # print(f"kappa_c = {kappa_c}")
    
    epsilon_all = calculate_energies(rho_c, Tc, X, Y)
    epsilon_c = epsilon_all[0] + epsilon_all[1]
    # print(f"epsilon_c = {epsilon_c}")

    Tr = 0

    Tr_rad = (Tc**4 - 1/(2*a*c) * (3/(4*np.pi))**(2/3) * kappa_c * epsilon_c * rho_c**(4/3) * M_r**(2/3))**(1/4)
    logTr = np.log10(Tc) - (np.pi/6)**(1/3) * G * nabla_adc * rho_c**(4/3)/(Pc) * M_r**(2/3)
    Tr_conv = 10**(logTr)
    Lr = epsilon_c * M_r

    #Determine if we are in a convective or radiative layer
    nabla_rad = 3/(16*np.pi*a*c) * Pr * kappa_c/Tr_rad**4 * Lr/(G * M_r)
    nabla_ad = 0.4
    if nabla_rad < nabla_ad:
        Tr = Tr_rad
        nabla = nabla_rad
    else:
        Tr = Tr_conv
        nabla = nabla_ad
    return (Lr, Pr, Tr, r_c, rho_c, nabla)

def load2(M, composition, Lstar, Rstar, interpolator):
    X, Y = composition
    mu = 4/(3+5*X)
    K = 1.38*10**(-16)
    N_a = 6.0221408 * 10**(23)

    a = 7.56557705*10**(-15)
    c = 2.998*10**(10)
    G = 6.674*10**(-8)
    sigma = 5.67*10**(-5)

    if Lstar <= 0 or Rstar <= 0 or not np.isfinite(Lstar) or not np.isfinite(Rstar):
        print(f"  load2: Invalid inputs - Lstar={Lstar:.2e}, Rstar={Rstar:.2e}")
        
        return (1e33, 1e10, 5000, Rstar if Rstar > 0 else 1e10, 1e-10)

    T = (Lstar/(4 * np.pi * Rstar**2 * sigma))**(1/4)
    #print(f"T_eff = {T}")

    rho_surface = scipy.optimize.fsolve(P_equations, x0 = [1e-10], args=(T, M, Rstar, mu, interpolator))[0]
    #print(f"Surface density = {rho_surface}")

    kappa_surface = interp_opacity(np.log10(rho_surface), np.log10(T), interpolator)
    #print(f"kappa = {kappa_surface}")
    P = G * M/Rstar**2 * 2/3 * 1/kappa_surface

    
    #print(f"Ps = {P}")

    return (Lstar, P, T, Rstar, rho_surface)

