import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import odeint
import os
from tqdm import tqdm

from utils import *
from opacity import interp_opacity

class StellarStructure:
    def __init__(self, Mstar, composition, Lstar, Rstar, M_c, Pc_guess, Tc_guess, num_iters, interpolator):
        self.Mstar = Mstar
        self.composition = composition
        self.Lstar = Lstar
        self.Rstar = Rstar
        self.M_c = M_c
        self.Pc_guess = Pc_guess
        self.Tc_guess = Tc_guess
        self.num_iters = num_iters
        self.interpolator = interpolator
        
        self.M_fit = 0.2*self.Mstar #empirically-tested fitting point, seems to result in the best convergence
        X, Y = self.composition
        Z = 1-X-Y
        self.mu = 4/(3+5*X)
        self.iteration_count = [0]
        
    def integrate_outward(self, num_iters, Pc_guess, Tc_guess):
        print("initiating outward integration")
        L_inner, P_inner, T_inner, r_inner, rho_inner, nabla = load1(self.M_c, self.composition, Pc_guess, Tc_guess, self.interpolator)

        if np.any(np.isnan((L_inner, P_inner, T_inner, r_inner, rho_inner, nabla))) or np.any(np.isinf((L_inner, P_inner, T_inner, r_inner, rho_inner, nabla))):
            print("nan", np.any(np.isnan((L_inner, P_inner, T_inner, r_inner, rho_inner, nabla))))

            print("inf", np.any(np.isinf((L_inner, P_inner, T_inner, r_inner, rho_inner, nabla))))

            print(L_inner)
            print(P_inner)
            print(T_inner)
            print(r_inner)
        M_out = np.linspace(self.M_c, self.M_fit, self.num_iters)
        #M_out = None
        outward_result = scipy.integrate.solve_ivp(derivs, t_span = (self.M_c, self.M_fit), y0=(L_inner, P_inner, T_inner, r_inner), t_eval = M_out, args=(self.composition, self.interpolator), method='Radau', 
                                                 rtol=1e-6, atol=1e-10)
        if outward_result.status == -1:
            print(f"outward integration failed with this message: {outward_soln.message}")
            return None, None
        else:
            return outward_result.y, M_out

    def integrate_inward(self, num_iters, Lstar = None, Rstar = None):
        print("initiating inward integration")
        L_outer, P_outer, T_outer, r_outer, rho_outer = load2(self.Mstar, self.composition, Lstar, Rstar, self.interpolator)
        
        M_in = np.linspace(self.Mstar, self.M_fit, self.num_iters)
        #M_in = None
        inward_result = scipy.integrate.solve_ivp(derivs, t_span = (self.Mstar, self.M_fit), y0=(L_outer, P_outer, T_outer, r_outer), t_eval = M_in, args = (self.composition, self.interpolator), method='Radau', 
                                                 rtol=1e-6, atol=1e-10)
        #inward_result = scipy.integrate.solve_ivp(derivs, t_span = (self.Mstar, self.M_fit), y0=(L_outer, P_outer, T_outer, r_outer), t_eval = M_in, args = (self.composition, self.interpolator), method='Radau')

        if inward_result.status == -1:
            print(f"inward integration failed with this message: {inward_result.message}")
            return None, None
        else:
            return inward_result.y, M_in
        
    def residuals(self, params):
        Pc, Tc, Rstar_guess, Lstar_guess = params
        if not np.isfinite(Pc) or not np.isfinite(Tc) or not np.isfinite(Rstar_guess) or not np.isfinite(Lstar_guess):
            print(f"residuals: NaN/Inf params - Pc={Pc}, Tc={Tc}")
            print(f"residuals: NaN/Inf params - Rstar={Rstar_guess}, Lstar={Lstar_guess}")
            return [1e20, 1e20, 1e20, 1e20]
    
        if Pc <= 0 or Tc <= 0 or Rstar_guess < 0 or Lstar_guess < 0:
            print(f"residuals: Non-positive params - Pc={Pc:.2e}, Tc={Tc:.2e}")
            print(f"residuals: Non-positive params - Rstar={Rstar_guess:.2e}, Lstar={Lstar_guess:.2e}")
            return [1e20, 1e20, 1e20, 1e20]
        soln_outward, M_out = self.integrate_outward(self.num_iters, Pc, Tc) 
        #print(y_outward)
        soln_inward, M_in = self.integrate_inward(self.num_iters, Lstar = Lstar_guess, Rstar = Rstar_guess)

        self.iteration_count[0] += 1
        if soln_outward is None or soln_inward is None:
            print(f"solution failed with following message: {soln_outward}")
            return [1e20, 1e20, 1e20, 1e20]
        else:
            #print(soln_outward.y[0])
            L_outward = soln_outward[0, -1]
            P_outward = soln_outward[1, -1]
            T_outward = soln_outward[2, -1]
            r_outward = soln_outward[3, -1]
            
            L_inward = soln_inward[0, -1]
            P_inward = soln_inward[1, -1]
            T_inward = soln_inward[2, -1]
            r_inward = soln_inward[3, -1]
    
            # calculate residuals
            L_diff = np.abs(L_outward - L_inward)/np.abs(L_outward)
            P_diff = np.abs(P_outward - P_inward)/np.abs(P_outward)
            T_diff = np.abs(T_outward - T_inward)/np.abs(T_outward)
            r_diff = np.abs(r_outward - r_inward)/np.abs(r_outward)
            # L_diff = (L_outward**2 - L_inward**2)
            # P_diff = (P_outward**2 - P_inward**2)
            # T_diff = (T_outward**2 - T_inward**2)
            # r_diff = (r_outward**2 - r_inward**2)

            print(f"Iter {self.iteration_count[0]:3d} | Pc={Pc:.4e} | Tc={Tc:.4e} | Lstar={Lstar_guess:.4e} | Rstar={Rstar_guess:.4e} " f"residual_P={P_diff:+.2e} | residual_T={T_diff:+.2e} | residual_L={L_diff:+.2e} | residual_R={r_diff:+.2e}")
    
            return [P_diff, T_diff, r_diff, L_diff]
    def shootf(self):
        params0 = [self.Pc_guess, self.Tc_guess, self.Rstar, self.Lstar]
        
        result = scipy.optimize.fsolve(self.residuals, params0, full_output=True)
        
        params_final, info, ier, mesg = result
        Pc_final, Tc_final, Rstar_final, Lstar_final = params_final

        # If model has converged, then we can create the final stellar model by integrating outward
        Lc_final, Pc_final, Tc_final, rc_final, rhoc_final, nabla_c_final = load1(self.M_c, self.composition, Pc_final, Tc_final, self.interpolator)
        Lstar_final, Pstar_final, T_eff, Rstar_final, rho_surface_final = load2(self.M_c, self.composition, Lstar_final, Rstar_final, self.interpolator)
        print(f"Final central params: Pc: {Pc_final:e}, Tc: {Tc_final:e}, Rc: {rc_final:e}, rho_c: {rhoc_final:e}")
        print(f"Final total params: P_surface = {Pstar_final:e}, T_eff = {T_eff:e}, Rstar = {Rstar_final:e}, rho_surface={rho_surface_final:e}")
        print(f"Constructing final model...")
        M_out = np.linspace(self.M_c, self.M_fit, self.num_iters)
        M_in = np.linspace(self.Mstar, self.M_fit, self.num_iters)
        soln_out, _ = self.integrate_outward(self.num_iters, Pc_final, Tc_final)
        soln_in, _ = self.integrate_inward(self.num_iters, Lstar = Lstar_final, Rstar = Rstar_final)
        M_in = M_in[::-1]
        soln_in = soln_in[:, ::-1] # so that solution and mass increases monotonically from M_fit

        # print(f"{R_outer_final:e}")
        # print(np.shape(soln_out))
        masses = np.concatenate((M_out, M_in[1:]))
        print(np.shape(masses))
        soln = np.concatenate((soln_out, soln_in[:, 1:]), axis=1)
        print(np.shape(soln))

        L_array = soln[0, :]
        P_array = soln[1, :]
        T_array = soln[2, :]
        R_array = soln[3, :]
        
        rho_array = calculate_rho(P_array, T_array, self.mu)
        
        # Calculate nabla values with the final stellar properties. This will enable us to determine which parts of the star are convective or radiative. 
        kappa_array = []
        e_pp_array = []
        e_cno_array = []
        X, Y = self.composition
        for T, rho in zip(T_array, rho_array):
            kappa = interp_opacity(np.log10(rho), np.log10(T), self.interpolator)
            kappa_array.append(kappa)
            
            e_pp, e_cno = calculate_energies(rho, T, X, Y)
            e_pp_array.append(e_pp)
            e_cno_array.append(e_cno)
        
        #kappa_array = interp_opacity(np.log10(rho_array), np.log10(T_array), self.interpolator)
        
        nabla_array, nablas, convective_bool = calculate_nabla(T_array, masses, L_array, P_array, kappa_array)
        
        # Calculate energy-generation rates for different temperatures in the star.


        return L_array, P_array, T_array, R_array, rho_array, masses, kappa_array, nabla_array, nablas, convective_bool, (e_pp_array, e_cno_array)
        
    