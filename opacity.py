import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def interp_opacity(log_rho, logT, interpolator):
    """
    Function for calculating opacities from rho and T, interpolating from the OPAL opacity tables. 
    log_rho: log density (in g cm^-3)
    logT: log temperature (in K)
    interp_k: interpolated opacity
    """

    logT_min, logT_max = interpolator.grid[0].min(), interpolator.grid[0].max()
    logR_min, logR_max = interpolator.grid[1].min(), interpolator.grid[1].max()

    if np.any(logT < logT_min) or np.any(logT > logT_max) or np.isnan(logT):
        print(f"logT = {logT} value is outside range for provided table.")
        return 1e10

    # if ((logT < 7.5) and (logT > 3.75)) and ((log_rho < 3) and (log_rho > -9)):
    T = 10**logT
    rho = 10**log_rho
    R = rho/(T/10**6)**3
    
    logR = np.log10(R)

    if np.any(logR < logR_min) or np.any(logR > logR_max) or np.isnan(logR):
        print(f"logR = {logR} value is oustide range for provided table.")
        return 1e10

    #print(f"Calculating opacity for logR = {logR:.2f} and logT = {logT:.2f}...")

    interp_k = interpolator([logT, logR])
    kappa = 10**(interp_k[0])
    #print(interp_k)
    return kappa 