#=========================Functions_Binned_Analysis.py==========================#
# Authors: Benjamín Grinstein, Carlos Miró, and Pablo Quílez
# Title: Counting axions with IAXO
# Cite: arXiv:26XX.XXXXX
# Contents:
# - Conversion factors to natural units system
# - Experimental configuration of CAST, BabyIAXO, IAXO, and IAXO+
# - Earth-Sun distance and Earth-Supernova (Spica/alpha-Virginis/HD 116658) distance
# - Differential emission spectrum (flux) of solar axions via Primakoff production
# - Differential emission spectrum (flux) of SN axions via NN-bremsstrahlung and pionic Compton scattering
# - Axion-photon conversion probability in vacuum
# - Gaussian smearing
# - Differential detection spectrum of X-rays from solar axions
# - Differential detection spectrum of gamma-rays from SN axions
# - Energy binning
# - Number of photon counts in each energy bin
# - Log-Poisson probability distribution function (pdf)
# - Log-likelihood function
# - Conditional Maximum Likelihood Estimator (CMLE) of axion mass m2
# - Discovery limits
# - Exclusion limits
# - Additional functions not used in the paper
#==============================================================================#

import numpy as np
from scipy.optimize import minimize_scalar

#==============================================================================#
# Conversion factors to natural units
# Ref. PDG
CFL = 197.3269804e-18 # [GeV m] 
CFT = 6.582119569e-25 # [GeV s]
CFB = 1.95e-16 # [GeV^2 T^-1]
#==============================================================================#

#==============================================================================#
# Experimental configuration: CAST, BabyIAXO, IAXO, IAXO+
# Ref. 2406.16840
CAST = dict(
    B = 9, # [T]
    L = 9.26, # [m]
    S = 0.0015, # [m^2]
    t = 314.6/(24*365), # [yr]
    eps_T = 0.6,
    eps_D = 0.3
)

# Ref. 2010.12076
BabyIAXO = dict(
    B = 2, # [T]
    L = 10.0, # [m]
    S = 0.77, # [m^2]
    t = 1.5*0.5, # [yr]
    eps_T = 0.7,
    eps_D = 0.35
)

# Ref. 2010.12076
IAXO = dict(
    B = 2.5, # [T]
    L = 20.0, # [m]
    S = 2.3, # [m^2]
    t = 3*0.5, # [yr]
    eps_T = 0.8,
    eps_D = 0.7
)

# Ref. 2010.12076
IAXOplus = dict(
    B = 3.5, # [T]
    L = 22.0, # [m]
    S = 3.9, # [m^2]
    t = 5*0.5, # [yr]
    eps_T = 0.8,
    eps_D = 0.7
)
#==============================================================================#

#==============================================================================#
# Earth-Sun distance
L_ES = 1.5e11 # [m]
#==============================================================================#

#==============================================================================#
# Earth-Supernova (Spica/alpha-Virginis/HD 116658) distance
# Ref. 2502.19476
L_SN = 0.077*3.09e19 # [m]
#==============================================================================#

#==============================================================================#
# Ref. 1811.09290
def AxionFlux_Primakoff_Tilde(w):
    # Parametrized differential solar axion flux [cm^-2 s^-1 keV^-1 GeV^2] (axion-photon coupling g^2 factored out)
    # w = X-ray energy [keV]
    norm = 6.02e10*(1/1e-10)**2.0 # [cm^-2 s^-1 keV^-1 GeV^2]
    return norm*(w**2.481)*np.exp(-w/1.205) # [cm^-2 s^-1 keV^-1 GeV^2]

# Ref. 1811.09290
def AxionFlux_Primakoff(w,g):
    # Parametrized differential solar axion flux [cm^-2 s^-1 keV^-1]
    # w = X-ray energy [keV]
    # g = axion-photon coupling [GeV^-1]
    return g**2.0*AxionFlux_Primakoff_Tilde(w) # [cm^-2 s^-1 keV^-1]

# Ref. 2405.02395
def SN_AxionFlux(w,g_aN,delta):
    # Parametrized differential SN axion flux [MeV^-1]
    # w = gamma-ray energy [MeV]
    # g_aN = axion-nucleon coupling [dimensionless]
    # delta = weigth of the pionic Compton scattering production mechanism
    t_grid = np.linspace(1,8,200) # [s]
    dt = t_grid[1]-t_grid[0] # [s]
    def FitParams(t,norm,alpha1,alpha2):
        return norm*(t**alpha1)*np.exp(alpha2*t)
    # NN bremsstrahlung
    A_NN_tilde = 1.75e55 # [MeV^-1 s^-1]
    A_NN = FitParams(t_grid,A_NN_tilde,1.865,-1.345) # [MeV^-1 s^-1]
    E_NN_tilde = 102.10 # [MeV]
    E_NN = FitParams(t_grid,E_NN_tilde,0.755,-0.413) # [MeV]
    beta_NN_tilde = 1.53
    beta_NN = FitParams(t_grid,beta_NN_tilde,0.0410,-0.0542)
    term_NN = (
        A_NN[:,None]
        * (g_aN/5e-10)**2
        * (w[None,:]/E_NN[:,None])**beta_NN[:,None]
        * np.exp(-(beta_NN[:,None]+1)*w[None,:]/E_NN[:,None])
    ) # [MeV^-1 s^-1]
    # pionN Compton scattering
    A_pionN_tilde = 3.88e56 # [MeV^-1 s^-1]
    A_pionN = FitParams(t_grid,A_pionN_tilde,5.975,-4.944) # [MeV^-1 s^-1]
    E_pionN_tilde = 218.59 # [MeV]
    E_pionN = FitParams(t_grid,E_pionN_tilde,0.304,-0.542) # [MeV]
    beta_pionN_tilde = 1.27
    beta_pionN = FitParams(t_grid,beta_pionN_tilde,-0.503,-0.019)
    w_pionN_tilde = 40.07 # [MeV]
    w_pionN = w_pionN_tilde*(1+1.537*t_grid**0.050) # [MeV]
    x = (w[None,:]-w_pionN[:,None])/E_pionN[:,None]
    xpos = np.clip(x,0.0,None)
    term_pionN = (
        delta 
        * A_pionN[:,None] 
        * (g_aN/5e-10)**2
        * (xpos**beta_pionN[:,None])
        * np.exp(-(beta_pionN[:,None]+1)*xpos)
    ) # [MeV^-1 s^-1]
    dNdwdt = term_NN+term_pionN  # [MeV^-1 s^-1]
    dNdw_t = 0.5*(dNdwdt[:-1,:]+dNdwdt[1:,:])*dt # [MeV^-1]
    dNdw = np.sum(dNdw_t,axis=0) # [MeV^-1]
    return dNdw # [MeV^-1]
#==============================================================================#

#==============================================================================#
def P_Tilde(w,m2,delta_m21_sq,mixing,B,L,L_source,w_res=None):
    # Axion-photon conversion probability in vacuum (axion-photon coupling g^2 factored out)
    # w = X-ray energy [keV]
    # m2 = axion 2 mass [eV] (in our convention, this corresponds to the heaviest axion)
    # delta_m21_sq = squared mass difference m2^2 - m1^2 [eV^2] (in our convention, this is positive by definition)
    # mixing = mixing angle between the 2 axions
    # B = B-field of the laboratory magnet [T]
    # L = length of the laboratory magnet [m]
    # L_source = distance to the axion source [m] (the Sun or a supernova)
    # w_res = energy resolution of the detector [keV]
    w = np.maximum(w,1e-10)
    norm = ((B*L*CFB/CFL)/2.0)**2 # [GeV^2]
    m1_sq = m2**2 - delta_m21_sq # [eV^2]
    if np.any(m1_sq < 0):
        raise ValueError("The squared mass difference delta_m21_sq cannot be greater than m2**2.")
    x1 = ((m1_sq*L)/(4.0*w))/(CFL*1e12) # dimensionless
    x2 = ((m2**2*L)/(4.0*w))/(CFL*1e12) # dimensionless
    y = ((delta_m21_sq*L_source)/(2.0*w))/(CFL*1e12) # dimensionless
    sinc1 = np.sinc(x1/np.pi) # in our expressions sinc(x) = sin(x)/x, while np.sinc(x) = sin(np.pi*x)/(np.pi*x)
    sinc2 = np.sinc(x2/np.pi) # in our expressions sinc(x) = sin(x)/x, while np.sinc(x) = sin(np.pi*x)/(np.pi*x)
    if w_res is None:
        cosy = np.cos(y)
    else:
        sigma = y*(w_res/w) # dimensionless
        cosy = np.cos(y)*np.exp(-0.5*sigma**2) # exponential damping for rapid oscillations
    cphi = np.cos(mixing)
    sphi = np.sin(mixing)
    s2phi = np.sin(2.0*mixing)
    return norm*((cphi**4)*sinc1**2 + (sphi**4)*sinc2**2 + 0.5*(s2phi**2)*sinc1*sinc2*cosy) # [GeV^2]

def P(w,g,m2,delta_m21_sq,mixing,B,L,L_source,w_res=None):
    # Axion-photon conversion probability in vacuum
    # w = X-ray energy [keV]
    # g = axion-photon coupling [GeV^-1]
    # m2 = axion 2 mass [eV] (in our convention, this corresponds to the heaviest axion)
    # delta_m21_sq = squared mass difference m2^2 - m1^2 [eV^2] (in our convention, this is positive by definition)
    # mixing = mixing angle between the 2 axions
    # B = B-field of the laboratory magnet [T]
    # L = length of the laboratory magnet [m]
    # L_source = distance to the axion source [m] (the Sun or a supernova)
    # w_res = energy resolution of the detector [keV]
    return g**2.0*P_Tilde(w,m2,delta_m21_sq,mixing,B,L,L_source,w_res=w_res) # dimensionless
#==============================================================================#

#==============================================================================#
def smear(dN,w,w_res):
    # Smeared differential spectrum of photon counts by the energy resolution of the detector
    # dN = true (theoretical) spectrum [E^-1]
    # w = energies defining the spectrum [E]
    # w_res = energy resolution of the detector [E] (same units as w)
    dw = w[1]-w[0]
    norm = 1.0/np.sqrt(2.0*np.pi*w_res**2.0)
    diff = w[None,:]-w[:,None]
    K = norm*np.exp(-0.5*(diff/w_res)**2.0) # Gaussian kernel
    # the smeared (measured) spectrum at a given energy w[j] is the convolution of the whole true spectrum with a Gaussian centered at w[j]
    f = dN[None,:]*K 
    return np.sum(0.5*(f[:,:-1]+f[:,1:])*dw,axis=1)
#==============================================================================#

#==============================================================================#
def dNdw_Tilde(w,m2,delta_m21_sq,mixing,B,L,L_source,S,t,eps_T,eps_D,res_on=None,w_res=None):
    # Differential spectrum of X-ray counts: d\tilde{N}/dw [keV^-1 GeV^4] (axion-photon coupling g^4 factored out)
    # w = X-ray energy [keV]
    # m2 = axion 2 mass [eV] (in our convention, this corresponds to the heaviest axion)
    # delta_m21_sq = squared mass difference m2^2 - m1^2 [eV^2] (in our convention, this is positive by definition)
    # mixing = mixing angle between the 2 axions
    # B = B-field of the laboratory magnet [T]
    # L = length of the laboratory magnet [m]
    # L_source = distance to the axion source [m] (the Sun or a supernova)
    # S = total aperture area of the magnet bores [m^2]
    # t = total exposure time of the telescope [yr]
    # eps_T = telescope efficiency
    # eps_D = detector efficiency
    # res_on = True or False, depending on whether the effect of the detector resolution is included or not
    # w_res = energy resolution of the detector [keV] 
    norm = (S*1e4)*(t*365*24*60*60)*eps_T*eps_D # [cm^2 s]
    dNdw = norm*AxionFlux_Primakoff_Tilde(w)*P_Tilde(w,m2,delta_m21_sq,mixing,B,L,L_source,w_res=w_res) #[keV^-1 GeV^4]
    if res_on:
        if w_res is None:
            raise ValueError("If res_on=True you must provide w_res [keV].")
        else:
            dNdw = smear(dNdw,w,w_res) # [keV^-1 GeV^4]
    return dNdw # [keV^-1 GeV^4]

def dNdw(w,g,m2,delta_m21_sq,mixing,B,L,L_source,S,t,eps_T,eps_D,res_on=None,w_res=None):
    # Differential spectrum of X-ray counts: dN/dw [keV^-1]
    # w = X-ray energy [keV]
    # g = axion-photon coupling [GeV^-1]
    # m2 = axion 2 mass [eV] (in our convention, this corresponds to the heaviest axion)
    # delta_m21_sq = squared mass difference m2^2 - m1^2 [eV^2] (in our convention, this is positive by definition)
    # mixing = mixing angle between the 2 axions
    # B = B-field of the laboratory magnet [T]
    # L = length of the laboratory magnet [m]
    # L_source = distance to the axion source [m] (the Sun or a supernova)
    # S = total aperture area of the magnet bores [m^2]
    # t = total exposure time of the telescope [yr]
    # eps_T = telescope efficiency
    # eps_D = detector efficiency
    # res_on = True or False, depending on whether the effect of the detector resolution is included or not
    # w_res = energy resolution of the detector [keV] 
    return g**4.0*dNdw_Tilde(w,m2,delta_m21_sq,mixing,B,L,L_source,S,t,eps_T,eps_D,
                             res_on=res_on,w_res=w_res) # [keV^-1]

def SN_dNdw_Tilde(w,g_aN,delta,m2,delta_m21_sq,mixing,B,L,L_source,S,eps,res_on=False,w_res=None):
    # Differential spectrum of gamma-rays for supernovae analysis: d\tilde{N}/dw [MeV^-1 GeV^2] (axion-photon coupling g^2 factored out)
    # g_aN = axion-nucleon coupling [dimensionless]
    # delta = weigth of the pionic Compton scattering production mechanism
    # m2 = axion 2 mass [eV] (in our convention, this corresponds to the heaviest axion)
    # delta_m21_sq = squared mass difference m2^2 - m1^2 [eV^2] (in our convention, this is positive by definition)
    # mixing = mixing angle between the 2 axions
    # B = B-field of the laboratory magnet [T]
    # L = length of the laboratory magnet [m]
    # L_source = distance to the supernova [m]
    # S = total aperture area of the magnet bores [m^2]
    # eps = gamma-ray detector efficiency
    # res_on = True or False, depending on whether the effect of the detector resolution is included or not
    # w_res = energy resolution of the detector [MeV] 
    dPhidw = SN_AxionFlux(w,g_aN,delta) # [MeV^-1]
    w_keV = w*1e3 # [keV] for P_Tilde
    w_res_keV = None if w_res is None else (w_res*1e3) # [keV] for P_Tilde
    Prob_Tilde = P_Tilde(w_keV,m2,delta_m21_sq,mixing,B,L,L_source,w_res_keV) # [GeV^2]
    dNdw = (1/(4*np.pi*L_source**2))*S*eps*dPhidw*Prob_Tilde # [MeV^-1 GeV^2]
    if res_on:
        if w_res is None:
            raise ValueError("If res_on=True you must provide w_res [MeV].")
        else:
            dNdw = smear(dNdw,w,w_res) # [MeV^-1 GeV^2]
    return dNdw # [MeV^-1 GeV^2]

def SN_dNdw(w,g_aN,delta,g,m2,delta_m21_sq,mixing,B,L,L_source,S,eps,res_on=False,w_res=None):
    # Differential spectrum of gamma-rays for supernovae analysis: dN/dw [MeV^-1]
    # g_aN = axion-nucleon coupling [dimensionless]
    # delta = weigth of the pionic Compton scattering production mechanism
    # g = axion-photon coupling [GeV^-1]
    # m2 = axion 2 mass [eV] (in our convention, this corresponds to the heaviest axion)
    # delta_m21_sq = squared mass difference m2^2 - m1^2 [eV^2] (in our convention, this is positive by definition)
    # mixing = mixing angle between the 2 axions
    # B = B-field of the laboratory magnet [T]
    # L = length of the laboratory magnet [m]
    # L_source = distance to the supernova [m]
    # S = total aperture area of the magnet bores [m^2]
    # eps = gamma-ray detector efficiency
    # res_on = True or False, depending on whether the effect of the detector resolution is included or not
    # w_res = energy resolution of the detector [MeV] 
    return g**2.0*SN_dNdw_Tilde(w,g_aN,delta,m2,delta_m21_sq,mixing,B,L,L_source,S,eps,res_on=res_on,w_res=w_res) # [MeV^-1]
#==============================================================================#

#==============================================================================#
def EnergyBins(w_min,w_max,N_bins):
    # Define energy bins
    # w_min = minimum energy (detector threshold) [E]
    # w_max = maximum energy [E]
    # N_bins = number of energy bins between w_min and w_max
    return np.linspace(w_min,w_max,int(N_bins)+1)
#==============================================================================#

#==============================================================================#
def BinnedPhotonNumber_Tilde(m2,delta_m21_sq,mixing,B,L,L_source,S,t,eps_T,eps_D,w_min,w_max,N_bins,res_on=False,w_res=None):
    # Expected number of photons in each energy bin: \tilde{N}_i (axion-photon coupling g^4 factored out)
    # m2 = axion 2 mass [eV] (in our convention, this corresponds to the heaviest axion)
    # delta_m21_sq = squared mass difference m2^2 - m1^2 [eV^2] (in our convention, this is positive by definition)
    # mixing = mixing angle between the 2 axions
    # B = B-field of the laboratory magnet [T]
    # L = length of the laboratory magnet [m]
    # L_source = distance to the axion source [m] (the Sun or a supernova)
    # S = total aperture area of the magnet bores [m^2]
    # t = total exposure time of the telescope [yr]
    # eps_T = telescope efficiency
    # eps_D = detector efficiency
    # w_min = minimum energy (detector threshold) [keV]
    # w_max = maximum energy [keV]
    # N_bins = number of energy bins between w_min and w_max
    # res_on = True or False, depending on whether the effect of the detector resolution is included or not
    # w_res = energy resolution of the detector [keV] 
    w_grid = EnergyBins(w_min,w_max,N_bins) # [keV]
    dw = w_grid[1]-w_grid[0] # [keV] 
    dNdw_grid = dNdw_Tilde(w_grid,m2,delta_m21_sq,mixing,B,L,L_source,S,t,eps_T,eps_D,
                           res_on=res_on,w_res=w_res) # [GeV^4 keV^-1]
    return 0.5*(dNdw_grid[:-1]+dNdw_grid[1:])*dw # [GeV^4]

def BinnedPhotonNumber(g,m2,delta_m21_sq,mixing,B,L,L_source,S,t,eps_T,eps_D,w_min,w_max,N_bins,res_on=False,w_res=None):
    # Expected number of photons in each energy bin: N_i
    # g = axion-photon coupling [GeV^-1]
    # m2 = axion 2 mass [eV] (in our convention, this corresponds to the heaviest axion)
    # delta_m21_sq = squared mass difference m2^2 - m1^2 [eV^2] (in our convention, this is positive by definition)
    # mixing = mixing angle between the 2 axions
    # B = B-field of the laboratory magnet [T]
    # L = length of the laboratory magnet [m]
    # L_source = distance to the axion source [m] (the Sun or a supernova)
    # S = total aperture area of the magnet bores [m^2]
    # t = total exposure time of the telescope [yr]
    # eps_T = telescope efficiency
    # eps_D = detector efficiency
    # w_min = minimum energy (detector threshold) [keV]
    # w_max = maximum energy [keV]
    # N_bins = number of energy bins between w_min and w_max
    # res_on = True or False, depending on whether the effect of the detector resolution is included or not
    # w_res = energy resolution of the detector [keV] 
    return g**4.0*BinnedPhotonNumber_Tilde(m2,delta_m21_sq,
                                           mixing,B,L,L_source,S,t,eps_T,eps_D,
                                           w_min,w_max,N_bins,
                                           res_on=res_on,w_res=w_res) # dimensionless

def SN_BinnedPhotonNumber_Tilde(g_aN,delta,m2,delta_m21_sq,mixing,B,L,L_source,S,eps,w_min,w_max,N_bins,res_on=False,w_res=None):
    # Expected number of photons in each energy bin for supernovae analysis: \tilde{N}_i (axion-photon coupling g^2 factored out)
    # g_aN = axion-nucleon coupling [dimensionless]
    # delta = weigth of the pionic Compton scattering production mechanism
    # m2 = axion 2 mass [eV] (in our convention, this corresponds to the heaviest axion)
    # delta_m21_sq = squared mass difference m2^2 - m1^2 [eV^2] (in our convention, this is positive by definition)
    # mixing = mixing angle between the 2 axions
    # B = B-field of the laboratory magnet [T]
    # L = length of the laboratory magnet [m]
    # L_source = distance to the supernova [m]
    # S = total aperture area of the magnet bores [m^2]
    # eps = gamma-ray detector efficiency
    # w_min = minimum energy (detector threshold) [MeV]
    # w_max = maximum energy [MeV]
    # N_bins = number of energy bins between w_min and w_max
    # res_on = True or False, depending on whether the effect of the detector resolution is included or not
    # w_res = energy resolution of the detector [MeV] 
    w_grid = EnergyBins(w_min,w_max,N_bins) # [MeV]
    dw = w_grid[1]-w_grid[0] # [MeV]
    dNdw_grid = SN_dNdw_Tilde(w_grid,g_aN,delta,m2,delta_m21_sq,mixing,B,L,L_source,S,eps,
                            res_on=res_on,w_res=w_res) # [MeV^-1 GeV^2]
    return 0.5*(dNdw_grid[:-1]+dNdw_grid[1:])*dw # [GeV^2]

def SN_BinnedPhotonNumber(g_aN,delta,g,m2,delta_m21_sq,mixing,B,L,L_source,S,eps,w_min,w_max,N_bins,res_on=False,w_res=None):
    # Expected number of photons in each energy bin for supernovae analysis: N_i 
    # g_aN = axion-nucleon coupling [dimensionless]
    # delta = weigth of the pionic Compton scattering production mechanism
    # g = axion-photon coupling [GeV^-1]
    # m2 = axion 2 mass [eV] (in our convention, this corresponds to the heaviest axion)
    # delta_m21_sq = squared mass difference m2^2 - m1^2 [eV^2] (in our convention, this is positive by definition)
    # mixing = mixing angle between the 2 axions
    # B = B-field of the laboratory magnet [T]
    # L = length of the laboratory magnet [m]
    # L_source = distance to the supernova [m]
    # S = total aperture area of the magnet bores [m^2]
    # eps = gamma-ray detector efficiency
    # w_min = minimum energy (detector threshold) [MeV]
    # w_max = maximum energy [MeV]
    # N_bins = number of energy bins between w_min and w_max
    # res_on = True or False, depending on whether the effect of the detector resolution is included or not
    # w_res = energy resolution of the detector [MeV] 
    return g**2.0*SN_BinnedPhotonNumber_Tilde(g_aN,delta,
                                              m2,delta_m21_sq,
                                              mixing,B,L,L_source,S,eps,
                                              w_min,w_max,N_bins,
                                              res_on=res_on,w_res=w_res) # dimensionless
#==============================================================================#

#==============================================================================#
def LogPoisson(N_obs,N_exp):
    # Log-Poisson pdf
    # N_obs = observed number of events 
    # N_exp = expected number of events
    N_exp = np.maximum(N_exp,1e-300)
    return N_obs*np.log(N_exp)-N_exp # constant factors that do not depend on the parameters are omitted

def LogLikelihood(N_obs,g,m2,delta_m21_sq,mixing,B,L,L_source,S,t,eps_T,eps_D,w_min,w_max,N_bins,res_on=False,w_res=None):
    # Log-likelihood function
    # N_obs = observed number of events in each bin
    # g = axion-photon coupling [GeV^-1]
    # m2 = axion 2 mass [eV] (in our convention, this corresponds to the heaviest axion)
    # delta_m21_sq = squared mass difference m2^2 - m1^2 [eV^2] (in our convention, this is positive by definition)
    # mixing = mixing angle between the 2 axions
    # B = B-field of the laboratory magnet [T]
    # L = length of the laboratory magnet [m]
    # L_source = distance to the axion source [m] (the Sun or a supernova)
    # S = total aperture area of the magnet bores [m^2]
    # t = total exposure time of the telescope [yr]
    # eps_T = telescope efficiency
    # eps_D = detector efficiency
    # w_min = minimum energy (detector threshold) [keV]
    # w_max = maximum energy [keV]
    # N_bins = number of energy bins between w_min and w_max
    # res_on = True or False, depending on whether the effect of the detector resolution is included or not
    # w_res = energy resolution of the detector [keV] 
    N_exp = BinnedPhotonNumber(g,m2,delta_m21_sq,mixing,B,L,L_source,S,t,eps_T,eps_D,w_min,w_max,N_bins,
                               res_on=res_on,w_res=w_res)
    return np.sum(LogPoisson(N_obs,N_exp))

def SN_LogLikelihood(g_aN,delta,N_obs,g,m2,delta_m21_sq,mixing,B,L,L_source,S,eps,w_min,w_max,N_bins,res_on=False,w_res=None):
    # Log-likelihood function for supernovae analysis
    # g_aN = axion-nucleon coupling [dimensionless]
    # delta = weigth of the pionic Compton scattering production mechanism
    # N_obs = observed number of events in each bin
    # g = axion-photon coupling [GeV^-1]
    # m2 = axion 2 mass [eV] (in our convention, this corresponds to the heaviest axion)
    # delta_m21_sq = squared mass difference m2^2 - m1^2 [eV^2] (in our convention, this is positive by definition)
    # mixing = mixing angle between the 2 axions
    # B = B-field of the laboratory magnet [T]
    # L = length of the laboratory magnet [m]
    # L_source = distance to the supernova [m]
    # S = total aperture area of the magnet bores [m^2]
    # eps = gamma-ray detector efficiency
    # w_min = minimum energy (detector threshold) [MeV]
    # w_max = maximum energy [MeV]
    # N_bins = number of energy bins between w_min and w_max
    # res_on = True or False, depending on whether the effect of the detector resolution is included or not
    # w_res = energy resolution of the detector [MeV] 
    N_exp = SN_BinnedPhotonNumber(g_aN,delta,
                                  g,m2,delta_m21_sq,
                                  mixing,B,L,L_source,S,eps,
                                  w_min,w_max,N_bins,
                                  res_on=res_on,w_res=w_res) # dimensionless
    return np.sum(LogPoisson(N_obs,N_exp)) # dimensionless
#==============================================================================#

#==============================================================================#
def CMLE_m2(N_obs,mixing,B,L,L_source,S,t,eps_T,eps_D,w_min,w_max,N_bins,res_on=False,w_res=None,m2_bounds=(1e-9,1e1)):
    # Conditional Maximum Likelihood Estimator (CMLE) for m2 given delta_m21_sq = 0
    # N_obs = observed number of events in each bin
    # mixing = mixing angle between the 2 axions
    # B = B-field of the laboratory magnet [T]
    # L = length of the laboratory magnet [m]
    # L_source = distance to the axion source [m] (the Sun or a supernova)
    # S = total aperture area of the magnet bores [m^2]
    # t = total exposure time of the telescope [yr]
    # eps_T = telescope efficiency
    # eps_D = detector efficiency
    # w_min = minimum energy (detector threshold) [keV]
    # w_max = maximum energy [keV]
    # N_bins = number of energy bins between w_min and w_max
    # res_on = True or False, depending on whether the effect of the detector resolution is included or not
    # w_res = energy resolution of the detector [keV] 
    # m2_bounds = range of masses to search for the CMLE [eV]
    N_tot = np.sum(N_obs)
    def NegativeLogLikelihood(logm2):
        m2 = 10.0**logm2 # [eV]
        N_exp_tilde = BinnedPhotonNumber_Tilde(m2,0.0, 
                                               mixing,B,L,L_source,S,t,eps_T,eps_D,
                                               w_min,w_max,N_bins,
                                               res_on=res_on,w_res=w_res) # [GeV^4]
        g4 = N_tot/np.sum(N_exp_tilde) # CMLE of g^4 given delta_m21_sq = 0 [GeV^-4]
        g = g4**0.25 # CMLE of g given delta_m21_sq = 0 [GeV^-1]
        return -LogLikelihood(N_obs,g,m2,0.0,
                              mixing,B,L,L_source,S,t,eps_T,eps_D,
                              w_min,w_max,N_bins,
                              res_on=res_on,w_res=w_res)
    lo,hi = np.log10(m2_bounds[0]),np.log10(m2_bounds[1])
    res = minimize_scalar(NegativeLogLikelihood,bounds=(lo,hi),method="bounded")
    m2hh = 10.0**res.x # [eV]
    return m2hh # [eV]
    
def SN_CMLE_m2(g_aN,delta,N_obs,mixing,B,L,L_source,S,eps,w_min,w_max,N_bins,res_on=False,w_res=None,m2_bounds=(1e-9,1e1)):
    # Conditional Maximum Likelihood Estimator for m2 given delta_m21_sq = 0 for supernovae analysis
    # g_aN = axion-nucleon coupling [dimensionless]
    # delta = weigth of the pionic Compton scattering production mechanism
    # N_obs = observed number of events in each bin
    # mixing = mixing angle between the 2 axions
    # B = B-field of the laboratory magnet [T]
    # L = length of the laboratory magnet [m]
    # L_source = distance to the supernova [m]
    # S = total aperture area of the magnet bores [m^2]
    # eps = gamma-ray detector efficiency
    # w_min = minimum energy (detector threshold) [MeV]
    # w_max = maximum energy [MeV]
    # N_bins = number of energy bins between w_min and w_max
    # res_on = True or False, depending on whether the effect of the detector resolution is included or not
    # w_res = energy resolution of the detector [MeV] 
    # m2_bounds = range of masses to search for the CMLE [eV]
    N_tot = np.sum(N_obs)
    def NegativeLogLikelihood(logm2):
        m2 = 10.0**logm2 # [eV]
        N_exp_tilde = SN_BinnedPhotonNumber_Tilde(g_aN,delta,
                                                  m2,0.0, 
                                                  mixing,B,L,L_source,S,eps,
                                                  w_min,w_max,N_bins,
                                                  res_on=res_on,w_res=w_res) # [GeV^2]
        g2 = N_tot/np.sum(N_exp_tilde) # CMLE of g^2 given delta_m21_sq = 0 [GeV^-2]
        g = g2**0.5 # CMLE of g given delta_m21_sq = 0 [GeV^-1]
        return -SN_LogLikelihood(g_aN,delta,
                                 N_obs,g,m2,0.0,
                                 mixing,B,L,L_source,S,eps,
                                 w_min,w_max,N_bins,
                                 res_on=res_on,w_res=w_res)   
    lo,hi = np.log10(m2_bounds[0]),np.log10(m2_bounds[1])
    res = minimize_scalar(NegativeLogLikelihood,bounds=(lo,hi),method="bounded")
    m2hh = 10.0**res.x # [eV]
    return m2hh # [eV]
#==============================================================================#

#==============================================================================#
def Discovery_Limit(q0,g_arb,m2,delta_m21_sq,mixing,B,L,L_source,S,t,eps_T,eps_D,w_min,w_max,N_bins,res_on=False,w_res=None,
                    m2_bounds=(1e-9,1e1)):
    # Discovery limit for 2-axion theory at \sqrt{q0} sigmas
    # q0 = test statistic
    # g_arb = arbitrary value of axion-photon coupling for computing CMLE of m2 (the result does not depend on g) [GeV^-1]
    # m2 = axion 2 mass [eV] (in our convention, this corresponds to the heaviest axion)
    # delta_m21_sq = squared mass difference m2^2 - m1^2 [eV^2] (in our convention, this is positive by definition)
    # mixing = mixing angle between the 2 axions
    # B = B-field of the laboratory magnet [T]
    # L = length of the laboratory magnet [m]
    # L_source = distance to the axion source [m] (the Sun or a supernova)
    # S = total aperture area of the magnet bores [m^2]
    # t = total exposure time of the telescope [yr]
    # eps_T = telescope efficiency
    # eps_D = detector efficiency
    # w_min = minimum energy (detector threshold) [keV]
    # w_max = maximum energy [keV]
    # N_bins = number of energy bins between w_min and w_max
    # res_on = True or False, depending on whether the effect of the detector resolution is included or not
    # w_res = energy resolution of the detector [keV] 
    # m2_bounds = range of masses to search for the CMLE [eV]
    numerator = q0
    N_obs = BinnedPhotonNumber(g_arb,m2,delta_m21_sq,
                               mixing,B,L,L_source,S,t,eps_T,eps_D,
                               w_min,w_max,N_bins,
                               res_on=res_on,w_res=w_res) # dimensionless
    m2hh = CMLE_m2(N_obs,mixing,B,L,L_source,S,t,eps_T,eps_D,w_min,w_max,N_bins,res_on=res_on,w_res=w_res,
                   m2_bounds=m2_bounds) # [eV]
    N_exp_num = BinnedPhotonNumber_Tilde(m2,delta_m21_sq,mixing,
                                         B,L,L_source,S,t,eps_T,eps_D,
                                         w_min,w_max,N_bins,
                                         res_on=res_on,w_res=w_res) # [GeV^4]
    N_exp_num_summed = np.sum(N_exp_num) # [GeV^4]
    N_exp_den = BinnedPhotonNumber_Tilde(m2hh,0.0,mixing,
                                         B,L,L_source,S,t,eps_T,eps_D,
                                         w_min,w_max,N_bins,
                                         res_on=res_on,w_res=w_res) # [GeV^4]
    N_exp_den_summed = np.sum(N_exp_den) # [GeV^4]
    h = N_exp_num_summed/N_exp_den_summed # dimensionless
    n = N_exp_num # [GeV^4]
    mu = h*N_exp_den # [GeV^4]
    t = (n-mu)/mu # dimensionless
    denominator = np.sum(2.0*mu*((1.0+t)*np.log1p(t)-t)) # [GeV^4]
    g_disc = (numerator/denominator)**0.25 # [GeV^-1]
    return g_disc # [GeV^-1]

def SN_Discovery_Limit(g_aN,delta,q0,g_arb,m2,delta_m21_sq,mixing,B,L,L_source,S,eps,w_min,w_max,N_bins,res_on=False,w_res=None,
                       m2_bounds=(1e-9,1e1)):
    # Discovery limit for 2-axion theory at \sqrt{q0} sigmas for supernovae analysis
    # g_aN = axion-nucleon coupling [dimensionless]
    # delta = weigth of the pionic Compton scattering production mechanism
    # q0 = test statistic
    # g_arb = arbitrary value of axion-photon coupling for computing CMLE of m2 (the result does not depend on g) [GeV^-1]
    # m2 = axion 2 mass [eV] (in our convention, this corresponds to the heaviest axion)
    # delta_m21_sq = squared mass difference m2^2 - m1^2 [eV^2] (in our convention, this is positive by definition)
    # mixing = mixing angle between the 2 axions
    # B = B-field of the laboratory magnet [T]
    # L = length of the laboratory magnet [m]
    # L_source = distance to the supernova [m]
    # S = total aperture area of the magnet bores [m^2]
    # eps = detector efficiency
    # w_min = minimum energy (detector threshold) [keV]
    # w_max = maximum energy [keV]
    # N_bins = number of energy bins between w_min and w_max
    # res_on = True or False, depending on whether the effect of the detector resolution is included or not
    # w_res = energy resolution of the detector [keV] 
    # m2_bounds = range of masses to search for the CMLE [eV]
    numerator = q0
    N_obs = SN_BinnedPhotonNumber(g_aN,delta,
                                  g_arb,m2,delta_m21_sq,
                                  mixing,B,L,L_source,S,eps,
                                  w_min,w_max,N_bins,
                                  res_on=res_on,w_res=w_res) # dimensionless
    m2hh = SN_CMLE_m2(g_aN,delta,N_obs,mixing,B,L,L_source,S,eps,w_min,w_max,N_bins,res_on=res_on,w_res=w_res,
                      m2_bounds=m2_bounds) # [eV]
    N_exp_num = SN_BinnedPhotonNumber_Tilde(g_aN,delta,
                                            m2,delta_m21_sq,mixing,
                                            B,L,L_source,S,eps,
                                            w_min,w_max,N_bins,
                                            res_on=res_on,w_res=w_res) # [GeV^2]
    N_exp_num_summed = np.sum(N_exp_num) # [GeV^2]
    N_exp_den = SN_BinnedPhotonNumber_Tilde(g_aN,delta,
                                            m2hh,0.0,mixing,
                                            B,L,L_source,S,eps,
                                            w_min,w_max,N_bins,
                                            res_on=res_on,w_res=w_res) # [GeV^2]
    N_exp_den_summed = np.sum(N_exp_den) # [GeV^2]
    h = N_exp_num_summed/N_exp_den_summed # dimensionless
    n = N_exp_num # [GeV^2]
    mu = h*N_exp_den # [GeV^2]
    t = (n-mu)/mu # dimensionless
    denominator = np.sum(2*mu*((1.0+t)*np.log1p(t)-t)) # [GeV^2]
    g_disc = (numerator/denominator)**0.5 # [GeV^-1]
    return g_disc # [GeV^-1]
#==============================================================================#

#==============================================================================#
def Exclusion_Limit(CL,b,m2,delta_m21_sq,mixing,B,L,L_source,S,t,eps_T,eps_D,w_min,w_max,N_bins,res_on=False,w_res=None):
    # Exclusion limit of 2-axion theory at CL*100 confidence level
    # CL = confidence level (0 < CL < 1)
    # b = total number of background events
    # m2 = axion 2 mass [eV] (in our convention, this corresponds to the heaviest axion)
    # delta_m21_sq = squared mass difference m2^2 - m1^2 [eV^2] (in our convention, this is positive by definition)
    # mixing = mixing angle between the 2 axions
    # B = B-field of the laboratory magnet [T]
    # L = length of the laboratory magnet [m]
    # L_source = distance to the axion source [m] (the Sun or a supernova)
    # S = total aperture area of the magnet bores [m^2]
    # t = total exposure time of the telescope [yr]
    # eps_T = telescope efficiency
    # eps_D = detector efficiency
    # w_min = minimum energy (detector threshold) [keV]
    # w_max = maximum energy [keV]
    # N_bins = number of energy bins between w_min and w_max
    # res_on = True or False, depending on whether the effect of the detector resolution is included or not
    # w_res = energy resolution of the detector [keV] 
    if not (0.0 <= CL <= 1.0):
        raise ValueError("CL must be between 0 and 1.")
    N_crit = -np.log(1.0-CL)-b # dimensionless
    if N_crit <= 0.0:
        raise ValueError("Background is too high for the targeted confidence level.")
    N_exp_tilde_total = np.sum(BinnedPhotonNumber_Tilde(m2,delta_m21_sq,
                                                        mixing,B,L,L_source,S,t,eps_T,eps_D,
                                                        w_min,w_max,N_bins,
                                                        res_on=res_on,w_res=w_res)) # [GeV^4]
    if (not np.isfinite(N_exp_tilde_total)) or (N_exp_tilde_total <= 0.0):
        return np.inf
    g_crit = (N_crit/N_exp_tilde_total)**0.25 # [GeV^-1]
    return g_crit # [GeV^-1]
#==============================================================================#

#==============================================================================#
# Not used in the paper
#==============================================================================#
def SN_Exclusion_Limit(CL,b,g_aN,delta,m2,delta_m21_sq,mixing,B,L,L_source,S,eps,w_min,w_max,N_bins,res_on=False,w_res=None):
    # Exclusion limit of 2-axion theory at CL*100 confidence level for supernovae analysis
    # CL = confidence level (0 < CL < 1)
    # b = total number of background events
    # g_aN = axion-nucleon coupling [dimensionless]
    # delta = weigth of the pionic Compton scattering production mechanism
    # m2 = axion 2 mass [eV] (in our convention, this corresponds to the heaviest axion)
    # delta_m21_sq = squared mass difference m2^2 - m1^2 [eV^2] (in our convention, this is positive by definition)
    # mixing = mixing angle between the 2 axions
    # B = B-field of the laboratory magnet [T]
    # L = length of the laboratory magnet [m]
    # L_source = distance to the supernova [m]
    # S = total aperture area of the magnet bores [m^2]
    # eps = detector efficiency
    # w_min = minimum energy (detector threshold) [keV]
    # w_max = maximum energy [keV]
    # N_bins = number of energy bins between w_min and w_max
    # res_on = True or False, depending on whether the effect of the detector resolution is included or not
    # w_res = energy resolution of the detector [keV] 
    if not (0.0 < CL < 1.0):
        raise ValueError("CL must be between 0 and 1.")
    N_crit = -np.log(1.0-CL)-b
    if N_crit <= 0.0:
        raise ValueError("Background is too high for the targeted confidence level.")
    N_exp_tilde_total = np.sum(SN_BinnedPhotonNumber_Tilde(g_aN,delta,
                                                           m2,delta_m21_sq,
                                                           mixing,B,L,L_source,S,eps,
                                                           w_min,w_max,N_bins,
                                                           res_on=res_on,w_res=w_res)) # [GeV^2]
    if (not np.isfinite(N_exp_tilde_total)) or (N_exp_tilde_total <= 0.0):
        return np.inf
    g_crit = (N_crit/N_exp_tilde_total)**0.5 # [GeV^-1]
    return g_crit # [GeV^-1]
    
def ConstantPhotonNumber_Curve(N_gamma,m2,delta_m21_sq,mixing,B,L,L_source,S,t,eps_T,eps_D,w_min,w_max,N_bins,res_on=False,w_res=None):
    # Curve of constant total photon number N_gamma
    # N_gamma = fixed total number of photons defining the curve
    # m2 = axion 2 mass [eV] (in our convention, this corresponds to the heaviest axion)
    # delta_m21_sq = squared mass difference m2^2 - m1^2 [eV^2] (in our convention, this is positive by definition)
    # mixing = mixing angle between the 2 axions
    # B = B-field of the laboratory magnet [T]
    # L = length of the laboratory magnet [m]
    # L_source = distance to the axion source [m] (the Sun or a supernova)
    # S = total aperture area of the magnet bores [m^2]
    # t = total exposure time of the telescope [yr]
    # eps_T = telescope efficiency
    # eps_D = detector efficiency
    # w_min = minimum energy (detector threshold) [keV]
    # w_max = maximum energy [keV]
    # N_bins = number of energy bins between w_min and w_max
    # res_on = True or False, depending on whether the effect of the detector resolution is included or not
    # w_res = energy resolution of the detector [keV]
    N_exp_tilde_total = np.sum(BinnedPhotonNumber_Tilde(m2,delta_m21_sq,
                                                        mixing,B,L,L_source,S,t,eps_T,eps_D,
                                                        w_min,w_max,N_bins,
                                                        res_on=res_on,w_res=w_res)) # [GeV^4]
    g = (N_gamma/N_exp_tilde_total)**0.25 # [GeV^-1]
    return g # [GeV^-1]

def SN_ConstantPhotonNumber_Curve(N_gamma,g_aN,delta,m2,delta_m21_sq,mixing,B,L,L_source,S,eps,w_min,w_max,N_bins,res_on=False,w_res=None):
    # Curve of constant total photon number N_gamma
    # N_gamma = fixed total number of photons defining the curve
    # g_aN = axion-nucleon coupling [dimensionless]
    # delta = weigth of the pionic Compton scattering production mechanism
    # m2 = axion 2 mass [eV] (in our convention, this corresponds to the heaviest axion)
    # delta_m21_sq = squared mass difference m2^2 - m1^2 [eV^2] (in our convention, this is positive by definition)
    # mixing = mixing angle between the 2 axions
    # B = B-field of the laboratory magnet [T]
    # L = length of the laboratory magnet [m]
    # L_source = distance to the supernova [m]
    # S = total aperture area of the magnet bores [m^2]
    # eps = gamma-ray detector efficiency
    # w_min = minimum energy (detector threshold) [MeV]
    # w_max = maximum energy [MeV]
    # N_bins = number of energy bins between w_min and w_max
    # res_on = True or False, depending on whether the effect of the detector resolution is included or not
    # w_res = energy resolution of the detector [MeV]
    N_exp_tilde_total = np.sum(SN_BinnedPhotonNumber_Tilde(g_aN,delta,
                                                           m2,delta_m21_sq,
                                                           mixing,B,L,L_source,S,eps,
                                                           w_min,w_max,N_bins,
                                                           res_on=res_on,w_res=w_res)) # [GeV^2]
    g = (N_gamma/N_exp_tilde_total)**0.5 # [GeV^-1]
    return g # [GeV^-1]
    
def BackgroundBins(N_bins,distribution="flat"):
    # Distribution of background events over energy bins
    # N_bins = number of energy bins
    # distribution = shape of the distribution of background events
    if distribution=="flat":
        return np.ones(N_bins)/N_bins
    raise ValueError("Unknown background distribution")

def q0_Asimov(g,N_exp_tilde,b):
    s = g**4*N_exp_tilde   
    if np.any(b <= 0):
        raise ValueError("All b_i must be positive")
    return 2.0*np.sum((b+s)*np.log1p(s/b)-s)

def g_from_q0_bisection(q0,N_exp_tilde,b,g_ref=1e-8,it=80):
    # Solving the equation q0_Asimov(g) = q0 via bisection method
    # q0: fixed value of the test statistic
    # N_exp_tilde: expected number of signal events in each bin [GeV^4] (g factored out)
    # b = number of background events
    # g_ref = reference coupling to initialize the bisection method
    # it = number of iterations used in the bisection method
    
    # In the absence of signal events, there is no discovery
    if np.all(N_exp_tilde <= 0):
        return np.inf

    def f(g):
        return q0_Asimov(g,N_exp_tilde,b)-q0

    g_low = 0.0
    g_high = g_ref
    # Increase g_high until f(g_high) > 0
    while f(g_high) < 0:
        g_high *= 2.0
        if g_high > 1e2:  
            return np.inf

    for _ in range(it):
        g_mid = 0.5*(g_low+g_high)
        if f(g_mid)<0:
            g_low = g_mid
        else:
            g_high = g_mid
    return 0.5*(g_low+g_high)

def Discovery_Prospects(q0,m2,delta_m21_sq,mixing,B,L,L_source,S,t,eps_T,eps_D,w_min,w_max,N_bins,b_mode="flat",res_on=False,w_res=None):
    b = BackgroundBins(N_bins,distribution=b_mode)
    b = np.maximum(b,1e-300)
    b = b/np.sum(b)
    N_exp_tilde = BinnedPhotonNumber_Tilde(m2,delta_m21_sq,mixing,
                                           B,L,L_source,S,t,eps_T,eps_D,
                                           w_min,w_max,N_bins,
                                           res_on=res_on,w_res=w_res)
    g_disc = g_from_q0_bisection(q0,N_exp_tilde,b,g_ref=1e-10,it=80)
    return g_disc
#==============================================================================#