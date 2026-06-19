"""
IAXO experimental parameters, physical constants, and utility functions.

Authors: B. Grinstein, C. Miro, and P. Quilez
Title: Counting axions with IAXO
Cite: arXiv:2606.XXXXX

Usage:
    from iaxo_params import IAXO, hbar_c, L_ES_eV, L_magnet_eV
    from iaxo_params import norm_Ngamma, flux_primakoff, apply_energy_resolution
"""

import numpy as np
from numpy import pi
from scipy.ndimage import gaussian_filter1d


# =============================================================================
# IAXO Experimental Parameters
# =============================================================================
class IAXO:
    """IAXO baseline parameters."""
    Bfield = 2.5          # Magnetic field [T]
    Exposure = 1.5        # Exposure time [years]
    Length = 20.0         # Magnet length [m]
    N_bores = 8           # Number of bore tubes
    BoreDiameter = 60.0   # Bore diameter [cm]
    eps_D = 0.7           # Detector efficiency
    eps_T = 0.8           # Telescope efficiency


# =============================================================================
# Physical Constants
# =============================================================================
hbar_c = 1.97e-7          # [eV * m] -- conversion factor
L_ES_m = 1.496e11         # Earth-Sun distance [m]
L_ES_eV = L_ES_m / hbar_c # [eV^-1]
R_sun_m = 6.957e8         # Solar radius [m]
R_sun_eV = R_sun_m / hbar_c  # Solar radius [eV^-1]
L_magnet_eV = IAXO.Length / hbar_c  # Magnet length [eV^-1]


# =============================================================================
# Derived Quantities
# =============================================================================
S_cm2 = IAXO.N_bores * pi * (IAXO.BoreDiameter / 2.0)**2  # Total bore area [cm^2]
t_secs = IAXO.Exposure * 3600 * 24 * 365                   # Exposure time [s]
B_natural = IAXO.Bfield * (1e-19 * 195)                    # B-field in natural units

# Normalization for photon number at g = 10^-10 GeV^-1
norm_Ngamma = (6.02e10 * t_secs * S_cm2 * IAXO.eps_D * IAXO.eps_T
               * (B_natural * L_magnet_eV / 2.0)**2)


# =============================================================================
# Energy defaults
# =============================================================================
omega_min = 0.1    # keV (energy threshold)
omega_max = 20.0   # keV (max energy)
g_ref = 1e-10      # Reference coupling [GeV^-1]


# =============================================================================
# Primakoff flux shape
# =============================================================================
def flux_primakoff(omega_keV):
    """
    Primakoff flux shape: E^2.481 / exp(E / 1.205).

    Parameters
    ----------
    omega_keV : array
        Energy [keV]

    Returns
    -------
    flux : array
        Flux shape (unnormalized, coupling stripped out)
    """
    return (omega_keV**2.481) / np.exp(omega_keV / 1.205)


# =============================================================================
# Energy resolution smearing
# =============================================================================
def apply_energy_resolution(spectrum, omega_keV, E_res_keV,
                            n_uniform=80000, omega_range=None):
    """
    Apply Gaussian energy resolution smearing.

    Interpolates to a uniform grid, applies gaussian_filter1d, returns
    (spectrum_smeared, omega_uniform).

    Parameters
    ----------
    spectrum : array
        Input dN/dE values
    omega_keV : array
        Energy grid (can be non-uniform)
    E_res_keV : float
        Energy resolution sigma [keV]
    n_uniform : int
        Number of points in uniform output grid
    omega_range : tuple, optional
        (min, max) energy in keV. Default: (omega_min, omega_max)

    Returns
    -------
    spectrum_smeared : array
        Smeared spectrum on uniform grid
    omega_uniform : array
        Uniform energy grid
    """
    if omega_range is None:
        omega_range = (omega_min, omega_max)

    # Build output grid on the physical detector range
    omega_uniform = np.linspace(omega_range[0], omega_range[1], n_uniform)
    dE = omega_uniform[1] - omega_uniform[0]
    sigma_indices = E_res_keV / dE

    # Extend grid by 4σ on each side so the Gaussian kernel is fully supported
    n_pad = int(np.ceil(4 * sigma_indices))
    omega_lo = omega_range[0] - n_pad * dE
    omega_hi = omega_range[1] + n_pad * dE
    omega_extended = np.linspace(omega_lo, omega_hi, n_uniform + 2 * n_pad)

    # Interpolate onto extended grid (extrapolated points get zero flux)
    spectrum_extended = np.interp(omega_extended, omega_keV, spectrum,
                                  left=0.0, right=0.0)

    # Smear on the extended grid, then keep only the physical range
    smeared_extended = gaussian_filter1d(spectrum_extended,
                                         sigma=sigma_indices, mode='nearest')
    spectrum_smeared = smeared_extended[n_pad:n_pad + n_uniform]

    return spectrum_smeared, omega_uniform


# =============================================================================
# Plotting style
# =============================================================================
def setup_plot_style():
    """Set standard plotting style for publication-quality figures."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "text.usetex": True,
        "font.size": 25,
        "axes.formatter.limits": [-3, 3],
        "axes.linewidth": 2,
        "lines.linewidth": 2,
        "xtick.top": True,
        "xtick.bottom": True,
        "ytick.left": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 2,
        "xtick.minor.width": 1,
        "ytick.major.width": 2,
        "ytick.minor.width": 1,
        "xtick.major.size": 10,
        "xtick.minor.size": 5,
        "ytick.major.size": 10,
        "ytick.minor.size": 5,
        "figure.figsize": [8, 7],
        "savefig.dpi": 300,
        "savefig.transparent": True,
    })
