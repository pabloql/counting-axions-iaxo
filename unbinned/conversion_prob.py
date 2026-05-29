"""
Conversion probabilities for axion-photon conversion in a helioscope.

All cases are special cases of one general two-axion formula:

    P(omega; m1, m2) = 0.25 sinc^2(x1) + 0.25 sinc^2(x2)
                      + 0.5 sinc(x1) sinc(x2) cos(phase_ES) * damping

where:
    x_i       = m_i^2 * L_magnet / (4 * omega)        [magnet phase]
    phase_ES  = (m2^2 - m1^2) * L_ES / (2 * omega)    [Earth-Sun phase]
    damping   = exp(-0.5 * sigma_phase^2)
    sigma_phase = |m2^2 - m1^2| * L_ES * E_res / (2 * omega^2)

Special cases:
    Single massless:  m1 = m2 = 0           -> P = 1
    Single massive:   m1 = m2 = m           -> P = sinc^2(m^2 L / 4E)
    Quasi-degenerate: m1 = 0, m2 = sqrt(Dm2) -> P = 0.5 + 0.5 cos(Dm2 L_ES/2E) * damping
    Hierarchical:     m1 = 0, m2 = m_heavy  -> P = 0.25 + 0.25 sinc^2(...) + 0.5 sinc(...) cos(...) * damping
    Mass ratio r:     m2 = r * m1           -> Full formula

Usage:
    from conversion_prob import P_conversion, dN_domega
"""

import numpy as np
from iaxo_params import L_magnet_eV, L_ES_eV, R_sun_eV, norm_Ngamma, flux_primakoff


# =============================================================================
# General conversion probability
# =============================================================================
def P_conversion(omega_keV, m1=0.0, m2=0.0, phi=np.pi/4, E_res_keV=0.0, n_Rsun=0.0, no_disappearance=False):
    """
    General axion-photon conversion probability for two axions.

    Parameters
    ----------
    omega_keV : array
        Photon energy [keV]
    m1 : float
        Mass of axion 1 [eV]. Default 0.
    m2 : float
        Mass of axion 2 [eV]. Default 0.
    phi : float
        Mixing angle [rad]. Default pi/4 (maximal mixing).
    E_res_keV : float
        Energy resolution for damping [keV]. Default 0 (no damping).
    n_Rsun : float
        Offset to Earth-Sun distance in units of solar radii.
        L = L_ES + n_Rsun * R_sun.  Default 0 (standard 1 AU).

    Returns
    -------
    P : array
        Conversion probability at each energy.

    Notes
    -----
    Uses np.sinc(x/pi) = sin(x)/x convention.
    """
    omega_eV = omega_keV * 1000.0
    L = L_ES_eV + n_Rsun * R_sun_eV

    m1sq = m1**2
    m2sq = m2**2

    # Magnet phases: x_i = m_i^2 * L_magnet / (4 * omega)
    x1 = m1sq * L_magnet_eV / (4.0 * omega_eV)
    x2 = m2sq * L_magnet_eV / (4.0 * omega_eV)

    # sinc(x/pi) = sin(x)/x
    sinc_x1 = np.sinc(x1 / np.pi)
    sinc_x2 = np.sinc(x2 / np.pi)

    # Earth-Sun oscillation phase: (m2^2 - m1^2) * L_ES / (2 * omega)
    delta_m2 = m2sq - m1sq
    phase_ES = delta_m2 * L / (2.0 * omega_eV)

    # Damping from energy resolution
    if E_res_keV > 0 and abs(delta_m2) > 0:
        sigma_phase = (abs(delta_m2) * L * E_res_keV * 1000.0
                       / (2.0 * omega_eV**2))
        damping = np.exp(-0.5 * sigma_phase**2)
    else:
        damping = 1.0

    c_phi = np.cos(phi)
    s_phi = np.sin(phi)
    s_2phi = np.sin(2.0 * phi)

    if no_disappearance:
        P = (c_phi**2 * sinc_x1 + s_phi**2 * sinc_x2)**2
    else:
        P = (c_phi**4 * sinc_x1**2
             + s_phi**4 * sinc_x2**2
             + 0.5 * s_2phi**2 * sinc_x1 * sinc_x2 * damping * np.cos(phase_ES))

    return P


# =============================================================================
# Convenience wrappers
# =============================================================================
def P_single_massive(omega_keV, m):
    """Single massive axion: P = sinc^2(m^2 L / 4E)."""
    return P_conversion(omega_keV, m1=m, m2=m, E_res_keV=0.0)


def P_quasi_degenerate(omega_keV, delta_m2, E_res_keV=0.0, n_Rsun=0.0):
    """
    Quasi-degenerate two axions: P = 0.5 + 0.5 cos(Dm2 L_ES / 2E) * damping.

    Both axion masses are ~0, only the mass-squared difference matters.

    Parameters
    ----------
    delta_m2 : float
        Mass squared difference m2^2 - m1^2 [eV^2]
    n_Rsun : float
        Offset to Earth-Sun distance in units of solar radii. Default 0.
    """
    m2_eff = np.sqrt(abs(delta_m2))
    return P_conversion(omega_keV, m1=0.0, m2=m2_eff, E_res_keV=E_res_keV,
                        n_Rsun=n_Rsun)


# =============================================================================
# Differential photon count
# =============================================================================
def dN_domega(omega_keV, m1=0.0, m2=0.0, phi=np.pi/4, E_res_keV=0.0, n_Rsun=0.0, no_disappearance=False):
    """
    Differential photon count dN/dE at g = g_ref = 10^-10 GeV^-1.

    dN/dE = norm_Ngamma * flux_primakoff(E) * P_conversion(E; m1, m2, phi)

    Parameters
    ----------
    omega_keV : array
        Energy [keV]
    m1 : float
        Mass of axion 1 [eV]
    m2 : float
        Mass of axion 2 [eV]
    phi : float
        Mixing angle [rad]. Default pi/4 (maximal mixing).
    E_res_keV : float
        Energy resolution [keV] for damping
    n_Rsun : float
        Offset to Earth-Sun distance in units of solar radii. Default 0.

    Returns
    -------
    dN_dE : array
        Differential photon count [keV^-1]
    """
    return (norm_Ngamma * flux_primakoff(omega_keV)
            * P_conversion(omega_keV, m1, m2, phi=phi,
                           E_res_keV=E_res_keV, n_Rsun=n_Rsun,
                           no_disappearance=no_disappearance))
