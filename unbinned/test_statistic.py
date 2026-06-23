"""
Likelihood ratio test statistic for two-axion discovery.

Authors: B. Grinstein, C. Miro, and P. Quilez
Title: Counting axions with IAXO
Cite: arXiv:2606.20826

    q0 = 2 * integral[ dN_H1 * ln(dN_H1 / (h * dN_H0)) ] dE

where h = integral(dN_H1) / integral(dN_H0) normalizes to equal total counts.

    g_discovery = (n_sigma^2 / (q0 / g_ref^4))^0.25

Usage:
    from test_statistic import compute_q0, g_discovery_from_q0
    from test_statistic import compute_discovery_limit
"""

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import minimize_scalar
from iaxo_params import apply_energy_resolution, g_ref, omega_min, omega_max


# =============================================================================
# Core test statistic
# =============================================================================
def compute_q0(dN_H1_func, dN_H0_func, omega_keV, E_res_keV,
               n_uniform=80000, omega_range=None):
    """
    Compute the test statistic q0/g^4 for distinguishing H1 from H0.

    Parameters
    ----------
    dN_H1_func : callable(omega_keV) -> array
        Spectrum under alternative hypothesis (at g = g_ref)
    dN_H0_func : callable(omega_keV) -> array
        Spectrum under null hypothesis (at g = g_ref)
    omega_keV : array
        Energy grid for initial evaluation (can be non-uniform)
    E_res_keV : float
        Energy resolution [keV]
    n_uniform : int
        Points in uniform grid for smearing
    omega_range : tuple, optional
        (min, max) energy in keV

    Returns
    -------
    q0_over_g4 : float
        Test statistic divided by g_ref^4
    h : float
        Normalization ratio integral(H1) / integral(H0)
    """
    if omega_range is None:
        omega_range = (omega_min, omega_max)

    # Evaluate spectra on input grid
    dN1_raw = dN_H1_func(omega_keV)
    dN0_raw = dN_H0_func(omega_keV)

    # Apply energy resolution (interpolates to uniform grid)
    dN1, omega_grid = apply_energy_resolution(dN1_raw, omega_keV, E_res_keV,
                                               n_uniform, omega_range)
    dN0, _ = apply_energy_resolution(dN0_raw, omega_keV, E_res_keV,
                                      n_uniform, omega_range)

    # Normalization: h = N_total(H1) / N_total(H0)
    int_N1 = trapezoid(dN1, omega_grid)
    int_N0 = trapezoid(dN0, omega_grid)
    h = int_N1 / int_N0

    # Rescaled null hypothesis spectrum
    dN0_scaled = h * dN0

    # Integrand: dN1 * ln(dN1 / dN0_scaled)
    mask = (dN1 > 1e-100) & (dN0_scaled > 1e-100)  # type: ignore[operator]
    integrand = np.zeros_like(dN1)
    integrand[mask] = dN1[mask] * np.log(dN1[mask] / dN0_scaled[mask])

    # Test statistic at g = g_ref
    q0_at_gref = 2.0 * trapezoid(integrand, omega_grid)

    # q0 scales as g^4
    q0_over_g4 = q0_at_gref / g_ref**4

    return q0_over_g4, h


# =============================================================================
# Discovery limit from q0
# =============================================================================
def g_discovery_from_q0(q0_over_g4, n_sigma=3):
    """
    Convert q0/g^4 to discovery coupling.

    g_disc = (n_sigma^2 / q0_over_g4)^0.25

    Parameters
    ----------
    q0_over_g4 : float or array
        Test statistic divided by g^4
    n_sigma : float
        Significance level (default 3, so threshold is n_sigma^2 = 9)

    Returns
    -------
    g_disc : float or array
        Discovery coupling [GeV^-1]
    """
    q0 = np.asarray(q0_over_g4, dtype=float)
    mask = q0 > 0
    result = np.full_like(q0, np.nan)
    result[mask] = (n_sigma**2 / q0[mask])**0.25
    return float(result) if result.ndim == 0 else result


# =============================================================================
# Convenience: scan over mass parameter
# =============================================================================
def compute_discovery_limit(mass_values, dN_H1_maker, dN_H0_func,
                            E_res_keV, n_uniform=80000, n_sigma=3,
                            verbose=False):
    """
    Compute discovery limit g_disc(mass) over an array of mass values.

    Parameters
    ----------
    mass_values : array
        Array of mass parameter values to scan
    dN_H1_maker : callable(mass) -> callable(omega)
        Factory: given mass parameter, returns H1 spectrum function.
        The H1 spectrum should include analytic damping with the correct
        E_res_keV to handle fast oscillations exactly.
    dN_H0_func : callable(omega)
        H0 spectrum function (fixed, not mass-dependent)
    E_res_keV : float
        Energy resolution [keV] for Gaussian smearing of smooth envelope
    n_uniform : int
        Uniform grid points
    n_sigma : float
        Significance threshold
    verbose : bool
        Print progress

    Returns
    -------
    g_discovery : array
    q0_over_g4 : array
    """
    omega = np.linspace(omega_min, omega_max, n_uniform)
    n = len(mass_values)
    q0_vals = np.zeros(n)

    for i, mass in enumerate(mass_values):
        if verbose and i % max(1, n // 10) == 0:
            print(f"  [{i+1}/{n}] mass = {mass:.2e}")

        dN_H1 = dN_H1_maker(mass)
        q0_vals[i], _ = compute_q0(dN_H1, dN_H0_func, omega, E_res_keV,
                                    n_uniform)

    g_disc = g_discovery_from_q0(q0_vals, n_sigma)
    return g_disc, q0_vals


# =============================================================================
# Profile likelihood: scan H0 over mass
# =============================================================================
def compute_profile_q0(dN_H1_func, dN_H0_maker, omega_keV, E_res_keV,
                        m_scan_min=1e-3, m_scan_max=6e-1, n_coarse=8,
                        n_uniform=80000, focus_masses=None, n_focus=20):
    """
    Profile likelihood test statistic: minimize q0 over H0 mass parameter.

    For a given H1, finds m_hat that minimizes q0 (the H0 mass that best
    mimics H1). Uses a coarse grid to locate the basin, then refines with
    scipy.optimize.minimize_scalar (Brent's method, ~10-15 evaluations).

    Parameters
    ----------
    dN_H1_func : callable(omega_keV) -> array
        H1 spectrum (fixed). Should include analytic damping with the
        correct E_res_keV to handle fast oscillations exactly.
    dN_H0_maker : callable(m) -> callable(omega_keV)
        Factory: given H0 mass, returns H0 spectrum function
    omega_keV : array
        Energy grid
    E_res_keV : float
        Energy resolution [keV]
    m_scan_min, m_scan_max : float
        Mass range to scan [eV]
    n_coarse : int
        Number of coarse grid points to locate the basin (default 8)
    n_uniform : int
        Uniform grid points for smearing
    focus_masses : list/tuple of float, optional
        Extra masses around which to densely sample (e.g. [m1, m2]).
        Adds n_focus points in a band around each focus mass.
    n_focus : int
        Number of extra points per focus mass (default 20)

    Returns
    -------
    q0_over_g4 : float
        Profile likelihood test statistic / g^4
    m_hat : float
        Best-fit H0 mass [eV]
    diagnostics : dict
        Contains m_coarse, D_coarse, m_hat for plotting
    """
    def objective(log_m):
        m = 10**log_m
        dN_H0 = dN_H0_maker(m)
        D, _ = compute_q0(dN_H1_func, dN_H0, omega_keV,
                           E_res_keV, n_uniform)
        return D

    log_m_min = np.log10(m_scan_min)
    log_m_max = np.log10(m_scan_max)

    # Coarse grid to find the right basin
    m_coarse = np.logspace(log_m_min, log_m_max, n_coarse)

    # Add dense sampling around focus masses
    if focus_masses is not None:
        focus_pts = []
        for mf in focus_masses:
            if mf <= 0:
                continue
            # Dense band: ±0.3 dex around each focus mass
            log_mf = np.log10(mf)
            lo = max(log_m_min, log_mf - 0.3)
            hi = min(log_m_max, log_mf + 0.3)
            focus_pts.append(np.logspace(lo, hi, n_focus))
        # Also add dense sampling between the focus masses
        if len(focus_masses) >= 2:
            fm_sorted = sorted([m for m in focus_masses if m > 0])
            for k in range(len(fm_sorted) - 1):
                lo = np.log10(fm_sorted[k])
                hi = np.log10(fm_sorted[k + 1])
                focus_pts.append(np.logspace(lo, hi, n_focus))
        if focus_pts:
            m_coarse = np.unique(np.concatenate([m_coarse] + focus_pts))

    D_coarse = np.array([objective(np.log10(m)) for m in m_coarse])

    idx_min = np.argmin(D_coarse)

    # Bracket: neighbors on the coarse grid (or full range at edges)
    n_total = len(m_coarse)
    log_lower = np.log10(m_coarse[max(0, idx_min - 1)])
    log_upper = np.log10(m_coarse[min(n_total - 1, idx_min + 1)])

    # Refine with Brent's method in log-space
    res = minimize_scalar(objective, bounds=(log_lower, log_upper),
                          method='bounded',
                          options={'xatol': 1e-3})

    m_hat = 10**res.x  # type: ignore[union-attr]
    D_min = res.fun  # type: ignore[union-attr]

    # Check if coarse grid had a better point (safety)
    if D_coarse[idx_min] < D_min:
        m_hat = m_coarse[idx_min]
        D_min = D_coarse[idx_min]

    diagnostics = {
        'm_coarse': m_coarse,
        'D_coarse': D_coarse,
        'm_hat': m_hat,
        'D_profile': D_min,
    }

    return D_min, m_hat, diagnostics
