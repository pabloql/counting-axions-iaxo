"""
General-purpose spectrum plotter for two-axion models.

Usage:
    from plot_spectra import plot_spectrum_comparison

    configs = [
        {'label': 'Single massless', 'm1': 0, 'm2': 0, 'color': 'k', 'lw': 4, 'alpha': 0.3},
        {'label': r'$r=1.2$', 'm1': 0.02, 'm2': 0.024, 'color': 'crimson'},
    ]
    fig, ax = plot_spectrum_comparison(configs, E_res_keV=0.1)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import os

from iaxo_params import (apply_energy_resolution, omega_min, omega_max,
                          setup_plot_style)
from conversion_prob import dN_domega


# =============================================================================
# LaTeX label helper
# =============================================================================
def sci_label(val):
    """Format a number in LaTeX scientific notation: 1.6 x 10^{-7}."""
    exp = int(np.floor(np.log10(abs(val))))
    coeff = val / 10**exp
    if abs(coeff - 1.0) < 0.01:
        return r'10^{%d}' % exp
    return r'%.2f \times 10^{%d}' % (coeff, exp)


# =============================================================================
# Resolve mass parameters from config dict
# =============================================================================
def resolve_masses(cfg):
    """
    Extract (m1, m2) from a config dict. Accepts any two of:
        'm1', 'm2'       -- direct masses [eV]
        'sqrt_dm2'        -- sqrt(m2^2 - m1^2) [eV], needs m1 or m2
        'r'               -- mass ratio m2/m1, needs m1

    Examples:
        {'m1': 0.02, 'm2': 0.024}           -> (0.02, 0.024)
        {'m1': 0.02, 'sqrt_dm2': 1e-6}      -> (0.02, sqrt(0.02^2 + 1e-12))
        {'m1': 0.02, 'r': 1.2}              -> (0.02, 0.024)
        {'m2': 0.05, 'sqrt_dm2': 0.01}      -> (sqrt(0.05^2 - 0.01^2), 0.05)
        {'sqrt_dm2': 1e-6}                   -> (0, 1e-6)  [quasi-degenerate]
    """
    m1 = cfg.get('m1', None)
    m2 = cfg.get('m2', None)
    sqrt_dm2 = cfg.get('sqrt_dm2', None)
    r = cfg.get('r', None)

    if m1 is not None and m2 is not None:
        return float(m1), float(m2)
    if m1 is not None and r is not None:
        return float(m1), float(r * m1)
    if m1 is not None and sqrt_dm2 is not None:
        dm2 = sqrt_dm2**2
        return float(m1), float(np.sqrt(m1**2 + dm2))
    if m2 is not None and sqrt_dm2 is not None:
        dm2 = sqrt_dm2**2
        return float(np.sqrt(max(m2**2 - dm2, 0))), float(m2)
    if sqrt_dm2 is not None:
        return 0.0, float(sqrt_dm2)

    return float(m1 or 0.0), float(m2 or 0.0)


# =============================================================================
# Default plot output directory
# =============================================================================
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Plots')
OVERLEAF_DIR = '/home/pabloql/Cloud/Dropbox/Aplicaciones/Overleaf/Counting axions with helioscopes/plots'


# =============================================================================
# Main plotting function
# =============================================================================
def plot_spectrum_comparison(spectra_configs, E_res_keV=0.1, n_uniform=10000,
                             normalize=True, xlim=None, ylim=None,
                             title=None, xlabel=None, ylabel=None,
                             save_name=None, save_overleaf=False,
                             figsize=(8, 6), ax=None):
    """
    Plot one or more spectra on the same axes.

    Parameters
    ----------
    spectra_configs : list of dict
        Each dict has keys:
            'label' : str             -- legend label
            'm1'    : float           -- mass 1 [eV] (default 0)
            'm2'    : float           -- mass 2 [eV] (default 0)
            'color' : str             -- line color
            'lw'    : float           -- line width (default 2)
            'ls'    : str             -- line style (default '-')
            'alpha' : float           -- transparency (default 0.8)
            'zorder': float           -- draw order (default auto)
    E_res_keV : float
        Energy resolution [keV]
    n_uniform : int
        Grid points for evaluation and smearing
    normalize : bool
        If True, normalize all spectra to same total counts as first spectrum.
    xlim : tuple, optional
        (xmin, xmax) for plot
    ylim : tuple, optional
        (ymin, ymax) for plot
    title : str, optional
        Plot title
    save_name : str, optional
        Filename (without extension). Saved as .png in Plots/.
    save_overleaf : bool
        Also save .pdf to Overleaf directory.
    figsize : tuple
        Figure size

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    setup_plot_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    omega = np.linspace(omega_min, omega_max, n_uniform)
    reference_integral = None

    for cfg in spectra_configs:
        m1, m2 = resolve_masses(cfg)
        n_Rsun = cfg.get('n_Rsun', 0.0)
        no_disapp = cfg.get('no_disappearance', False)

        # Compute spectrum
        dN = dN_domega(omega, m1=m1, m2=m2, E_res_keV=E_res_keV,
                       n_Rsun=n_Rsun, no_disappearance=no_disapp)

        # Apply energy resolution smearing
        dN_smeared, omega_grid = apply_energy_resolution(
            dN, omega, E_res_keV, n_uniform=n_uniform)

        # Normalize to same total counts as first spectrum
        if normalize:
            integral = trapezoid(dN_smeared, omega_grid)
            if reference_integral is None:
                reference_integral = integral
            dN_smeared *= reference_integral / integral

        ax.plot(omega_grid, dN_smeared,
                color=cfg.get('color', 'blue'),
                lw=cfg.get('lw', 2),
                ls=cfg.get('ls', '-'),
                alpha=cfg.get('alpha', 0.8),
                label=cfg.get('label', ''),
                zorder=cfg.get('zorder', 2))

    ax.set_xlabel(xlabel or r'Energy [keV]', fontsize=16)
    ax.set_ylabel(ylabel or r'dN/dE [keV$^{-1}$]', fontsize=16)
    if title:
        ax.set_title(title, fontsize=16)
    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(omega_min, omega_max)
    if ylim:
        ax.set_ylim(ylim)

    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)

    ax.tick_params(which='major', direction='in', length=8, width=1.5,
                   top=True, right=True)
    ax.tick_params(which='minor', direction='in', length=4, width=1,
                   top=True, right=True)

    plt.tight_layout()

    if save_name:
        os.makedirs(PLOT_DIR, exist_ok=True)
        plot_path = os.path.join(PLOT_DIR, f'{save_name}.png')
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {plot_path}")

        if save_overleaf and os.path.exists(OVERLEAF_DIR):
            pdf_path = os.path.join(OVERLEAF_DIR, f'{save_name}.pdf')
            fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
            print(f"Saved to Overleaf: {pdf_path}")

    return fig, ax


# =============================================================================
# Ratio plot (spectrum / single massless)
# =============================================================================
def plot_spectrum_ratio(spectra_configs, E_res_keV=0.1, n_uniform=10000,
                        xlim=None, title=None, save_name=None,
                        save_overleaf=False, figsize=(7, 4.5)):
    """
    Plot the ratio of each spectrum to the single massless axion.

    Same spectra_configs format as plot_spectrum_comparison.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)

    omega = np.linspace(omega_min, omega_max, n_uniform)

    # Reference: single massless
    dN_ref = dN_domega(omega, m1=0, m2=0)
    dN_ref_smeared, omega_grid = apply_energy_resolution(
        dN_ref, omega, E_res_keV, n_uniform=n_uniform)

    ax.axhline(1.0, color='k', lw=2, alpha=0.3, label='Single axion (P=1)')

    for cfg in spectra_configs:
        m1, m2 = resolve_masses(cfg)
        n_Rsun = cfg.get('n_Rsun', 0.0)
        if m1 == 0 and m2 == 0 and n_Rsun == 0.0:
            continue  # Skip the reference itself

        dN = dN_domega(omega, m1=m1, m2=m2, E_res_keV=E_res_keV,
                       n_Rsun=n_Rsun)
        dN_smeared, _ = apply_energy_resolution(
            dN, omega, E_res_keV, n_uniform=n_uniform)

        # Normalize to same integral
        ratio_norm = trapezoid(dN_ref_smeared, omega_grid) / trapezoid(dN_smeared, omega_grid)
        ratio = (dN_smeared * ratio_norm) / dN_ref_smeared

        ax.plot(omega_grid, ratio,
                color=cfg.get('color', 'blue'),
                lw=cfg.get('lw', 2),
                ls=cfg.get('ls', '-'),
                alpha=cfg.get('alpha', 0.8),
                label=cfg.get('label', ''))

    ax.set_xlabel(r'Energy [keV]', fontsize=16)
    ax.set_ylabel(r'Ratio to single massless axion', fontsize=16)
    if title:
        ax.set_title(title, fontsize=16)
    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(omega_min, omega_max)

    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3)

    ax.tick_params(which='major', direction='in', length=8, width=1.5,
                   top=True, right=True)
    ax.tick_params(which='minor', direction='in', length=4, width=1,
                   top=True, right=True)

    plt.tight_layout()

    if save_name:
        os.makedirs(PLOT_DIR, exist_ok=True)
        plot_path = os.path.join(PLOT_DIR, f'{save_name}.png')
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {plot_path}")

        if save_overleaf and os.path.exists(OVERLEAF_DIR):
            pdf_path = os.path.join(OVERLEAF_DIR, f'{save_name}.pdf')
            fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
            print(f"Saved to Overleaf: {pdf_path}")

    return fig, ax
