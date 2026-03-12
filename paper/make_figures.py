"""Generate paper figures from stored FDTD data.

Usage: python3 make_figures.py
Produces: fig1_sigma_tr.pdf, fig2_integrand.pdf, fig3_enhancement.pdf
"""
import sys
sys.path.insert(0, '../tests')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.sigma_ring import k_vals, sigma_ring, sigma_alpha_nn
from helpers.config import k_vals_7, ALPHA_REF, V_ref
from helpers.born import V_eff
from helpers.ms import build_G_matrix
from helpers.geometry import disk_bonds

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'legend.fontsize': 11,
    'figure.figsize': (6, 5),  # review size (scales down for PRL)
    'lines.linewidth': 1.8,
    'lines.markersize': 7,
    'savefig.dpi': 300,
})


# ── Figure 1: σ_tr(k) spectrum ──────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel (a): σ_tr(k) at multiple α
for alpha, label in [(0.10, r'$\alpha=0.10$'),
                     (0.20, r'$\alpha=0.20$'),
                     (0.30, r'$\alpha=0.30$'),
                     (0.50, r'$\alpha=0.50$')]:
    ax1.semilogy(k_vals, sigma_alpha_nn[alpha], 'o-', label=label)

ax1.set_xlabel(r'$k$')
ax1.set_ylabel(r'$\sigma_{\rm tr}(k)$')
ax1.legend()
ax1.set_title('(a)')

# Panel (b): σ_tr vs R at fixed k (R^{3/2} scaling)
Rs = np.array([3, 5, 7, 9])
for ik, kv in [(0, 0.3), (2, 0.7), (4, 1.1)]:
    sigmas = np.array([sigma_ring[R][ik] for R in Rs])
    ax2.loglog(Rs, sigmas, 'o-', label=f'$k={kv}$')

# R^{3/2} reference line
R_ref = np.linspace(2.5, 10, 50)
ax2.loglog(R_ref, 2.0 * R_ref**1.5, 'k--', alpha=0.4, label=r'$R^{3/2}$')

ax2.set_xlabel(r'$R$')
ax2.set_ylabel(r'$\sigma_{\rm tr}$')
ax2.legend(fontsize=8)
ax2.set_title('(b)')

fig.tight_layout()
fig.savefig('fig1_sigma_tr.pdf', bbox_inches='tight')
print('Wrote fig1_sigma_tr.pdf')


# ── Figure 2: Boltzmann integrand ────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 5))

for alpha, style in [(0.05, 'v--'), (0.10, 's--'),
                     (0.20, 'D-'), (0.30, 'o-'), (0.50, '^-')]:
    if alpha == 0.05:
        # α=0.05 not in stored data — skip or approximate
        continue
    integrand = np.sin(k_vals)**2 * sigma_alpha_nn[alpha]
    ax.plot(k_vals, integrand, style, label=rf'$\alpha={alpha}$')

ax.axhline(y=np.mean(np.sin(k_vals)**2 * sigma_alpha_nn[0.30]),
           color='gray', ls=':', alpha=0.5)
ax.set_xlabel(r'$k$')
ax.set_ylabel(r'$\sin^2(k)\,\sigma_{\rm tr}(k)$')
ax.legend()

fig.tight_layout()
fig.savefig('fig2_integrand.pdf', bbox_inches='tight')
print('Wrote fig2_integrand.pdf')


# ── Figure 3: MS/Born enhancement ratio ─────────────────────────

def compute_enhancement(R, k_arr):
    """Σ|Tb|²/Σ|Vb|² at each k — same as TestForwardDecoherence."""
    dx, dy = disk_bonds(R)
    N = len(dx)
    V = V_ref
    enh = np.zeros(len(k_arr))
    for ik, kv in enumerate(k_arr):
        G = build_G_matrix(dx, dy, kv)
        b = np.exp(1j * kv * dx)
        Vb = V * b
        T = np.linalg.solve(np.eye(N) - V * G, V * np.eye(N))
        Tb = T @ b
        enh[ik] = np.sum(np.abs(Tb)**2) / np.sum(np.abs(Vb)**2)
    return enh


fig, ax = plt.subplots(figsize=(6, 5))

for R, marker in [(5, 'o'), (7, 's'), (9, 'D')]:
    enh = compute_enhancement(R, k_vals_7)
    ax.plot(k_vals_7, enh, f'{marker}-', label=f'$R={R}$')

ax.axhline(y=1.0, color='gray', ls=':', alpha=0.5)
ax.set_xlabel(r'$k$')
ax.set_ylabel(r'MS/Born power ratio')
ax.legend()

fig.tight_layout()
fig.savefig('fig3_enhancement.pdf', bbox_inches='tight')
print('Wrote fig3_enhancement.pdf')

plt.close('all')
print('Done.')
