"""Can we predict the transport shift from cross ~ k^{-1/2}?

The transport shift (F4) is +0.448, R-independent.
The total power shift (enh exponent) is +0.354 at R=5 and keeps growing.

The transport cross-section uses angular integration with (1-cosθ_s) weight.
The total power uses Σ|Tb|² (no angular weight).

Let's compute the transport-weighted decomposition directly.
"""
import sys
sys.path.insert(0, '/Users/alextoader/Sites/st_vortex_transport/tests')

import numpy as np
from helpers.config import K1, K2, c_lat, ALPHA_REF, V_ref, k_vals_7
from helpers.geometry import disk_bonds
from helpers.lattice import k_eff
from helpers.ms import build_G_matrix, T_matrix, sigma_tr_born_ms
from helpers.born import V_eff, sigma_bond_born
from helpers.stats import cv, log_log_slope

R = 5
alpha = ALPHA_REF
V = V_ref
dx, dy = disk_bonds(R)
N = len(dx)
k_vals = list(k_vals_7)
k_arr = np.array(k_vals)

print(f"R={R}, α={alpha}, V={V:.4f}, N={N}")

# ── Compute transport σ and integrand ───────────────────────────
print("\n" + "=" * 70)
print("TRANSPORT INTEGRAND DECOMPOSITION")
print("=" * 70)

# Get Born and MS sigma_tr
sigma_born, sigma_ms = sigma_tr_born_ms(dx, dy, k_arr, alpha, n_theta=50, n_phi=100)

sin2k = np.sin(k_arr)**2
integrand_born = sin2k * sigma_born
integrand_ms = sin2k * sigma_ms

p_sigma_born = log_log_slope(k_arr, sigma_born)[0]
p_sigma_ms = log_log_slope(k_arr, sigma_ms)[0]
shift = p_sigma_ms - p_sigma_born

print(f"\n{'k':>5} {'σ_Born':>10} {'σ_MS':>10} {'ratio':>8} {'I_Born':>10} {'I_MS':>10}")
for i, k in enumerate(k_vals):
    print(f"{k:5.1f} {sigma_born[i]:10.4f} {sigma_ms[i]:10.4f} "
          f"{sigma_ms[i]/sigma_born[i]:8.4f} "
          f"{integrand_born[i]:10.4f} {integrand_ms[i]:10.4f}")

print(f"\nσ_Born exponent: {p_sigma_born:.3f}")
print(f"σ_MS exponent:   {p_sigma_ms:.3f}")
print(f"Shift:           {shift:+.3f}")
print(f"Integrand CV (Born): {100*cv(integrand_born):.1f}%")
print(f"Integrand CV (MS):   {100*cv(integrand_ms):.1f}%")


# ── N_eff decomposition in transport space ──────────────────────
print("\n" + "=" * 70)
print("N_eff = σ_tr / σ_bond (TRANSPORT-WEIGHTED)")
print("=" * 70)

neff_born = []
neff_ms = []
for i, k in enumerate(k_vals):
    sb = sigma_bond_born(k, alpha)
    neff_born.append(sigma_born[i] / sb)
    neff_ms.append(sigma_ms[i] / sb)

p_nb = log_log_slope(k_arr, np.array(neff_born))[0]
p_nm = log_log_slope(k_arr, np.array(neff_ms))[0]
shift_neff = p_nm - p_nb

print(f"\n{'k':>5} {'N_Born':>10} {'N_MS':>10} {'ratio':>8}")
for i, k in enumerate(k_vals):
    print(f"{k:5.1f} {neff_born[i]:10.3f} {neff_ms[i]:10.3f} {neff_ms[i]/neff_born[i]:8.4f}")

print(f"\nN_eff_Born exponent: {p_nb:.3f}")
print(f"N_eff_MS exponent:   {p_nm:.3f}")
print(f"N_eff shift:         {shift_neff:+.3f}")


# ── Enh in transport space ──────────────────────────────────────
print("\n" + "=" * 70)
print("TRANSPORT ENHANCEMENT = σ_MS / σ_Born")
print("=" * 70)

enh_tr = sigma_ms / sigma_born
p_enh_tr = log_log_slope(k_arr, enh_tr)[0]

print(f"\n{'k':>5} {'enh_tr':>10} {'enh_pow':>10}")
for i, k in enumerate(k_vals):
    print(f"{k:5.1f} {enh_tr[i]:10.4f}")

print(f"\nenh_tr exponent: {p_enh_tr:.3f}")
print(f"enh_pow (total power) exponent: 0.354 (from F10)")
print(f"Difference: {p_enh_tr - 0.354:.3f}")

# ── Compare total power vs transport enhancement ────────────────
print("\n" + "=" * 70)
print("TOTAL POWER vs TRANSPORT ENHANCEMENT")
print("=" * 70)

# Total power enh
enh_pow = []
for k in k_vals:
    b = np.exp(1j * k * dx)
    T_mat = T_matrix(dx, dy, k, alpha)
    Tb = T_mat @ b
    Vb = V * b
    enh_pow.append(np.sum(np.abs(Tb)**2) / np.sum(np.abs(Vb)**2))

p_enh_pow = log_log_slope(k_arr, np.array(enh_pow))[0]

print(f"\n{'k':>5} {'enh_pow':>10} {'enh_tr':>10} {'ratio':>10}")
for i, k in enumerate(k_vals):
    print(f"{k:5.1f} {enh_pow[i]:10.4f} {enh_tr[i]:10.4f} {enh_tr[i]/enh_pow[i]:10.4f}")

print(f"\nenh_pow exponent: {p_enh_pow:.3f}")
print(f"enh_tr exponent:  {p_enh_tr:.3f}")
print(f"Transport enhancement is STEEPER than total power because:")
print(f"transport weight suppresses forward cone where Born/MS differ LEAST")

# ── Check R-dependence of transport enhancement ─────────────────
print("\n" + "=" * 70)
print("TRANSPORT ENHANCEMENT EXPONENT vs R")
print("=" * 70)

for R_test in [3, 5, 7, 9, 12]:
    dx_t, dy_t = disk_bonds(R_test)
    N_t = len(dx_t)

    sb, sm = sigma_tr_born_ms(dx_t, dy_t, k_arr, alpha, n_theta=50, n_phi=100)
    enh = sm / sb
    p = log_log_slope(k_arr, enh)[0]

    # Also total power
    enh_p = []
    for k in k_vals:
        b = np.exp(1j * k * dx_t)
        T_mat = T_matrix(dx_t, dy_t, k, alpha)
        Tb = T_mat @ b
        Vb = V * b
        enh_p.append(np.sum(np.abs(Tb)**2) / np.sum(np.abs(Vb)**2))

    p_pow = log_log_slope(k_arr, np.array(enh_p))[0]

    print(f"R={R_test:2d}, N={N_t:4d}: enh_tr_exp={p:+.3f}, enh_pow_exp={p_pow:+.3f}, diff={p-p_pow:+.3f}")
