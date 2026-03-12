"""Quick check: does the cross_12 sign change quantitatively predict incoh?

cross_12 = 2Re<VG* · VG²> is the leading Born correction to |R_ij|².
At k=0.3 it's -11%, at k=1.5 it's +79%. This sign change drives incoh.

Can we predict the incoh exponent from the cross_12 variation alone?
"""
import sys
sys.path.insert(0, '/Users/alextoader/Sites/st_vortex_transport/tests')

import numpy as np
from helpers.config import ALPHA_REF, V_ref
from helpers.geometry import disk_bonds
from helpers.ms import build_G_matrix
from helpers.stats import log_log_slope

k_vals = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
k_arr = np.array(k_vals)

print("Cross_12 vs incoh at various R")
print("=" * 70)
print(f"{'R':>4} {'p_incoh':>8} {'p_cross12':>10} {'sign_chg_k':>10} {'pi/<path>':>10}")

for R in [3, 5, 7, 9, 12]:
    dx, dy = disk_bonds(R)
    N = len(dx)
    V = V_ref

    incoh_vals = []
    cross12_vals = []

    for k in k_vals:
        G = build_G_matrix(dx, dy, k)
        VG = V * G
        VG2 = VG @ VG
        mask = ~np.eye(N, dtype=bool)

        term1 = np.mean(np.abs(VG[mask])**2)
        cross12 = 2 * np.mean(np.real(VG[mask].conj() * VG2[mask]))

        resolvent = np.linalg.inv(np.eye(N) - VG)
        incoh = np.mean(np.abs((V * resolvent)[mask])**2)

        incoh_vals.append(incoh)
        cross12_vals.append(cross12)

    p_inc = log_log_slope(k_arr, np.array(incoh_vals))[0]

    # Find sign change k (where cross12 crosses zero)
    c12 = np.array(cross12_vals)
    sign_chg = None
    for i in range(len(c12)-1):
        if c12[i] < 0 and c12[i+1] > 0:
            # Linear interpolation
            sign_chg = k_vals[i] + (k_vals[i+1]-k_vals[i]) * (-c12[i])/(c12[i+1]-c12[i])
            break

    # Mean path excess
    dist = np.sqrt((dx[:, None] - dx[None, :])**2 + (dy[:, None] - dy[None, :])**2)
    pe_list = []
    for i in range(min(N, 15)):
        for j in range(min(N, 15)):
            if i == j: continue
            r_ij = dist[i, j]
            if r_ij < 0.5: continue
            for l in range(min(N, 30)):
                if l == i or l == j: continue
                r_il = dist[i, l]
                r_lj = dist[l, j]
                if r_il < 0.5 or r_lj < 0.5: continue
                pe_list.append(r_il + r_lj - r_ij)
    mean_pe = np.mean(pe_list)
    pi_pe = np.pi / mean_pe

    # cross12 normalized
    norm_c12 = c12 / np.mean(np.abs(VG[~np.eye(N, dtype=bool)])**2)

    print(f"{R:4d} {p_inc:+8.3f} {'' if sign_chg is None else '':>10} "
          f"{'%.3f' % sign_chg if sign_chg else 'N/A':>10} {pi_pe:10.3f}")
    print(f"     cross12/|VG|²: [{norm_c12[0]:+.3f}, {norm_c12[2]:+.3f}, {norm_c12[4]:+.3f}, {norm_c12[6]:+.3f}]")


# ── Does 1st-order perturbation theory (cross_12 only) predict incoh exponent?
print("\n" + "=" * 70)
print("Perturbative prediction: incoh ≈ |VG|² × (1 + cross12/|VG|²)")
print("=" * 70)

for R in [5, 9]:
    dx, dy = disk_bonds(R)
    N = len(dx)
    V = V_ref

    incoh_true = []
    incoh_pert = []

    for k in k_vals:
        G = build_G_matrix(dx, dy, k)
        VG = V * G
        VG2 = VG @ VG
        mask = ~np.eye(N, dtype=bool)

        term1 = np.mean(np.abs(VG[mask])**2)
        cross12 = 2 * np.mean(np.real(VG[mask].conj() * VG2[mask]))
        term2 = np.mean(np.abs(VG2[mask])**2)

        resolvent = np.linalg.inv(np.eye(N) - VG)
        incoh = np.mean(np.abs((V * resolvent)[mask])**2)

        incoh_true.append(incoh / (V**2))  # = <|R_ij|²>
        incoh_pert.append(term1 + cross12 + term2)  # perturbative to order 2

    p_true = log_log_slope(k_arr, np.array(incoh_true))[0]
    p_pert = log_log_slope(k_arr, np.array(incoh_pert))[0]

    print(f"\nR={R}:")
    print(f"  True <|R_ij|²> exponent:         {p_true:+.3f}")
    print(f"  Perturbative (order 2) exponent:  {p_pert:+.3f}")
    print(f"  Agreement: {abs(p_true - p_pert)/abs(p_true)*100:.0f}% error")
