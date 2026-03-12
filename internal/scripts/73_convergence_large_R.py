"""Verify convergence of incoh and cos(Φ) exponents at larger R.

Expected large-R limits:
  N_coop → k^{-3/2} (geometric, Fresnel zone)
  incoh  → k^{+1/2}? (Born series alternation)
  cos(Φ) → k^{0}   (becomes constant)
  offdiag_amp = √(N_coop × incoh) → k^{(-3/2 + 1/2)/2} = k^{-1/2}
  cross → offdiag_amp (since cos(Φ)→const) → k^{-1/2}

If cross ~ k^{-1/2}: enh = D - |cross| + offdiag/born
Since offdiag/born ~ k^{-0.97} ≪ |cross/born| ~ k^{-0.5},
  enh ≈ D - A·k^{-1/2}

This means Δp = d(ln enh)/d(ln k) ≈ (1/2)·A·k^{-1/2}/(D - A·k^{-1/2})

Not a simple power law, but the average over the k-range gives the shift.
"""
import sys
sys.path.insert(0, '/Users/alextoader/Sites/st_vortex_transport/tests')

import numpy as np
from helpers.config import K1, K2, c_lat, ALPHA_REF, V_ref
from helpers.geometry import disk_bonds
from helpers.lattice import k_eff
from helpers.ms import build_G_matrix, T_matrix
from helpers.born import V_eff
from helpers.stats import log_log_slope

alpha = ALPHA_REF
V = V_ref
k_vals = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
k_arr = np.array(k_vals)

print("Convergence of exponents with R")
print("=" * 70)

print(f"\n{'R':>4} {'N':>5} {'p_Ncoop':>8} {'p_incoh':>8} {'p_cosΦ':>8} {'p_offamp':>8} {'p_cross':>8} {'p_enh':>8}")

for R in [3, 5, 7, 9, 12, 15]:
    dx, dy = disk_bonds(R)
    N = len(dx)

    incoh_vals = []
    ncoop_vals = []
    cos_vals = []
    offdiag_amp_vals = []
    cross_vals = []
    enh_vals = []
    born_vals = []

    for k in k_vals:
        b = np.exp(1j * k * dx)
        G = build_G_matrix(dx, dy, k)
        VG = V * G
        T_mat = V * np.linalg.inv(np.eye(N) - VG)
        Tb = T_mat @ b
        Vb = V * b

        # Per-site decomposition
        diag_part = np.diag(T_mat) * b
        offdiag_part = Tb - diag_part

        # Powers
        born_pow = np.sum(np.abs(Vb)**2)
        diag_pow = np.sum(np.abs(diag_part)**2)
        offdiag_pow = np.sum(np.abs(offdiag_part)**2)
        cross = 2 * np.real(np.sum(np.conj(diag_part) * offdiag_part))
        total_pow = diag_pow + offdiag_pow + cross

        # Incoh: mean |T_ij|² off-diagonal
        mask = ~np.eye(N, dtype=bool)
        T_offdiag = T_mat[mask]
        incoh = np.mean(np.abs(T_offdiag)**2)

        # N_coop = offdiag_pow / (N × incoh)
        # Actually: offdiag_pow = Σ_i |Σ_{j≠i} T_ij b_j|²
        # incoh × N(N-1) = Σ_i Σ_{j≠i} |T_ij|²
        # N_coop = offdiag_pow / (Σ_i Σ_{j≠i} |T_ij|² × <|b_j|²>)
        # Since |b_j|² = 1, N_coop = offdiag_pow / (N × (N-1) × incoh)
        # But conventionally we use N_coop = offdiag_pow / incoherent_sum
        # where incoherent_sum = Σ_i Σ_{j≠i} |T_ij|² |b_j|² = N(N-1)×incoh
        incoherent_sum = N * (N - 1) * incoh
        ncoop = offdiag_pow / incoherent_sum if incoherent_sum > 0 else 0

        # cos(Φ)
        diag_amp = np.sqrt(diag_pow)
        offdiag_amp = np.sqrt(offdiag_pow)
        cos_eff = abs(cross / (2 * diag_amp * offdiag_amp))

        enh = total_pow / born_pow

        incoh_vals.append(incoh)
        ncoop_vals.append(ncoop)
        cos_vals.append(cos_eff)
        offdiag_amp_vals.append(offdiag_amp / np.sqrt(born_pow))
        cross_vals.append(abs(cross / born_pow))
        enh_vals.append(enh)
        born_vals.append(born_pow)

    p_nc = log_log_slope(k_arr, np.array(ncoop_vals))[0]
    p_inc = log_log_slope(k_arr, np.array(incoh_vals))[0]
    p_cos = log_log_slope(k_arr, np.array(cos_vals))[0]
    p_oamp = log_log_slope(k_arr, np.array(offdiag_amp_vals))[0]
    p_cr = log_log_slope(k_arr, np.array(cross_vals))[0]
    p_enh = log_log_slope(k_arr, np.array(enh_vals))[0]

    print(f"{R:4d} {N:5d} {p_nc:+8.3f} {p_inc:+8.3f} {p_cos:+8.3f} {p_oamp:+8.3f} {p_cr:+8.3f} {p_enh:+8.3f}")


# ── Predicted chain at large R ──────────────────────────────────
print("\n" + "=" * 70)
print("PREDICTED vs MEASURED at large R")
print("=" * 70)

for R in [5, 9, 15]:
    dx, dy = disk_bonds(R)
    N = len(dx)

    ncoop_vals = []
    incoh_vals = []

    for k in k_vals:
        b = np.exp(1j * k * dx)
        G = build_G_matrix(dx, dy, k)
        VG = V * G
        T_mat = V * np.linalg.inv(np.eye(N) - VG)
        Tb = T_mat @ b
        diag_part = np.diag(T_mat) * b
        offdiag_part = Tb - diag_part

        offdiag_pow = np.sum(np.abs(offdiag_part)**2)
        mask = ~np.eye(N, dtype=bool)
        incoh = np.mean(np.abs(T_mat[mask])**2)
        incoherent_sum = N * (N - 1) * incoh
        ncoop = offdiag_pow / incoherent_sum if incoherent_sum > 0 else 0

        ncoop_vals.append(ncoop)
        incoh_vals.append(incoh)

    p_nc = log_log_slope(k_arr, np.array(ncoop_vals))[0]
    p_inc = log_log_slope(k_arr, np.array(incoh_vals))[0]

    # Predicted offdiag_amp exponent = (p_nc + p_inc) / 2
    # (since offdiag_pow/N ~ N_coop × (N-1) × incoh, and offdiag_amp ~ √offdiag_pow)
    # offdiag_amp/√born ~ √(N_coop × (N-1) × incoh × N) / √(N|V|²)
    # The N-dep cancels leaving √(N_coop × (N-1) × incoh / |V|²)
    # exponent of offdiag_amp = (p_nc + p_inc)/2
    p_oamp_pred = (p_nc + p_inc) / 2

    print(f"\nR={R}, N={N}:")
    print(f"  N_coop exponent: {p_nc:+.3f} (→ -3/2 = -1.500)")
    print(f"  incoh exponent:  {p_inc:+.3f} (→ +1/2 = +0.500?)")
    print(f"  Predicted offdiag_amp: {p_oamp_pred:+.3f}")
    print(f"  Predicted cross (if cos→const): {p_oamp_pred:+.3f}")


# ── Verify: what is the analytic enh shift? ─────────────────────
print("\n" + "=" * 70)
print("TRANSPORT SHIFT PREDICTION")
print("=" * 70)

# At large R:
#   cross/born ~ k^{-1/2} (from N_coop^{-3/2} × incoh^{+1/2}, cos→const)
#   enh ≈ D - |cross/born|, where D ≈ diag/born ≈ 1.35 (constant)
#   offdiag/born ~ k^{-1} (much smaller than |cross|)
#
# The enh is NOT a power law, but its average slope over [0.3, 1.5] gives Δp.
# Let me compute what slope a model enh = D - A·k^{-1/2} would give.

R = 5
dx, dy = disk_bonds(R)
N = len(dx)

D_vals = []
cross_abs_vals = []
enh_model_vals = []

for k in k_vals:
    b = np.exp(1j * k * dx)
    G = build_G_matrix(dx, dy, k)
    VG = V * G
    T_mat = V * np.linalg.inv(np.eye(N) - VG)
    Tb = T_mat @ b
    Vb = V * b

    diag_part = np.diag(T_mat) * b
    offdiag_part = Tb - diag_part

    born_pow = np.sum(np.abs(Vb)**2)
    diag_pow = np.sum(np.abs(diag_part)**2)
    offdiag_pow = np.sum(np.abs(offdiag_part)**2)
    cross = 2 * np.real(np.sum(np.conj(diag_part) * offdiag_part))
    enh = (diag_pow + offdiag_pow + cross) / born_pow

    D_vals.append(diag_pow / born_pow)
    cross_abs_vals.append(abs(cross / born_pow))

# Fit cross/born to A·k^{p_cross}
p_cr, logA = log_log_slope(k_arr, np.array(cross_abs_vals))
A = np.exp(logA)
D_avg = np.mean(D_vals)

print(f"D (diag/born) average: {D_avg:.4f}")
print(f"|cross/born| fit: A={A:.4f} × k^{p_cr:.3f}")
print(f"offdiag/born: small, ~{np.mean([v for v in [0.2505, 0.2164, 0.1584, 0.1183, 0.0863, 0.0609, 0.0622]]):.3f} average")

# Model: enh(k) = D + offdiag(k) - |cross(k)|
# For simplicity, enh(k) ≈ D_avg - A × k^p_cr + offdiag_avg
# The slope of log(enh) is approximately:
# d ln(enh)/d ln(k) = k/enh × d(enh)/dk = k/enh × (-A × p_cr × k^{p_cr - 1})
#                    = -A × p_cr × k^{p_cr} / enh

print(f"\nLocal slope d ln(enh)/d ln k at each k:")
for i, k in enumerate(k_vals):
    enh_i = enh_model_vals[i] if enh_model_vals else (D_avg - cross_abs_vals[i])
    local_slope = cross_abs_vals[i] * abs(p_cr) / (D_avg - cross_abs_vals[i])
    print(f"  k={k}: |cross/born|={cross_abs_vals[i]:.4f}, enh≈{D_avg - cross_abs_vals[i] + 0.13:.4f}, Δp_local≈{local_slope:.3f}")

# Average slope
# Δp ≈ <|cross|/born × |p_cr| / (D - |cross|/born + offdiag/born)>
offdiag_avg = 0.13  # rough average
slopes = []
for i in range(len(k_vals)):
    enh_approx = D_avg - cross_abs_vals[i] + offdiag_avg
    slopes.append(cross_abs_vals[i] * abs(p_cr) / enh_approx)
avg_slope = np.mean(slopes)
print(f"\nAverage Δp (total power): {avg_slope:.3f}")
print(f"Measured Δp (total power): 0.354")
print(f"Measured Δp (transport): 0.448")
