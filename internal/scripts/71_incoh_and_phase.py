"""Investigate incoh exponent (+0.48) and cos(phase) exponent (-0.155).

Two questions:
1. Why does per-element |T_ij|² grow as k^{+0.48} when |G_ij|² is k-independent?
2. Why does |cos(phase)| between diag and offdiag decrease as k^{-0.155}?
"""
import sys
sys.path.insert(0, '/Users/alextoader/Sites/st_vortex_transport/tests')

import numpy as np
from helpers.config import K1, K2, c_lat, ALPHA_REF, V_ref
from helpers.geometry import disk_bonds
from helpers.lattice import k_eff
from helpers.ms import build_G_matrix, T_matrix
from helpers.born import V_eff
from helpers.stats import cv, log_log_slope

R = 5
alpha = ALPHA_REF
V = V_ref
dx, dy = disk_bonds(R)
N = len(dx)
k_vals = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]

print(f"R={R}, α={alpha}, V={V:.4f}, N={N}")
print()

# ── Part 1: Incoh analysis ──────────────────────────────────────
print("=" * 70)
print("PART 1: INCOH — per-element |T_ij|² vs k")
print("=" * 70)

incoh_vals = []
G_incoh_vals = []
R_incoh_vals = []
born1_incoh_vals = []

for k in k_vals:
    b = np.exp(1j * k * dx)

    G = build_G_matrix(dx, dy, k)
    VG = V * G

    # Resolvent
    resolvent = np.linalg.inv(np.eye(N) - VG)
    T_mat = V * resolvent

    # Off-diagonal masks
    mask = ~np.eye(N, dtype=bool)

    # |T_ij|² off-diagonal
    T_offdiag = T_mat[mask]
    incoh = np.mean(np.abs(T_offdiag)**2)
    incoh_vals.append(incoh)

    # Bare |G_ij|²
    G_offdiag = G[mask]
    G_incoh = np.mean(np.abs(G_offdiag)**2)
    G_incoh_vals.append(G_incoh)

    # |R_ij|² off-diagonal
    R_offdiag = resolvent[mask]
    R_incoh = np.mean(np.abs(R_offdiag)**2)
    R_incoh_vals.append(R_incoh)

    # Born 1st order: T_ij^(1) = V² G_ij
    born1_incoh = np.mean(np.abs(V**2 * G_offdiag)**2)
    born1_incoh_vals.append(born1_incoh)

print(f"\n{'k':>5} {'incoh':>12} {'G_incoh':>12} {'R_incoh':>12} {'born1':>12} {'|T/VG|²':>10}")
for i, k in enumerate(k_vals):
    # T_ij = V * R_ij, born1 = V² G_ij → ratio = |R_ij/G_ij|² per-element
    ratio = incoh_vals[i] / born1_incoh_vals[i]
    print(f"{k:5.1f} {incoh_vals[i]:12.6f} {G_incoh_vals[i]:12.8f} {R_incoh_vals[i]:12.6f} "
          f"{born1_incoh_vals[i]:12.6f} {ratio:10.3f}")

k_arr = np.array(k_vals)
p_incoh = log_log_slope(k_arr, np.array(incoh_vals))[0]
p_G = log_log_slope(k_arr, np.array(G_incoh_vals))[0]
p_R = log_log_slope(k_arr, np.array(R_incoh_vals))[0]
p_b1 = log_log_slope(k_arr, np.array(born1_incoh_vals))[0]

print(f"\nExponents: incoh={p_incoh:.3f}, G_incoh={p_G:.3f}, R_incoh={p_R:.3f}, born1={p_b1:.3f}")
print(f"Since G_incoh exp={p_G:.3f} ≈ 0 (k-independent), all k-dep is in resolvent.")
print(f"incoh = V² × R_incoh → incoh exp ≈ R_incoh exp = {p_R:.3f}")

# ── VG eigenvalue spectrum ──────────────────────────────────────
print(f"\n{'k':>5} {'ρ(VG)':>10} {'Re(λ_max)':>12} {'Re(λ_min)':>12}")
spec_radii = []
for k in k_vals:
    G = build_G_matrix(dx, dy, k)
    VG = V * G
    eigs = np.linalg.eigvals(VG)
    rho = np.max(np.abs(eigs))
    spec_radii.append(rho)
    print(f"{k:5.1f} {rho:10.4f} {np.max(np.real(eigs)):12.4f} {np.min(np.real(eigs)):12.4f}")

p_spec = log_log_slope(k_arr, np.array(spec_radii))[0]
print(f"Spectral radius exponent: {p_spec:.3f}")

# ── Born series order-by-order ──────────────────────────────────
print("\n" + "=" * 70)
print("Born series order-by-order (off-diagonal |T_ij|)")
print("=" * 70)

for k_test in [0.3, 0.9, 1.5]:
    G = build_G_matrix(dx, dy, k_test)
    VG = V * G
    mask = ~np.eye(N, dtype=bool)

    R_partial = np.zeros((N, N), dtype=complex)
    VG_power = np.eye(N, dtype=complex)

    rms_by_order = []
    for order in range(1, 8):
        VG_power = VG_power @ VG
        R_partial = R_partial + VG_power  # cumulative: Σ (VG)^n for n=1..order

        T_partial = V * R_partial  # off-diagonal T through order n
        T_offdiag = T_partial[mask]
        rms = np.sqrt(np.mean(np.abs(T_offdiag)**2))
        rms_by_order.append(rms)

    # Full T-matrix
    T_full = T_matrix(dx, dy, k_test, alpha)
    T_full_offdiag = T_full[mask]
    rms_full = np.sqrt(np.mean(np.abs(T_full_offdiag)**2))

    print(f"\nk={k_test}: RMS |T_ij| by Born series order (cumulative, off-diag only)")
    for n, rms in enumerate(rms_by_order, 1):
        ratio_to_full = rms / rms_full
        print(f"  order {n}: {rms:.6f} (ratio to full: {ratio_to_full:.4f})")
    print(f"  FULL:    {rms_full:.6f}")


# ── Part 2: Phase analysis ──────────────────────────────────────
print("\n" + "=" * 70)
print("PART 2: PHASE between diagonal and off-diagonal forward amplitudes")
print("=" * 70)

phase_vals = []
diag_amp_vals = []
offdiag_amp_vals = []
cos_phase_vals = []

for k in k_vals:
    b = np.exp(1j * k * dx)

    T_mat = T_matrix(dx, dy, k, alpha)
    Tb = T_mat @ b

    # Decompose per-site
    diag_part = np.diag(T_mat) * b      # T_ii b_i
    offdiag_part = Tb - diag_part        # Σ_{j≠i} T_ij b_j

    # Forward sums
    S_diag = np.sum(diag_part)
    S_offdiag = np.sum(offdiag_part)

    phase = np.angle(S_offdiag / S_diag)
    cos_p = np.cos(phase)

    diag_amp_vals.append(np.abs(S_diag))
    offdiag_amp_vals.append(np.abs(S_offdiag))
    phase_vals.append(phase)
    cos_phase_vals.append(cos_p)

print(f"\n{'k':>5} {'|S_diag|':>10} {'|S_offdiag|':>12} {'phase/π':>8} {'cos(phase)':>12}")
for i, k in enumerate(k_vals):
    print(f"{k:5.1f} {diag_amp_vals[i]:10.2f} {offdiag_amp_vals[i]:12.2f} "
          f"{phase_vals[i]/np.pi:8.4f} {cos_phase_vals[i]:12.4f}")

cos_abs = np.abs(np.array(cos_phase_vals))
p_cos = log_log_slope(k_arr, cos_abs)[0]
print(f"\n|cos(phase)| exponent: {p_cos:.3f}")
print(f"Phase deviation from π: {phase_vals[0]/np.pi:.4f}π → {phase_vals[-1]/np.pi:.4f}π")

# ── What determines the phase? ──────────────────────────────────
print("\n" + "=" * 70)
print("PART 3: PHASE MECHANISM")
print("=" * 70)

# Phase of S_offdiag relative to S_diag.
# S_diag = Σ_i T_ii b_i. Since T_ii = V/(1-VG_00) and b_i = exp(ikx_i),
#   S_diag = T_11 × Σ exp(ikx_i) (all T_ii are equal by translational invariance of G_00)
#
# S_offdiag = Σ_i Σ_{j≠i} T_ij b_j = Σ_i Σ_{j≠i} T_ij exp(ikx_j)
#   The T_ij carry complex phases from exp(ik_eff r_ij) in the Green function.
#
# Key: the FORWARD sum weights by exp(ikx_j), which at low k ≈ 1 (all in phase).
# The off-diagonal T_ij ∝ V² G_ij at first order, which has phase exp(ik_eff r_ij).

for k in [0.3, 0.9, 1.5]:
    b = np.exp(1j * k * dx)
    G = build_G_matrix(dx, dy, k)
    T_mat = T_matrix(dx, dy, k, alpha)

    # T_ii is the same for all i
    T_diag_val = T_mat[0, 0]

    # Forward sum of b: Σ exp(ikx_i) — the "form factor"
    F_inc = np.sum(b)

    S_diag = T_diag_val * F_inc

    # Off-diagonal: Σ_i Σ_{j≠i} T_ij exp(ikx_j)
    # = Σ_j [Σ_{i≠j} T_ij] exp(ikx_j)
    # = Σ_j [column sum of T minus diagonal] × b_j

    col_sum_offdiag = np.sum(T_mat, axis=0) - np.diag(T_mat)  # Σ_{i≠j} T_ij for each j
    S_offdiag = np.sum(col_sum_offdiag * b)

    # Born approximation comparison
    # T_ij^(1) = V² G_ij (first order only, i≠j)
    # G_ij has phase exp(ik_eff r_ij)/(4πc²r_ij)
    T_born_offdiag = V**2 * (G - np.diag(np.diag(G)))  # zero diagonal
    col_sum_born = np.sum(T_born_offdiag, axis=0)
    S_offdiag_born = np.sum(col_sum_born * b)

    phase_full = np.angle(S_offdiag / S_diag)
    phase_born = np.angle(S_offdiag_born / S_diag)

    print(f"\nk={k}:")
    print(f"  S_diag      = {S_diag:.4f}   (phase/π = {np.angle(S_diag)/np.pi:.4f})")
    print(f"  S_off_full  = {S_offdiag:.4f}  (phase/π = {np.angle(S_offdiag)/np.pi:.4f})")
    print(f"  S_off_born1 = {S_offdiag_born:.4f}  (phase/π = {np.angle(S_offdiag_born)/np.pi:.4f})")
    print(f"  Phase_full/π  = {phase_full/np.pi:.4f},  cos = {np.cos(phase_full):.4f}")
    print(f"  Phase_born/π  = {phase_born/np.pi:.4f},  cos = {np.cos(phase_born):.4f}")
    print(f"  |S_off_born/S_off_full| = {np.abs(S_offdiag_born/S_offdiag):.4f}")


# ── Part 4: Resolvent amplification mechanism ───────────────────
print("\n" + "=" * 70)
print("PART 4: RESOLVENT (VG)^n COHERENCE")
print("=" * 70)

# Hypothesis: at low k, (VG)^n off-diagonal elements are more COHERENT
# (aligned phases) → they CANCEL in alternating Born series (since V<0, signs
# alternate). At high k, phases random → less cancellation → larger |Σ (VG)^n|.

for k in [0.3, 0.9, 1.5]:
    G = build_G_matrix(dx, dy, k)
    VG = V * G
    mask = ~np.eye(N, dtype=bool)

    print(f"\nk={k}:")
    VG_power = np.eye(N, dtype=complex)
    for n in range(1, 6):
        VG_power = VG_power @ VG
        offdiag = VG_power[mask]

        # Phase coherence: |mean(z)|/mean(|z|)
        mean_complex = np.abs(np.mean(offdiag))
        mean_abs = np.mean(np.abs(offdiag))
        coherence = mean_complex / mean_abs if mean_abs > 0 else 0

        # Mean value and sign
        mean_real = np.mean(np.real(offdiag))
        mean_imag = np.mean(np.imag(offdiag))

        print(f"  (VG)^{n}: mean={mean_real:+.6f}{mean_imag:+.6f}j, "
              f"|mean|={mean_complex:.6f}, <|z|>={mean_abs:.6f}, coh={coherence:.4f}")

    # Cumulative sum: R_ij = Σ_{n=1}^∞ [(VG)^n]_ij
    # At low k, high coherence → terms add then cancel
    # At high k, low coherence → terms are random → less cancellation

    # Compare resolvent off-diag to individual terms
    resolvent = np.linalg.inv(np.eye(N) - VG)
    R_offdiag = resolvent[mask]
    R_rms = np.sqrt(np.mean(np.abs(R_offdiag)**2))

    # First order only
    VG_offdiag = VG[mask]
    VG_rms = np.sqrt(np.mean(np.abs(VG_offdiag)**2))

    print(f"  Resolvent off-diag rms: {R_rms:.6f}")
    print(f"  VG off-diag rms:        {VG_rms:.6f}")
    print(f"  Ratio (resolvent/VG):   {R_rms/VG_rms:.4f}")


# ── Part 5: Direct test of cancellation hypothesis ──────────────
print("\n" + "=" * 70)
print("PART 5: CANCELLATION IN BORN SERIES")
print("=" * 70)

# If cancellation at low k is stronger, then:
# |R_ij| < Σ_n |(VG)^n_ij|  (triangle inequality, equality = all aligned)
# At low k: more alignment → more cancellation between alternating terms
# At high k: less alignment → less cancellation

for k in [0.3, 0.9, 1.5]:
    G = build_G_matrix(dx, dy, k)
    VG = V * G
    mask = ~np.eye(N, dtype=bool)

    # Compute |resolvent_ij| and Σ |(VG)^n_ij|
    resolvent = np.linalg.inv(np.eye(N) - VG)
    R_offdiag = np.abs(resolvent[mask].reshape(N, N-1))
    R_mean = np.mean(R_offdiag)

    # Sum of absolute values of each Born order
    VG_power = np.eye(N, dtype=complex)
    sum_abs = np.zeros((N, N))
    for n in range(1, 15):
        VG_power = VG_power @ VG
        sum_abs += np.abs(VG_power)

    sum_abs_offdiag = sum_abs[~np.eye(N, dtype=bool)].reshape(N, N-1)
    sum_abs_mean = np.mean(sum_abs_offdiag)

    cancel_ratio = R_mean / sum_abs_mean

    print(f"k={k}: <|R_ij|>={R_mean:.6f}, <Σ|(VG)^n|>={sum_abs_mean:.6f}, "
          f"cancel_ratio={cancel_ratio:.4f}")

print("\nCancel ratio < 1 means destructive interference between Born orders.")
print("Lower cancel_ratio = more cancellation.")
