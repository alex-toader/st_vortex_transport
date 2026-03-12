"""Total power decomposition: diag_pow, offdiag_pow, cross, and the effective phase.

The cross exponent -0.63 came from total power decomposition, not forward sums.
Let's verify and understand the effective phase between diag and offdiag N-vectors.

Also: understand incoh in terms of Born series cancellation quantitatively.
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

R = 5
alpha = ALPHA_REF
V = V_ref
dx, dy = disk_bonds(R)
N = len(dx)
k_vals = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
k_arr = np.array(k_vals)

print(f"R={R}, α={alpha}, V={V:.4f}, N={N}")
print()

# ── Total power decomposition ───────────────────────────────────
print("=" * 70)
print("TOTAL POWER DECOMPOSITION: born_pow, diag_pow, offdiag_pow, cross")
print("=" * 70)

born_pow_arr = []
diag_pow_arr = []
offdiag_pow_arr = []
cross_arr = []
diag_amp_arr = []
offdiag_amp_arr = []
eff_phase_arr = []

for k in k_vals:
    b = np.exp(1j * k * dx)
    T_mat = T_matrix(dx, dy, k, alpha)
    Tb = T_mat @ b
    Vb = V * b

    # Per-site decomposition
    diag_part = np.diag(T_mat) * b       # T_ii b_i (N-vector)
    offdiag_part = Tb - diag_part         # Σ_{j≠i} T_ij b_j (N-vector)

    # Powers
    born_pow = np.sum(np.abs(Vb)**2)       # = N|V|²
    diag_pow = np.sum(np.abs(diag_part)**2)
    offdiag_pow = np.sum(np.abs(offdiag_part)**2)
    cross = 2 * np.real(np.sum(np.conj(diag_part) * offdiag_part))

    # Verify: total = diag + offdiag + cross
    total_pow = np.sum(np.abs(Tb)**2)
    assert abs(total_pow - (diag_pow + offdiag_pow + cross)) < 1e-8 * total_pow

    # Effective phase: cos(Φ) = cross / (2 |d| |o|)
    diag_amp = np.sqrt(diag_pow)
    offdiag_amp = np.sqrt(offdiag_pow)
    cos_eff = cross / (2 * diag_amp * offdiag_amp)

    born_pow_arr.append(born_pow)
    diag_pow_arr.append(diag_pow)
    offdiag_pow_arr.append(offdiag_pow)
    cross_arr.append(cross)
    diag_amp_arr.append(diag_amp)
    offdiag_amp_arr.append(offdiag_amp)
    eff_phase_arr.append(cos_eff)

# Normalize by born_pow
born_arr = np.array(born_pow_arr)
diag_arr = np.array(diag_pow_arr) / born_arr
offdiag_arr = np.array(offdiag_pow_arr) / born_arr
cross_norm = np.array(cross_arr) / born_arr
enh_arr = (np.array(diag_pow_arr) + np.array(offdiag_pow_arr) + np.array(cross_arr)) / born_arr
cos_eff_arr = np.array(eff_phase_arr)

print(f"\n{'k':>5} {'diag/born':>10} {'off/born':>10} {'cross/born':>11} {'enh':>8} {'cos(Φ)':>10}")
for i, k in enumerate(k_vals):
    print(f"{k:5.1f} {diag_arr[i]:10.4f} {offdiag_arr[i]:10.4f} {cross_norm[i]:11.4f} "
          f"{enh_arr[i]:8.4f} {cos_eff_arr[i]:10.4f}")

# Exponents
p_diag_amp = log_log_slope(k_arr, np.array(diag_amp_arr))[0]
p_offdiag_amp = log_log_slope(k_arr, np.array(offdiag_amp_arr))[0]
p_cross = log_log_slope(k_arr, np.abs(np.array(cross_arr)))[0]
p_cos = log_log_slope(k_arr, np.abs(cos_eff_arr))[0]
p_enh = log_log_slope(k_arr, enh_arr)[0]

print(f"\nExponents (normalized by √born for amps, by born for powers):")
print(f"  diag_amp/√born:      {p_diag_amp:.3f} (≈ 0 expected)")
print(f"  offdiag_amp/√born:   {p_offdiag_amp:.3f} (≈ -0.48 expected)")
print(f"  |cross|/born:        {p_cross:.3f} (≈ -0.63 expected)")
print(f"  |cos(Φ)|:            {p_cos:.3f} (≈ -0.15 expected)")
print(f"  enh:                 {p_enh:.3f} (≈ +0.35 expected)")
print(f"\n  Check: diag_amp + offdiag_amp + cos = {p_diag_amp:.3f} + {p_offdiag_amp:.3f} + {p_cos:.3f} = {p_diag_amp + p_offdiag_amp + p_cos:.3f}")
print(f"  vs cross exponent:   {p_cross:.3f}")


# ── Effective phase: what determines it? ────────────────────────
print("\n" + "=" * 70)
print("EFFECTIVE PHASE ANALYSIS")
print("=" * 70)

# cos(Φ) = Σ_i Re(d_i* o_i) / (|d||o|)
# where d_i = T_ii b_i, o_i = Σ_{j≠i} T_ij b_j
#
# All T_ii are equal (same G_00), so d_i = T_11 b_i
# o_i = Σ_{j≠i} T_ij b_j has both amplitude and phase varying with i
#
# The dot product d·o = T_11* Σ_i b_i* o_i = T_11* Σ_i b_i* Σ_{j≠i} T_ij b_j
# = T_11* Σ_i Σ_{j≠i} T_ij b_j b_i*
# = T_11* Σ_i Σ_{j≠i} T_ij exp(ik(x_j - x_i))

for k in [0.3, 0.9, 1.5]:
    b = np.exp(1j * k * dx)
    T_mat = T_matrix(dx, dy, k, alpha)
    T_diag_val = T_mat[0, 0]

    # Per-site: o_i = Σ_{j≠i} T_ij b_j
    Tb = T_mat @ b
    diag_part = T_diag_val * b
    offdiag_part = Tb - diag_part

    # Per-site contribution to dot product
    per_site = np.conj(diag_part) * offdiag_part  # complex contributions

    # Phase distribution of per-site contributions
    phases = np.angle(per_site)
    amplitudes = np.abs(per_site)

    # How coherent are per-site contributions?
    sum_complex = np.sum(per_site)
    sum_abs = np.sum(amplitudes)
    coherence = np.abs(sum_complex) / sum_abs

    print(f"\nk={k}:")
    print(f"  T_11 = {T_diag_val:.4f}")
    print(f"  Per-site phase: mean={np.mean(phases):.3f}, std={np.std(phases):.3f}")
    print(f"  Per-site |contrib|: mean={np.mean(amplitudes):.4f}, CV={100*np.std(amplitudes)/np.mean(amplitudes):.1f}%")
    print(f"  Dot product: {sum_complex:.4f}")
    print(f"  Sum |contrib|: {sum_abs:.4f}")
    print(f"  Coherence (|sum|/Σ|z|): {coherence:.4f}")
    print(f"  cos(Φ) = Re(dot)/(|d||o|) = {np.real(sum_complex)/(np.sqrt(np.sum(np.abs(diag_part)**2))*np.sqrt(np.sum(np.abs(offdiag_part)**2))):.4f}")


# ── Born series: does incoh exponent come from alternation? ─────
print("\n" + "=" * 70)
print("INCOH FROM BORN SERIES STRUCTURE")
print("=" * 70)

# R_ij = (VG)_ij + (VG)²_ij + ...
# At low k, (VG)^n_ij are coherent (high coherence in Part 4 of script 71)
# and ALTERNATE in sign (since V < 0, odd powers have one sign, even powers another)
# → cancellation → smaller |R_ij| → smaller incoh
#
# At high k, (VG)^n_ij are incoherent → no systematic cancellation
# → larger |R_ij| → larger incoh
#
# Let's verify: compare |R_ij| to |R_ij| with ABSOLUTE values of Born terms

print("\nTest: resolve by Born order contribution to incoh")

for k in [0.3, 0.9, 1.5]:
    G = build_G_matrix(dx, dy, k)
    VG = V * G
    mask = ~np.eye(N, dtype=bool)

    # True resolvent
    resolvent = np.linalg.inv(np.eye(N) - VG)
    R_offdiag = resolvent[mask].reshape(N, N-1)
    incoh_true = np.mean(np.abs(R_offdiag)**2)

    # Born series terms individually
    VG_power = np.eye(N, dtype=complex)
    terms = []
    for n in range(1, 10):
        VG_power = VG_power @ VG
        terms.append(VG_power.copy())

    # "Static" incoh: |Σ (VG)^n|² but taking |.| of each term before summing
    # This gives the incoh WITHOUT alternation cancellation
    sum_abs_sq = np.zeros((N, N))
    for term in terms:
        sum_abs_sq += np.abs(term)**2
    incoh_no_cancel = np.mean(sum_abs_sq[mask].reshape(N, N-1))

    # Alternation pattern: sign of Re((VG)^n_ij) for a typical pair
    i0, j0 = 0, 1  # typical pair
    signs = []
    for n, term in enumerate(terms, 1):
        val = term[i0, j0]
        signs.append(f"  n={n}: {val:+.6f}")

    print(f"\nk={k}:")
    print(f"  incoh_true (from resolvent):    {incoh_true:.8f}")
    print(f"  incoh_no_cancel (Σ|VG^n|²):     {incoh_no_cancel:.8f}")
    print(f"  ratio (cancellation factor):    {incoh_true/incoh_no_cancel:.4f}")
    print(f"  Typical pair (0,1) Born terms:")
    for s in signs[:5]:
        print(f"    {s}")


# ── Incoh at different R: is the exponent R-independent? ────────
print("\n" + "=" * 70)
print("INCOH EXPONENT vs R")
print("=" * 70)

for R_test in [3, 5, 7, 9]:
    dx_t, dy_t = disk_bonds(R_test)
    N_t = len(dx_t)

    incoh_R = []
    for k in k_vals:
        G = build_G_matrix(dx_t, dy_t, k)
        VG = V * G
        resolvent = np.linalg.inv(np.eye(N_t) - VG)
        T_off = (V * resolvent)[~np.eye(N_t, dtype=bool)]
        incoh_R.append(np.mean(np.abs(T_off)**2))

    p_inc = log_log_slope(k_arr, np.array(incoh_R))[0]
    print(f"R={R_test:2d}, N={N_t:4d}: incoh exponent = {p_inc:+.3f}")


# ── cos(Φ) at different R ───────────────────────────────────────
print("\n" + "=" * 70)
print("cos(Φ) EXPONENT vs R")
print("=" * 70)

for R_test in [3, 5, 7, 9]:
    dx_t, dy_t = disk_bonds(R_test)
    N_t = len(dx_t)

    cos_R = []
    for k in k_vals:
        b = np.exp(1j * k * dx_t)
        T_mat = T_matrix(dx_t, dy_t, k, alpha)
        Tb = T_mat @ b
        diag_part = np.diag(T_mat) * b
        offdiag_part = Tb - diag_part

        cross = 2 * np.real(np.sum(np.conj(diag_part) * offdiag_part))
        diag_amp = np.sqrt(np.sum(np.abs(diag_part)**2))
        offdiag_amp = np.sqrt(np.sum(np.abs(offdiag_part)**2))
        cos_eff = cross / (2 * diag_amp * offdiag_amp)
        cos_R.append(np.abs(cos_eff))

    p_cos_R = log_log_slope(k_arr, np.array(cos_R))[0]
    print(f"R={R_test:2d}, N={N_t:4d}: |cos(Φ)| exponent = {p_cos_R:+.3f}")
    print(f"  values: {['%.4f' % c for c in cos_R]}")
