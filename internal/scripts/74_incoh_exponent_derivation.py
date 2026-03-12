"""Can we derive the incoh exponent +1/2?

incoh = <|T_ij|²> = |V|² <|R_ij|²> where R = (I-VG)^{-1}.
G_ij is k-independent in magnitude (|G_ij|² ≈ const).
The k-dependence comes from the PHASE structure of G.

Key question: what controls the cancel_ratio = <|R_ij|²> / <Σ|(VG)^n_ij|²>?

Hypothesis: cancel_ratio ∝ 1/N_coop_G, where N_coop_G is the cooperation
number of the BARE Green function phases. At low k, N_coop_G is large
(phases aligned) → more cancellation → lower cancel_ratio.

If cancel_ratio ~ 1/N_coop_G and N_coop_G ~ k^{-3/2} (Fresnel zone),
then cancel_ratio ~ k^{+3/2}.
incoh = incoh_incoherent × cancel_ratio ~ k^0 × k^{+3/2} = k^{+3/2}.
But measured incoh ~ k^{+1/2}. So this simple argument overshoots.

Let's test the N_coop_G hypothesis directly.
"""
import sys
sys.path.insert(0, '/Users/alextoader/Sites/st_vortex_transport/tests')

import numpy as np
from helpers.config import K1, K2, c_lat, ALPHA_REF, V_ref
from helpers.geometry import disk_bonds
from helpers.lattice import k_eff
from helpers.ms import build_G_matrix
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

# ── N_coop of bare G vs N_coop of T ────────────────────────────
print("=" * 70)
print("N_coop of BARE G vs T (off-diagonal)")
print("=" * 70)

# N_coop_G: how coherent are G_ij elements when summed with phase b_j?
# Σ_j G_ij b_j (coherent) vs Σ_j |G_ij|² (incoherent)

ncoop_G_vals = []
ncoop_T_vals = []
incoh_G_vals = []
incoh_T_vals = []

for k in k_vals:
    b = np.exp(1j * k * dx)
    G = build_G_matrix(dx, dy, k)
    VG = V * G
    resolvent = np.linalg.inv(np.eye(N) - VG)
    T_mat = V * resolvent

    mask = ~np.eye(N, dtype=bool)

    # For G: off-diagonal
    # Per-site coherent sum: sum_j≠i G_ij b_j
    Gb_offdiag = G @ b - np.diag(G) * b
    coh_G = np.sum(np.abs(Gb_offdiag)**2)
    incoh_G = np.sum(np.abs(G[mask])**2)  # Σ_i Σ_{j≠i} |G_ij|²
    ncoop_G = coh_G / incoh_G

    # For T: off-diagonal
    Tb_offdiag = T_mat @ b - np.diag(T_mat) * b
    coh_T = np.sum(np.abs(Tb_offdiag)**2)
    incoh_T = np.sum(np.abs(T_mat[mask])**2)
    ncoop_T = coh_T / incoh_T

    ncoop_G_vals.append(ncoop_G)
    ncoop_T_vals.append(ncoop_T)
    incoh_G_vals.append(incoh_G / (N * (N-1)))
    incoh_T_vals.append(incoh_T / (N * (N-1)))

p_ncG = log_log_slope(k_arr, np.array(ncoop_G_vals))[0]
p_ncT = log_log_slope(k_arr, np.array(ncoop_T_vals))[0]
p_iG = log_log_slope(k_arr, np.array(incoh_G_vals))[0]
p_iT = log_log_slope(k_arr, np.array(incoh_T_vals))[0]

print(f"\n{'k':>5} {'Ncoop_G':>10} {'Ncoop_T':>10} {'incoh_G':>12} {'incoh_T':>12}")
for i, k in enumerate(k_vals):
    print(f"{k:5.1f} {ncoop_G_vals[i]:10.3f} {ncoop_T_vals[i]:10.3f} "
          f"{incoh_G_vals[i]:12.8f} {incoh_T_vals[i]:12.8f}")

print(f"\nExponents:")
print(f"  N_coop_G: {p_ncG:+.3f}")
print(f"  N_coop_T: {p_ncT:+.3f}")
print(f"  incoh_G:  {p_iG:+.3f} (should be ≈ 0)")
print(f"  incoh_T:  {p_iT:+.3f} (should be ≈ +0.48)")

print(f"\nRatio N_coop_T/N_coop_G:")
for i, k in enumerate(k_vals):
    ratio = ncoop_T_vals[i] / ncoop_G_vals[i]
    print(f"  k={k}: {ratio:.4f}")
p_ratio = log_log_slope(k_arr, np.array(ncoop_T_vals) / np.array(ncoop_G_vals))[0]
print(f"  Ratio exponent: {p_ratio:+.3f}")

# ── The relationship: incoh_T = incoh_G × amplification ────────
print("\n" + "=" * 70)
print("RESOLVENT AMPLIFICATION: incoh_T / incoh_G")
print("=" * 70)

amp_ratio = np.array(incoh_T_vals) / np.array(incoh_G_vals)
p_amp = log_log_slope(k_arr, amp_ratio)[0]

print(f"\n{'k':>5} {'incoh_T/incoh_G':>15}")
for i, k in enumerate(k_vals):
    print(f"{k:5.1f} {amp_ratio[i]:15.4f}")
print(f"\nAmplification exponent: {p_amp:+.3f}")
print(f"This is the RESOLVENT amplification. incoh exponent = 0 + amp exponent = {p_amp:+.3f}")

# ── Where does amplification come from? ────────────────────────
print("\n" + "=" * 70)
print("AMPLIFICATION DECOMPOSITION")
print("=" * 70)

# T_ij = V × [(I-VG)^{-1}]_ij = V × R_ij
# R_ij = VG_ij + (VG²)_ij + (VG³)_ij + ...
# |R_ij|² = |VG_ij|² × (series correction factor)
#
# The correction factor depends on how higher Born orders contribute
# relative to the first order.
#
# At first order: T_ij = V²G_ij, so incoh₁ = V⁴ × incoh_G
# Actual: incoh_T = V² × <|R_ij|²>
# Amplification = incoh_T / (V² × incoh_G) = <|R_ij|²> / <|G_ij|²>
# But also incoh_T / incoh_G = V² × <|R_ij|²> / <|G_ij|²>

# Wait, let me be more careful:
# incoh_G = <|G_ij|²> (per element)
# incoh_T = <|T_ij|²> = |V|² × <|R_ij|²>
# So incoh_T / incoh_G = |V|² × <|R_ij|²> / <|G_ij|²>

# But since incoh_G is k-independent, the k-dep of the ratio is just
# the k-dep of <|R_ij|²>.

# <|R_ij|²> = <|Σ_n (VG)^n_ij|²>
# = <Σ_n |VG^n_ij|² + Σ_{m≠n} VG^m_ij × (VG^n_ij)*>
# = Σ_n <|VG^n|²> + cross terms

# The cross terms have interference pattern controlled by phase coherence of VG^n.
# At low k: coherent → cross terms large and alternating → cancellation
# At high k: incoherent → cross terms small → just sum of |VG^n|²

# Test: compute <|R_ij|²> with and without cross terms
print(f"\n{'k':>5} {'<|R|²>':>12} {'Σ<|VG^n|²>':>12} {'ratio':>8} {'cross_frac':>12}")

for k in k_vals:
    G = build_G_matrix(dx, dy, k)
    VG = V * G
    mask = ~np.eye(N, dtype=bool)

    resolvent = np.linalg.inv(np.eye(N) - VG)
    # Remove identity (we want off-diagonal only, but resolvent has I on diagonal)
    # Actually R = I + VG + (VG)² + ..., so R_ij for i≠j has no I contribution
    R_offdiag = resolvent[mask]
    R_sq = np.mean(np.abs(R_offdiag)**2)

    # Sum of |VG^n_ij|² for each order
    VG_power = np.eye(N, dtype=complex)
    sum_sq = np.zeros(N * (N-1))
    for n in range(1, 12):
        VG_power = VG_power @ VG
        term_offdiag = VG_power[mask]
        sum_sq += np.abs(term_offdiag)**2

    incoherent_sum = np.mean(sum_sq)

    ratio = R_sq / incoherent_sum
    cross_frac = (R_sq - incoherent_sum) / R_sq

    print(f"{k:5.1f} {R_sq:12.8f} {incoherent_sum:12.8f} {ratio:8.4f} {cross_frac:+12.4f}")

# So the cross_frac tells us: positive = constructive interference between orders,
# negative = destructive interference.


# ── VG eigenvalue structure: analytic prediction ────────────────
print("\n" + "=" * 70)
print("VG EIGENVALUE SPECTRUM")
print("=" * 70)

# If VG has eigenvalues λ_i, then resolvent eigenvalues are 1/(1-λ_i).
# <|R_ij|²> relates to Σ |1/(1-λ)|⁴ (fourth moment) for off-diagonal.
# Actually, for a random matrix with known eigenvalue distribution,
# <|R_ij|²> can be computed from the resolvent trace moments.
#
# But VG is NOT random — it has geometric structure.
# Let's just look at the eigenvalue spectrum.

for k in [0.3, 0.9, 1.5]:
    G = build_G_matrix(dx, dy, k)
    VG = V * G
    eigs = np.linalg.eigvals(VG)

    print(f"\nk={k}:")
    print(f"  |λ| histogram: ", end="")
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    counts, _ = np.histogram(np.abs(eigs), bins=bins)
    for b, c in zip(bins, counts):
        if c > 0:
            print(f"[{b:.1f},{bins[bins.index(b)+1]:.1f}):{c} ", end="")
    print()

    # Resolvent eigenvalues
    res_eigs = 1 / (1 - eigs)
    print(f"  |1/(1-λ)| range: [{np.min(np.abs(res_eigs)):.3f}, {np.max(np.abs(res_eigs)):.3f}]")
    print(f"  <|1/(1-λ)|²>: {np.mean(np.abs(res_eigs)**2):.4f}")
    print(f"  <|1/(1-λ)|⁴>: {np.mean(np.abs(res_eigs)**4):.4f}")

    # The trace: Tr(R†R) / N = <|R_ii|²> + (N-1)<|R_ij|²>
    # Actually Tr(R†R) = Σ |1/(1-λ_i)|² for eigenvalues
    trRR = np.sum(np.abs(res_eigs)**2)
    print(f"  Tr(R†R) = {trRR:.4f}")
    print(f"  Tr(R†R)/N = {trRR/N:.4f}")


# ── Final: does incoh exponent = spectral radius exponent? ──────
print("\n" + "=" * 70)
print("SPECTRAL RADIUS vs INCOH")
print("=" * 70)

spec_max = []
for k in k_vals:
    G = build_G_matrix(dx, dy, k)
    VG = V * G
    eigs = np.linalg.eigvals(VG)
    spec_max.append(np.max(np.abs(eigs)))

p_spec = log_log_slope(k_arr, np.array(spec_max))[0]
print(f"Spectral radius exponent: {p_spec:+.3f}")
print(f"Incoh exponent:           {p_iT:+.3f}")
print(f"Ratio:                    {p_iT / p_spec:.2f}")

# If incoh ~ ρ(VG)^2 (from dominant eigenvalue), then incoh exp ≈ 2 × spec exp
print(f"2 × spec_exp:             {2*p_spec:+.3f}")
print(f"Expected from ρ²:         incoh ~ ρ^2 if dominant eigenvalue controls")
