"""Why incoh ~ k^{+1/2}?

incoh = |V|² <|R_ij|²>  where R = (I-VG)^{-1}, off-diagonal.

Since G is symmetric (complex), VG is symmetric.
Eigendecomposition: VG = Σ λ_n |v_n><v_n| (complex symmetric, Takagi).
But for practical purposes, numpy.eigh works on the Hermitian part.

Actually, VG is complex symmetric but NOT Hermitian.
Let's work with the SVD or direct eigenvalue approach.

For symmetric VG: R = (I-VG)^{-1} = Σ_n 1/(1-λ_n) |v_n><v_n|
where λ_n are eigenvalues and v_n are (right) eigenvectors.

<|R_ij|²>_offdiag = [Tr(R†R) - Σ_i |R_ii|²] / [N(N-1)]

Tr(R†R) = Tr[(I-VG†)^{-1}(I-VG)^{-1}]

For normal VG: Tr(R†R) = Σ |1/(1-λ_n)|²
For non-normal VG: more complex.

VG here IS symmetric (not Hermitian), so it's diagonalizable but
eigenvectors are NOT orthonormal under standard inner product.

Let's take an empirical approach: what FUNCTION of VG eigenvalues
predicts incoh?
"""
import sys
sys.path.insert(0, '/Users/alextoader/Sites/st_vortex_transport/tests')

import numpy as np
from helpers.config import K1, K2, c_lat, ALPHA_REF, V_ref
from helpers.geometry import disk_bonds
from helpers.lattice import k_eff
from helpers.ms import build_G_matrix
from helpers.born import V_eff
from helpers.stats import log_log_slope

alpha = ALPHA_REF
V = V_ref
k_vals = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]
k_arr = np.array(k_vals)

R = 5
dx, dy = disk_bonds(R)
N = len(dx)

print(f"R={R}, N={N}, V={V:.4f}")
print()

# ── Eigenvalue statistics vs k ──────────────────────────────────
print("=" * 70)
print("VG EIGENVALUE STATISTICS")
print("=" * 70)

incoh_vals = []
trRR_vals = []       # Tr(R†R)/N
trRR_diag_vals = []  # Σ|R_ii|²/N
mean_res2_vals = []  # <|1/(1-λ)|²>
mean_res4_vals = []  # <|1/(1-λ)|⁴>
sum_lam2_vals = []   # <|λ|²>
sum_lam4_vals = []   # <|λ|⁴>
spec_rad_vals = []

for k in k_vals:
    G = build_G_matrix(dx, dy, k)
    VG = V * G

    eigs = np.linalg.eigvals(VG)

    resolvent = np.linalg.inv(np.eye(N) - VG)

    # incoh
    mask = ~np.eye(N, dtype=bool)
    T_mat = V * resolvent
    incoh = np.mean(np.abs(T_mat[mask])**2)
    incoh_vals.append(incoh)

    # Tr(R†R)
    RhR = resolvent.conj().T @ resolvent
    trRR = np.real(np.trace(RhR)) / N
    trRR_vals.append(trRR)

    # Diagonal part
    diag_sq = np.mean(np.abs(np.diag(resolvent))**2)
    trRR_diag_vals.append(diag_sq)

    # From eigenvalues (if VG were normal)
    res_eigs = 1 / (1 - eigs)
    mean_res2 = np.mean(np.abs(res_eigs)**2)
    mean_res2_vals.append(mean_res2)
    mean_res4 = np.mean(np.abs(res_eigs)**4)
    mean_res4_vals.append(mean_res4)

    sum_lam2 = np.mean(np.abs(eigs)**2)
    sum_lam4 = np.mean(np.abs(eigs)**4)
    sum_lam2_vals.append(sum_lam2)
    sum_lam4_vals.append(sum_lam4)

    spec_rad_vals.append(np.max(np.abs(eigs)))

# incoh = V² × offdiag_R²
# offdiag_R² = [Tr(R†R)/N - diag_R²] × N/(N-1)
offdiag_R2 = [(trRR_vals[i] - trRR_diag_vals[i]) * N / (N-1) for i in range(len(k_vals))]

print(f"\n{'k':>5} {'incoh':>10} {'Tr(R†R)/N':>10} {'<|R_ii|²>':>10} {'offdiag_R²':>11} {'<|1/(1-λ)|²>':>13}")
for i, k in enumerate(k_vals):
    print(f"{k:5.1f} {incoh_vals[i]:10.6f} {trRR_vals[i]:10.4f} {trRR_diag_vals[i]:10.4f} "
          f"{offdiag_R2[i]:11.6f} {mean_res2_vals[i]:13.4f}")

print(f"\n  Is Tr(R†R)/N = <|1/(1-λ)|²>?")
for i, k in enumerate(k_vals):
    ratio = trRR_vals[i] / mean_res2_vals[i]
    print(f"    k={k}: ratio = {ratio:.6f}")

print(f"\nExponents:")
p_incoh = log_log_slope(k_arr, np.array(incoh_vals))[0]
p_trRR = log_log_slope(k_arr, np.array(trRR_vals))[0]
p_diag = log_log_slope(k_arr, np.array(trRR_diag_vals))[0]
p_offR2 = log_log_slope(k_arr, np.array(offdiag_R2))[0]
p_res2 = log_log_slope(k_arr, np.array(mean_res2_vals))[0]
p_lam2 = log_log_slope(k_arr, np.array(sum_lam2_vals))[0]
print(f"  incoh:          {p_incoh:+.3f}")
print(f"  Tr(R†R)/N:      {p_trRR:+.3f}")
print(f"  <|R_ii|²>:      {p_diag:+.3f}")
print(f"  offdiag_R²:     {p_offR2:+.3f}")
print(f"  <|1/(1-λ)|²>:   {p_res2:+.3f}")
print(f"  <|λ|²>:         {p_lam2:+.3f}")


# ── Key insight: is VG nearly normal? ───────────────────────────
print("\n" + "=" * 70)
print("IS VG NEARLY NORMAL? (VG·VG† ≈ VG†·VG?)")
print("=" * 70)

for k in [0.3, 0.9, 1.5]:
    G = build_G_matrix(dx, dy, k)
    VG = V * G

    # Normality check: ||[VG, VG†]|| / ||VG||²
    comm = VG @ VG.conj().T - VG.conj().T @ VG
    norm_comm = np.linalg.norm(comm, 'fro')
    norm_VG = np.linalg.norm(VG, 'fro')

    print(f"k={k}: ||[VG,VG†]||/||VG||² = {norm_comm/norm_VG**2:.6f}")

    # Also: is G Hermitian? G_ij = exp(ik r)/r → complex
    # G is SYMMETRIC (G_ij = G_ji) but NOT Hermitian (G_ij ≠ G_ji*)
    # The anti-Hermitian part: A = (G - G†)/2
    A = (G - G.conj().T) / 2
    H = (G + G.conj().T) / 2
    print(f"        ||Im(G)||/||Re(G)|| = {np.linalg.norm(A,'fro')/np.linalg.norm(H,'fro'):.4f}")


# ── Perturbative formula for offdiag R² ────────────────────────
print("\n" + "=" * 70)
print("PERTURBATIVE EXPANSION: <|R_ij|²> to order (VG)²")
print("=" * 70)

# R = I + VG + (VG)² + ...
# R†R = I + VG† + VG + VG†VG + VG² + (VG†)² + ...
# Off-diagonal of R†R:
# [R†R]_ij = [VG]_ij + [VG†]_ij + [VG†VG]_ij + [VG²]_ij + [(VG†)²]_ij + ...
#          = 2Re(VG_ij) + [VG†VG + VG²]_ij + ...
#
# <|R_ij|²> = <|VG_ij|²> + 2Re<VG_ij* × [(VG²)_ij]> + <|(VG²)_ij|²> + ...
#            ≈ <|VG_ij|²> × (1 + correction)
#
# But wait: R_ij for i≠j starts at order 1 (VG)_ij.
# R_ij = (VG)_ij + (VG²)_ij + (VG³)_ij + ...
#
# |R_ij|² = |VG_ij|² + 2Re[(VG)_ij* (VG²)_ij] + |VG²_ij|² + 2Re[(VG)_ij* (VG³)_ij] + ...
#
# The first term |VG_ij|² = V² |G_ij|² is k-independent.
# The CORRECTION comes from cross terms between orders.

for k in [0.3, 0.9, 1.5]:
    G = build_G_matrix(dx, dy, k)
    VG = V * G
    VG2 = VG @ VG
    VG3 = VG2 @ VG
    mask = ~np.eye(N, dtype=bool)

    # Order 1
    term1 = np.mean(np.abs(VG[mask])**2)

    # Cross 1-2: 2 Re[VG* × VG²]
    cross12 = 2 * np.mean(np.real(VG[mask].conj() * VG2[mask]))

    # Order 2: |VG²|²
    term2 = np.mean(np.abs(VG2[mask])**2)

    # Cross 1-3
    cross13 = 2 * np.mean(np.real(VG[mask].conj() * VG3[mask]))

    # Full resolvent
    resolvent = np.linalg.inv(np.eye(N) - VG)
    R_full = np.mean(np.abs(resolvent[mask])**2)

    print(f"\nk={k}:")
    print(f"  |VG|²:              {term1:.8f}")
    print(f"  2Re(VG*·VG²):       {cross12:+.8f}  ({100*cross12/term1:+.1f}% of order 1)")
    print(f"  |VG²|²:             {term2:.8f}  ({100*term2/term1:.1f}% of order 1)")
    print(f"  2Re(VG*·VG³):       {cross13:+.8f}  ({100*cross13/term1:+.1f}% of order 1)")
    print(f"  Sum (1+cross+2):    {term1+cross12+term2:.8f}")
    print(f"  Full <|R_ij|²>:     {R_full:.8f}")
    print(f"  Perturbative ratio: {(term1+cross12+term2)/R_full:.4f}")


# ── The key cross term: 2Re<VG* × VG²> ─────────────────────────
print("\n" + "=" * 70)
print("CROSS TERM 2Re<VG*·VG²> vs k")
print("=" * 70)

# VG²_ij = Σ_l (VG)_il (VG)_lj = V² Σ_l G_il G_lj
# Cross = 2Re<VG_ij* VG²_ij> = 2V³ Re<G_ij* Σ_l G_il G_lj>
# = 2V³ Re Σ_l <G_ij* G_il G_lj>
#
# This is a 3-point correlator of the Green function.
# G_ij = exp(ik_eff r_ij)/(4πc²r_ij)
#
# The PHASE of G_ij* G_il G_lj = exp(ik_eff(-r_ij + r_il + r_lj))/(...)
# The phase argument is k_eff × (r_il + r_lj - r_ij)
# By triangle inequality: r_il + r_lj ≥ r_ij
# So the phase is always k_eff × (positive quantity)
# At low k: phases ≈ 1 → cross term ~ Σ 1/(r_ij r_il r_lj) × cos(0) > 0
# Since V < 0: V³ < 0, so 2V³ × (positive) < 0 → NEGATIVE cross
# This means |R_ij|² < |VG_ij|² → SUPPRESSION at low k. ✓
#
# At high k: phases oscillate → cross averages to smaller value → less suppression

cross12_vals = []
term1_vals = []

for k in k_vals:
    G = build_G_matrix(dx, dy, k)
    VG = V * G
    VG2 = VG @ VG
    mask = ~np.eye(N, dtype=bool)

    term1 = np.mean(np.abs(VG[mask])**2)
    cross12 = 2 * np.mean(np.real(VG[mask].conj() * VG2[mask]))

    term1_vals.append(term1)
    cross12_vals.append(cross12)

print(f"\n{'k':>5} {'|VG|²':>12} {'cross_12':>12} {'cross/|VG|²':>12}")
for i, k in enumerate(k_vals):
    print(f"{k:5.1f} {term1_vals[i]:12.8f} {cross12_vals[i]:+12.8f} {cross12_vals[i]/term1_vals[i]:+12.4f}")

p_cross12 = log_log_slope(k_arr, np.abs(np.array(cross12_vals)))[0]
print(f"\n|cross_12| exponent: {p_cross12:.3f}")
print(f"|VG|² exponent: {log_log_slope(k_arr, np.array(term1_vals))[0]:.3f} (≈ 0, k-independent)")

# ── Phase of 3-point correlator ─────────────────────────────────
print("\n" + "=" * 70)
print("3-POINT CORRELATOR PHASE")
print("=" * 70)

# G_ij* G_il G_lj = exp(ik_eff × Δ_path) / (...)
# where Δ_path = r_il + r_lj - r_ij ≥ 0 (triangle inequality)
# At low k: all phases ≈ 1, correlator is REAL POSITIVE
# At high k: phases oscillate, correlator averages toward 0

for k in [0.3, 0.9, 1.5]:
    ke = k_eff(k)
    G = build_G_matrix(dx, dy, k)
    mask = ~np.eye(N, dtype=bool)

    # Compute path length excess for all (i,j,l) triplets
    dist = np.sqrt((dx[:, None] - dx[None, :])**2 + (dy[:, None] - dy[None, :])**2)

    # For a sample of pairs (i,j), compute Σ_l Re[G_ij* G_il G_lj]
    # = Σ_l cos(ke × (r_il + r_lj - r_ij)) / (r_ij r_il r_lj) / (4πc²)³

    # Average path excess
    path_excess_list = []
    for i in range(min(N, 20)):
        for j in range(min(N, 20)):
            if i == j:
                continue
            r_ij = dist[i, j]
            if r_ij < 1e-10:
                continue
            for l in range(N):
                if l == i or l == j:
                    continue
                r_il = dist[i, l]
                r_lj = dist[l, j]
                if r_il < 1e-10 or r_lj < 1e-10:
                    continue
                path_excess_list.append(r_il + r_lj - r_ij)

    pe = np.array(path_excess_list)
    # Average cos(ke × path_excess)
    avg_cos = np.mean(np.cos(ke * pe))
    avg_pe = np.mean(pe)

    print(f"k={k}, k_eff={ke:.3f}:")
    print(f"  <path_excess> = {avg_pe:.3f}")
    print(f"  <cos(k_eff × path_excess)> = {avg_cos:.4f}")
    print(f"  k_eff × <path_excess> = {ke * avg_pe:.3f} rad")
