"""
Route 19: Fine k-grid scan — test for oscillations in σ_tr(k).

If ring acts as coherent interferometer, σ_tr(k) should show oscillations
with period Δk ≈ 1/R. Fine grid: Δk=0.05, 25 points in k=0.3–1.5.

Multi-R test: R=3,5,7 at α=0.3. If period scales as 1/R → confirmed.

Results (L=80, r_m=20, α=0.3, 25 k-pts, Δk=0.05):

  NO OSCILLATIONS. σ_tr(k) is smooth.

    R  RMS%  peak_T  verdict       κ_data
    3   3.3%   1.10  INCONCLUSIVE   0.252
    5   1.8%   1.10  SMOOTH         0.942
    7   1.2%   1.10  SMOOTH         2.151

  Dominant "period" = 1.10 at ALL R values — does NOT scale as 1/R.
  This is systematic drift (detrending edge), not physical oscillation.

  RMS residuals decrease with R: 3.3% → 1.8% → 1.2%.
  R=5,7: residuals < 2% → σ_tr is genuinely smooth.

  Conclusion: ring does NOT act as coherent interferometer.
  Scattering is incoherent (independent arc segments).

  κ_data(R=5) = 0.942 on 25-pt grid matches 0.944 on 7-pt grid (file 16).

Run: OPENBLAS_NUM_THREADS=1 /usr/bin/python3 wip/w_21_kappa/19_oscillation_scan.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from elastic_3d import scalar_laplacian_3d, make_damping_3d, run_fdtd_3d
from gauge_3d import make_vortex_force
from scattering_3d import (make_wave_packet_3d, make_sphere_points,
                           compute_sphere_f2, integrate_sigma_3d,
                           estimate_n_steps_3d)

K1, K2 = 1.0, 0.5
L = 80
DW = 15
DS = 1.5
DT = 0.25
sx = 8.0
r_m = 20
ALPHA = 0.3
N_POL = 2
R_vals = [3, 5, 7]

thetas = np.linspace(0, np.pi, 13)
phis = np.linspace(0, 2 * np.pi, 24, endpoint=False)
x_start = DW + 5
iz, iy, ix = make_sphere_points(L, r_m, thetas, phis)
gamma_pml = make_damping_3d(L, DW, DS)

# Fine k-grid: Δk = 0.05
k_vals = np.arange(0.30, 1.51, 0.05)
print(f"Fine k-grid: {len(k_vals)} points, k = {k_vals[0]:.2f} to {k_vals[-1]:.2f}, "
      f"Δk = 0.05")
for R in R_vals:
    print(f"  R={R}: expected period Δk = 1/R = {1/R:.3f}, "
          f"~{1.2/(1/R):.1f} cycles in [0.3,1.5]")
print()

from numpy.fft import rfft, rfftfreq


def force_plain(ux, uy, uz):
    return scalar_laplacian_3d(ux, uy, uz, K1, K2)


def analyze_oscillations(sigma_tr, k_vals, R):
    """Detrend σ_tr, compute residuals and FFT. Return dict of results."""
    # Detrend: quadratic in log-log, fit on k > 0.4 to avoid edge effects
    mask_fit = k_vals > 0.4
    log_k = np.log(k_vals[mask_fit])
    log_s = np.log(sigma_tr[mask_fit])
    poly = np.polyfit(log_k, log_s, 2)

    # Residuals only on fitted region (mask_fit)
    trend_fit = np.exp(np.polyval(poly, np.log(k_vals[mask_fit])))
    residual_fit = sigma_tr[mask_fit] / trend_fit - 1.0

    # Also compute on full range for display
    trend_all = np.exp(np.polyval(poly, np.log(k_vals)))
    residual_all = sigma_tr / trend_all - 1.0

    rms = np.sqrt(np.mean(residual_fit**2))

    # FFT on fitted-region residuals
    N = len(residual_fit)
    dk = k_vals[1] - k_vals[0]
    fft_amp = np.abs(rfft(residual_fit))
    fft_freq = rfftfreq(N, d=dk)

    # Peak detection
    peak_idx = np.argmax(fft_amp[1:]) + 1
    peak_freq = fft_freq[peak_idx]
    peak_period = 1.0 / peak_freq if peak_freq > 0 else float('inf')
    peak_amp = fft_amp[peak_idx]
    noise_level = np.median(fft_amp[1:])

    return {
        'residual_all': residual_all, 'residual_fit': residual_fit,
        'rms': rms, 'fft_amp': fft_amp, 'fft_freq': fft_freq,
        'peak_freq': peak_freq, 'peak_period': peak_period,
        'peak_amp': peak_amp, 'noise_level': noise_level,
        'mask_fit': mask_fit,
    }


# ── References (shared across R) ─────────────────────
t0 = time.time()
print(f"Computing {len(k_vals)} references (L={L})...")
refs = {}
for k0 in k_vals:
    ux0, vx0 = make_wave_packet_3d(L, k0, x_start, sx, K1, K2)
    ns = estimate_n_steps_3d(k0, L, x_start, sx, r_m, DT, 50, K1, K2)
    ref = run_fdtd_3d(force_plain, ux0.copy(), vx0.copy(), gamma_pml,
                      DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
    refs[k0] = (ref, ux0, vx0, ns)
t_ref = time.time() - t0
print(f"  Done ({t_ref:.0f}s)\n")


# ── Multi-R scattering + oscillation analysis ────────
all_results = []

for R in R_vals:
    f_def = make_vortex_force(ALPHA, R, L, K1, K2)
    sigma_tr = np.zeros(len(k_vals))

    print(f"Scattering: α={ALPHA}, R={R}")
    t1 = time.time()
    for i, k0 in enumerate(k_vals):
        ref, ux0, vx0, ns = refs[k0]
        d = run_fdtd_3d(f_def, ux0.copy(), vx0.copy(), gamma_pml,
                        DT, ns, rec_iz=iz, rec_iy=iy, rec_ix=ix, rec_n=ns)
        f2 = compute_sphere_f2(d['ux'], d['uy'], d['uz'],
                               ref['ux'], ref['uy'], ref['uz'], r_m)
        _, st = integrate_sigma_3d(f2, thetas, phis)
        sigma_tr[i] = st
    t_scat = time.time() - t1
    print(f"  Done ({t_scat:.0f}s)")

    # σ_tr table
    integrand = np.sin(k_vals)**2 * sigma_tr
    print(f"\n  {'k':>5s}  {'kR':>5s}  {'σ_tr':>8s}  {'sin²·σ':>7s}")
    print(f"  {'-'*30}")
    for i in range(len(k_vals)):
        print(f"  {k_vals[i]:5.2f}  {k_vals[i]*R:5.2f}  "
              f"{sigma_tr[i]:8.2f}  {integrand[i]:7.3f}")

    # Oscillation analysis
    res = analyze_oscillations(sigma_tr, k_vals, R)
    residual_all = res['residual_all']
    rms = res['rms']

    print(f"\n  Oscillation analysis (R={R}, 1/R={1/R:.3f}):")
    print(f"  Fractional residuals (fitted region k>0.4):")
    print(f"  {'k':>5s}  {'residual':>8s}")
    k_fit = k_vals[res['mask_fit']]
    for j in range(len(k_fit)):
        bar = "+" * int(abs(res['residual_fit'][j]) * 100) if res['residual_fit'][j] > 0 \
              else "-" * int(abs(res['residual_fit'][j]) * 100)
        print(f"  {k_fit[j]:5.2f}  {res['residual_fit'][j]:+8.4f}  {bar}")

    print(f"  RMS residual: {rms:.4f} ({rms*100:.1f}%)")

    # FFT (low-resolution sanity check: N~21 pts, Δf=1/(N·dk)≈0.95)
    print(f"\n  FFT of residuals (N={len(res['residual_fit'])}, "
          f"low resolution — sanity check only):")
    print(f"  {'f (1/k)':>8s}  {'period':>7s}  {'amplitude':>9s}")
    fft_amp = res['fft_amp']
    fft_freq = res['fft_freq']
    for j in range(1, len(fft_freq)):
        period = 1.0 / fft_freq[j] if fft_freq[j] > 0 else float('inf')
        marker = " <-- 1/R" if abs(period - 1.0/R) < 0.05 or \
                 abs(fft_freq[j] - R) < 1.0 else ""
        if fft_amp[j] > 0.3 * fft_amp[1:].max() or marker:
            print(f"  {fft_freq[j]:8.3f}  {period:7.2f}  "
                  f"{fft_amp[j]:9.4f}{marker}")

    peak_period = res['peak_period']
    peak_amp = res['peak_amp']
    noise_level = res['noise_level']
    snr = peak_amp / noise_level if noise_level > 0 else 0

    print(f"\n  Dominant: f={res['peak_freq']:.3f} (period={peak_period:.2f})")
    print(f"  Peak amp: {peak_amp:.4f}, noise: {noise_level:.4f}, SNR: {snr:.1f}")

    expected_period = 1.0 / R
    if abs(peak_period - expected_period) < 0.5 * expected_period and snr > 3:
        verdict = "OSCILLATIONS"
    elif rms < 0.02:
        verdict = "SMOOTH"
    else:
        verdict = "INCONCLUSIVE"

    print(f"  Verdict: {verdict}")

    # κ consistency check
    prefactor = N_POL * R / (4 * np.pi**2)
    kd = prefactor * np.trapz(integrand, k_vals)
    print(f"  κ_data (fine grid) = {kd:.3f}")

    all_results.append({
        'R': R, 'sigma_tr': sigma_tr.copy(), 'rms': rms,
        'peak_period': peak_period, 'snr': snr, 'verdict': verdict,
        'kd': kd,
    })
    print()


# ── Cross-R summary ──────────────────────────────────
print("=" * 55)
print("Multi-R oscillation summary")
print("=" * 55)
print(f"{'R':>3s}  {'1/R':>6s}  {'RMS%':>5s}  {'peak_T':>7s}  "
      f"{'SNR':>4s}  {'κ_data':>7s}  {'verdict':>12s}")
print("-" * 55)
for r in all_results:
    print(f"{r['R']:3d}  {1/r['R']:6.3f}  {r['rms']*100:4.1f}%  "
          f"{r['peak_period']:7.3f}  {r['snr']:4.1f}  "
          f"{r['kd']:7.3f}  {r['verdict']:>12s}")

# Check if peak period scales as 1/R across R values
periods = [r['peak_period'] for r in all_results]
R_arr = np.array([r['R'] for r in all_results])
period_arr = np.array(periods)
expected = 1.0 / R_arr

print(f"\nPeriod scaling test (expect period ∝ 1/R):")
for i in range(len(all_results)):
    ratio = period_arr[i] / expected[i] if expected[i] > 0 else float('inf')
    print(f"  R={R_arr[i]}: period={period_arr[i]:.3f}, "
          f"1/R={expected[i]:.3f}, ratio={ratio:.2f}")

# Overall conclusion
all_smooth = all(r['verdict'] == 'SMOOTH' for r in all_results)
all_osc = all(r['verdict'] == 'OSCILLATIONS' for r in all_results)
if all_smooth:
    print(f"\n** NO OSCILLATIONS at any R: scattering is smooth **")
    print(f"   Ring does NOT act as coherent interferometer.")
elif all_osc:
    print(f"\n** OSCILLATIONS at all R: ring interference confirmed **")
else:
    verdicts = [(r['R'], r['verdict']) for r in all_results]
    print(f"\n** MIXED: {verdicts} **")

t_total = time.time() - t0
print(f"\nTotal time: {t_total:.0f}s ({t_total/60:.1f} min)")
