# Supplemental Material

## S1. FDTD simulation details

The scattering problem is solved by finite-difference time-domain (FDTD)
integration of the lattice equation of motion

$$m\ddot{u}_i = \sum_j K_{ij}(u_j - u_i)$$

on a simple cubic lattice with $K_1 = 1.0$, $K_2 = 0.5$ ($K_1 = 2K_2$ for
isotropic sound speed $c = \sqrt{K_1 + 4K_2} = \sqrt{3}$). The time step is
$\Delta t = 0.25$, well within the CFL stability limit.

**Peierls gauge.** Bonds crossing the Dirac disk of the disclination loop are
modified: $u_j \to \mathcal{R}(2\pi\alpha)\,u_j$, where $\mathcal{R}$ is a rotation matrix of
angle $2\pi\alpha$ in the $(x,y)$ plane. This produces a coupling
$\delta K = K_1(R - I)$ on each gauged bond. Since $R$ acts only on
$(u_x, u_y)$, the $u_z$ component decouples: $N_{\rm pol} = 2$ independent
polarizations contribute to transport.

**Boundary conditions.** Perfectly matched layers (PML) of width $D_W = 15$
lattice spacings with damping coefficient $\sigma_s = 1.5$ absorb outgoing
waves on all faces. PML convergence is verified: the spread across
$D_W = 10, 15, 20$ is less than 2% at all $k \geq 0.5$. The $k = 0.3$ mode
is sensitive to box size rather than PML (see below).

**Wave packet.** A Gaussian wave packet with spatial width $\sigma_x = 8$ is
launched along $+x$ at wavenumber $k$. Bandwidth independence is verified:
$\sigma_x = 6, 8, 12$ give results within 5% at $k = 0.9$ and $k = 1.5$.

**Box size.** The default box is $L = 100$ lattice spacings. Convergence is
verified by comparison with $L = 120$: agreement within 5% at all $k$. At
$L = 80$, the measurement sphere is too close to the PML and $k = 0.3$
overestimates $\sigma_{\rm tr}$ by 44%. At $k \geq 0.9$, $L = 80$ bias is
below 15%.

**Measurement.** Scattered waves are recorded on a sphere of radius
$r_m = 20$ surrounding the defect, centered at the loop center. The recording
starts at $x_{\rm start} = L/2 - r_m$ to capture the wave packet as it
enters the measurement sphere. The transport cross-section is

$$\sigma_{\rm tr} = \frac{1}{P_{\rm inc}} \int (1 - \cos\theta_s)\,\frac{d\sigma}{d\Omega}\,d\Omega$$

where $\theta_s$ is the scattering angle relative to the incident direction
$+x$, and $P_{\rm inc}$ is the incident power.

---

## S2. Transport formula derivation

The Boltzmann phonon drag coefficient for a single defect of radius $R$ in a
Debye solid is

$$\kappa = \frac{R}{2\pi^2} \int_0^{k_{\rm max}} \sin^2(k)\,\sigma_{\rm tr}(k)\,dk$$

**Derivation.** Start from the standard drag force per unit thermal gradient:

$$F_{\rm drag} = \frac{1}{V} \sum_{\mathbf{k},s} \hbar\omega\,v_g\,\tau^{-1}\,\frac{\partial n}{\partial T}\,\sigma_{\rm tr}$$

In the Debye approximation with lattice dispersion $\omega(k) = 2c\sin(k/2)$,
the density of states along one cubic axis is

$$g(k) = \frac{k^2}{2\pi^2} \to \frac{\sin^2(k)}{2\pi^2}$$

after the change of variable from continuum $k$ to lattice dispersion (the
Jacobian $dk/d\omega = 1/v_g$ combined with the Boltzmann weight $v_g$
produces $\sin^2(k)$ from $\omega^2/c^2$). The resulting integral has no tail
extrapolation: we sample 7 wavenumbers $k \in [0.3, 1.5]$ covering the
dominant part of the Brillouin zone.

**Polarization.** Since $\mathcal{R}(2\pi\alpha)$ acts only on $(u_x, u_y)$, the $u_z$
polarization is unscattered. Two transverse polarizations contribute:
$N_{\rm pol} = 2$.

---

## S3. Systematics

The dominant systematic uncertainties are the gauging prescription (NN vs NNN)
and the $k$-space cutoff.

### S3.1 NN vs NNN gauging

The Peierls gauge can be applied to nearest-neighbor bonds only (NN) or also
to next-nearest-neighbor bonds (NNN). Both gaugings produce identical holonomy
$\mathcal{R}(2\pi\alpha)$ around the disclination, but distribute the gauge differently
at the lattice scale.

NNN gauging gives systematically higher $\sigma_{\rm tr}$ at all $k$ and
produces a less flat integrand (CV = 29% at $\alpha = 0.30$ vs 5.6% for NN).
The physical reason: NN $z$-bonds have $\Delta x = 0$, so the incident phase
$\exp(ikx)$ is identically 1 at all $k$, making the per-bond vertex
$k$-independent. NNN bonds have $\Delta x = 1$, introducing $k$-dependent
phases.

At $\alpha = 0.30$, $k \leq 1.5$: $\kappa_{\rm NN} = 0.844$,
$\kappa_{\rm NNN} = 2.227$. The NNN gauging overshoots $\kappa = 1$.

### S3.2 $k$-cutoff

The integral is evaluated at two cutoffs: $k \leq 0.9$ (conservative) and
$k \leq 1.5$ (full range). The ratio is 1.9--2.7$\times$ depending on
$\alpha$.

### S3.3 Gauge invariance

Rotating the Dirac disk orientation (gauge choice) changes $\sigma_{\rm tr}$
by less than 2.6% at all $k$, confirming that the result is a property of the
disclination topology, not the gauge surface.

### S3.4 Other systematics

| Source                                       | Effect                 |
|----------------------------------------------|------------------------|
| Angular grid ($N_\theta = 13, N_\phi = 24$)  | Converged within 5%    |
| Near-field correction ($r_m = 20$)            | Small (< 3%)           |
| $k$/$k_{\rm eff}$ phase mismatch             | $< 9\%$ at $k = 1.5$  |
| $K_1/K_2$ variation                           | $\kappa = 0.75$--$1.23$|

The $T$-matrix calculation uses the lattice wavenumber $k$ for the incident
phase and $k_{\rm eff} = 2\sin(k/2)$ for the outgoing phase (from the
continuum Green function). The mismatch $k - k_{\rm eff} = O(k^3/24)$
reaches $\sim 9\%$ at $k = 1.5$. This is validated empirically: the MS
$T$-matrix reproduces 75--110% of the FDTD exponent correction at $R = 5$.

### S3.5 Uncertainty budget

| Systematic                      | Factor                  |
|---------------------------------|-------------------------|
| NN vs NNN gauging               | $2.1$--$4.9\times$      |
| $k$-cutoff                      | $1.9$--$2.7\times$      |
| Gauge invariance                | $< 2.6\%$               |
| Angular grid                    | $< 5\%$                 |
| PML width                       | $< 2\%$                 |
| Box size ($L = 100$ vs $120$)   | $< 5\%$                 |
| Bandwidth ($\sigma_x$)          | $< 5\%$                 |
| $k$/$k_{\rm eff}$ phase mismatch | $< 9\%$ at $k = 1.5$  |
| $K_1/K_2$                       | $\kappa = 0.75$--$1.23$ |

---

## S4. Per-bond vertex $V(k)$

The Born vertex $V(k) = c_{m1}^2 Z_{\rm mono}(k) + s_\phi^2 Z_{\rm dipo}(k)$
combines a monopole angular integral $Z_{\rm mono} = 8\pi(1 + \sin k/k)$ and
a dipole integral $Z_{\rm dipo} = 8\pi(1 - \sin k/k)$. The monopole term is
approximately constant ($Z_{\rm mono}$ CV = 5.9%), while the dipole varies
strongly ($Z_{\rm dipo}$ CV = 72.7%). At strong coupling ($\alpha \geq 0.25$),
$|c_{m1}|^2 \gg s_\phi^2$ and the monopole dominates, giving $V(k) \approx
{\rm const}$.

$V(k)$ at $\alpha = 0.30$:

| $k$    | 0.3   | 0.5   | 0.7   | 0.9   | 1.1   | 1.3   | 1.5   |
|--------|-------|-------|-------|-------|-------|-------|-------|
| $V(k)$ | 85.83 | 85.30 | 84.51 | 83.50 | 82.27 | 80.87 | 79.32 |

CV = 2.7%. The variation is less than 8% peak-to-peak across the sampled
range. At $\alpha = 0.25$, $|c_{m1}| = |s_\phi|$ exactly, making $V(k) =
(c_{m1}^2 + s_\phi^2)(Z_{\rm mono} + Z_{\rm dipo})/2 = {\rm const}$
(CV = 0.0%, since $Z_{\rm mono} + Z_{\rm dipo} = 16\pi$).

$V(k)$ CV vs $\alpha$:

| $\alpha$ | 0.05  | 0.10  | 0.20 | 0.25 | 0.30 | 0.50 |
|----------|-------|-------|------|------|------|------|
| CV       | 54.1% | 28.2% | 4.6% | 0.0% | 2.7% | 5.9% |

The vertex constancy has a sharp threshold near $\alpha = 0.25$ (the
monopole-dipole crossover), with CV < 6% for $\alpha \geq 0.20$.

---

## S5. Assembly chain details

The flat integrand factorizes as

$$\sin^2(k) \cdot \sigma_{\rm ring} = \underbrace{4\sin^2(k/2)\cos^2(k/2)}_{\sin^2(k)} \cdot \underbrace{\frac{C_0\,V(k)}{\cos^2(k/2)}}_{1/v_g^2} \cdot N_{\rm eff}(k)$$

**Step 1: Algebraic cancellation.** $\cos^2(k/2)$ from $\sin^2(k)$ cancels
$\cos^2(k/2)$ from $1/v_g^2$, leaving $4\sin^2(k/2) \cdot V(k) \cdot N_{\rm eff}(k)$.

**Step 2: Constant vertex.** At $\alpha \geq 0.25$, the monopole coupling
$|c_{m1}|^2 Z_{\rm mono}$ dominates the vertex. Since $Z_{\rm mono}$ is
approximately constant (CV < 7%), $V(k) \approx {\rm const}$ (CV < 6%).

**Step 3: $N_{\rm eff}$ balance.** The remaining factor
$4\sin^2(k/2) \cdot N_{\rm eff}$ must be approximately constant, requiring
$N_{\rm eff} \propto 1/\sin^2(k/2)$, i.e., exponent $\approx -2.0$.

The Born exponent is $-5/2$: the filled disk forward cone has angular widths
$\delta \sim 1/\sqrt{kR}$, $\eta \sim 1/(kR)$, giving
$\sigma_{\rm tot} \sim (kR)^{-3/2}$; the transport weight $(1-\cos\theta)$
adds $-1$. Multiple scattering shifts this to $\approx -2.0$ (shift $+0.45$
at $\alpha = 0.30$).

The lattice form $1/\sin^2(k/2)$ fits better than the continuum $1/k^2$:
CV = 5.8% vs 7.1%. This lattice dispersion correction reflects the
$\sin(k/2)$ appearing in the exact dispersion relation.

**Balance point.** The integrand is flattest at $\alpha \approx 0.25$
(CV = 3.7%). At $\alpha = 0.30$: CV = 5.6%. The window
$\alpha \in [0.20, 0.40]$ gives CV < 11%.

---

## S6. FDTD convergence

Box size convergence at $R = 5$, $\alpha = 0.30$. All three box sizes are
run with identical FDTD parameters; $L = 120$ serves as reference.

| Test                          | Result                                  |
|-------------------------------|-----------------------------------------|
| $L = 100$ vs $L = 120$       | Agreement within 5% at all $k$          |
| $L = 80$, $k = 0.3$          | Overestimates by 44% (PML too close)    |
| $L = 80$, $k \geq 0.9$       | Bias < 15%                              |
| Integrand CV at $L = 100$    | 5.6% (converged)                        |

Reference $\sigma_{\rm tr}(k)$ values at $L = 100$, $R = 5$, $\alpha = 0.30$:

| $k$                            |  0.3 |  0.5 |  0.7 |  0.9 |  1.1 |  1.3 |  1.5 |
|--------------------------------|------|------|------|------|------|------|------|
| $\sigma_{\rm tr}$              | 28.3 | 12.3 | 7.09 | 4.69 | 3.50 | 2.91 | 2.60 |
| $\sin^2(k)\,\sigma_{\rm tr}$  | 2.47 | 2.83 | 2.94 | 2.88 | 2.78 | 2.70 | 2.59 |

---

## S7. Negative tests

The flat integrand is not a generic property. It fails under the following
modifications:

**Born approximation only.** Without multiple scattering, the Born integrand
has CV > 30% (approximately 19$\times$ variation across the BZ). The Born
exponent $-5/2$ makes low-$k$ modes dominate.

**Weak coupling.** At $\alpha = 0.05$, the per-bond vertex $V(k)$ varies
strongly (CV = 54%) because the dipole term $|s_\phi|^2 Z_{\rm dipo}$
dominates over the monopole. At $\alpha = 0.10$, the integrand CV exceeds 15%.

**NNN gauging.** With NNN Peierls gauge, the integrand CV = 29% at
$\alpha = 0.30$. The $k$-dependent phases $\exp(ik\Delta x)$ from NNN bonds
break the per-bond vertex constancy.

**Aharonov-Bohm formula.** The AB cross-section $\sigma_{\rm AB} \sim 1/k$
gives an integrand CV > 20%, amplitude $6.5\times$ below FDTD, and does not
reproduce the observed $k$-dependence, $\alpha$-dependence, or $R$-scaling.

These negative tests establish that the flat integrand requires: (i) multiple
scattering ($N_{\rm eff}$ exponent shift), (ii) strong coupling
($V(k) \approx {\rm const}$), and (iii) NN gauging (per-bond phase
independence).

---

## S8. $\kappa(\alpha)$ tables

### NN gauging

| $\alpha$ | $\kappa$ ($k \leq 0.9$) | $\kappa$ ($k \leq 1.5$) |
|----------|-------------------------|-------------------------|
| 0.10     | 0.022                   | 0.049                   |
| 0.15     | 0.071                   | 0.145                   |
| 0.20     | 0.162                   | 0.315                   |
| 0.25     | 0.287                   | 0.557                   |
| 0.30     | 0.428                   | 0.844                   |
| 0.40     | 0.674                   | 1.386                   |
| 0.50     | 0.769                   | 1.614                   |

### NNN gauging

| $\alpha$ | $\kappa$ ($k \leq 0.9$) | $\kappa$ ($k \leq 1.5$) |
|----------|-------------------------|-------------------------|
| 0.10     | 0.087                   | 0.238                   |
| 0.15     | 0.209                   | 0.556                   |
| 0.20     | 0.393                   | 1.016                   |
| 0.25     | 0.628                   | 1.593                   |
| 0.30     | 0.889                   | 2.227                   |

$\kappa = 1$ crossing: $\alpha \approx 0.20$--$0.33$ (NN, bracketed by
$k$-cutoff). NNN overshoots at all $\alpha \geq 0.20$.

---

*Mar 2026*
