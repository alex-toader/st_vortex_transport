# Phonon scattering by a disclination loop: forward decoherence and a flat Boltzmann transport integrand

Alexandru Toader

Independent researcher, Buzău, Romania

toader_alexandru@yahoo.com

---

## Abstract

We compute the phonon transport cross-section $\sigma_{\rm tr}(k)$ of a
disclination loop in a three-dimensional cubic lattice across the full
Brillouin zone. The Boltzmann integrand $\sin^2(k)\,\sigma_{\rm tr}(k)$
is nearly constant (coefficient of variation 5.6% at Frank angle
$\alpha = 0.30$), yielding a transport coefficient $\kappa = O(1)$.
This flat spectrum results from three ingredients:
(i) an algebraic cancellation between the lattice group velocity and the
Boltzmann weight,
(ii) an approximately $k$-independent per-bond vertex at strong coupling,
and (iii) a multiple-scattering correction that shifts the collective
exponent from $-5/2$ to $\approx -2$ through forward decoherence.
Rescattering between bonds on the Dirac disk partially breaks the forward
coherence of the Born approximation at low $k$, redistributing scattering
power across the spectrum.

---

## I. Introduction

Phonon drag on lattice defects determines the defect diffusion constant and
has been studied extensively for point defects (Rayleigh scattering,
$\sigma \propto \omega^4$) [1,2] and dislocations (vibrating
string, $\sigma \propto \omega$) [3]. In superfluids, phonon scattering
from quantized vortices has been analyzed for rotons [4] and in connection
with the Magnus force [5]. For disclination loops
--- topological defects characterized by a rotational holonomy [6,7] --- no
transport cross-section computation exists to our knowledge.

A disclination loop of Frank angle $2\pi\alpha$ threads the lattice on a Dirac
disk: every bond crossing this disk acquires a Peierls phase, rotating the
displacement field by $\mathcal{R}(2\pi\alpha)$. A phonon wave packet impinging on the
disk is deflected by each gauged bond. In the Born approximation, these bonds
scatter coherently, producing a strongly forward-peaked angular pattern whose
transport cross-section scales as $k^{-5/2}$ due to the shrinking forward cone
of a disk-shaped scatterer in a 3D medium. Multiple scattering between
bonds partially breaks this forward coherence, flattening the spectrum and
producing a nearly uniform contribution from all wavelengths.

In this Letter we compute $\sigma_{\rm tr}(k)$ for a disclination loop in a
simple cubic lattice with nearest-neighbor Peierls gauging, using
finite-difference time-domain (FDTD) simulations validated against analytic
Born and $T$-matrix predictions. We identify the mechanism that produces the
flat Boltzmann integrand and determine $\kappa(\alpha)$. The scattering
problem is of independent interest; in the stochastic mechanics framework,
$\kappa = 1$ gives the Schr\"odinger diffusion constant
$D = \hbar/(2m)$.

---

## II. Model and method

We consider a simple cubic lattice with nearest-neighbor (NN) and
next-nearest-neighbor (NNN) springs, $K_1 = 2K_2$, ensuring isotropic sound
speed $c = \sqrt{K_1 + 4K_2}$. The equation of motion for the vector
displacement $u_i$ at site $i$ is

$$m\ddot{u}_i = \sum_j K_{ij}(u_j - u_i)$$

The Dirac disk of the disclination loop is introduced by the Peierls
substitution [8] $u_j \to \mathcal{R}(2\pi\alpha)\,u_j$ for bonds crossing the
disk, where $\mathcal{R}$ is a rotation matrix of angle $2\pi\alpha$ in the
$(x,y)$ plane. Since $\mathcal{R}$ acts only on $(u_x, u_y)$, the $u_z$
polarization decouples and two independent polarizations contribute to
transport. The disclination loop has radius $R$
(measured in lattice spacings); results are presented at $R = 5$ unless
stated otherwise, with $R = 3$--$9$ used for scaling tests. The simulation
box ($L = 100$, with $L \gg R$) and PML ensure that finite-size and boundary
effects are below 5% (SM, Sec. S1, S6).

We solve the scattering problem using FDTD; a Gaussian wave packet is
launched along $+x$ and scattered waves are recorded on a sphere of radius
$r_m = 20$ surrounding the defect. The transport cross-section is extracted
from the angular power spectrum (details in SM, Sec. S1).

The Boltzmann transport coefficient is

$$\kappa = \frac{R}{2\pi^2} \int_0^{k_{\rm max}} \sin^2(k)\,\sigma_{\rm tr}(k)\,dk$$

where the integral runs over the Brillouin zone (derivation in SM, Sec. S2).
We evaluate this at 7 wavenumbers $k \in [0.3, 1.5]$, which covers the
region where the integrand varies; the high-$k$ tail contributes less than
10% (SM, Sec. S5). No extrapolation is used.

---

## III. Results

**Transport cross-section.** Figure 1 shows $\sigma_{\rm tr}(k)$ for four
values of $\alpha$. The cross-section decreases across the measured range
$k \in [0.3, 1.5]$. The scaling with loop radius is sub-geometric:
$\sigma_{\rm tr} \sim R^p$ with $p \approx 1.6$ (sub-geometric; the
stationary phase prediction $p = 3/2$ is the leading order), not $R^2$. The $k \to 0$ acoustic limit lies outside the simulation window
($kR \geq 1.5$); a crossover to Rayleigh-type behavior is expected only for
$kR \ll 1$.

**Flat integrand.** Figure 2 shows the Boltzmann integrand
$\sin^2(k)\,\sigma_{\rm tr}$. At $\alpha \geq 0.2$ with NN gauging, the
integrand is nearly constant: CV = 5.6% at $\alpha = 0.30$. The flatness
requires both strong coupling ($\alpha \geq 0.20$, where the per-bond vertex
$V(k) \approx {\rm const}$) and NN gauging (NNN gauging gives CV = 29%).
NN gauging is special because the gauged $z$-bonds have $\Delta x = 0$, so
the incident phase $e^{ikx}$ is identically 1 at all $k$, making the
per-bond vertex $k$-independent (SM, Sec. S3).
At weak coupling ($\alpha = 0.05$), the per-bond vertex $V(k)$ varies
strongly (CV = 54%) and the integrand is far from flat.

**Transport coefficient.** $\kappa(\alpha)$ is smooth and monotonic, reaching
$O(1)$ at $\alpha \approx 0.2$--$0.3$. The $\kappa = 1$ crossing lies at
$\alpha \approx 0.20$--$0.33$, bracketed by gauging and $k$-cutoff
systematics (SM, Sec. S3). The dominant systematic is the NN/NNN gauging
choice (factor $2.1$--$4.9\times$), followed by the $k$-cutoff ($1.9$--$2.7\times$).
Gauge invariance is verified by rotating the Dirac disk orientation
(branch cut displacement): $\sigma_{\rm tr}$ changes by less than 2.6% at
all $k$.

---

## IV. Mechanism

Why is $\sin^2(k)\,\sigma_{\rm tr} \approx {\rm const}$? Three ingredients
combine: an algebraic cancellation, an approximately constant vertex, and
forward decoherence by multiple scattering.

**Per-bond scattering.** Each gauged bond scatters with Born cross-section
$\sigma_{\rm bond} = C_0\,V(k)/v_g^2(k)$, where $v_g = c\cos(k/2)$ is the
lattice group velocity. The Boltzmann weight $\sin^2(k) = 4\sin^2(k/2)\cos^2(k/2)$
cancels the $\cos^2(k/2)$ from $1/v_g^2$: an algebraic identity that leaves
$\sin^2(k/2) \cdot V \cdot N_{\rm eff}$. At $\alpha \geq 0.25$, the monopole
term dominates the Peierls vertex and $V(k)$ becomes approximately constant
(CV < 6%). The FDTD angular shape matches the Born prediction within 2.8%.

**Born exponent.** A filled disk of $N \approx \pi R^2$ bonds produces an
asymmetric forward cone with angular widths $\delta \sim 1/\sqrt{kR}$ and
$\eta \sim 1/(kR)$. The total cross-section scales as $\sigma_{\rm tot} \sim
(kR)^{-3/2}$; the transport weight $(1-\cos\theta) \sim \delta^2$ adds $-1$,
giving $\sigma_{\rm tr}^{\rm Born} \sim (kR)^{-5/2}$. This exponent is
verified at $R = 200$ ($p = -2.491$, within 0.4% of $-5/2$) and is geometric:
random disks, square disks, and lattice disks all give the same scaling.

**Forward decoherence: mechanism.** Multiple scattering [9] through the
$T$-matrix $T = (I - VG)^{-1}V$, with continuum Green function
$G(r) = e^{ik_{\rm eff}r}/(4\pi c^2 r)$, shifts the exponent from $-5/2$ to
approximately $-2.0$ (Fig. 3). The shift originates from partial loss of
forward coherence. At low $k$, the Born amplitude
$|\sum V \cdot e^{ikx_j}|^2 \approx N^2|V|^2$ due to phase alignment in the
forward cone. Rescattering between bonds breaks this alignment, reducing the
forward-scattered power at low $k$ more than at high $k$ and thereby
flattening the spectrum.

**Forward decoherence: evidence.** The MS/Born power ratio is 0.61 at
$k = 0.3$ (suppression) and grows to 1.18 at $k = 1.5$ (Fig. 3). The forward
amplitude ratio $|\sum T_j b_j|^2/|\sum V_j b_j|^2 = 0.54$ at $k = 0.3$
confirms that forward coherence is broken. The suppression is monotonic in
disk radius: $0.61 \to 0.54 \to 0.47$ at $R = 5, 7, 9$. The shift magnitude
is $\Delta p = C\,|V|$ with $C \approx 0.31$--$0.35$ (UV-cutoff dependent);
the system is not resonant ($|\lambda_{\rm max}| = 0.3$--$0.86$ for
$R \leq 9$). Eighty percent of the shift comes from inter-bond $1/r$
propagation, with only 9% from the self-energy $G_{00}$. At short range
($r \leq 2$), the oscillating phases $e^{ikr}$ add constructively; beyond
$r \approx 3$ they cancel, limiting the cooperation range and explaining why
the physical shift (+0.45) is roughly half the static $1/r$ shift (+0.83).

**Assembly.** Combining the three ingredients:
$\sin^2(k) \cdot \sigma_{\rm ring} = 4\sin^2(k/2) \cdot [C_0 V / v_g^2] \cdot N_{\rm eff}$.
After cancellation, $4\sin^2(k/2) \cdot N_{\rm eff} \approx {\rm const}$
when $N_{\rm eff} \propto 1/\sin^2(k/2)$ (exponent $\approx -2$). The
residual CV of 5.6% reflects the approximate nature of the exponent match.

---

## V. Discussion

The mechanism has geometric and lattice-specific components. The Born exponent
$-5/2$ and the positive direction of the MS shift are geometric consequences
of a filled 2D scatterer cluster in a 3D medium with $1/r$ Green function:
random disks, square disks, and lattice disks all show the same behavior. The
exact value of the shift coefficient $C$, the $\cos^2(k/2)$ cancellation from
the group velocity, and the flatness of $V(k)$ are specific to the cubic
lattice with NN gauging. On other lattices (Kelvin foam, Weaire-Phelan), the
mechanism direction is preserved but $\kappa$ requires recalculation.

The result passes several negative tests: the integrand is not flat without
multiple scattering (Born CV > 30%), not flat at weak coupling ($\alpha = 0.05$,
CV = 54%), not flat with NNN gauging (CV = 29%), and not reproduced by the
Aharonov-Bohm formula (amplitude $6.5\times$ too small, wrong $k$-dependence).
These controls establish that the flatness is a specific consequence of
strong-coupling NN-gauged disclination scattering, not a numerical coincidence.

In the stochastic mechanics framework, $\kappa = 1$ gives the
Schr\"odinger diffusion constant $D = \hbar/(2m)$. Our result $\kappa = O(1)$
at $\alpha \approx 0.2$--$0.3$ establishes structural compatibility without
fine-tuning. Whether $\kappa = 1$ exactly is a separate question requiring
$\alpha$ from defect energetics and a physical $k$-cutoff, to be addressed
elsewhere.

To our knowledge, no prior computation of the phonon transport cross-section
of a disclination loop exists, nor has a flat Boltzmann integrand
$\sin^2(k)\,\sigma_{\rm tr} \approx {\rm const}$ been reported for any
lattice defect.

---

## Figures

1. $\sigma_{\rm tr}(k)$ spectrum at $\alpha = 0.10, 0.20, 0.30, 0.50$.
   Full BZ, log-log. Inset or second panel: $\sigma_{\rm tr}$ vs $R$ at
   fixed $k$ showing $R^{3/2}$ scaling.

2. Boltzmann integrand $\sin^2(k)\,\sigma_{\rm tr}$: flat at $\alpha \geq 0.2$
   vs non-flat at $\alpha = 0.10$.

3. MS/Born enhancement ratio vs $k$ at $R = 5, 7, 9$. Shows forward
   decoherence: ratio < 1 at low $k$, growing to > 1 at high $k$.
   Suppression monotonic in $R$. One-curve-explains-mechanism figure.

---

## Reproducibility

All computations use Python 3.9, NumPy, SciPy. Source code and a structured
test suite (8 files, 133 tests) are available at
https://github.com/alex-toader/st_vortex_transport. Each test verifies an
analytic prediction or physical constraint --- not implementation correctness
--- and mirrors the mechanism chain of the paper: per-bond Born scattering
($\sigma_{\rm bond} = C_0 V/v_g^2$, monopole/dipole decomposition), Born disk
exponent ($-5/2$ at large $R$, universality across disk geometries), multiple
scattering correction (exponent shift, forward decoherence, propagation
range), assembly (algebraic cancellation, $N_{\rm eff}$ balance, residual CV),
coupling requirements (weak coupling fails, polarization independence),
mechanism elimination (scalar/vectorial $T$-matrix, resonance bounds, geometry
dependence), null gauging (NNN integrand, AB formula), and FDTD convergence
(box size, PML, bandwidth).

---

## Data availability

FDTD cross-section data ($\sigma_{\rm tr}(k)$ at multiple $R$, $\alpha$,
gauging prescriptions) and $\kappa(\alpha)$ tables are included in the
repository under `tests/data/`.

---

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work the author used Claude (Anthropic)
in order to assist with code review, test suite consolidation, and manuscript
formatting. After using this tool, the author reviewed and edited the content
as needed and takes full responsibility for the content of the published
article.

---

## References

[1] J. M. Ziman, *Electrons and Phonons* (Oxford University Press, 1960).

[2] P. G. Klemens, Thermal conductivity and lattice vibrational modes,
    Solid State Phys. **7**, 1 (1958).

[3] A. Granato and K. Lücke, Theory of mechanical damping due to
    dislocations, J. Appl. Phys. **27**, 583 (1956).

[4] R. M. Cleary, Scattering of superfluid helium rotons and phonons
    from quantized vortices, Phys. Rev. **175**, 587 (1968).

[5] E. B. Sonin, Magnus force in superfluids and superconductors,
    Phys. Rev. B **55**, 485 (1997).

[6] N. D. Mermin, The topological theory of defects in ordered media,
    Rev. Mod. Phys. **51**, 591 (1979).

[7] R. de Wit, Theory of disclinations: IV. Straight disclinations,
    J. Res. Natl. Bur. Stand. A **77A**, 607 (1973).

[8] R. Peierls, Zur Theorie des Diamagnetismus von Leitungselektronen,
    Z. Phys. **80**, 763 (1933).

[9] L. L. Foldy, The multiple scattering of waves,
    Phys. Rev. **67**, 107 (1945).

---

*Mar 2026*
