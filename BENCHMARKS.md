# Benchmark of the LBM permeability and thermal-conductivity model

Tests of the solvers in `LBM_thermal_conductivity_moving_fluid.ipynb` (D3Q19 BGK flow, D3Q7 BGK heat)
on a case with published exact solutions: a **simple cubic (SC) array of spheres**.
The benchmark scripts load the solver functions straight from the notebook, so this is exactly the code
used for the rock samples.

| Item | Value |
|---|---|
| Geometry | SC array of spheres, solid fraction c = 0.125 and 0.216 |
| Resolution | unit cell L = 24, 32, 48 voxels (sphere diameter 15–36 voxels) |
| Fluid / solid | water k_f = 0.6 W/(m K), quartz k_s = 7.7 W/(m K) (k_s/k_f = 12.8) |
| Flow solver | τ = 1, Zou-He pressure inlet/outlet, periodic sides, 4 unit cells along x |
| Heat solver | hot wall T = 1 / cold wall T = 0 (anti-bounce-back, half a voxel outside), periodic sides |
| Scripts | `benchmark_sc_spheres.py`, `benchmark_sc_fields.py`, `plot_benchmark_summary.py` |
| Data | `benchmark_sc_spheres.json`, `benchmark_sc_alpha.json` |

## References

**Permeability — Zick & Homsy (1982)**, exact Stokes flow through SC arrays of spheres.
The drag on a sphere is `F = 6 π μ a U_D K_d(c)`, which gives the permeability

`k = 2 a² / (9 c K_d)`

with `K_d(0.125) = 4.292` and `K_d(0.216) = 7.442`. The Sangani & Acrivos (1982) series
`1/K_d = 1 − 1.7601 c^(1/3) + c − 1.5593 c² + 3.9799 c^(8/3) − 3.0734 c^(10/3)` (4.29 at c = 0.125)
corrects for the small difference between the voxel and the nominal solid fraction.

**Conductivity — Rayleigh (1892)**, SC array of spheres of conductivity `k_s` in a matrix `k_f`, `ν = k_s/k_f`:

`k_eff / k_f = 1 − 3c / [ (2+ν)/(1−ν) + c − 1.65 (1−ν)/(4+3ν) · c^(10/3) ]`

For the heat-flow direction along a lattice axis, one unit cell between isothermal walls is exact for the
infinite array (the planes midway between spheres are isothermal by symmetry).

**Finite volume (same voxels).** An independent finite-volume solution of the identical voxel geometry.
It separates the LBM error from the staircase-sphere error: the voxel sphere is not a true sphere.
It is also the reference for the with-flow case, which has no closed-form solution.

## 1. Permeability

Permeability in voxel²; error vs Zick & Homsy in brackets.

| c | L | voxel c | Zick & Homsy | Notebook solver (pressure inlet/outlet) | Fully periodic (body force) |
|---|---|---|---|---|---|
| 0.125 | 24 | 0.1256 | 22.83 | 22.52 (−1.40 %) | 22.68 (−0.66 %) |
| 0.125 | 32 | 0.1230 | 41.54 | 40.73 (−1.95 %) | 40.96 (−1.40 %) |
| 0.125 | 48 | 0.1246 | 92.18 | 90.71 (−1.59 %) | 91.05 (−1.22 %) |
| 0.216 | 24 | 0.2182 | 10.86 | 10.60 (−2.43 %) | 10.68 (−1.68 %) |
| 0.216 | 32 | 0.2163 | 19.57 | 19.12 (−2.30 %) | 19.23 (−1.74 %) |
| 0.216 | 48 | 0.2169 | 43.85 | 42.98 (−1.99 %) | 43.14 (−1.62 %) |

- The notebook solver is **within 1.4–2.4 %** of the exact solution, always slightly low.
- About 0.4–0.8 % of that comes from the pressure inlet/outlet (compare with the fully periodic column).
- The rest is the known BGK / bounce-back wall-position error at τ = 1. It changes only slowly with resolution.
  A TRT or MRT collision operator would remove it.

## 2. Effective conductivity, fluid at rest

k_eff / k_f; error vs Rayleigh in brackets. `α_f` is the lattice diffusivity of the fluid.
The solid's relaxation time is `τ_s = 0.5 + 4 α_f k_s/k_f`, so the notebook default α_f = 0.125 gives τ_s = 6.9.

| c | L | Rayleigh | Finite volume | LBM α_f = 0.125 (default) | LBM α_f = 0.05 | LBM α_f = 0.02 |
|---|---|---|---|---|---|---|
| 0.125 | 24 | 1.3341 | 1.3348 (+0.05 %) | 1.3140 (−1.51 %) | 1.3306 (−0.26 %) | 1.3391 (+0.37 %) |
| 0.125 | 32 | 1.3267 | 1.3281 (+0.11 %) | 1.3120 (−1.10 %) | 1.3243 (−0.18 %) | 1.3310 (+0.32 %) |
| 0.125 | 48 | 1.3311 | 1.3325 (+0.10 %) | 1.3213 (−0.74 %) | 1.3297 (−0.11 %) | 1.3343 (+0.24 %) |
| 0.216 | 24 | 1.6339 | 1.6390 (+0.31 %) | 1.6046 (−1.80 %) | 1.6298 (−0.25 %) | 1.6448 (+0.67 %) |
| 0.216 | 32 | 1.6273 | 1.6329 (+0.34 %) | 1.6058 (−1.33 %) | 1.6256 (−0.11 %) | 1.6375 (+0.62 %) |
| 0.216 | 48 | 1.6293 | 1.6343 (+0.30 %) | 1.6148 (−0.89 %) | 1.6289 (−0.03 %) | 1.6371 (+0.48 %) |

- The finite-volume solution matches Rayleigh to 0.05–0.34 %. What remains is the staircase-sphere effect.
- **With the default α_f = 0.125, the LBM is 0.7–1.8 % low.** The error shrinks with resolution, but slowly.
  It comes from the large relaxation time in the conductive solid at curved interfaces. Flat layers (series /
  parallel) are exact at any α_f.
- **With α_f = 0.05, the LBM is within 0.03–0.26 % of Rayleigh** and converges with resolution.
  α_f = 0.02 overshoots slightly (+0.2–0.7 %).
- Cost: a smaller α_f slows diffusion, so the heat runs take about 2.5× more steps at α_f = 0.05.

## 3. Effective conductivity with flow, Pe_d = 1

Pe_d = U_D d / α_f (d = sphere diameter), Re_d = 0.14 for water. There is no closed-form reference here.
The same finite-volume solver is fed the LBM velocity field, and both give the apparent conductivity along
the flow from the steady 1D model `q = U_D ΔT / (1 − exp(−U_D L / k))`. The LBM α_f was set automatically
(≤ 0.125) to keep the lattice velocity below 0.05.

| c | L | k_eff / k_f at rest (LBM, α_f = 0.125) | k_eff / k_f with flow (LBM) | Finite volume | LBM vs FV |
|---|---|---|---|---|---|
| 0.125 | 24 | 1.3140 | 1.3975 | 1.3688 | +2.09 % |
| 0.125 | 32 | 1.3120 | 1.3757 | 1.3628 | +0.94 % |
| 0.125 | 48 | 1.3213 | 1.3692 | — | FV too large to solve |
| 0.216 | 24 | 1.6046 | 1.6804 | 1.6802 | +0.01 % |
| 0.216 | 32 | 1.6058 | 1.6664 | 1.6743 | −0.47 % |
| 0.216 | 48 | 1.6148 | 1.6647 | — | FV too large to solve |

- The LBM and finite volume agree within **0.01–2.1 %**, and the gap narrows with resolution (2.1 → 0.9 % at c = 0.125).
- The flow raises k_eff over the at-rest value (thermal dispersion). With the same α_f = 0.05 for both runs
  (c = 0.216, L = 48, the field plots below) the increase is +2.6 %: 1.6289 → 1.6720 k_f. The at-rest column
  above uses α_f = 0.125 and is ~1 % low, so it overstates the increase if compared directly.

## Summary plot

![Benchmark errors](benchmark_sc_summary.png)

Left: permeability error vs Zick & Homsy. Middle: at-rest conductivity error vs Rayleigh for three lattice
diffusivities, plus finite volume. Right: with-flow conductivity, LBM vs finite volume.

## Field plots (c = 0.216, L = 48, α_f = 0.05)

![Benchmark fields](benchmark_sc_fields.png)

- **Top left:** Stokes-flow speed on the mid-plane through 4 unit cells. Flow is fastest in the gaps between
  sphere rows and nearly stagnant just upstream and downstream of each sphere.
  Permeability 42.98 voxel² (Zick & Homsy 43.85).
- **Top middle / right:** u_x across the flow, through a sphere centre and midway between spheres.
  The flow concentrates in the open channels at the cell corners.
- **Bottom left:** temperature at rest in one unit cell. The conductive sphere is nearly isothermal.
  k_eff = 1.6289 k_f vs Rayleigh 1.6293 k_f (−0.03 %).
- **Bottom middle:** heat flux q_x at rest. Heat funnels through the quartz sphere (2–3× the mean flux), and
  little passes through the fluid beside it.
- **Bottom right:** steady temperature with flow (Pe_d = 1), 4 cells. The profile is pushed downstream.
  k_eff = 1.6720 k_f (+2.6 % over the at-rest value).

## Conclusions

| Quantity | Accuracy at 24–48 voxels per unit cell | Recommendation |
|---|---|---|
| Permeability | 1.4–2.4 % low | acceptable; use TRT/MRT if < 1 % is needed |
| k_eff at rest | 0.7–1.8 % low with the default α_f = 0.125 | **set `ALPHA_MAX = 0.05`** → within 0.3 % |
| k_eff with flow | within 0.01–2.1 % of finite volume, improving with resolution | same `ALPHA_MAX = 0.05` |

For the `dir00_n_p1.000_p0.000_p0.000` sample (about 27 voxels per grain), this suggests its permeability
(214 D) is about 2 % low, and its k_eff (3.842 and 4.091 k_f) about 1–1.5 % low with the settings that were used.

## Other checks run earlier (same solvers; files since removed)

| Test | Reference | Result |
|---|---|---|
| Series layers, k_s/k_f = 0.1–100 | exact 1/k = φ/k_f + (1−φ)/k_s | error < 1e-4 % |
| Parallel layers, k_s/k_f = 0.1–100 | exact k = φ k_f + (1−φ) k_s | error < 1e-6 % |
| Plug flow in a uniform medium, Pe_L up to 2.6 | exact k = k_f | error ≤ 5e-4 (known lattice error ~ u²) |
| Plane Poiseuille channel, 15 cells across | exact k = H²/12 · H/N_y | +0.5 % |
| Taylor–Aris dispersion between plates, gap 10–40 cells | exact K/α = 1 + Pe²/210 | error of the dispersion part 14 % → 3.5 % → 0.8 % (2nd order) |
