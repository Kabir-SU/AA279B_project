# Using the papers for low-thrust constellation deployment

This assessment treats “constellation development” as deploying or reconfiguring satellites from injection orbits into a prescribed constellation. Choosing the constellation geometry itself could become an outer design problem, but is a larger scope. The recommendations below are a proposed synthesis, not a method already demonstrated by all seven papers.

## Main recommendation

Build a hierarchical optimizer: choose drift orbits, deployment times, and satellite-to-slot assignments in an outer problem; generate and refine continuous low-thrust transfers in an inner problem. Validate the resulting trajectories with the existing Cartesian J2 + drag + thrust simulator.

The central research opportunity is **jointly controlling differential nodal drift and along-track phasing during finite-thrust deployment**. Reaching an orbit is insufficient: each spacecraft must occupy its assigned slot at the required epoch. This matters especially for a circular Walker constellation, where argument of periapsis is undefined and cannot serve as an independent terminal requirement.

## What each paper contributes

| Paper | Useful contribution | Application here | Important limit |
|---|---|---|---|
| [Apa, Hudson, Romano (2026), drift orbit optimization](../papers/s40295-026-00580-4.pdf), §§3–5 | Thrust–drift–thrust strategy; optimize drift altitude and inclination; account for secular J2, drag, eclipse availability, and mass depletion; tabulate time- and mass-dependent transfer costs | Primary architecture for spreading satellites between orbital planes, and for inexpensive evaluation of deployment alternatives | Nearly circular orbits, e ≤ 0.05. Main optimization neglects phasing cost; §6.5.2 estimates its impact afterward. Multi-client servicing is sequential; independent satellite deployment requires a different assignment/scheduling problem. |
| [Ruggiero et al. (2011), efficient orbital element correction](../papers/Ruggiero,%20Low%20Thrust%20Maneuvers%20for%20the%20Efficient%20Correction%20of%20Orbital%20Elements.pdf), §§II–III | Gauss-equation steering laws for individual and combined element changes; efficiency thresholds introduce coasts | First feedback baseline, checks on thrust direction, and transfer initial guesses | Two-body + thrust model. Explicitly omits true-anomaly targeting. Instantaneously optimal element correction does not establish a globally optimal coupled transfer. |
| [Varga and Sánchez Pérez, equinoctial Q-law](../papers/Final_Paper_ICATT.pdf), §§2–4 | Nonsingular treatment near circular/equatorial orbits; feedback guidance with J2 and eclipse effects; NSGA-II tuning for time/propellant tradeoffs | General orbit-acquisition controller and source of initial trajectories | Q targets five orbit variables and leaves final anomaly free. A separate or extended phasing formulation is needed. The feedback result is a heuristic, not proof of global optimality. Standard prograde equinoctial elements still have a retrograde singularity. |
| [Leomanni et al. (2021), direct approach](../papers/jsr21.pdf), §§II–V | Regularized ideal-element dynamics; Lyapunov initial guess; direct collocation with eclipse constraints and time/fuel tradeoff | Strong reference for refining deployment transfers with mass dynamics and explicit endpoint constraints | More implementation work than feedback guidance. Demonstrations use GPOPS-II. Equation (27) permits fixed terminal longitude and time, but constellation slots must be propagated to that time rather than treated as static targets. |
| [Tracy and Manchester (2021), KS transformation](../papers/ks_low_thrust.pdf), trajectory optimization and numerical experiments | Regularized dynamics with favorable linearization; ALTRO optimization; demonstrated GTO–GEO convergence from zero-thrust initial controls | Alternative state representation to benchmark if long, eccentric transfers make refinement expensive | Demonstrations use a quadratic control/orbit-error objective, not the full deployment fuel objective. Add mass, physical-time endpoint constraints, slot targeting, and mission constraints. Reported convergence is scenario-specific. |
| [Kim (2005), dissertation](../papers/MischaKim.pdf), abstract and symmetry/homotopy development | Cascaded global/local solution strategy, continuation, and symmetry reuse across related optimal-control problems | Reuse solved transfers across nearby slots, inclinations, epochs, thrust levels, and deployment deadlines | Earth J2 preserves axial rotational symmetry, but Sun-dependent eclipse conditions break arbitrary RAAN reuse at a fixed epoch. Reused solutions need validation and often refinement. |
| [Morante et al. (2021), survey](../papers/aerospace-08-00088-v3.pdf), §§2–7, 9 | Framework covering continuous/discrete decisions, direct/indirect methods, feedback, averaging, and hybrid optimal control | Justify separating constellation architecture from continuous trajectory optimization; guide later method comparisons | Background and taxonomy, not a ready deployment algorithm. |

## Why drift orbits connect these papers

For first-order secular J2 dynamics,

$
n=\sqrt{\mu/a^3},\quad p=a(1-e^2),\qquad
\dot\Omega=-\frac32 J_2 n\left(\frac{R_E}{p}\right)^2\cos i.
$

Changing altitude or inclination changes nodal precession. A spacecraft can enter a drift orbit and accumulate a relative RAAN change against a moving target plane. The relevant prediction is

$
\Delta\Omega(t_f)=\Delta\Omega(t_0)+
\int_{t_0}^{t_f}(\dot\Omega_s-\dot\Omega_{\mathrm{target}})\,dt.
$

Include both powered arcs in this integral. Precession does not stop while thrusting, and the target plane also precesses.

Altitude changes simultaneously change mean motion. For nearby circular orbits, the leading Keplerian contribution to relative along-track phase rate is

$
\Delta\dot\theta\simeq\Delta n\simeq-\frac32 n\frac{\Delta a}{a}.
$

Thus the same drift-altitude choice affects both plane separation and slot spacing. J2 corrections and changing inclination must be included in the actual mean-angle propagation. Treat these equations as planning approximations, then validate with the Cartesian model. Large plane separations may demand long drift durations; near polar inclination, altitude-driven nodal drift is weak because of the cosine factor. Feasibility depends on the actual launch geometry, thrust, and deadline.

For circular inclined orbits use a consistent mean argument of latitude or nonsingular longitude for phasing, together with the plane orientation. Do not independently constrain argument of periapsis and mean anomaly at zero eccentricity. Define whether terminal targets are mean or osculating elements; short-period J2 oscillations otherwise create misleading endpoint errors.

## Proposed optimization

Start with a prescribed Walker target and known injection states. Let the outer variables include satellite-to-slot assignment, drift semi-major axis and inclination, coast duration, departure time, and arrival time. Initially keep inclination fixed if the launch and final planes permit useful altitude-driven drift; add inclination as a design variable when needed.

For each candidate, solve or approximate an inner transfer with thrust magnitude/direction as controls. In the existing km–s–kg convention, with thrust force T in newtons,

$
\dot r=v,\qquad
\dot v=a_{2b}+a_{J2}+a_D+10^{-3}\frac{T}{m},\qquad
\dot m=-\frac{\|T\|}{I_{sp}g_0}.
$

Enforce thrust limits, available propellant, safe perigee, allowed transfer duration, and target orbit plus slot tolerances at arrival. Add eclipse restrictions if the power system requires them. A later joint validation must assess separation between deploying spacecraft; independent single-satellite feasibility does not establish constellation-wide feasibility.

Use total propellant versus time until the last satellite is operational as the initial tradeoff. For example,

$
J=w_m\frac{\sum_s(m_{s,0}-m_{s,f})}{m_{\mathrm{ref}}}
+w_t\frac{\max_s(t_{s,f}-t_0)}{t_{\mathrm{ref}}}.
$

Normalization makes the weights interpretable. Alternatively minimize propellant for a sequence of deployment deadlines. If operational coverage during deployment is important, add a coverage-time objective later; terminal geometry alone does not measure that benefit.

Adapt Apa's transfer tables into costs from injection states to moving constellation slots. Store dependence on departure epoch, mass, deadline, and relative phase as needed. Do not discard phase merely to keep the tables small. Interpolation can misrepresent feasibility boundaries, so re-solve selected candidates with the underlying transfer model.

## Suggested implementation sequence

1. **Establish one-satellite baselines.** Implement Ruggiero-style steering for a small circular orbit raise and a plane change; verify steering signs, units, and propellant accounting. Existing `Satellite.thrust_command(t, state)` already offers an integration point. A feedback controller can convert the Cartesian state to its chosen elements and rotate local radial/transverse/normal thrust into ECI.
2. **Build a two-satellite drift/phasing case.** Start from nearby injection states, target one final altitude/inclination and specified relative RAAN/along-track phase, and optimize drift altitude plus maneuver timing. Use averaged dynamics for the search and Cartesian propagation for acceptance.
3. **Refine continuous controls.** Use the direct-collocation architecture in Leomanni, initialized by feedback or the drift plan. Begin with two-body dynamics, then continue into J2, drag, and eclipse constraints. Compare against the unrefined plan at identical endpoint tolerances.
4. **Expand to a small Walker deployment.** Introduce assignment and shared deployment deadlines. Cache transfer evaluations and reuse nearby solutions. Use symmetry only under the assumptions that preserve it.
5. **Benchmark KS if needed.** Compare equal endpoint accuracy, physical constraints, propellant objective, and runtime. The published eccentric GTO result does not establish that KS is preferable for near-circular LEO deployment.

Report fuel per satellite and total fuel, last arrival time, final plane and phase errors, failures to converge, and runtime. Validate with a tighter integration and finer control discretization, then perturb injection state and thrust execution. A controller that uses less fuel but misses its assigned slot is not an improved deployment solution.

## Findings in the current repository

- `sim.py` and `dynamics.py` already support finite thrust, mass depletion, dry-mass cutoff, J2, and drag. README's deferred low-thrust statement is outdated.
- `planner.py:12` uses `OMEGA_EARTH` in the secular J2 coefficient. Replace that factor with orbital mean motion `sqrt(MU_EARTH/a**3)` before using its predictions. Earth's spin belongs in atmosphere-relative velocity, not this precession coefficient. The returned `d_M0` is a secular correction; propagation of mean anomaly also needs the Keplerian `n` term.
- `plan.md` describes selecting element offsets to obtain desired relative rates. This is a useful reduced planning problem, but finite burn durations, mass depletion, coupled RAAN/phase evolution, and moving terminal slots must be included before it represents low-thrust deployment.
- The inspected model has no Sun/eclipse availability calculation. `Thruster.min_thrust` is stored but `force()` currently scales continuously from zero to maximum; discrete throttle or minimum-on-thrust constraints would need explicit implementation if required by the hardware.

No existing implementation files were changed for this assessment.

## Focused research question

**For deployment from a common launch into a prescribed LEO constellation, how much propellant can be saved at a fixed deployment deadline by jointly optimizing J2 drift and along-track phasing, relative to separate orbit acquisition and phasing?**

Compare sequential acquisition/phasing, a coupled drift plan, and a collocation-refined coupled plan. Keep injection conditions, available thrust, perturbations, terminal tolerances, and deadline identical. This combines the supplied papers into a testable contribution without claiming novelty from the literature review alone.

Reading priority: Apa for architecture and assumptions; Ruggiero for the first controller; Varga for nonsingular feedback; Leomanni for refinement; Kim for solution reuse; KS as a representation benchmark; the survey throughout for context.

Source scope: the seven supplied PDFs were reviewed through extracted text, with emphasis on formulations and assumptions. One page of the Apa PDF did not extract; its principal assumptions, formulation, and conclusions were readable. Selected primary online copies were also checked: [Apa publisher version](https://link.springer.com/article/10.1007/s40295-026-00580-4), [Leomanni author copy](https://www.giannibi.net/research/papers/jsr21.pdf), [KS author copy](https://rexlab.ri.cmu.edu/papers/ks_low_thrust.pdf), and [KS accompanying implementation](https://github.com/RoboticExplorationLab/KSLowThrust).
