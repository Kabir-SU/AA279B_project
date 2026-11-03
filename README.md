# Constellation Orbit Propagator

Vectorized multi-satellite orbit propagator with Earth as the central body. Supports J2 oblateness and piecewise-exponential atmospheric drag (Earth-rotation aware). Walker T/P/F constellation generator and reusable plot panels included.

Phase 1 of a larger low-thrust constellation optimization pipeline.

## Install

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running

From the repo root:

```bash
# Verification suite: two-body closure, J2 nodal regression, drag decay,
# Walker geometry, vectorization scaling.
python -m example

# Sample scenario: Walker 24/6/2 at 700 km LEO, 1 day, J2 + drag.
# Writes PNGs under scenarios/plots/walker_leo/ and shows them interactively.
python -m scenarios.walker_leo
```

## Core API

```python
import numpy as np
from sim import Satellite, Simulator, walker
from plots import summary

# Walker T/P/F constellation (or build a list of Satellite objects by hand)
sats = walker(T=24, P=6, F=2,
              a=7078.0, e=0.0, inc_deg=55.0,
              mass=500.0, area=5.0, cd=2.2)

sim = Simulator(sats, perturbations=("J2", "drag"), earth_rotation=True)
times, states = sim.propagate(t_span=(0.0, 86400.0),
                              t_eval=np.linspace(0, 86400.0, 1000))
# states has shape (N_sats, 6, M_times) in ECI km / km·s^-1

summary(times, states, sats, name="my_run")
```

## Adding a scenario

Copy `scenarios/walker_leo.py` and adjust the constellation / horizon / perturbations:

```python
import numpy as np
from sim import Simulator, walker
from constants import DAY_TO_SEC
from plots import summary

NAME = "my_scenario"

def run():
    sats = walker(T=..., P=..., F=..., a=..., e=..., inc_deg=...,
                  mass=..., area=..., cd=2.2)
    sim = Simulator(sats, perturbations=("J2", "drag"))
    times, states = sim.propagate((0.0, DAY_TO_SEC),
                                  t_eval=np.linspace(0, DAY_TO_SEC, 1000))
    summary(times, states, sats, name=NAME)

if __name__ == "__main__":
    run()
```

Then: `python -m scenarios.my_scenario`.

## Modules

| File | Purpose |
|---|---|
| `constants.py` | Physical constants + piecewise-exponential atmosphere table |
| `astro.py` | Classical ↔ Cartesian element conversions, Kepler solver, PQW rotations |
| `dynamics.py` | Vectorized accelerations: central body + J2 + drag |
| `sim.py` | `Satellite`, `walker()`, `Simulator.propagate()` |
| `plots.py` | `plot_trajectories_3d`, `plot_elements`, `plot_altitude`, `plot_ground_tracks`, `summary()` |
| `example.py` | Verification suite |
| `scenarios/` | One script per scenario; outputs land in `scenarios/plots/` |

## Model notes

- **Integrator**: `scipy.integrate.solve_ivp` with DOP853, `rtol=atol=1e-12`.
- **J2**: Vallado closed-form perturbing acceleration (Eq. 8-37).
- **Atmosphere**: Vallado Table 8-4 piecewise exponential, 0-1000 km. Returns zero above.
- **Drag**: `a = -½ρ(C_dA/m)|v_rel|v_rel` with `v_rel = v − ω_⊕ × r` (toggle via `earth_rotation=False`).
- **Walker**: standard T/P/F — RAAN spacing 360°/P, in-plane spacing 360°/(T/P), inter-plane phase F·360°/T.

## Out of scope (deferred)

SRP, third-body (Moon/Sun), tesseral harmonics beyond J2, NRLMSISE-00 atmosphere, low-thrust actuation, variational / sensitivity propagation.
