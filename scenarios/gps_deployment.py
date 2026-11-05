"""300 km equatorial injection -> GPS-like Walker 24/6/1.

Run: python -m scenarios.gps_deployment --mode one   # or third / full
The planner targets orbit elements; final slot phasing is only an initial guess.
GPS orbit reference: https://www.gps.gov/space-segment
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid

from astro import j2_reference, cart_to_coe, coe_to_cart, wrap_to_pi
from constants import DAY_TO_SEC, G0, J2, MU_EARTH, R_EARTH
from planner import algorithm_1, make_w_ecl
from plots import summary, plot_target_constellation, finish_plots
from sim import Satellite, Simulator, Thruster, walker

GPS_A = R_EARTH + 20200.0
GPS_I = np.deg2rad(55.0)
NSTEPS = 10000


def target_rates():
    n = np.sqrt(MU_EARTH/GPS_A**3)
    k = .75*J2*n*(R_EARTH/GPS_A)**2
    return -2*k*np.cos(GPS_I), n+k*(8*np.cos(GPS_I)**2-2)


def make_command(profile, eclipses=True):
    """Edelbaum steering from Algorithm 1, with actual eclipse switching."""

    def command(t, state):
        r, v = state[:3], state[3:6]
        radial = r/np.linalg.norm(r)
        normal = np.cross(r, v)
        normal /= np.linalg.norm(normal)
        tangent = np.cross(normal, radial)
        if t >= profile["t"][-1] or (eclipses and r[0] < 0 and np.linalg.norm(r[1:]) < R_EARTH):
            return 0.0, tangent
        beta = np.interp(t, profile["t"], profile["beta"])
        # Use the planned plane's node, including at zero inclination where
        # the instantaneous ascending node is undefined.
        node = np.interp(t, profile["t"], profile["Omega"])
        node_axis = np.array([np.cos(node), np.sin(node), 0.0])
        direction = np.cos(beta)*tangent + np.sin(beta)*np.sign(radial @ node_axis)*normal
        return 1.0, direction

    return command


def build(mode="one", eclipses=True):
    count = {"one": 1, "third": 8, "full": 24}[mode]
    indices = [0] if count == 1 else np.arange(0, 24, 24//count)
    targets = walker(24, 6, 1, GPS_A, 0.0, 55.0, 500.0, 5.0)
    eclipse = make_w_ecl(lambda t: [1.0, 0.0, 0.0]) if eclipses else lambda *args: 1.0
    node_rate, u_rate = target_rates()
    satellites, profiles, selected = [], [], []
    for index in indices:
        plane, slot = divmod(int(index), 4)
        target_node = plane*np.pi/3
        target_u = 2*np.pi*(slot/4 + plane/24)  # Walker F=1
        initial_node = target_node
        # Pick the otherwise-free initial steering axis to account for the
        # planner's predicted J2 precession. This does not modify Algorithm 1.
        for _ in range(4):
            dv, duration, dn, profile = algorithm_1(
                0, R_EARTH+300, GPS_A, 0, GPS_I, initial_node,
                500, 3000, 1.0,
                2.418e-11*1e9*np.exp(300/53.628), 53.628,
                2.2, 5e-6, 500, eclipse, return_details=True,
            )
            corrected = target_node + node_rate*duration - dn
            if abs(wrap_to_pi(corrected-initial_node)) < 1e-8:
                break
            initial_node = corrected
        # Keep the command's equatorial steering axis consistent with profile.
        initial_node = profile["Omega"][0]
        n = np.sqrt(MU_EARTH/profile["a"]**3)
        c = np.cos(profile["i"])
        u_dot = n+.75*J2*n*(R_EARTH/profile["a"])**2*(8*c*c-2)
        accumulated_u = trapezoid(u_dot, profile["t"])
        initial_longitude = initial_node + target_u + u_rate*duration - accumulated_u
        satellites.append(Satellite.from_elements(
            (R_EARTH+300, 0, 0, 0, 0, initial_longitude),
            mass=500, area=5, cd=2.2, name=f"w{plane}_{slot}",
            thruster=Thruster(1.0, 3000), dry_mass=250,
            thrust_command=make_command(profile, eclipses),
        ))
        profiles.append(profile)
        selected.append(targets[int(index)])
        fuel = 500*(-np.expm1(-dv*1000/(3000*G0)))
        print(f"w{plane}_{slot}: {duration/DAY_TO_SEC:.2f} days, "
              f"{dv*1000:.1f} m/s, planned fuel {fuel:.2f} kg", flush=True)
    return satellites, profiles, selected


def run(mode="one", show=True, plan_only=False, days=None, eclipses=True):
    sats, profiles, targets = build(mode, eclipses)
    end = max(p["t"][-1] for p in profiles)
    if plan_only:
        return sats, profiles, targets
    if days is not None:
        end = max(end, days*DAY_TO_SEC)
    print(f"Propagating {len(sats)} satellites for {end/DAY_TO_SEC:.2f} days...", flush=True)
    sim = Simulator(sats, perturbations=("J2", "drag"), rtol=1e-8, atol=1e-10)
    times, states = sim.propagate((0, end), t_eval=np.linspace(0, end, NSTEPS), max_step=180)
    name = f"gps_deployment_{mode}" + ("" if eclipses else "_no_eclipse")
    references = np.array([j2_reference(cart_to_coe(s.r, s.v, MU_EARTH), times) for s in targets])
    summary(times, states, sats, name=name, show=False, plot_endpoints=True,
            target_elements=references)

    node_rate, u_rate = target_rates()
    planes = [(GPS_A, 0, GPS_I, plane*np.pi/3+node_rate*end, 0, 0)
              for plane in range(6)]
    expected = []
    print("Osculating errors at the propagation end relative to moving Walker targets:")
    for j, target in enumerate(targets):
        a, e, inc, node, omega, mean = cart_to_coe(states[j, :3, -1], states[j, 3:6, -1], MU_EARTH)
        target_coe = list(cart_to_coe(target.r, target.v, MU_EARTH))
        target_coe[3] += node_rate*end
        target_coe[5] += u_rate*end
        expected.append(coe_to_cart(target_coe, MU_EARTH)[0])
        phase_error = wrap_to_pi(omega+mean-target_coe[5])
        print(f"{sats[j].name}: altitude={a-R_EARTH:.1f} km, e={e:.4f}, "
              f"i={np.rad2deg(inc):.2f} deg, "
              f"RAAN error={np.rad2deg(wrap_to_pi(node-target_coe[3])):.2f} deg, "
              f"phase error={np.rad2deg(phase_error):.2f} deg, "
              f"fuel={500-states[j, 6, -1]:.2f} kg")
    plot_target_constellation(
        states[:, :3, -1], expected, planes, name=name,
        title="GPS-like Walker 24/6/1 target and propagated endpoints",
        final_velocities=states[:, 3:6, -1],
    )
    output = Path(__file__).parent / "plots" / name
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "propagation.npz", times=times, states=states)
    print("Slot phases are initial guesses; the planner does not enforce final phasing.")
    finish_plots(show)
    return times, states


def main(default_eclipses=True):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("one", "third", "full"), default="one")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--days", type=float, help="Truncate propagation for a quick check.")
    parser.add_argument("--eclipses", action=argparse.BooleanOptionalAction,
                        default=default_eclipses)
    args = parser.parse_args()
    run(args.mode, show=not args.no_show, plan_only=args.plan_only, days=args.days,
        eclipses=args.eclipses)


if __name__ == "__main__":
    main()
