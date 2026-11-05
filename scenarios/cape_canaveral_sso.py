"""300 km, 51.6 deg circular insertion above Cape Canaveral -> 600 km dawn-dusk SSO.

Run: python -m scenarios.cape_canaveral_sso [--no-eclipses] [--no-show]
Starts after rocket ascent. Launch orientation is chosen for a dawn-dusk plane;
time zero is an idealized equinox, not a specified calendar launch date.
"""

import argparse

import numpy as np

from astro import j2_reference, cart_to_coe, coe_to_cart, wrap_to_pi
from constants import DAY_TO_SEC, G0, J2, MU_EARTH, R_EARTH
from planner import algorithm_1, make_w_ecl
from plots import summary, plot_target_constellation, finish_plots
from sim import Satellite, Simulator, Thruster

SUN_RATE = 2*np.pi/(365.2422*DAY_TO_SEC)
INITIAL_I = np.deg2rad(51.6)  # ISS orbital inclination
TARGET_A = R_EARTH+600
TARGET_I = np.arccos(-SUN_RATE/(1.5*J2*np.sqrt(MU_EARTH/TARGET_A**3)*(R_EARTH/TARGET_A)**2))
NSTEPS = 5000

def sun_vector(t):
    """Circular annual solar motion with obliquity, starting at equinox."""
    longitude = SUN_RATE*t
    obliquity = np.deg2rad(23.44)
    return np.array([np.cos(longitude), np.cos(obliquity)*np.sin(longitude),
                     np.sin(obliquity)*np.sin(longitude)])


def build(eclipses=True):
    latitude, longitude = np.deg2rad([28.392, -80.607])
    # Northeast circular velocity into an ISS-inclination orbit.
    heading = np.arcsin(np.cos(INITIAL_I)/np.cos(latitude))
    up = np.array([np.cos(latitude)*np.cos(longitude),
                   np.cos(latitude)*np.sin(longitude), np.sin(latitude)])
    east = np.array([-np.sin(longitude), np.cos(longitude), 0])
    north = np.cross(up, east)
    r = (R_EARTH+300)*up
    v = np.sqrt(MU_EARTH/(R_EARTH+300))*(np.cos(heading)*north+np.sin(heading)*east)
    node = cart_to_coe(r, v, MU_EARTH)[3]
    sunlight = make_w_ecl(sun_vector) if eclipses else lambda *args: 1.0
    initial_node = np.pi/2
    # Choose launch time to aim the planner's final plane at dawn-dusk SSO.
    for _ in range(6):
        dv, duration, dn, profile = algorithm_1(
            0, R_EARTH+300, TARGET_A, INITIAL_I, TARGET_I, initial_node,
            500, 3000, 1.0,
            2.418e-11*1e9*np.exp(300/53.628), 53.628,
            2.2, 5e-6, 300, sunlight, return_details=True,
        )
        corrected = np.pi/2+SUN_RATE*duration-dn
        if abs(wrap_to_pi(corrected-initial_node)) < 1e-8:
            break
        initial_node = corrected
    # Rotate the site into the chosen inertial orientation at insertion.
    angle = profile["Omega"][0]-node
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    r, v = rotation@r, rotation@v

    def command(t, state):
        radial = state[:3]/np.linalg.norm(state[:3])
        normal = np.cross(state[:3], state[3:6])
        tangent = np.cross(normal/np.linalg.norm(normal), radial)
        sun = sun_vector(t)
        along_sun = state[:3]@sun
        shadow = along_sun < 0 and np.linalg.norm(state[:3]-along_sun*sun) < R_EARTH
        beta = np.interp(t, profile["t"], profile["beta"])
        node = np.interp(t, profile["t"], profile["Omega"])
        node_axis = np.array([np.cos(node), np.sin(node), 0.0])
        direction = np.cos(beta)*tangent + np.sin(beta)*np.sign(radial@node_axis)*normal/np.linalg.norm(normal)
        return float(t < duration and not (eclipses and shadow)), direction

    sat = Satellite(r, v, mass=500, area=5, cd=2.2, name="Cape Canaveral to SSO",
                    thruster=Thruster(1.0, 3000), dry_mass=250, thrust_command=command)
    return sat, profile, dv, duration, heading


def run(show=True, eclipses=True):
    sat, profile, dv, duration, heading = build(eclipses)
    fuel = 500*(-np.expm1(-dv*1000/(3000*G0)))
    print(f"Initial orbit: 300 km above Cape Canaveral, i={np.rad2deg(INITIAL_I):.3f} deg")
    print(f"Target: 600 km dawn-dusk SSO, i={np.rad2deg(TARGET_I):.3f} deg")
    print(f"Inertial velocity heading: {np.rad2deg(heading):.2f} deg clockwise from north")
    print(f"Planner: {duration/DAY_TO_SEC:.3f} days, {dv*1000:.2f} m/s, {fuel:.3f} kg propellant")
    sim = Simulator([sat], perturbations=("J2", "drag"), rtol=1e-9, atol=1e-11)
    times, states = sim.propagate(
        (0, duration), t_eval=np.linspace(0, duration, NSTEPS),
        max_step=120, min_altitude=120,
    )
    if times[-1] < duration:
        print(f"Stopped at 120 km altitude after {times[-1]/DAY_TO_SEC:.3f} days; SSO not reached.")
    final = cart_to_coe(states[0, :3, -1], states[0, 3:6, -1], MU_EARTH)
    target_node = np.pi/2+SUN_RATE*times[-1]
    target = (TARGET_A, 0, TARGET_I, target_node, 0, 0)
    # No along-track slot requirement: mark the nearest phase on the target circle.
    p, _ = coe_to_cart(target, MU_EARTH)
    q, _ = coe_to_cart((*target[:5], np.pi/2), MU_EARTH)
    phase = np.arctan2(states[0, :3, -1]@q, states[0, :3, -1]@p)
    target_position, _ = coe_to_cart((*target[:5], phase), MU_EARTH)
    name = "cape_canaveral_sso" + ("" if eclipses else "_no_eclipse")
    summary(times, states, [sat], name=name, show=False, plot_endpoints=True,
            target_elements=np.array([j2_reference(
                (TARGET_A, 0, TARGET_I, np.pi/2, 0, 0), times)]))
    plot_target_constellation(
        states[:, :3, -1], [target_position], [target], name=name,
        title="600 km dawn-dusk SSO target and final osculating orbit",
        final_velocities=states[:, 3:6, -1],
    )
    print(f"Propagated: a-R_E={final[0]-R_EARTH:.2f} km, e={final[1]:.5f}, "
          f"i={np.rad2deg(final[2]):.3f} deg")
    print(f"RAAN error={np.rad2deg(wrap_to_pi(final[3]-target_node)):.3f} deg, "
          f"fuel={500-states[0, 6, -1]:.3f} kg")
    finish_plots(show)
    return times, states


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--eclipses", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    run(show=not args.no_show, eclipses=args.eclipses)
