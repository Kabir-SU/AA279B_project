"""Paper-style deployment: 500 km injection -> 600 km drift -> 700 km target.

Run: python -m scenarios.low_thrust_drift
Also propagates the planned thrust schedule with J2, drag, and eclipses.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from astro import cart_to_coe
from constants import DAY_TO_SEC, G0, R_EARTH, MU_EARTH, OMEGA_EARTH, J2
from dynamics import density
from planner import algorithm_1, algorithm_2, make_w_ecl
from plots import summary
from sim import Satellite, Simulator, Thruster


def run(show=True):
    mass = 500.0  # kg
    isp = 1500.0  # s
    inclination = np.deg2rad(55.0)
    # Match the atmosphere table at 600 km: rho = 1.454e-13 kg/m^3.
    # Planner uses rho_0 * exp(-altitude/H_0), with density in kg/km^3.
    scale_height = 71.835  # km
    rho_0 = 1.454e-13 * 1e9 * np.exp(600.0 / scale_height)
    common = {
        "t_i": 0.0,
        "a_i": R_EARTH + 500.0,
        "a_f": R_EARTH + 700.0,
        "i_i": inclination, "i_f": inclination,
        "Omega_i": 0.0,
        "m_i": mass, "I_sp": isp, "T_max": 0.1,  # thrust [N]
        "rho_0": rho_0, "H_0": scale_height,
        "C_D": 2.2, "S": 5e-6, "N": 200,        # area [km^2]
    }

    # Fixed Sun direction is an illustrative geometry, not a solar ephemeris.
    eclipse = make_w_ecl(lambda t: [1.0, 0.0, 0.0])
    print("500 kg spacecraft, 0.1 N thrust, 1500 s Isp; drag on")
    print("Target RAAN starts 5 deg behind injection; drift altitude = 600 km")
    print(f"{'Case':<26} {'Delta-v [m/s]':>14} {'Time [days]':>13} {'Fuel [kg]':>11}")
    for label, sunlight in [("Full sunlight", lambda *args: 1.0),
                            ("Fixed-Sun eclipses", eclipse)]:
        dv, duration = algorithm_2(
            **common, Omega_c_i=np.deg2rad(-5.0),
            a_D=R_EARTH + 600.0, i_D=inclination, w_ecl=sunlight,
        )
        fuel = mass * (-np.expm1(-dv * 1000.0 / (isp * G0)))
        print(f"{label:<26} {dv*1000:14.3f} {duration/DAY_TO_SEC:13.3f} {fuel:11.3f}")

    # Algorithm 1 alone raises altitude but does not target the final RAAN.
    dv, duration, node_change = algorithm_1(**common, w_ecl=eclipse)
    print(f"\nAltitude-only comparison (Algorithm 1): {dv*1000:.3f} m/s, "
          f"{duration/DAY_TO_SEC:.3f} days, RAAN change {np.rad2deg(node_change):.3f} deg")

    dv, duration, plan = algorithm_2(
        **common, Omega_c_i=np.deg2rad(-5.0),
        a_D=R_EARTH + 600.0, i_D=inclination, w_ecl=eclipse,
        return_details=True,
    )
    return propagate_plan(common, plan, dv, duration, show)


def propagate_plan(common, plan, planned_dv, duration, show):
    """Execute the planner's open-loop schedule; endpoint differences are expected."""
    t1, t2, t3 = plan["durations"]
    boundaries = np.array([0.0, t1, t1+t2, t1+t2+t3])

    sunlight = make_w_ecl(lambda t: [1, 0, 0])

    def command(t, state):
        r, v, mass = state[:3], state[3:6], state[6]
        radial = r / np.linalg.norm(r)
        normal = np.cross(r, v)
        normal /= np.linalg.norm(normal)
        tangent = np.cross(normal, radial)
        # Same fixed Sun (+x) and cylindrical shadow as the averaged planner.
        shadow = r[0] < 0 and np.linalg.norm(r[1:]) < R_EARTH
        if shadow or t >= boundaries[-1]:
            return 0.0, tangent
        if t1 <= t < t1+t2:
            # Offset rotating-atmosphere drag during sunlit drift intervals.
            v_rel = v - np.cross([0, 0, OMEGA_EARTH], r)
            force = .5*density(np.linalg.norm(r)-R_EARTH)*common["C_D"]
            force *= common["S"]*1e6 * (np.linalg.norm(v_rel)*1000)**2
            node = np.arctan2(normal[0], -normal[1])
            inc = np.arccos(np.clip(normal[2], -1, 1))
            a = cart_to_coe(r, v, MU_EARTH)[0]
            throttle = force / (common["T_max"]*sunlight(t, a, inc, node))
            return throttle, v_rel
        leg = plan["leg1"] if t < t1 else plan["leg3"]
        local_time = t if t < t1 else t-(t1+t2)
        beta = np.interp(local_time, leg["t"]-leg["t"][0], leg["beta"])
        # Edelbaum normal steering reverses at maximum/minimum latitude.
        node_vector = np.cross([0, 0, 1], normal)
        cos_u = np.dot(radial, node_vector) / np.linalg.norm(node_vector)
        change_i = leg["i"][-1]-leg["i"][0]
        direction = np.cos(beta)*tangent + np.sin(beta)*np.sign(change_i)*np.sign(cos_u)*normal
        return 1.0, direction

    sat = Satellite.from_elements(
        (common["a_i"], 0, common["i_i"], common["Omega_i"], 0, 0),
        mass=common["m_i"], area=common["S"]*1e6, cd=common["C_D"],
        name="deployment", thruster=Thruster(common["T_max"], common["I_sp"]),
        dry_mass=450.0, thrust_command=command,
    )
    sim = Simulator([sat], perturbations=("J2", "drag"), rtol=1e-9, atol=1e-11)
    print("\nPropagating the planned schedule with instantaneous eclipse switching...")
    times, states = sim.propagate((0, duration), t_eval=np.linspace(0, duration, 3000), max_step=120)
    elements = np.array([cart_to_coe(s[:3], s[3:6], MU_EARTH) for s in states[0].T])
    node = np.unwrap(elements[:, 3])
    target_rate = -1.5*J2*np.sqrt(MU_EARTH/common["a_f"]**3)*(R_EARTH/common["a_f"])**2*np.cos(common["i_f"])
    target_node = np.deg2rad(-5)+target_rate*times
    fuel = common["m_i"]-states[0, 6, -1]
    expected_fuel = common["m_i"]*(-np.expm1(-planned_dv*1000/(common["I_sp"]*G0)))
    error_node = np.arctan2(np.sin(node[-1]-target_node[-1]), np.cos(node[-1]-target_node[-1]))
    print(f"Final altitude from a: {elements[-1, 0]-R_EARTH:.3f} km (target 700 km)")
    print(f"Final inclination: {np.rad2deg(elements[-1, 2]):.4f} deg (target 55 deg)")
    print(f"Final RAAN error: {np.rad2deg(error_node):.4f} deg")
    print(f"Fuel: propagated {fuel:.3f} kg; planner {expected_fuel:.3f} kg")

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    days = times/DAY_TO_SEC
    axes[0].plot(days, elements[:, 0]-R_EARTH, label="Propagated (osculating)")
    axes[0].axhline(700, color="k", ls=":", label="Target")
    for leg, start in [(plan["leg1"], 0), (plan["leg3"], t1+t2)]:
        axes[0].plot((leg["t"]-leg["t"][0]+start)/DAY_TO_SEC, leg["a"]-R_EARTH, "--", color="tab:orange")
    axes[0].plot(np.array([t1, t1+t2])/DAY_TO_SEC, [600, 600], "--", color="tab:orange", label="Planner")
    axes[0].set_ylabel("a − Earth radius [km]")
    axes[0].legend()
    axes[1].plot(days, np.rad2deg(elements[:, 2]))
    axes[1].axhline(55, color="k", ls=":")
    axes[1].set_ylabel("Inclination [deg]")
    axes[2].plot(days, np.rad2deg(node), label="Propagated")
    axes[2].plot(days, np.rad2deg(target_node), "--", label="Target plane")
    axes[2].set_ylabel("RAAN [deg]")
    axes[2].legend()
    axes[3].plot(days, states[0, 6])
    axes[3].axhline(common["m_i"]-expected_fuel, color="k", ls=":", label="Planner final mass")
    axes[3].set_ylabel("Mass [kg]")
    axes[3].set_xlabel("Time [days]")
    axes[3].legend()
    for ax in axes:
        for boundary in boundaries[1:-1]:
            ax.axvline(boundary/DAY_TO_SEC, color="gray", ls=":", alpha=.5)
        ax.grid(alpha=.2)
    fig.suptitle("Planned low-thrust deployment: thrust / drift / thrust")
    fig.tight_layout()
    output = Path(__file__).parent / "plots" / "low_thrust_drift"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "planner_vs_propagation.png", dpi=150)
    summary(times, states, [sat], name="low_thrust_drift",
            show=show, plot_endpoints=True)
    if not show:
        plt.close("all")
    return times, states


if __name__ == "__main__":
    run()
