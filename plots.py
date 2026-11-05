"""Standardized plot panels for constellation propagator output.

The standard propagation panels share the signature

    plot_X(times, states, sats, name="")

where `times` is (M,), `states` is (N, N_STATES, M), including mass at index 6, `sats` is the Satellite list.
`summary()` runs all four panels, saves PNGs under
    scenarios/plots/<name>/,
and (optionally) shows the figures interactively. Planner comparisons and
constellation endpoint comparisons are available as separate panels.
"""

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

from astro import cart_to_coe, coe_to_cart
from constants import MU_EARTH, R_EARTH, OMEGA_EARTH, VELOCITY_SLICE, DAY_TO_SEC

PLOT_DIR = Path(__file__).parent / "scenarios" / "plots"
_WALKER_NAME = re.compile(r"^w(\d+)_(\d+)$")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def plot_earth(ax, n=60, alpha=0.25):
    """Translucent Earth sphere on a 3D axes."""
    u = np.linspace(0.0, 2.0 * np.pi, n)
    v = np.linspace(0.0, np.pi, n)
    x = R_EARTH * np.outer(np.cos(u), np.sin(v))
    y = R_EARTH * np.outer(np.sin(u), np.sin(v))
    z = R_EARTH * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, alpha=alpha, linewidth=0, color="steelblue")


def set_axes_equal(ax):
    """Make 3D axes have equal scale so orbits aren't visually distorted."""
    xlim, ylim, zlim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()
    xm, ym, zm = np.mean(xlim), np.mean(ylim), np.mean(zlim)
    half = 0.5 * max(xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0])
    ax.set_xlim3d(xm - half, xm + half)
    ax.set_ylim3d(ym - half, ym + half)
    ax.set_zlim3d(zm - half, zm + half)
    ax.set_box_aspect((1, 1, 1))


def _plane_colors(sats):
    """Return a per-satellite color list. If every name matches wP_S, group by plane;
    otherwise color continuously by satellite index."""
    planes = []
    for s in sats:
        m = _WALKER_NAME.match(s.name or "")
        planes.append(int(m.group(1)) if m else None)
    if None not in planes:
        n_planes = max(planes) + 1
        cmap = plt.get_cmap("tab10" if n_planes <= 10 else "tab20")
        return [cmap(p % cmap.N) for p in planes], planes, n_planes
    cmap = plt.get_cmap("viridis")
    N = len(sats)
    return [cmap(i / max(N - 1, 1)) for i in range(N)], [None] * N, 0


def _savefig(fig, name, panel):
    if not name:
        return
    out_dir = PLOT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{panel}.png", dpi=120, bbox_inches="tight")


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

def plot_trajectories_3d(times, states, sats, name="", plot_endpoints=False, *, target_elements=None):
    """3D ECI trajectories; optionally mark starts with Xs and ends with dots."""
    colors, planes, n_planes = _plane_colors(sats)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    plot_earth(ax)

    angle = np.linspace(0, 2*np.pi, 181)
    ax.plot(R_EARTH*np.cos(angle), R_EARTH*np.sin(angle), np.zeros_like(angle),
            "--", color="black", linewidth=1, label="Earth equator")
    ax.plot([0, 0], [0, 0], [-1.2*R_EARTH, 1.2*R_EARTH], ":", color="gray")
    ax.text(0, 0, 1.25*R_EARTH, "North (+z)", fontsize=8)

    labeled_planes = set()
    for i, _ in enumerate(sats):
        r = states[i, 0:3, :]
        label = None
        if planes[i] is not None and planes[i] not in labeled_planes:
            label = f"plane {planes[i]}"
            labeled_planes.add(planes[i])
        ax.plot(r[0], r[1], r[2], color=colors[i], linewidth=0.8, label=label)
        if plot_endpoints:
            ax.scatter(r[0, 0], r[1, 0], r[2, 0], color=colors[i],
                       marker="x", s=45, linewidths=1.5, depthshade=False,
                       label="starting position" if i == 0 else None)
            ax.scatter(r[0, -1], r[1, -1], r[2, -1], color=colors[i],
                       s=35, edgecolors="black", linewidths=0.5, depthshade=False,
                       label="final position" if i == 0 else None)
        if len(sats) == 1:
            start = cart_to_coe(r[:, 0], states[i, VELOCITY_SLICE, 0], MU_EARTH)
            end = cart_to_coe(r[:, -1], states[i, VELOCITY_SLICE, -1], MU_EARTH)
            ax.text2D(.02, .95, f"Inclination: {np.rad2deg(start[2]):.3f}° → "
                      f"{np.rad2deg(end[2]):.3f}°\nMeasured from Earth's equator",
                      transform=ax.transAxes, fontsize=9)
            velocity = states[i, VELOCITY_SLICE, -1]
            arrow = velocity/np.linalg.norm(velocity)*.2*R_EARTH
            ax.quiver(*r[:, -1], *arrow, color=colors[i], linewidth=2,
                      arrow_length_ratio=.3, label="Final velocity direction")

    if target_elements is not None:
        for j, reference in enumerate(target_elements):
            elements = reference[:, -1]
            anomalies = np.sort(np.r_[np.linspace(0, 2*np.pi, 361), elements[5]])
            ring = np.array([coe_to_cart((*elements[:5], u), MU_EARTH)[0] for u in anomalies])
            ax.plot(*ring.T, "--", color=colors[j % len(colors)], alpha=.6,
                    label="Target orbit at final time" if j == 0 else None)
            point, _ = coe_to_cart(elements, MU_EARTH)
            ax.scatter(*point, marker="+", color=colors[j % len(colors)], depthshade=False,
                       label="Reference position at final time" if j == 0 else None)

    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("z [km]")
    ax.set_title(f"ECI trajectories ({len(sats)} sats, {times[-1] / 3600:.1f} h){' - ' + name if name else ''}")
    set_axes_equal(ax)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    _savefig(fig, name, "trajectories")
    return fig


def plot_elements(times, states, sats, name="", *, target_elements=None):
    """Classical orbital elements vs time (2x3 grid)."""
    colors, planes, _ = _plane_colors(sats)

    N, _, M = states.shape
    elems = np.empty((N, 6, M))
    for i in range(N):
        for k in range(M):
            elems[i, :, k] = cart_to_coe(states[i, :3, k], states[i, VELOCITY_SLICE, k], MU_EARTH)

    t_h = times / 3600.0

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    titles = ["a [km]", "e [-]", "inc [deg]", "RAAN [deg]", "arg. peri. [deg]", "M [deg]"]
    # RAAN is unwrapped so J2 secular drift is visible.
    # arg_peri and M are left wrapped: for near-circular orbits arg_peri is degenerate
    # and unwrapping its numerical noise produces a bogus secular trend.
    data = [elems[:, 0, :], elems[:, 1, :],
            np.rad2deg(elems[:, 2, :]),
            np.rad2deg(np.unwrap(elems[:, 3, :], axis=1)),
            np.rad2deg(elems[:, 4, :]),
            np.rad2deg(elems[:, 5, :])]

    for ax, title, d in zip(axes.flat, titles, data):
        for i in range(N):
            ax.plot(t_h, d[i], color=colors[i], linewidth=0.6)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
    if target_elements is not None:
        for j, reference in enumerate(target_elements):
            values = [reference[0], reference[1], np.rad2deg(reference[2]),
                      np.rad2deg(np.unwrap(reference[3])),
                      np.rad2deg(reference[4] % (2*np.pi)),
                      np.rad2deg(reference[5] % (2*np.pi))]
            for ax, value in zip(axes.flat, values):
                ax.plot(t_h, value, "--", color=colors[j % N], linewidth=1,
                        label="Target at time t" if j == 0 else None)
            # Periapsis is undefined for circular targets; avoid suggesting
            # that the arbitrary zero reference is a tracking constraint.
            if np.all(reference[1] < 1e-10):
                axes[1, 1].lines[-1].remove()
        axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time [h]")

    fig.suptitle(f"Classical orbital elements{' - ' + name if name else ''}")
    fig.tight_layout()
    _savefig(fig, name, "elements")
    return fig


def plot_altitude(times, states, sats, name="", *, target_states=None):
    """Instantaneous altitude |r| - R_E vs time, one line per sat."""
    colors, _, _ = _plane_colors(sats)
    N = states.shape[0]
    t_h = times / 3600.0

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(N):
        r_norm = np.linalg.norm(states[i, :3, :], axis=0)
        ax.plot(t_h, r_norm - R_EARTH, color=colors[i], linewidth=0.6)
    if target_states is not None:
        for j, reference in enumerate(target_states):
            ax.plot(t_h, np.linalg.norm(reference[:3], axis=0)-R_EARTH, "--",
                    color=colors[j % N], label="Target at time t" if j == 0 else None)
        ax.legend(fontsize=8)
    ax.set_xlabel("time [h]")
    ax.set_ylabel("altitude [km]")
    ax.set_title(f"Altitude vs time{' - ' + name if name else ''}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, name, "altitude")
    return fig


def plot_ground_tracks(times, states, sats, name="", *, target_states=None):
    """Ground tracks on a rectangular lat/lon map (simple ECEF via -omega*t rotation)."""
    colors, planes, _ = _plane_colors(sats)
    N = states.shape[0]

    gmst = OMEGA_EARTH * times
    cos_g = np.cos(gmst)
    sin_g = np.sin(gmst)

    fig, ax = plt.subplots(figsize=(12, 6))
    labeled_planes = set()
    tracks = [(states[i], colors[i], False, i) for i in range(N)]
    if target_states is not None:
        tracks += [(ref, colors[j % N], True, j) for j, ref in enumerate(target_states)]
    for track, color, reference, i in tracks:
        x_e = cos_g * track[0] + sin_g * track[1]
        y_e = -sin_g * track[0] + cos_g * track[1]
        z_e = track[2]
        r_norm = np.sqrt(x_e**2 + y_e**2 + z_e**2)
        lat = np.rad2deg(np.arcsin(z_e / r_norm))
        lon = np.rad2deg(np.arctan2(y_e, x_e))

        # Break track at longitude wraparound so matplotlib doesn't draw straight lines.
        jumps = np.where(np.abs(np.diff(lon)) > 180.0)[0]
        lat_p = np.insert(lat, jumps + 1, np.nan)
        lon_p = np.insert(lon, jumps + 1, np.nan)

        label = None
        if reference:
            label = "Reference ground track" if i == 0 else None
        elif planes[i] is not None and planes[i] not in labeled_planes:
            label = f"plane {planes[i]}"
            labeled_planes.add(planes[i])
        ax.plot(lon_p, lat_p, color=color, linestyle="--" if reference else "-",
                linewidth=0.5, label=label)

    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xticks(np.arange(-180.0, 181.0, 30.0))
    ax.set_yticks(np.arange(-90.0, 91.0, 30.0))
    ax.set_xlabel("longitude [deg]")
    ax.set_ylabel("latitude [deg]")
    ax.set_title(f"Ground tracks{' - ' + name if name else ''}")
    ax.grid(True, alpha=0.3)
    if labeled_planes or target_states is not None:
        ax.legend(loc="lower left", fontsize=8, ncol=max(1, min(len(labeled_planes), 6)))
    fig.tight_layout()
    _savefig(fig, name, "ground_tracks")
    return fig


def plot_planner_comparison(times, states, elements, plan, common,
                            target_node, expected_fuel, name=""):
    """Compare a single satellite with its averaged thrust-drift-thrust plan."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    t1, t2, _ = plan["durations"]
    boundaries = np.array([0, t1, t1+t2, times[-1]])
    node = np.unwrap(elements[:, 3])
    days = times/DAY_TO_SEC
    axes[0].plot(days, elements[:, 0]-R_EARTH, label="Propagated (osculating)")
    axes[0].axhline(common["a_f"]-R_EARTH, color="k", ls=":", label="Target")
    for leg, start in [(plan["leg1"], 0), (plan["leg3"], t1+t2)]:
        axes[0].plot((leg["t"]-leg["t"][0]+start)/DAY_TO_SEC, leg["a"]-R_EARTH, "--", color="tab:orange")
    axes[0].plot(np.array([t1, t1+t2])/DAY_TO_SEC, [plan["leg1"]["a"][-1]-R_EARTH]*2, "--", color="tab:orange", label="Planner")
    axes[0].set_ylabel("a − Earth radius [km]")
    axes[0].legend()
    axes[1].plot(days, np.rad2deg(elements[:, 2]))
    axes[1].axhline(np.rad2deg(common["i_f"]), color="k", ls=":")
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
    _savefig(fig, name, "planner_vs_propagation")
    return fig


def plot_target_constellation(final_positions, target_positions, plane_elements,
                              name="", title="Target constellation and propagated endpoints",
                              final_velocities=None):
    """Show orbit rings and compare actual and target ECI endpoint positions.

    Position arrays have shape (N, 3); plane_elements contains (a, e, i, RAAN,
    argument of periapsis, M) tuples at the comparison epoch.
    Supply final_velocities to draw the actual final osculating orbits too.
    """
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    plot_earth(ax)
    for j, elements in enumerate(plane_elements):
        ring = np.array([coe_to_cart((*elements[:5], u), MU_EARTH)[0]
                         for u in np.linspace(0, 2*np.pi, 721)])
        ax.plot(*ring.T, alpha=.35, label="Target orbits" if j == 0 else None)
    if final_velocities is not None:
        for j, (r, v) in enumerate(zip(final_positions, final_velocities)):
            elements = cart_to_coe(r, v, MU_EARTH)
            # Include the endpoint's exact anomaly as a vertex of its orbit.
            anomalies = np.sort(np.r_[np.linspace(0, 2*np.pi, 721), elements[5]])
            ring = np.array([coe_to_cart((*elements[:5], u), MU_EARTH)[0]
                             for u in anomalies])
            ax.plot(*ring.T, "--", color="red", linewidth=.8, alpha=.6,
                    label="Final osculating orbits" if j == 0 else None)
    ax.scatter(*np.asarray(target_positions).T, marker="x", color="black",
               depthshade=False, label="Target slots")
    ax.scatter(*np.asarray(final_positions).T, color="red",
               depthshade=False, label="Propagated endpoints")
    ax.set(xlabel="ECI x [km]", ylabel="ECI y [km]", zlabel="ECI z [km]", title=title)
    ax.legend()
    set_axes_equal(ax)
    _savefig(fig, name, "target_constellation")
    return fig


def finish_plots(show=True):
    """Display the current figures, or close them for a noninteractive run."""
    if show:
        plt.show()
    else:
        plt.close("all")


def plot_control_effort(times, states, sats, name=""):
    """Sample commanded throttle and applied thrust acceleration (no gravity/drag).

    Commands are evaluated at saved states; narrow eclipse switches between
    saved samples are not resolved. Callbacks should be deterministic.
    """
    colors, _, _ = _plane_colors(sats)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for j, sat in enumerate(sats):
        throttle = np.zeros(len(times))
        acceleration = np.zeros(len(times))
        if sat.thrust_command is not None:
            for k, t in enumerate(times):
                state = states[j, :, k]
                if state[6] <= sat.dry_mass:
                    continue
                command, _ = sat.thrust_command(t, state.copy())
                force = sat.thruster.force(command)
                throttle[k] = force / sat.thruster.max_thrust
                acceleration[k] = force / state[6]  # N/kg = m/s^2
        axes[0].step(times/DAY_TO_SEC, throttle, where="post", color=colors[j],
                     linewidth=.7, label=sat.name or f"sat {j}")
        axes[1].plot(times/DAY_TO_SEC, acceleration, color=colors[j], linewidth=.7)
    axes[0].set_ylabel("Throttle [-]")
    axes[0].set_ylim(-.05, 1.05)
    if len(sats) <= 8:
        axes[0].legend(fontsize=8)
    axes[1].set_ylabel("Thrust acceleration [m/s²]")
    axes[1].set_xlabel("Time [days]")
    for ax in axes:
        ax.grid(alpha=.3)
    fig.suptitle(f"Sampled control effort{' - ' + name if name else ''}")
    fig.tight_layout()
    _savefig(fig, name, "control_effort")
    return fig


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def summary(times, states, sats, name="", show=True, plot_endpoints=False, *, target_elements=None):
    """Save/show panels, optionally overlaying time-dependent target orbits.

    target_elements is (N_targets, 6, len(times)), in km/radians, sampled at
    the same epochs as states. Use astro.j2_reference() for secular J2 motion
    or provide your own target ephemeris. None preserves the original plots.
    The 3D target rings/markers are at times[-1]; other references evolve
    with time. Reference phase is only a slot requirement if you specify one.
    """
    target_states = None
    if target_elements is not None:
        target_elements = np.asarray(target_elements)
        target_states = np.array([[np.concatenate(coe_to_cart(ref[:, k], MU_EARTH))
                                   for k in range(len(times))] for ref in target_elements]).transpose(0, 2, 1)
    plot_trajectories_3d(times, states, sats, name, plot_endpoints=plot_endpoints,
                         target_elements=target_elements)
    plot_elements(times, states, sats, name, target_elements=target_elements)
    plot_altitude(times, states, sats, name, target_states=target_states)
    plot_ground_tracks(times, states, sats, name, target_states=target_states)
    if any(sat.thrust_command is not None for sat in sats):
        plot_control_effort(times, states, sats, name)
    if name:
        print(f"[plots] saved to {PLOT_DIR / name}/")
    if show:
        plt.show()
