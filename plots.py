"""Standardized plot panels for constellation propagator output.

All public panel functions share the signature

    plot_X(times, states, sats, name="")

where `times` is (M,), `states` is (N, 6, M), `sats` is the Satellite list.
`summary()` runs all four panels, saves PNGs under
    scenarios/plots/<name>/,
and (optionally) shows the figures interactively.
"""

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

from astro import cart_to_coe
from constants import MU_EARTH, R_EARTH, OMEGA_EARTH

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

def plot_trajectories_3d(times, states, sats, name=""):
    """3D ECI trajectories around an Earth sphere, colored by Walker plane."""
    colors, planes, n_planes = _plane_colors(sats)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    plot_earth(ax)

    labeled_planes = set()
    for i, _ in enumerate(sats):
        r = states[i, 0:3, :]
        label = None
        if planes[i] is not None and planes[i] not in labeled_planes:
            label = f"plane {planes[i]}"
            labeled_planes.add(planes[i])
        ax.plot(r[0], r[1], r[2], color=colors[i], linewidth=0.8, label=label)

    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("z [km]")
    ax.set_title(f"ECI trajectories ({len(sats)} sats, {times[-1] / 3600:.1f} h){' - ' + name if name else ''}")
    set_axes_equal(ax)
    if labeled_planes:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    _savefig(fig, name, "trajectories")
    return fig


def plot_elements(times, states, sats, name=""):
    """Classical orbital elements vs time (2x3 grid)."""
    colors, planes, _ = _plane_colors(sats)

    N, _, M = states.shape
    elems = np.empty((N, 6, M))
    for i in range(N):
        for k in range(M):
            elems[i, :, k] = cart_to_coe(states[i, :3, k], states[i, 3:, k], MU_EARTH)

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
    for ax in axes[-1]:
        ax.set_xlabel("time [h]")

    fig.suptitle(f"Classical orbital elements{' - ' + name if name else ''}")
    fig.tight_layout()
    _savefig(fig, name, "elements")
    return fig


def plot_altitude(times, states, sats, name=""):
    """Instantaneous altitude |r| - R_E vs time, one line per sat."""
    colors, _, _ = _plane_colors(sats)
    N = states.shape[0]
    t_h = times / 3600.0

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(N):
        r_norm = np.linalg.norm(states[i, :3, :], axis=0)
        ax.plot(t_h, r_norm - R_EARTH, color=colors[i], linewidth=0.6)
    ax.set_xlabel("time [h]")
    ax.set_ylabel("altitude [km]")
    ax.set_title(f"Altitude vs time{' - ' + name if name else ''}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, name, "altitude")
    return fig


def plot_ground_tracks(times, states, sats, name=""):
    """Ground tracks on a rectangular lat/lon map (simple ECEF via -omega*t rotation)."""
    colors, planes, _ = _plane_colors(sats)
    N = states.shape[0]

    gmst = OMEGA_EARTH * times
    cos_g = np.cos(gmst)
    sin_g = np.sin(gmst)

    fig, ax = plt.subplots(figsize=(12, 6))
    labeled_planes = set()
    for i in range(N):
        x_e = cos_g * states[i, 0, :] + sin_g * states[i, 1, :]
        y_e = -sin_g * states[i, 0, :] + cos_g * states[i, 1, :]
        z_e = states[i, 2, :]
        r_norm = np.sqrt(x_e**2 + y_e**2 + z_e**2)
        lat = np.rad2deg(np.arcsin(z_e / r_norm))
        lon = np.rad2deg(np.arctan2(y_e, x_e))

        # Break track at longitude wraparound so matplotlib doesn't draw straight lines.
        jumps = np.where(np.abs(np.diff(lon)) > 180.0)[0]
        lat_p = np.insert(lat, jumps + 1, np.nan)
        lon_p = np.insert(lon, jumps + 1, np.nan)

        label = None
        if planes[i] is not None and planes[i] not in labeled_planes:
            label = f"plane {planes[i]}"
            labeled_planes.add(planes[i])
        ax.plot(lon_p, lat_p, color=colors[i], linewidth=0.5, label=label)

    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-90.0, 90.0)
    ax.set_xticks(np.arange(-180.0, 181.0, 30.0))
    ax.set_yticks(np.arange(-90.0, 91.0, 30.0))
    ax.set_xlabel("longitude [deg]")
    ax.set_ylabel("latitude [deg]")
    ax.set_title(f"Ground tracks{' - ' + name if name else ''}")
    ax.grid(True, alpha=0.3)
    if labeled_planes:
        ax.legend(loc="lower left", fontsize=8, ncol=min(len(labeled_planes), 6))
    fig.tight_layout()
    _savefig(fig, name, "ground_tracks")
    return fig


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def summary(times, states, sats, name="", show=True):
    """Run every panel, save PNGs under scenarios/plots/<name>/, optionally show."""
    plot_trajectories_3d(times, states, sats, name)
    plot_elements(times, states, sats, name)
    plot_altitude(times, states, sats, name)
    plot_ground_tracks(times, states, sats, name)
    if name:
        print(f"[plots] saved to {PLOT_DIR / name}/")
    if show:
        plt.show()
