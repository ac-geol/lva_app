"""Equal-area (Schmidt) lower-hemisphere stereonets, matplotlib only.

Deliberately outside core/ -- the engine stays plotting-free. Projection is the
same one the prototype notebook used, so figures remain comparable.
"""
from __future__ import annotations

import numpy as np


def line_to_equal_area_xy(trend_deg, plunge_deg):
    trend = np.radians(np.asarray(trend_deg, dtype=float))
    plunge = np.radians(np.asarray(plunge_deg, dtype=float))
    colat = 0.5 * np.pi - plunge
    r = np.sqrt(2.0) * np.sin(colat / 2.0)
    return r * np.sin(trend), r * np.cos(trend)


def draw_frame(ax, title=None, subtitle=None):
    import matplotlib.pyplot as plt
    ax.add_patch(plt.Circle((0, 0), np.sqrt(2), fill=False, lw=1.1, color='0.35'))
    for ang in range(0, 360, 30):
        a = np.radians(ang)
        ax.plot([1.36 * np.sin(a), np.sqrt(2) * np.sin(a)],
                [1.36 * np.cos(a), np.sqrt(2) * np.cos(a)],
                lw=0.6, color='0.7')
    ax.text(0, 1.52, 'N', ha='center', va='center', fontsize=8, color='0.35')
    ax.set_aspect('equal')
    ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.65, 1.65)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9.5, pad=10)
    if subtitle:
        # Drawn INSIDE the axes, in the margin between the circle (r=sqrt2) and
        # the axis limit. set_aspect('equal') shrinks this axes within whatever
        # slot the layout engine gives it, and an xlabel placed outside the box
        # then lands unpredictably -- at three panels it collided with the title
        # of the row below. A caption in data coordinates moves with the axes.
        ax.text(0, -1.57, subtitle, ha='center', va='center', fontsize=7.5,
                color='0.35')


def density_contour(ax, trend, plunge, *, gridsize=180, sigma=0.09, levels=8,
                    cmap='magma'):
    """Kernel density on the projected disc.

    Smoothing happens in projection space, which is adequate for comparing
    panels drawn identically; it is not a substitute for a Kamb contour.
    """
    x, y = line_to_equal_area_xy(trend, plunge)
    lim = np.sqrt(2.0)
    gx = np.linspace(-lim, lim, gridsize)
    gxx, gyy = np.meshgrid(gx, gx)
    inside = gxx ** 2 + gyy ** 2 <= lim ** 2

    dens = np.zeros_like(gxx)
    flat = np.stack([gxx[inside], gyy[inside]], axis=1)
    chunk = 20000
    acc = np.zeros(len(flat))
    for k in range(0, len(x), chunk):
        dx = flat[:, 0][:, None] - x[k:k + chunk][None, :]
        dy = flat[:, 1][:, None] - y[k:k + chunk][None, :]
        acc += np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2)).sum(axis=1)
    dens[inside] = acc / max(len(x), 1)

    dens_masked = np.where(inside, dens, np.nan)
    ax.contourf(gxx, gyy, dens_masked, levels=levels, cmap=cmap)
    return dens_masked


def scatter(ax, trend, plunge, *, s=3, alpha=0.25, color='#1f4e79'):
    x, y = line_to_equal_area_xy(trend, plunge)
    ax.scatter(x, y, s=s, alpha=alpha, color=color, linewidths=0)
