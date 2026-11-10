#!/usr/bin/env python3
"""Validation figure for the forced Poiseuille channel (apps/poiseuille).

Usage: .venv/bin/python scripts/plot_poiseuille.py [--input-dir output]
           [--output docs/images/poiseuille.png] [--H 16] [--nu 0.1666667]
"""
import argparse
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

U_MAX = 0.01  # as in apps/poiseuille (kUmax)
LAMBDAS = (3.0 / 16.0, 1.0 / 4.0)
EPS = float(np.finfo(np.float64).eps)


def fmt(x, digits=6):
    """Exact simple fraction (denominator <= 64, exact to 1e-12) or decimal."""
    fr = Fraction(x).limit_denominator(64)
    if abs(float(fr) - x) <= 1e-12 and fr.denominator > 1:
        return f"{fr.numerator}/{fr.denominator}"
    return f"{x:.{digits}g}"


def load(path):
    if not path.is_file():
        sys.exit(f"error: missing input file {path} "
                 "(run the poiseuille app first, or check --input-dir/--H/--nu)")
    data = np.genfromtxt(path, delimiter=",", names=True)
    return data["y"], data["u_numeric"], data["u_analytic"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input-dir", default="output")
    ap.add_argument("--output", default="docs/images/poiseuille.png")
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--nu", type=float, default=1.0 / 6.0)
    args = ap.parse_args()
    H, nu = args.H, args.nu

    fx = 8.0 * nu * U_MAX / (H * H)
    colors = ("tab:blue", "tab:red")
    markers = ("o", "s")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    ax_a, ax_b, ax_c = axes
    analytic_drawn = False
    res_min, res_max = np.inf, 0.0
    summary = []

    for lam, col, mk in zip(LAMBDAS, colors, markers):
        # File names exactly as the app writes them (%.6g).
        name = f"poiseuille_H{H}_nu{nu:.6g}_lambda{lam:.6g}.csv"
        y, un, ua = load(Path(args.input_dir) / name)
        # Slip prediction of the TRT halfway bounce-back wall.
        du = fx / (2.0 * nu) * (16.0 * lam - 3.0) / 12.0
        label = rf"$\Lambda$ = {fmt(lam)}"

        if not analytic_drawn:
            yy = np.linspace(0.5, H + 0.5, 400)
            ax_a.plot(yy, fx / (2.0 * nu) * (yy - 0.5) * (H + 0.5 - yy) / U_MAX,
                      "k-", lw=1.2, label="analytical parabola")
            analytic_drawn = True
        ax_a.plot(y, un / U_MAX, mk, color=col, mfc="none", label=f"numerical, {label}")

        dev = (un - ua) / U_MAX
        ax_b.plot(y, dev, mk, color=col, mfc="none", label=f"numerical, {label}")
        ax_b.axhline(du / U_MAX, color=col, ls="--", lw=1.2,
                     label=f"predicted $\\Delta u/u_{{max}}$ = {fmt(du / U_MAX, 4)}")

        res = np.abs(un - (ua + du)) / U_MAX
        res_min, res_max = min(res_min, res.min()), max(res_max, res.max())
        # Exact zeros cannot be drawn on a log axis (and would make matplotlib
        # drop or mask them silently): plot them at EPS/100, below the
        # reference line, so they remain visible as "zero residual".
        res_plot = np.where(res > 0.0, res, EPS / 100.0)
        ax_c.semilogy(y, res_plot, mk, color=col, mfc="none", label=label)
        summary.append((lam, du / U_MAX, dev.mean(), dev.min(), dev.max(),
                        res.min(), res.max()))

    ax_c.axhline(EPS, color="gray", ls=":", lw=1.2, label=r"double epsilon $2.2\times10^{-16}$")

    ax_a.set(xlabel="y (lattice nodes)", ylabel=r"$u_x / u_{max}$", title="a. Velocity profiles")
    ax_b.set(xlabel="y (lattice nodes)", ylabel=r"$(u_{num} - u_{analytic}) / u_{max}$",
             title="b. Deviation from the parabola")
    ax_c.set(xlabel="y (lattice nodes)",
             ylabel=r"$|u_{num} - (u_{analytic} + \Delta u)| / u_{max}$",
             title="c. Residual against the slip formula")
    ax_c.set_ylim(EPS / 1000.0, max(res_max, EPS) * 100.0)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    ax_b.margins(y=0.3)

    fig.suptitle(f"Poiseuille channel: D2Q9 TRT, Guo force, halfway bounce-back, "
                 f"H = {H}, $\\nu$ = {fmt(nu)}, double precision", fontsize=11)
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    for lam, pred, mean, lo, hi, rmin, rmax in summary:
        print(f"lambda={lam:g}: predicted du/umax={pred:.6e} observed mean={mean:.6e} "
              f"min={lo:.6e} max={hi:.6e} residual range=[{rmin:.3e}, {rmax:.3e}]")


if __name__ == "__main__":
    main()
