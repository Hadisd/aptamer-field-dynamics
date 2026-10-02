#!/usr/bin/env python3
"""
plot_thermodynamics_abcd.py
===========================
Ultra-clean, publication-quality 4-panel (2x2) thermodynamic dashboard (Panels A, B, C, D).
Fixes time alignment so Phase 2 starts seamlessly at t = 100.0 ns (no 1.5 ns gap).

Output:
  - analysis/plots/md_thermodynamic_diagnostics_abcd.png
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent.parent
PLOTS_DIR = BASE / "analysis" / "plots"
RUN_P2 = BASE / "run" / "prod_phase2_rev_wall"

def read_xvg(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('@', '#')):
                continue
            parts = [float(x) for x in line.split()]
            data.append(parts)
    return np.array(data)

def moving_avg(arr, w=40):
    pad = w // 2
    padded = np.pad(arr, (pad, pad), mode='edge')
    return np.convolve(padded, np.ones(w)/w, mode='valid')[:len(arr)]

def main():
    prod1_energy = read_xvg(PLOTS_DIR / "prod_energy.xvg")
    prod2_energy = read_xvg(RUN_P2 / "prod_energy.xvg")

    t_p1 = prod1_energy[:, 0] / 1000.0  # 0 to 100.0 ns
    # Seamless Phase 2 time alignment starting exactly at 100.0 ns
    t_p2 = 100.0 + ((prod2_energy[:, 0] - prod2_energy[0, 0]) / 1000.0)

    # Drop step 0 from Phase 2 arrays to remove single-frame unintegrated checkpoint artifact
    t_p2 = t_p2[1:]
    prod2_clean = prod2_energy[1:]

    t_prod = np.concatenate([t_p1, t_p2])

    pot_prod = np.concatenate([prod1_energy[:, 2], prod2_clean[:, 2]]) / 1e5
    tot_prod = np.concatenate([prod1_energy[:, 3], prod2_clean[:, 3]]) / 1e5
    temp_prod = np.concatenate([prod1_energy[:, 4], prod2_clean[:, 4]])
    press_prod = np.concatenate([prod1_energy[:, 5], prod2_clean[:, 5]])
    dens_prod = np.concatenate([prod1_energy[:, 7], prod2_clean[:, 7]])

    # Modern refined typography with normal non-bold weights
    plt.rcParams.update({
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.family': 'sans-serif',
        'font.weight': 'normal',
        'font.size': 11,
        'axes.labelweight': 'normal',
        'axes.titleweight': 'normal',
        'axes.labelsize': 11.5,
        'axes.titlesize': 12.5,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9.5,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    plt.subplots_adjust(hspace=0.28, wspace=0.22)

    def apply_clean_style(ax):
        ax.set_facecolor('#ffffff')
        ax.set_xlim(0, 150)
        ax.axvline(100.0, color='#94a3b8', linestyle='--', lw=0.9, alpha=0.7)
        ax.grid(True, linestyle=':', alpha=0.35, color='#cbd5e1')

    # -------------------------------------------------------------
    # Panel A: Energy
    # -------------------------------------------------------------
    ax = axes[0, 0]
    apply_clean_style(ax)
    ax.plot(t_prod, pot_prod, color='#2563eb', lw=1.1, label='Potential Energy')
    ax.plot(t_prod, tot_prod, color='#d97706', lw=1.1, label='Total Energy')
    ax.set_title("A. Energy", loc='left')
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(r"Energy ($10^5$ kJ/mol)")
    ax.set_ylim(-5.3, -3.8)
    ax.legend(loc='upper left', frameon=False)

    # -------------------------------------------------------------
    # Panel B: Temperature
    # -------------------------------------------------------------
    ax = axes[0, 1]
    apply_clean_style(ax)
    ax.plot(t_prod, temp_prod, color='#fda4af', alpha=0.45, lw=0.5)
    temp_smooth = moving_avg(temp_prod, 50)
    ax.plot(t_prod, temp_smooth, color='#be123c', lw=1.5, label='Temperature')
    ax.axhline(300.0, color='#0f172a', linestyle=':', lw=1.1, label='300 K Target')
    ax.set_title("B. Temperature", loc='left')
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Temperature (K)")
    ax.set_ylim(290, 310)
    ax.legend(loc='upper left', frameon=False)

    # -------------------------------------------------------------
    # Panel C: Density
    # -------------------------------------------------------------
    ax = axes[1, 0]
    apply_clean_style(ax)
    ax.plot(t_prod, dens_prod, color='#047857', alpha=0.35, lw=0.5)
    dens_smooth = moving_avg(dens_prod, 40)
    ax.plot(t_prod, dens_smooth, color='#047857', lw=1.6, label='Density (0.15 M NaCl)')
    ax.axhline(1012.5, color='#0f172a', linestyle=':', lw=1.1, label=r'Equilibrium (1012.5 kg/m$^3$)')
    ax.set_title("C. Density", loc='left')
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel(r"Density (kg/m$^3$)")
    ax.set_ylim(998, 1026)
    ax.legend(loc='upper left', frameon=False)

    # -------------------------------------------------------------
    # Panel D: Pressure
    # -------------------------------------------------------------
    ax = axes[1, 1]
    apply_clean_style(ax)
    ax.plot(t_prod, press_prod, color='#94a3b8', alpha=0.30, lw=0.4)
    press_smooth = moving_avg(press_prod, 60)
    ax.plot(t_prod, press_smooth, color='#4338ca', lw=1.6, label='Pressure (Moving Avg)')
    ax.axhline(1.0, color='#e11d48', linestyle=':', lw=1.1, label='1.0 bar Target')
    ax.set_title("D. Pressure", loc='left')
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Pressure (bar)")
    ax.set_ylim(-200, 200)
    ax.legend(loc='upper left', frameon=False)

    out_png = PLOTS_DIR / "md_thermodynamic_diagnostics_abcd.png"
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Seamless continuous ABCD plot saved: {out_png}")

if __name__ == "__main__":
    main()
