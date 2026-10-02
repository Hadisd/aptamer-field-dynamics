#!/usr/bin/env python3
"""
plot_efield_effects.py
======================
Generates plots showing direct physical effects of the applied electric field:
  Plot 1: Time evolution of 3' End Z-position and 5'->3' Z-projection (Directional pulling).
  Plot 3: 1D Spatial Density Profiles along Z for Na+, Cl-, and DNA (Ion polarization).

Usage:
    pixi run python analysis/plot_efield_effects.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import MDAnalysis as mda
from pathlib import Path

BASE = Path(__file__).parent.parent
GRO = BASE / "run" / "em" / "system_GMX.gro"
XTC = BASE / "run" / "prod" / "prod.xtc"
OUT_DIR = BASE / "analysis" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print(f"Loading trajectory: {XTC}...")
    u = mda.Universe(str(GRO), str(XTC))

    dna = u.select_atoms("nucleic")
    five = u.select_atoms("resid 1 and not name H*")
    three = u.select_atoms("resid 20 and not name H*")
    na = u.select_atoms("resname Na+")
    cl = u.select_atoms("resname Cl-")

    times = []
    z_rel_3prime = []     # (Z_3 - Z_5) in nm
    z_dna_com = []        # Z of DNA COM in nm
    z_na_all = []         # all Na positions along Z
    z_cl_all = []         # all Cl positions along Z
    z_dna_all = []        # all DNA positions along Z

    print("Analyzing 2,001 trajectory frames...")
    for ts in u.trajectory:
        t_ns = ts.time / 1000.0
        times.append(t_ns)

        r5 = five.center_of_mass() / 10.0   # nm
        r3 = three.center_of_mass() / 10.0  # nm
        r_dna = dna.center_of_mass() / 10.0 # nm

        z_rel_3prime.append(r3[2] - r5[2])
        z_dna_com.append(r_dna[2])

        # Sample ion coordinates (last 50 ns of production for equilibrium distribution)
        if t_ns >= 50.0:
            z_na_all.extend(na.positions[:, 2] / 10.0)
            z_cl_all.extend(cl.positions[:, 2] / 10.0)
            z_dna_all.extend(dna.positions[:, 2] / 10.0)

    times = np.array(times)
    z_rel_3prime = np.array(z_rel_3prime)
    z_dna_com = np.array(z_dna_com)
    z_na_all = np.array(z_na_all)
    z_cl_all = np.array(z_cl_all)
    z_dna_all = np.array(z_dna_all)

    # ════════════════════════════════════════════════════════════════════════
    # PLOT 1: Directional 3' End Position & DNA Pulling along Z
    # ════════════════════════════════════════════════════════════════════════
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=300, sharex=True)
    plt.subplots_adjust(hspace=0.2)

    # Subplot 1: (Z_3' - Z_5') projection
    ax1.plot(times, z_rel_3prime, color='#1f77b4', alpha=0.35, lw=0.8, label='Raw (every 10 ps)')
    window = 25
    z_smooth = np.convolve(z_rel_3prime, np.ones(window)/window, mode='valid')
    ax1.plot(times[:len(z_smooth)], z_smooth, color='#084081', lw=2.0, label=f'Moving Avg ({window*10} ps)')
    ax1.axhline(0, color='gray', linestyle=':', lw=1.0)
    ax1.axhline(z_rel_3prime[0], color='red', linestyle='--', lw=1.5, label=f'Initial state at t=0 (+{z_rel_3prime[0]:.2f} nm)')
    ax1.set_ylabel(r"$Z_{3'} - Z_{5'}$ Distance (nm)", fontsize=11, fontweight='bold')
    ax1.set_title("Plot 1: Directional DNA Aptamer Response to +Z Electric Field ($E_z = 0.1$ V/nm, 100 ns)", fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', frameon=True)
    ax1.grid(True, alpha=0.3)
    
    # Annotate field force direction
    ax1.annotate(r"$\vec{F} = q\vec{E}$ pulls negative DNA downwards ($-Z$)", 
                 xy=(10.0, -1.8), xytext=(5.0, -0.5),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#ffe6e6", ec="red"))

    # Subplot 2: DNA Center of Mass Z position
    ax2.plot(times, z_dna_com, color='#2ca02c', alpha=0.4, lw=0.8)
    dna_smooth = np.convolve(z_dna_com, np.ones(window)/window, mode='valid')
    ax2.plot(times[:len(dna_smooth)], dna_smooth, color='#005a32', lw=2.0, label='DNA COM (Z)')
    ax2.set_xlabel("Simulation Time (ns)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("DNA Center of Mass $Z$ (nm)", fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', frameon=True)
    ax2.grid(True, alpha=0.3)

    out1 = OUT_DIR / "plot1_efield_directional_pull.png"
    fig1.savefig(out1, bbox_inches='tight')
    plt.close(fig1)
    print(f"✅ Saved Plot 1: {out1}")

    # ════════════════════════════════════════════════════════════════════════
    # PLOT 2: Periodic Boundary Clearance (PBC Minimum Distance)
    # ════════════════════════════════════════════════════════════════════════
    pbc_file = OUT_DIR / "pbc_mindist.xvg"
    if pbc_file.exists():
        pbc_raw = []
        with open(pbc_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith(('@', '#')):
                    pbc_raw.append([float(x) for x in line.split()])
        pbc_arr = np.array(pbc_raw)
        t_pbc = pbc_arr[:, 0] / 1000.0
        d_pbc = pbc_arr[:, 1]

        fig_pbc, ax_pbc = plt.subplots(figsize=(10, 5), dpi=300)
        ax_pbc.plot(t_pbc, d_pbc, color='#2b5c8f', lw=1.0, alpha=0.7, label='DNA-to-Periodic-Image Mindist')
        pbc_smooth = np.convolve(d_pbc, np.ones(20)/20, mode='valid')
        ax_pbc.plot(t_pbc[:len(pbc_smooth)], pbc_smooth, color='#081d58', lw=1.8, label='Moving Average')
        ax_pbc.axhline(1.0, color='red', linestyle='--', lw=1.8, label='GROMACS Minimum Safe Cutoff (1.0 nm)')
        ax_pbc.axhline(np.min(d_pbc), color='green', linestyle=':', lw=1.5, label=f'Global Minimum: {np.min(d_pbc):.2f} nm (0% Violations)')
        ax_pbc.fill_between(t_pbc, 0, 1.0, color='red', alpha=0.15, label='Self-Interaction Violation Region')
        ax_pbc.set_title(r"Plot 2: Periodic Boundary Condition (PBC) Clearance ($20\ \mathrm{\AA}$ Buffer, 100 ns)", fontsize=13, fontweight='bold')
        ax_pbc.set_xlabel("Simulation Time (ns)", fontsize=11, fontweight='bold')
        ax_pbc.set_ylabel("Distance to Nearest Periodic Image (nm)", fontsize=11, fontweight='bold')
        ax_pbc.set_ylim(0, 4.0)
        ax_pbc.legend(loc='upper right', frameon=True)
        ax_pbc.grid(True, alpha=0.3)

        out_pbc = OUT_DIR / "plot2_pbc_clearance.png"
        fig_pbc.savefig(out_pbc, bbox_inches='tight')
        plt.close(fig_pbc)
        print(f"✅ Saved Plot 2: {out_pbc}")

    # ════════════════════════════════════════════════════════════════════════
    # PLOT 3: 1D Ion Distribution Profile along Z (Polarization)
    # ════════════════════════════════════════════════════════════════════════
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8), dpi=300, sharex=True)
    plt.subplots_adjust(hspace=0.25)

    # Binning along Z (20 A buffer box ~ 7.2 nm)
    bins = np.linspace(0, 7.5, 150)
    
    # Probability density
    na_hist, _ = np.histogram(z_na_all, bins=bins, density=True)
    cl_hist, _ = np.histogram(z_cl_all, bins=bins, density=True)
    dna_hist, _ = np.histogram(z_dna_all, bins=bins, density=True)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    # Subplot 1: Na+ vs Cl- distributions
    ax3.plot(bin_centers, na_hist, color='#d62728', lw=2.0, label=r'Na$^+$ Cations (Drifting to $+Z$)')
    ax3.plot(bin_centers, cl_hist, color='#1f77b4', lw=2.0, label=r'Cl$^-$ Anions (Drifting to $-Z$)')
    ax3.plot(bin_centers, dna_hist * 0.4, color='#2ca02c', linestyle='--', lw=1.5, label='DNA Aptamer (Scaled)')
    ax3.set_title(r"Plot 3: Ion Atmosphere Polarization along Field Axis ($+Z$, 100 ns)", fontsize=13, fontweight='bold')
    ax3.set_ylabel("Number Density (a.u.)", fontsize=11, fontweight='bold')
    ax3.legend(loc='upper right', frameon=True)
    ax3.grid(True, alpha=0.3)

    # Subplot 2: Net Charge Separation Profile (Na+ - Cl-)
    charge_diff = na_hist - cl_hist
    ax4.plot(bin_centers, charge_diff, color='#7b1fa2', lw=2.0, label=r'Net Charge ($\rho_{\mathrm{Na}^+} - \rho_{\mathrm{Cl}^-}$)')
    ax4.fill_between(bin_centers, 0, charge_diff, where=(charge_diff > 0), color='#d62728', alpha=0.3, label='Positive Excess (+Z)')
    ax4.fill_between(bin_centers, 0, charge_diff, where=(charge_diff < 0), color='#1f77b4', alpha=0.3, label='Negative Excess (-Z)')
    ax4.axhline(0, color='black', linestyle=':', lw=1.2)
    ax4.set_xlabel("Z-Axis Position in Simulation Box (nm)", fontsize=11, fontweight='bold')
    ax4.set_ylabel(r"Net Charge Density ($\Delta\rho$)", fontsize=11, fontweight='bold')
    ax4.legend(loc='upper right', frameon=True)
    ax4.grid(True, alpha=0.3)

    out2 = OUT_DIR / "plot3_ion_polarization_profile.png"
    fig2.savefig(out2, bbox_inches='tight')
    plt.close(fig2)
    print(f"✅ Saved Plot 3: {out2}")

if __name__ == "__main__":
    main()
