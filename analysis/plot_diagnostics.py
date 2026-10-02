#!/usr/bin/env python3
"""
plot_diagnostics.py
===================
Reads extracted GROMACS .xvg files and MD trajectories to generate a comprehensive
8-panel diagnostic dashboard covering thermodynamic stability, density, pressure,
position restraints, RMSD, Gyration, and End-to-End extension across the FULL 150 ns
simulation timeline (Phase 1: 0 - 100 ns + Phase 2: 100 - 150 ns).

Output:
    analysis/plots/md_diagnostics_dashboard.png
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import MDAnalysis as mda
from MDAnalysis.analysis import rms

BASE = Path(__file__).parent.parent
PLOTS_DIR = BASE / "analysis" / "plots"
RUN_P2 = BASE / "run" / "prod_phase2_rev_wall"

GRO_P1 = BASE / "analysis" / "aptamer_lab_ref.pdb"
XTC_P1 = BASE / "analysis" / "aptamer_labframe.xtc"
GRO_P2 = RUN_P2 / "prod_lab_ref.pdb"
XTC_P2 = RUN_P2 / "prod_labframe.xtc"
REF_T0 = BASE / "analysis" / "aptamer_t0_ref.pdb"

def read_xvg(filepath):
    """Parses GROMACS .xvg files ignoring comments and headers."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('@', '#')):
                continue
            parts = [float(x) for x in line.split()]
            data.append(parts)
    return np.array(data)

def moving_avg(arr, w=30):
    pad = w // 2
    padded = np.pad(arr, (pad, pad), mode='edge')
    return np.convolve(padded, np.ones(w)/w, mode='valid')[:len(arr)]

def main():
    print("Reading equilibration and production XVG energy files...")
    nvt_data = read_xvg(PLOTS_DIR / "nvt_energy.xvg")
    npt_data = read_xvg(PLOTS_DIR / "npt_energy.xvg")
    
    prod1_energy = read_xvg(PLOTS_DIR / "prod_energy.xvg")
    prod2_energy = read_xvg(RUN_P2 / "prod_energy.xvg")

    # Time axes in ns
    t_nvt = nvt_data[:, 0] / 1000.0
    t_npt = npt_data[:, 0] / 1000.0

    # Combine production energies across 0 - 150 ns seamlessly
    t_p1 = prod1_energy[:, 0] / 1000.0
    t_p2 = 100.0 + ((prod2_energy[:, 0] - prod2_energy[0, 0]) / 1000.0)

    # Drop step 0 from Phase 2
    t_p2 = t_p2[1:]
    prod2_clean = prod2_energy[1:]

    t_prod = np.concatenate([t_p1, t_p2])
    # Columns: 0: Time, 1: Posre, 2: Potential, 3: Total Energy, 4: Temp, 5: Press, 6: Vol, 7: Dens
    posre_prod = np.concatenate([prod1_energy[:, 1], prod2_clean[:, 1]])
    pot_prod = np.concatenate([prod1_energy[:, 2], prod2_clean[:, 2]])
    tot_prod = np.concatenate([prod1_energy[:, 3], prod2_clean[:, 3]])
    temp_prod = np.concatenate([prod1_energy[:, 4], prod2_clean[:, 4]])
    press_prod = np.concatenate([prod1_energy[:, 5], prod2_clean[:, 5]])
    dens_prod = np.concatenate([prod1_energy[:, 7], prod2_clean[:, 7]])

    # Load trajectories for structural analysis (stride = 10, 100 ps)
    print("Computing structural metrics across 0 - 150 ns...")
    u_t0 = mda.Universe(str(REF_T0))
    u1 = mda.Universe(str(GRO_P1), str(XTC_P1))
    u2 = mda.Universe(str(GRO_P2), str(XTC_P2))

    bb_sel = "nucleic and name P C3' C4' C5' O3' O5'"
    ref_pos = u_t0.select_atoms(bb_sel).positions.copy()

    bb1 = u1.select_atoms(bb_sel)
    dna1 = u1.select_atoms("nucleic")
    five1 = u1.select_atoms("resid 1 and not name H*")
    three1 = u1.select_atoms("resid 20 and not name H*")

    bb2 = u2.select_atoms(bb_sel)
    dna2 = u2.select_atoms("nucleic")
    five2 = u2.select_atoms("resid 1 and not name H*")
    three2 = u2.select_atoms("resid 20 and not name H*")

    stride = 10
    t_geom = []
    rmsd_geom = []
    rg_geom = []
    rg_z_geom = []
    e2e_geom = []

    # Phase 1
    for ts in u1.trajectory[::stride]:
        t_geom.append(ts.time / 1000.0)
        rmsd_geom.append(rms.rmsd(ts.positions[bb1.indices], ref_pos, superposition=True) / 10.0)
        rg_geom.append(dna1.radius_of_gyration() / 10.0)
        # Z-component of radius of gyration
        z_coords = dna1.positions[:, 2] / 10.0
        rg_z_geom.append(np.sqrt(np.mean((z_coords - np.mean(z_coords))**2)))
        e2e_geom.append(np.linalg.norm(three1.center_of_mass() - five1.center_of_mass()) / 10.0)

    # Phase 2
    for ts in u2.trajectory[::stride]:
        t_geom.append(100.0 + (ts.time / 1000.0))
        rmsd_geom.append(rms.rmsd(ts.positions[bb2.indices], ref_pos, superposition=True) / 10.0)
        rg_geom.append(dna2.radius_of_gyration() / 10.0)
        z_coords = dna2.positions[:, 2] / 10.0
        rg_z_geom.append(np.sqrt(np.mean((z_coords - np.mean(z_coords))**2)))
        e2e_geom.append(np.linalg.norm(three2.center_of_mass() - five2.center_of_mass()) / 10.0)

    t_geom = np.array(t_geom)
    rmsd_geom = np.array(rmsd_geom)
    rg_geom = np.array(rg_geom)
    rg_z_geom = np.array(rg_z_geom)
    e2e_geom = np.array(e2e_geom)

    # Plot styling
    plt.rcParams.update({
        'font.sans-serif': 'DejaVu Sans',
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9.5,
        'ytick.labelsize': 9.5,
        'legend.fontsize': 9.5,
        'figure.titlesize': 14,
    })

    fig, axes = plt.subplots(4, 2, figsize=(15, 17), dpi=300)
    plt.subplots_adjust(hspace=0.36, wspace=0.22)

    def add_phase_shading(ax, show_legend=False):
        ax.axvspan(0, 100, color='#e3f2fd', alpha=0.5, label='Phase 1: $E_z = +0.1$ V/nm (Solution Pull)' if show_legend else None)
        ax.axvspan(100, 150, color='#fff3e0', alpha=0.5, label='Phase 2: $E_z = -0.1$ V/nm (Substrate Floor)' if show_legend else None)
        ax.axvline(100.0, color='#d32f2f', linestyle='--', lw=1.5, alpha=0.85)

    # 1. Potential & Total Energy
    ax = axes[0, 0]
    add_phase_shading(ax, show_legend=True)
    ax.plot(t_prod, pot_prod, color='#1f77b4', lw=1.0, alpha=0.7, label='Potential Energy')
    ax.plot(t_prod, tot_prod, color='#ff7f0e', lw=1.0, alpha=0.7, label='Total Energy')
    ax.set_title("A. Potential & Total Energy (0 - 150 ns)", fontweight='bold')
    ax.set_xlabel("Cumulative Time (ns)")
    ax.set_ylabel("Energy (kJ/mol)")
    ax.set_xlim(0, 150)
    ax.legend(loc='lower left', framealpha=0.9, fontsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.3)

    # 2. System Temperature
    ax = axes[0, 1]
    add_phase_shading(ax)
    ax.plot(t_prod, temp_prod, color='#e57373', alpha=0.4, lw=0.6, label='Instantaneous')
    temp_smooth = moving_avg(temp_prod, 40)
    ax.plot(t_prod, temp_smooth, color='#b71c1c', lw=1.8, label='Moving Avg (400 ps)')
    ax.axhline(300, color='black', linestyle='--', lw=1.2, label='Target (300 K)')
    ax.set_title("B. System Temperature Stability (Target: 300 K)", fontweight='bold')
    ax.set_xlabel("Cumulative Time (ns)")
    ax.set_ylabel("Temperature (K)")
    ax.set_xlim(0, 150)
    ax.set_ylim(290, 310)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.3)

    # 3. Density Equilibration
    ax = axes[1, 0]
    # NPT equilibration (0 to 5 ns) + Production (5 to 155 ns)
    ax.plot(t_npt, npt_data[:, 2], color='#9c27b0', alpha=0.8, lw=1.2, label='NPT Equilibration (5 ns)')
    ax.plot(t_prod + 5.0, dens_prod, color='#2e7d32', alpha=0.7, lw=0.8, label='Production 150 ns (Phase 1 + 2)')
    ax.axvline(5.0, color='gray', linestyle=':', lw=1.2)
    ax.axvline(105.0, color='#d32f2f', linestyle='--', lw=1.5, alpha=0.85)
    ax.set_title("C. Mass Density Equilibration (0.15 M NaCl TIP3P)", fontweight='bold')
    ax.set_xlabel("Total Time including NPT (ns)")
    ax.set_ylabel(r"Density (kg/m$^3$)")
    ax.set_xlim(0, 155)
    ax.set_ylim(1000, 1025)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.3)

    # 4. Pressure
    ax = axes[1, 1]
    add_phase_shading(ax)
    press_smooth = moving_avg(press_prod, 60)
    ax.plot(t_prod, press_prod, color='#b0bec5', alpha=0.35, lw=0.5, label='Instantaneous')
    ax.plot(t_prod, press_smooth, color='#00838f', lw=1.8, label='Moving Avg (600 ps)')
    ax.axhline(1.0, color='red', linestyle='--', lw=1.2, label='Target (1.0 bar)')
    ax.set_title("D. System Pressure (Target: 1 bar)", fontweight='bold')
    ax.set_xlabel("Cumulative Time (ns)")
    ax.set_ylabel("Pressure (bar)")
    ax.set_xlim(0, 150)
    ax.set_ylim(-120, 120)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.3)

    # 5. Position Restraint Energy (5' Anchor)
    ax = axes[2, 0]
    add_phase_shading(ax)
    ax.plot(t_prod, posre_prod, color='#ab47bc', alpha=0.5, lw=0.7, label='Raw (10 ps)')
    posre_smooth = moving_avg(posre_prod, 30)
    ax.plot(t_prod, posre_smooth, color='#6a1b9a', lw=2.0, label='Moving Avg (300 ps)')
    ax.set_title("E. 5' DT-1 Anchor Position Restraint Energy", fontweight='bold')
    ax.set_xlabel("Cumulative Time (ns)")
    ax.set_ylabel("Restraint Energy (kJ/mol)")
    ax.set_xlim(0, 150)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.3)

    # 6. DNA Backbone RMSD
    ax = axes[2, 1]
    add_phase_shading(ax)
    ax.plot(t_geom, rmsd_geom, color='#ffb74d', alpha=0.5, lw=0.8, label='Raw (100 ps)')
    rmsd_smooth = moving_avg(rmsd_geom, 20)
    ax.plot(t_geom, rmsd_smooth, color='#e65100', lw=2.2, label='Moving Avg (2.0 ns)')
    ax.set_title("F. DNA Backbone RMSD vs Equilibrated Starting Structure", fontweight='bold')
    ax.set_xlabel("Cumulative Time (ns)")
    ax.set_ylabel("RMSD (nm)")
    ax.set_xlim(0, 150)
    ax.set_ylim(0.0, 0.70)
    ax.legend(loc='lower right', framealpha=0.9, fontsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.3)

    # 7. Radius of Gyration
    ax = axes[3, 0]
    add_phase_shading(ax)
    ax.plot(t_geom, rg_geom, color='#1565c0', lw=1.8, label=r'$R_g$ (Total)')
    ax.plot(t_geom, rg_z_geom, color='#c62828', lw=1.5, linestyle='--', label=r'$R_g$ ($Z$-axis)')
    ax.set_title(r"G. Radius of Gyration ($R_g$) – Compaction under Reversed Field", fontweight='bold')
    ax.set_xlabel("Cumulative Time (ns)")
    ax.set_ylabel("Radius of Gyration (nm)")
    ax.set_xlim(0, 150)
    ax.set_ylim(0.4, 1.6)
    ax.legend(loc='lower left', framealpha=0.9, fontsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.3)

    # 8. End-to-End Distance (5' to 3')
    ax = axes[3, 1]
    add_phase_shading(ax)
    ax.plot(t_geom, e2e_geom, color='#81c784', alpha=0.5, lw=0.8, label='Raw (100 ps)')
    e2e_smooth = moving_avg(e2e_geom, 20)
    ax.plot(t_geom, e2e_smooth, color='#1b5e20', lw=2.2, label='Moving Avg (2.0 ns)')
    ax.set_title("H. 5'-to-3' End-to-End Extension Across 150 ns", fontweight='bold')
    ax.set_xlabel("Cumulative Time (ns)")
    ax.set_ylabel("Distance (nm)")
    ax.set_xlim(0, 150)
    ax.set_ylim(0.8, 3.5)
    ax.legend(loc='lower left', framealpha=0.9, fontsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.3)

    out_png = PLOTS_DIR / "md_diagnostics_dashboard.png"
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Full 150 ns Dashboard saved successfully: {out_png}")

if __name__ == "__main__":
    main()
