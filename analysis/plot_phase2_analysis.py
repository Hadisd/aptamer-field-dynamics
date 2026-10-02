#!/usr/bin/env python3
"""
plot_phase2_analysis.py
=======================
Generates comprehensive publication-quality plots analyzing the full 150 ns trajectory:
- Phase 1 (0 - 100 ns): Ez = +0.1 V/nm (pulling negative DNA into solution, -Z)
- Phase 2 (100 - 150 ns): Ez = -0.1 V/nm (reversed field) + Substrate Wall at Z = 3.80 nm

Outputs saved directly to analysis/plots/:
- phase2_dashboard.png (Comprehensive 6-panel executive dashboard)
- phase2_combined_displacement.png (3' End Z-trajectory across 0-150 ns)
- phase2_rmsd_evolution.png (Backbone RMSD & Radius of Gyration evolution)
- phase2_z_density_wall.png (DNA Z-envelope & wall confinement verification)
- phase2_density_comparison.png (DNA spatial probability density P(Z) Phase 1 vs Phase 2)
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
ANALYSIS_DIR = BASE / "analysis"
PLOTS_DIR = ANALYSIS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

GRO_P1 = ANALYSIS_DIR / "aptamer_lab_ref.pdb"
XTC_P1 = ANALYSIS_DIR / "aptamer_labframe.xtc"
GRO_P2 = BASE / "run" / "prod_phase2_rev_wall" / "prod_lab_ref.pdb"
XTC_P2 = BASE / "run" / "prod_phase2_rev_wall" / "prod_labframe.xtc"
REF_T0 = ANALYSIS_DIR / "aptamer_t0_ref.pdb"

Z_WALL = 3.80  # nm (Substrate wall)
Z_5PRIME_NOMINAL = 3.33  # nm (Nominal 5' anchor restraint)

def main():
    print("Loading universes...")
    u_t0 = mda.Universe(str(REF_T0))
    u1 = mda.Universe(str(GRO_P1), str(XTC_P1))
    u2 = mda.Universe(str(GRO_P2), str(XTC_P2))

    bb_sel = "nucleic and name P C3' C4' C5' O3' O5'"
    bb_t0 = u_t0.select_atoms(bb_sel)
    ref_pos = bb_t0.positions.copy()

    bb1 = u1.select_atoms(bb_sel)
    five1 = u1.select_atoms("resid 1 and not name H*")
    three1 = u1.select_atoms("resid 20 and not name H*")
    dna1 = u1.select_atoms("nucleic")

    bb2 = u2.select_atoms(bb_sel)
    five2 = u2.select_atoms("resid 1 and not name H*")
    three2 = u2.select_atoms("resid 20 and not name H*")
    dna2 = u2.select_atoms("nucleic")

    stride = 5  # 50 ps stride (2000 points Phase 1, 1000 points Phase 2)
    print(f"Extracting trajectory metrics with stride = {stride} (50 ps)...")

    # Containers
    t_all = []
    z_rel_all = []
    z_5prime_all = []
    z_3prime_all = []
    z_min_all = []
    z_max_all = []
    e2e_all = []
    rmsd_all = []
    rg_all = []

    # Phase 1: 0 - 100 ns
    print(f"Processing Phase 1 ({len(u1.trajectory)} frames)...")
    for ts in u1.trajectory[::stride]:
        t_ns = ts.time / 1000.0
        r5 = five1.center_of_mass() / 10.0
        r3 = three1.center_of_mass() / 10.0
        dna_z = dna1.positions[:, 2] / 10.0
        cur_rmsd = rms.rmsd(ts.positions[bb1.indices], ref_pos, superposition=True) / 10.0
        cur_rg = dna1.radius_of_gyration() / 10.0

        t_all.append(t_ns)
        z_5prime_all.append(r5[2])
        z_3prime_all.append(r3[2])
        z_rel_all.append(r3[2] - r5[2])
        e2e_all.append(np.linalg.norm(r3 - r5))
        z_min_all.append(dna_z.min())
        z_max_all.append(dna_z.max())
        rmsd_all.append(cur_rmsd)
        rg_all.append(cur_rg)

    # Phase 2: 100 - 150 ns
    print(f"Processing Phase 2 ({len(u2.trajectory)} frames)...")
    for ts in u2.trajectory[::stride]:
        t_ns = 100.0 + (ts.time / 1000.0)
        r5 = five2.center_of_mass() / 10.0
        r3 = three2.center_of_mass() / 10.0
        dna_z = dna2.positions[:, 2] / 10.0
        cur_rmsd = rms.rmsd(ts.positions[bb2.indices], ref_pos, superposition=True) / 10.0
        cur_rg = dna2.radius_of_gyration() / 10.0

        t_all.append(t_ns)
        z_5prime_all.append(r5[2])
        z_3prime_all.append(r3[2])
        z_rel_all.append(r3[2] - r5[2])
        e2e_all.append(np.linalg.norm(r3 - r5))
        z_min_all.append(dna_z.min())
        z_max_all.append(dna_z.max())
        rmsd_all.append(cur_rmsd)
        rg_all.append(cur_rg)

    t_all = np.array(t_all)
    z_rel_all = np.array(z_rel_all)
    z_5prime_all = np.array(z_5prime_all)
    z_3prime_all = np.array(z_3prime_all)
    z_min_all = np.array(z_min_all)
    z_max_all = np.array(z_max_all)
    e2e_all = np.array(e2e_all)
    rmsd_all = np.array(rmsd_all)
    rg_all = np.array(rg_all)

    # Rolling average helper
    def moving_avg(arr, w=30):
        pad = w // 2
        padded = np.pad(arr, (pad, pad), mode='edge')
        return np.convolve(padded, np.ones(w)/w, mode='valid')[:len(arr)]

    smooth_z_rel = moving_avg(z_rel_all, 31)
    smooth_e2e = moving_avg(e2e_all, 31)
    smooth_rmsd = moving_avg(rmsd_all, 31)
    smooth_rg = moving_avg(rg_all, 31)

    # Styling settings
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

    # =========================================================================
    # PLOT 1: Full 6-Panel Executive Dashboard (analysis/plots/phase2_dashboard.png)
    # =========================================================================
    print("\nGenerating 6-Panel Executive Dashboard...")
    fig, axes = plt.subplots(3, 2, figsize=(15, 14), dpi=300)
    plt.subplots_adjust(hspace=0.32, wspace=0.22)

    # Helper for background spans
    def add_phase_spans(ax):
        ax.axvspan(0, 100, color='#e3f2fd', alpha=0.5, label='Phase 1: $E_z = +0.1$ V/nm (Solution Pull)' if ax == axes[0, 0] else None)
        ax.axvspan(100, 150, color='#fff3e0', alpha=0.5, label='Phase 2: $E_z = -0.1$ V/nm (Wall Floor)' if ax == axes[0, 0] else None)
        ax.axvline(100.0, color='#d32f2f', linestyle='--', lw=1.4, alpha=0.85)

    # Panel A: 3' End Z-Displacement
    ax = axes[0, 0]
    add_phase_spans(ax)
    ax.plot(t_all, z_rel_all, color='#90caf9', lw=0.7, alpha=0.5, label='Raw (50 ps)')
    ax.plot(t_all, smooth_z_rel, color='#0d47a1', lw=2.2, label='Rolling Average (1.5 ns)')
    ax.axhline(0, color='gray', linestyle=':', lw=1.2)
    ax.set_title(r"A. 3' End $Z$-Displacement relative to 5' Anchor ($\Delta Z = Z_{3'} - Z_{5'}$)", fontweight='bold')
    ax.set_xlabel("Simulation Time (ns)")
    ax.set_ylabel(r"$\Delta Z$ (nm)")
    ax.set_xlim(0, 150)
    ax.set_ylim(-2.2, 1.2)
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Panel B: Backbone RMSD
    ax = axes[0, 1]
    add_phase_spans(ax)
    ax.plot(t_all, rmsd_all, color='#ffcc80', lw=0.7, alpha=0.5, label='Raw (50 ps)')
    ax.plot(t_all, smooth_rmsd, color='#e65100', lw=2.2, label='Rolling Average (1.5 ns)')
    ax.set_title("B. Aptamer Backbone RMSD vs Equilibrated Starting Structure ($t=0$)", fontweight='bold')
    ax.set_xlabel("Simulation Time (ns)")
    ax.set_ylabel("RMSD (nm)")
    ax.set_xlim(0, 150)
    ax.set_ylim(0.0, 0.70)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Panel C: End-to-End Distance
    ax = axes[1, 0]
    add_phase_spans(ax)
    ax.plot(t_all, e2e_all, color='#a5d6a7', lw=0.7, alpha=0.5, label='Raw (50 ps)')
    ax.plot(t_all, smooth_e2e, color='#1b5e20', lw=2.2, label='Rolling Average (1.5 ns)')
    ax.set_title("C. 5'-to-3' End-to-End Extension Distance", fontweight='bold')
    ax.set_xlabel("Simulation Time (ns)")
    ax.set_ylabel("Distance (nm)")
    ax.set_xlim(0, 150)
    ax.set_ylim(0.8, 3.5)
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Panel D: Radius of Gyration
    ax = axes[1, 1]
    add_phase_spans(ax)
    ax.plot(t_all, rg_all, color='#ce93d8', lw=0.7, alpha=0.5, label='Raw (50 ps)')
    ax.plot(t_all, smooth_rg, color='#4a148c', lw=2.2, label='Rolling Average (1.5 ns)')
    ax.set_title(r"D. Radius of Gyration ($R_g$) – Structural Compaction", fontweight='bold')
    ax.set_xlabel("Simulation Time (ns)")
    ax.set_ylabel(r"$R_g$ (nm)")
    ax.set_xlim(0, 150)
    ax.set_ylim(1.0, 1.6)
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Panel E: DNA Z-Envelope vs Substrate Wall Boundary
    ax = axes[2, 0]
    add_phase_spans(ax)
    ax.plot(t_all, z_max_all, color='#6a1b9a', lw=1.1, alpha=0.85, label=r'Highest DNA Atom ($Z_{\mathrm{max}}$)')
    ax.plot(t_all, z_5prime_all, color='#c2185b', lw=1.5, label=r"5' Anchor ($Z_{5'} \approx 3.33$ nm)")
    ax.plot(t_all, z_min_all, color='#00838f', lw=1.1, alpha=0.85, label=r'Lowest DNA Atom ($Z_{\mathrm{min}}$)')
    ax.axhline(Z_WALL, color='#d32f2f', linestyle='--', lw=2.0, label=f'Substrate Wall ($Z = {Z_WALL:.2f}$ nm)')
    ax.fill_between(t_all, Z_WALL, 4.5, color='#d32f2f', alpha=0.15, label='Excluded Substrate Region')
    ax.set_title(r"E. DNA Coordinate Envelope & Wall Confinement ($Z \leq 3.80$ nm)", fontweight='bold')
    ax.set_xlabel("Simulation Time (ns)")
    ax.set_ylabel("Box $Z$ Position (nm)")
    ax.set_xlim(0, 150)
    ax.set_ylim(0.5, 4.2)
    ax.legend(loc='lower left', framealpha=0.9, fontsize=8.5, ncol=2)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Panel F: DNA Spatial Density Distribution P(Z) Phase 1 vs Phase 2
    ax = axes[2, 1]
    # Sample last 40 ns of Phase 1 and last 30 ns of Phase 2 (equilibrated regimes)
    dna_z_p1_eq = []
    for ts in u1.trajectory[int(len(u1.trajectory)*0.6)::5]:
        dna_z_p1_eq.extend(dna1.positions[:, 2] / 10.0)

    dna_z_p2_eq = []
    for ts in u2.trajectory[int(len(u2.trajectory)*0.4)::5]:
        dna_z_p2_eq.extend(dna2.positions[:, 2] / 10.0)

    bins = np.linspace(0.5, 4.5, 120)
    hist_p1, edges_p1 = np.histogram(dna_z_p1_eq, bins=bins, density=True)
    hist_p2, edges_p2 = np.histogram(dna_z_p2_eq, bins=bins, density=True)
    centers = 0.5 * (edges_p1[:-1] + edges_p1[1:])

    ax.plot(centers, hist_p1, color='#1565c0', lw=2.2, label='Phase 1: Extended in Solution (60-100 ns)')
    ax.fill_between(centers, 0, hist_p1, color='#1565c0', alpha=0.25)

    ax.plot(centers, hist_p2, color='#e65100', lw=2.2, label='Phase 2: Confined at Wall (120-150 ns)')
    ax.fill_between(centers, 0, hist_p2, color='#e65100', alpha=0.30)

    ax.axvline(Z_WALL, color='#d32f2f', linestyle='--', lw=2.0, label=f'Substrate Wall ($Z = {Z_WALL:.2f}$ nm)')
    ax.axvline(Z_5PRIME_NOMINAL, color='#c2185b', linestyle=':', lw=1.8, label=r"5' Anchor ($Z = 3.33$ nm)")
    ax.fill_between([Z_WALL, 4.5], 0, max(max(hist_p1), max(hist_p2))*1.15, color='#d32f2f', alpha=0.15)

    ax.set_title(r"F. DNA Spatial Density along $Z$ ($P(Z)$ Equilibrium)", fontweight='bold')
    ax.set_xlabel("Box $Z$ Coordinate (nm)")
    ax.set_ylabel("Probability Density")
    ax.set_xlim(0.8, 4.2)
    ax.set_ylim(0, max(max(hist_p1), max(hist_p2))*1.15)
    ax.legend(loc='upper left', framealpha=0.9, fontsize=8.5)
    ax.grid(True, linestyle='--', alpha=0.3)

    out_dash = PLOTS_DIR / "phase2_dashboard.png"
    fig.savefig(out_dash, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved: {out_dash}")

    # =========================================================================
    # PLOT 2: Combined Displacement Plot (analysis/plots/phase2_combined_displacement.png)
    # =========================================================================
    print("Generating Combined Displacement Plot...")
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    ax.axvspan(0, 100, color='#e3f2fd', alpha=0.55, label='Phase 1: $E_z = +0.1$ V/nm (Pulling negative DNA into solution, $-Z$)')
    ax.axvspan(100, 150, color='#fff3e0', alpha=0.55, label=f'Phase 2: $E_z = -0.1$ V/nm (Opposite Field) + Substrate Wall at $Z = {Z_WALL:.2f}$ nm')

    ax.plot(t_all, z_rel_all, color='#90caf9', lw=0.8, alpha=0.45, label='Raw Trajectory (50 ps stride)')
    ax.plot(t_all, smooth_z_rel, color='#0d47a1', lw=2.4, label='Rolling Average (1.5 ns window)')
    ax.axhline(0, color='gray', linestyle=':', lw=1.2)
    ax.axvline(100.0, color='#d32f2f', linestyle='--', lw=1.8, label='Electric Field Inversion ($t = 100$ ns)')

    ax.annotate('Extended into\nSolution Bulk', xy=(45, -1.3), xytext=(25, -0.4),
                arrowprops=dict(arrowstyle="->", color='#0d47a1', lw=1.5),
                fontsize=9.5, fontweight='bold', color='#0d47a1',
                bbox=dict(boxstyle="round,pad=0.3", fc="#e3f2fd", ec="#90caf9"))

    ax.annotate('Pushed toward\nSubstrate Floor', xy=(125, -0.4), xytext=(125, 0.45),
                arrowprops=dict(arrowstyle="->", color='#e65100', lw=1.5),
                fontsize=9.5, fontweight='bold', color='#e65100',
                bbox=dict(boxstyle="round,pad=0.3", fc="#fff3e0", ec="#ffb74d"))

    ax.set_title("Aptamer 3' End Displacement Across 150 ns Field Inversion & Wall Confinement", fontsize=12.5, fontweight='bold')
    ax.set_xlabel("Simulation Time (ns)", fontsize=11, fontweight='bold')
    ax.set_ylabel(r"3' End Relative Position ($\Delta Z = Z_{3'} - Z_{5'}$, nm)", fontsize=11, fontweight='bold')
    ax.set_xlim(0, 150)
    ax.set_ylim(-2.2, 1.2)
    ax.legend(loc='lower left', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.3)

    out_disp = PLOTS_DIR / "phase2_combined_displacement.png"
    fig.savefig(out_disp, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved: {out_disp}")

    # =========================================================================
    # PLOT 3: Dedicated RMSD & Rg Evolution (analysis/plots/phase2_rmsd_evolution.png)
    # =========================================================================
    print("Generating Dedicated RMSD & Rg Evolution Plot...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), dpi=300, sharex=True)
    plt.subplots_adjust(hspace=0.15)

    # Panel 1: RMSD
    ax1.axvspan(0, 100, color='#e3f2fd', alpha=0.5, label='Phase 1: Solution Pull')
    ax1.axvspan(100, 150, color='#fff3e0', alpha=0.5, label='Phase 2: Substrate Wall Active')
    ax1.axvline(100.0, color='#d32f2f', linestyle='--', lw=1.5)

    ax1.plot(t_all, rmsd_all, color='#ffb74d', lw=0.8, alpha=0.5, label='Raw RMSD (50 ps)')
    ax1.plot(t_all, smooth_rmsd, color='#e65100', lw=2.3, label='Rolling Mean RMSD')
    ax1.set_ylabel("Backbone RMSD (nm)", fontsize=11, fontweight='bold')
    ax1.set_title("DNA Aptamer Structural Stability & Compaction Across 150 ns", fontsize=12.5, fontweight='bold')
    ax1.set_ylim(0.0, 0.70)
    ax1.legend(loc='lower right', framealpha=0.9)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Panel 2: Radius of Gyration
    ax2.axvspan(0, 100, color='#e3f2fd', alpha=0.5)
    ax2.axvspan(100, 150, color='#fff3e0', alpha=0.5)
    ax2.axvline(100.0, color='#d32f2f', linestyle='--', lw=1.5)

    ax2.plot(t_all, rg_all, color='#ba68c8', lw=0.8, alpha=0.5, label='Raw $R_g$ (50 ps)')
    ax2.plot(t_all, smooth_rg, color='#4a148c', lw=2.3, label='Rolling Mean $R_g$')
    ax2.set_xlabel("Simulation Time (ns)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Radius of Gyration $R_g$ (nm)", fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 150)
    ax2.set_ylim(1.0, 1.55)
    ax2.legend(loc='lower left', framealpha=0.9)
    ax2.grid(True, linestyle='--', alpha=0.3)

    out_rmsd = PLOTS_DIR / "phase2_rmsd_evolution.png"
    fig.savefig(out_rmsd, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved: {out_rmsd}")

    # =========================================================================
    # PLOT 4: Wall Confinement Verification (analysis/plots/phase2_z_density_wall.png)
    # =========================================================================
    print("Generating Wall Confinement Verification Plot...")
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    ax.axvspan(0, 100, color='#e3f2fd', alpha=0.5, label='Phase 1: Extended in Bulk Solution')
    ax.axvspan(100, 150, color='#e8f5e9', alpha=0.5, label=f'Phase 2: Substrate Wall Confined ($Z_{{wall}} = {Z_WALL:.2f}$ nm)')

    ax.plot(t_all, z_max_all, color='#6a1b9a', lw=1.0, alpha=0.75, label=r'Highest DNA Atom ($Z_{\mathrm{max}}$)')
    ax.plot(t_all, z_5prime_all, color='#c2185b', lw=1.6, label=r"5' Anchor DT-1 ($Z_{5'} \approx 3.33$ nm)")
    ax.plot(t_all, z_min_all, color='#00838f', lw=1.0, alpha=0.75, label=r'Lowest DNA Atom ($Z_{\mathrm{min}}$)')

    ax.axhline(Z_WALL, color='#d32f2f', linestyle='--', lw=2.2, label=f'Substrate Wall Boundary ($Z = {Z_WALL:.2f}$ nm)')
    ax.fill_between(t_all, Z_WALL, 4.8, color='#d32f2f', alpha=0.18, label=r'Excluded Substrate Region ($Z > Z_{\mathrm{wall}}$)')
    ax.axvline(100.0, color='black', linestyle=':', lw=1.5)

    ax.set_title(rf"Verification of Aptamer Confinement by Substrate Wall ($Z \leq {Z_WALL:.2f}$ nm)", fontsize=12.5, fontweight='bold')
    ax.set_xlabel("Simulation Time (ns)", fontsize=11, fontweight='bold')
    ax.set_ylabel("DNA Atom $Z$ Positions in Box (nm)", fontsize=11, fontweight='bold')
    ax.set_xlim(0, 150)
    ax.set_ylim(0.5, 4.5)
    ax.legend(loc='lower left', framealpha=0.9, ncol=2)
    ax.grid(True, linestyle='--', alpha=0.3)

    out_wall = PLOTS_DIR / "phase2_z_density_wall.png"
    fig.savefig(out_wall, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved: {out_wall}")

    # =========================================================================
    # PLOT 5: Spatial Density Comparison (analysis/plots/phase2_density_comparison.png)
    # =========================================================================
    print("Generating Spatial Density Comparison Plot...")
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

    ax.plot(centers, hist_p1, color='#1565c0', lw=2.5, label='Phase 1 (60-100 ns): Extended into bulk solution')
    ax.fill_between(centers, 0, hist_p1, color='#1565c0', alpha=0.25)

    ax.plot(centers, hist_p2, color='#e65100', lw=2.5, label='Phase 2 (120-150 ns): Compressed against substrate wall')
    ax.fill_between(centers, 0, hist_p2, color='#e65100', alpha=0.30)

    ax.axvline(Z_WALL, color='#d32f2f', linestyle='--', lw=2.2, label=f'Substrate Wall ($Z = {Z_WALL:.2f}$ nm)')
    ax.axvline(Z_5PRIME_NOMINAL, color='#c2185b', linestyle=':', lw=1.8, label="5' Anchor ($Z = 3.33$ nm)")
    ax.fill_between([Z_WALL, 4.5], 0, max(max(hist_p1), max(hist_p2))*1.15, color='#d32f2f', alpha=0.18, label='Zero Density Region ($Z > Z_{wall}$)')

    ax.set_title("Equilibrium Spatial Probability Density Along Box $Z$ Axis", fontsize=12.5, fontweight='bold')
    ax.set_xlabel("Box $Z$ Coordinate (nm)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Probability Density", fontsize=11, fontweight='bold')
    ax.set_xlim(0.8, 4.2)
    ax.set_ylim(0, max(max(hist_p1), max(hist_p2))*1.15)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.3)

    out_dens = PLOTS_DIR / "phase2_density_comparison.png"
    fig.savefig(out_dens, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved: {out_dens}")

    print("\n🎉 All 5 publication-quality plots updated successfully!")

if __name__ == "__main__":
    main()
