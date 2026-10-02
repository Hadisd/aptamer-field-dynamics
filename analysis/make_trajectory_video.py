#!/usr/bin/env python3
"""
make_trajectory_video.py
========================
Renders a high-quality 10-second Full HD (1920x1080, 30 fps) scientific video
of the 100 ns DNA aptamer MD simulation in the laboratory frame under +Z electric field.

Features:
- PyMOL molecular rendering with 3D Cyan +Z Electric Field arrow.
- Synchronized live telemetry HUD (time progress bar, live metrics).
- Synchronized dual inset plots:
    1. 3' End Z-displacement (Z_3' - Z_5') over 100 ns with animated tracer.
    2. DNA Aptamer Backbone RMSD vs Time with animated tracer.
- Encodes both MP4 (H.264, 30 fps, 10 seconds exact) and high-quality GIF in analysis/.

Usage:
    pixi run python analysis/make_trajectory_video.py
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import MDAnalysis as mda
from pymol import cmd
from pymol.cgo import *

BASE = Path(__file__).parent.parent
ANALYSIS_DIR = BASE / "analysis"
PLOTS_DIR = ANALYSIS_DIR / "plots"
VIDEOS_DIR = ANALYSIS_DIR / "videos"
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR = BASE / "scratch" / "video_render"

REF_PDB = ANALYSIS_DIR / "aptamer_lab_ref.pdb"
TRAJ_XTC = ANALYSIS_DIR / "aptamer_labframe.xtc"
RMSD_XVG = PLOTS_DIR / "prod_rmsd.xvg"

OUT_MP4 = VIDEOS_DIR / "aptamer_100ns_trajectory.mp4"
OUT_GIF = VIDEOS_DIR / "aptamer_100ns_trajectory.gif"

TOTAL_FRAMES = 300
FPS = 30
DURATION_SEC = TOTAL_FRAMES / FPS  # 10.0 seconds

def extract_telemetry():
    print(f"Loading trajectory for telemetry analysis: {TRAJ_XTC}...")
    u = mda.Universe(str(REF_PDB), str(TRAJ_XTC))
    five = u.select_atoms("resid 1 and not name H*")
    three = u.select_atoms("resid 20 and not name H*")
    dna = u.select_atoms("nucleic")

    n_total = len(u.trajectory)
    frame_indices = np.linspace(0, n_total - 1, TOTAL_FRAMES, dtype=int)

    all_times = []
    all_z_rel = []
    all_e2e = []
    all_dna_z = []

    for idx in frame_indices:
        ts = u.trajectory[idx]
        t_ns = ts.time / 1000.0
        r5 = five.center_of_mass() / 10.0  # nm
        r3 = three.center_of_mass() / 10.0 # nm
        r_dna = dna.center_of_mass() / 10.0 # nm

        all_times.append(t_ns)
        all_z_rel.append(r3[2] - r5[2])
        all_e2e.append(np.linalg.norm(r3 - r5))
        all_dna_z.append(r_dna[2])

    # Extract RMSD from prod_rmsd.xvg
    rmsd_raw = []
    if RMSD_XVG.exists():
        with open(RMSD_XVG, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith(('@', '#')):
                    rmsd_raw.append([float(x) for x in line.split()])
        rmsd_arr = np.array(rmsd_raw)
        rmsd_interp = np.interp(all_times, rmsd_arr[:, 0] / 1000.0, rmsd_arr[:, 1])
    else:
        rmsd_interp = np.zeros(TOTAL_FRAMES)

    return (np.array(all_times),
            np.array(all_z_rel),
            np.array(all_e2e),
            np.array(all_dna_z),
            rmsd_interp,
            frame_indices)

def setup_pymol():
    print("Initializing PyMOL scene...")
    cmd.reinitialize()
    cmd.load(str(REF_PDB), "aptamer")
    cmd.load_traj(str(TRAJ_XTC), "aptamer")

    # High quality styling
    cmd.bg_color("white")
    cmd.show("cartoon", "aptamer")
    cmd.show("sticks", "aptamer and (resn DT5 or resn DA3)")
    cmd.set("stick_radius", 0.20)
    cmd.set("cartoon_ring_mode", 3)
    cmd.set("cartoon_nucleic_acid_mode", 4)
    cmd.set("cartoon_ladder_mode", 1)

    # Color palette
    cmd.color("ruby", "aptamer and resi 1")
    cmd.color("deepblue", "aptamer and resi 20")
    cmd.color("marine", "aptamer and not (resi 1 or resi 20)")

    # Spheres at 5' and 3' anchors
    cmd.show("spheres", "aptamer and resi 1 and name P")
    cmd.show("spheres", "aptamer and resi 20 and name C1'")
    cmd.set("sphere_scale", 0.75)
    cmd.color("ruby", "aptamer and resi 1 and name P")
    cmd.color("deepblue", "aptamer and resi 20 and name C1'")

    # 3D Cyan Electric Field Vector (+Z)
    x0, y0, z0 = 5.0, 20.0, 5.0
    z_top = z0 + 30.0
    arrow_E = [
        CYLINDER, x0, y0, z0,  x0, y0, z_top,  0.7,  0.0, 0.75, 0.9,  0.0, 0.75, 0.9,
        CONE,     x0, y0, z_top,  x0, y0, z_top + 8.0,  1.5, 0.0,  0.0, 0.75, 0.9,  0.0, 0.75, 0.9, 1.0, 1.0
    ]
    cmd.load_cgo(arrow_E, "Electric_Field_Vector")

    # Camera orientation (Lab frame: +Z straight UP, +X to the RIGHT)
    cmd.set_view((
         1.00000,    0.00000,    0.00000,
         0.00000,    0.00000,    1.00000,
         0.00000,   -1.00000,    0.00000,
         0.00000,    0.00000, -135.00000,
        20.00000,   20.00000,   15.00000,
        70.00000,  180.00000,  -20.00000
    ))
    cmd.zoom("aptamer", 12)

def generate_dual_telemetry_plot(times, z_rel, rmsd, cur_idx, out_path):
    """Generates a synchronized dual telemetry plot (Z-displacement + RMSD) for frame cur_idx."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.9, 5.4), dpi=150)
    fig.patch.set_facecolor('#ffffff')
    for ax in (ax1, ax2):
        ax.set_facecolor('#f8f9fa')

    cur_time = times[cur_idx]
    cur_z = z_rel[cur_idx]
    cur_rmsd = rmsd[cur_idx]

    # --- Plot 1: Z-displacement ---
    ax1.plot(times, z_rel, color='#a6bddb', lw=1.2, alpha=0.7)
    ax1.plot(times[:cur_idx+1], z_rel[:cur_idx+1], color='#08519c', lw=2.2)
    ax1.scatter([cur_time], [cur_z], color='#e31a1c', s=55, zorder=5, edgecolors='black', linewidth=1.1)
    ax1.axhline(0, color='gray', linestyle=':', lw=1.0)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-3.5, 1.5)
    ax1.set_title("3' End Z-Displacement ($Z_{3'} - Z_{5'}$)", fontsize=10, fontweight='bold', pad=4, color='#1a202c')
    ax1.set_ylabel("ΔZ (nm)", fontsize=8.5, labelpad=2)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.grid(True, linestyle='--', alpha=0.4)
    ax1.text(0.96, 0.12, f"ΔZ = {cur_z:+.2f} nm",
             transform=ax1.transAxes, fontsize=8.5, fontweight='bold',
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#e3f2fd', edgecolor='#2196f3', alpha=0.9))

    # --- Plot 2: Backbone RMSD ---
    ax2.plot(times, rmsd, color='#fdbb84', lw=1.2, alpha=0.7)
    ax2.plot(times[:cur_idx+1], rmsd[:cur_idx+1], color='#d95f0e', lw=2.2)
    ax2.scatter([cur_time], [cur_rmsd], color='#e31a1c', s=55, zorder=5, edgecolors='black', linewidth=1.1)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("DNA Backbone RMSD vs Initial Structure", fontsize=10, fontweight='bold', pad=4, color='#1a202c')
    ax2.set_xlabel("Time (ns)", fontsize=8.5, labelpad=2)
    ax2.set_ylabel("RMSD (nm)", fontsize=8.5, labelpad=2)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.text(0.96, 0.12, f"RMSD = {cur_rmsd:.2f} nm",
             transform=ax2.transAxes, fontsize=8.5, fontweight='bold',
             verticalalignment='bottom', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#fff3e0', edgecolor='#fb8c00', alpha=0.9))

    plt.tight_layout(pad=0.8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def compose_composite_frame(mol_png_path, plot_png_path, out_png_path, cur_time, cur_z, cur_e2e, cur_rmsd, frame_num):
    """Assembles the final 1920x1080 composite frame with professional header, dual plots, and telemetry."""
    canvas = Image.new("RGB", (1920, 1080), "#ffffff")
    draw = ImageDraw.Draw(canvas)

    # Load 3D molecular render
    mol_img = Image.open(mol_png_path).convert("RGB")
    mol_resized = mol_img.resize((1360, 960), Image.Resampling.LANCZOS)
    canvas.paste(mol_resized, (20, 70))

    # Load and paste dual telemetry plot on the upper-middle right
    plot_img = Image.open(plot_png_path).convert("RGBA")
    plot_resized = plot_img.resize((490, 540), Image.Resampling.LANCZOS)
    canvas.paste(plot_resized, (1400, 75), mask=plot_resized)

    # Fonts
    try:
        font_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
        font_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        font_card_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
        font_card_val = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 21)
        font_card_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        font_title = font_sub = font_card_title = font_card_val = font_card_sub = ImageFont.load_default()

    # 1. Top Header Banner
    draw.rectangle([0, 0, 1920, 65], fill="#0d1b2a")
    draw.text((30, 16), "20-mer DNA Aptamer (8TFD) All-Atom MD Simulation", fill="#ffffff", font=font_title)
    draw.text((1400, 20), "Electric Field:  Ez = +0.1 V/nm (+Z)", fill="#4cc9f0", font=font_sub)

    # 2. Right Telemetry Info Cards (4 compact cards)
    card_y = 630
    cards = [
        ("Simulation Time", f"{cur_time:.1f} ns / 100.0 ns", f"Frame {frame_num:03d} / {TOTAL_FRAMES}", "#2563eb"),
        ("Backbone RMSD", f"{cur_rmsd:.2f} nm", "Conformational Drift", "#d97706"),
        ("3' End Z-Displacement", f"{cur_z:+.2f} nm", "Pulled downwards (-Z)", "#dc2626" if cur_z < 0 else "#16a34a"),
        ("5' to 3' Extension", f"{cur_e2e:.2f} nm", "End-to-End Distance", "#0891b2"),
    ]

    for title, val, sub, text_col in cards:
        draw.rounded_rectangle([1400, card_y, 1890, card_y + 88], radius=8, fill="#f8fafc", outline="#e2e8f0", width=2)
        draw.text((1415, card_y + 8), title, fill="#64748b", font=font_card_title)
        draw.text((1415, card_y + 30), val, fill=text_col, font=font_card_val)
        draw.text((1415, card_y + 60), sub, fill="#94a3b8", font=font_card_sub)
        card_y += 98

    # 3. Bottom Progress Bar & Annotation
    draw.rectangle([0, 1035, 1920, 1080], fill="#f1f5f9")
    progress = cur_time / 100.0
    bar_width = int(1920 * progress)
    draw.rectangle([0, 1035, bar_width, 1042], fill="#3b82f6")

    # Legends on molecular viewport
    draw.rounded_rectangle([40, 970, 320, 1020], radius=6, fill="#ffffffcc", outline="#cbd5e1", width=1)
    draw.ellipse([50, 982, 64, 996], fill="#e63946")
    draw.text((72, 980), "5' Anchor DT-1 [Fixed]", fill="#1e293b", font=font_card_sub)

    draw.ellipse([50, 1000, 64, 1014], fill="#1d3557")
    draw.text((72, 998), "3' Free End DA-20", fill="#1e293b", font=font_card_sub)

    # Electric field label on canvas
    draw.rounded_rectangle([40, 80, 320, 130], radius=6, fill="#ffffffcc", outline="#4cc9f0", width=2)
    draw.text((50, 88), "⚡ Applied E-Field Vector (+Z)", fill="#0284c7", font=font_card_title)
    draw.text((50, 108), "Force F = qE pulls DNA in -Z", fill="#dc2626", font=font_card_sub)

    canvas.save(out_png_path, quality=95)

def main():
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    raw_mol_dir = TEMP_DIR / "mol_frames"
    raw_plot_dir = TEMP_DIR / "plot_frames"
    final_frames_dir = TEMP_DIR / "final_frames"

    raw_mol_dir.mkdir(exist_ok=True)
    raw_plot_dir.mkdir(exist_ok=True)
    final_frames_dir.mkdir(exist_ok=True)

    # 1. Telemetry
    times, z_rel, e2e, dna_z, rmsd, frame_indices = extract_telemetry()

    # 2. PyMOL Setup
    setup_pymol()

    print(f"\n🎬 Rendering {TOTAL_FRAMES} frames ({DURATION_SEC:.1f} s @ {FPS} fps)...")
    for i, f_idx in enumerate(frame_indices):
        state = f_idx + 1
        cmd.set("state", state)

        mol_png = raw_mol_dir / f"mol_{i:04d}.png"
        plot_png = raw_plot_dir / f"plot_{i:04d}.png"
        final_png = final_frames_dir / f"frame_{i:04d}.png"

        # Render PyMOL viewport
        cmd.png(str(mol_png), width=1600, height=1100, ray=0)

        # Render Dual Telemetry Plot (Z-displacement + RMSD)
        generate_dual_telemetry_plot(times, z_rel, rmsd, i, plot_png)

        # Composite Frame
        compose_composite_frame(mol_png, plot_png, final_png, times[i], z_rel[i], e2e[i], rmsd[i], i + 1)

        if (i + 1) % 50 == 0 or (i + 1) == TOTAL_FRAMES:
            print(f"  -> Rendered frame {i+1:03d}/{TOTAL_FRAMES} ({times[i]:.1f} ns)")

    # 3. Encode MP4 Video with FFmpeg
    print("\n🎥 Encoding MP4 Video (H.264 / 30 fps / 1080p)...")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-r", str(FPS),
        "-i", str(final_frames_dir / "frame_%04d.png"),
        "-c:v", "libx264",
        "-profile:v", "high",
        "-level:v", "4.0",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "slow",
        str(OUT_MP4)
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"✅ Video created: {OUT_MP4} ({OUT_MP4.stat().st_size / (1024*1024):.2f} MB)")

    # 4. Generate Animated GIF
    print("\n🎞️ Generating Animated GIF preview...")
    gif_cmd = [
        "ffmpeg", "-y",
        "-i", str(OUT_MP4),
        "-vf", "fps=15,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
        str(OUT_GIF)
    ]
    subprocess.run(gif_cmd, check=True)
    print(f"✅ GIF created: {OUT_GIF} ({OUT_GIF.stat().st_size / (1024*1024):.2f} MB)")

    print("\n🎉 Complete! 10-second high-resolution trajectory video with RMSD plot is ready in analysis/.")

if __name__ == "__main__":
    main()
