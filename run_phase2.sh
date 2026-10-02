#!/usr/bin/env bash
# =============================================================================
#  run_phase2.sh – 50 ns Continuation with Reversed E-field + Aptamer Floor Wall
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GMX="/home/hadi/.local/gromacs-2026.2/bin/gmx"
RUN_DIR="run/prod_phase2_rev_wall"
NTHREADS=6

echo "================================================================="
echo " Starting Production Phase 2 – 50 ns Continuation MD"
echo " Condition: Ez = -0.1 V/nm (Opposite Field) + Substrate Wall Z = 3.80 nm"
echo " Hardware: 1 GPU (RTX 3060) + 6 OpenMP threads (i5-13600KF)"
echo "================================================================="

cd "$RUN_DIR"

# 1. Run 50 ns Production Simulation
$GMX mdrun -v -deffnm prod -ntmpi 1 -ntomp "$NTHREADS" -nb gpu -pme gpu -update cpu

echo "================================================================="
echo " Production MD Finished! Beginning Post-Processing & Analysis..."
echo "================================================================="

# 2. Extract unrotated labframe trajectory and fitted trajectory
echo "DNA" | $GMX trjconv -s prod.tpr -f prod.xtc -o part2_labframe.xtc -pbc whole -quiet

# Merge Part 1 (0 - 1500 ps) and Part 2 (1500 - 50000 ps)
$GMX trjcat -f ../../prep/part1_labframe_1500ps.xtc part2_labframe.xtc -o prod_labframe.xtc -quiet

# Reference PDB at t=0 (Phase 1 final state / Phase 2 start)
cp ../../prep/dna_ref_unshifted.pdb prod_lab_ref.pdb

# Fitted trajectory
printf "DNA\nDNA\n" | $GMX trjconv -s prod_lab_ref.pdb -f prod_labframe.xtc -o prod_fit.xtc -fit rot+trans -quiet

# 3. Extract energies and structural metrics
printf "Potential\nTotal-Energy\nTemperature\nPressure\nDensity\nPosition-Rest.\nVolume\n0\n" | \
  $GMX energy -f prod.edr -o prod_energy.xvg -quiet

echo "DNA DNA" | $GMX rms -s prod_lab_ref.pdb -f prod_fit.xtc -o prod_rmsd.xvg -quiet
echo "DNA" | $GMX gyrate -s prod_lab_ref.pdb -f prod_fit.xtc -o prod_gyrate.xvg -quiet

cd "$SCRIPT_DIR"

# 4. Generate Comparative Analysis Plots
pixi run python analysis/plot_phase2_analysis.py

# 5. Render 10-Second Video and High-Res GIF
pixi run python analysis/make_phase2_video.py

echo "================================================================="
echo " Phase 2 Run & Analysis Complete!"
echo " Results available in $RUN_DIR and analysis/plots/"
echo "================================================================="
