#!/usr/bin/env bash
# =============================================================================
#  clean_trajectory.sh  –  Automated Trajectory Post-Processing
#  =============================================================================
#
#  Generates:
#    1. Laboratory Frame Trajectory (Unrotated for Electric Field Analysis):
#       - analysis/aptamer_labframe.xtc (DNA only, PBC whole, ~5 MB)
#       - analysis/aptamer_lab_ref.pdb  (Matching reference PDB)
#
#    2. Rot+Trans Fitted Trajectory (For RMSD / Internal Motion):
#       - analysis/aptamer_clean_fit.xtc (DNA only, rot+trans fitted)
#       - analysis/aptamer_ref.pdb       (Reference PDB)
#
#    3. Full System Fitted Trajectory:
#       - run/prod/prod_fit.xtc (DNA + water + ions, fitted)
#
#  Usage:
#    cd /home/hadi/Project/prot-apt/apt-alone
#    bash analysis/clean_trajectory.sh
# =============================================================================

set -euo pipefail

GMX=/home/hadi/.local/gromacs-2026.2/bin/gmx
mkdir -p analysis

echo "════════════════════════════════════════════════════════════"
echo "  [1/3] Generating True Laboratory-Frame Trajectory..."
echo "        (Preserves real Z-axis for Electric Field analysis)"
echo "════════════════════════════════════════════════════════════"

# A. DNA only, whole molecules (No rotation fitting)
echo "DNA" | $GMX trjconv \
  -s run/prod/prod.tpr \
  -f run/prod/prod.xtc \
  -o analysis/aptamer_labframe.xtc \
  -pbc whole -quiet

# B. Matching reference PDB at t = 0 ps
echo "DNA" | $GMX trjconv \
  -s run/prod/prod.tpr \
  -f run/prod/prod.gro \
  -o analysis/aptamer_lab_ref.pdb \
  -dump 0 -quiet

echo "  -> Created: analysis/aptamer_labframe.xtc"
echo "  -> Created: analysis/aptamer_lab_ref.pdb"


echo -e "\n════════════════════════════════════════════════════════════"
echo "  [2/3] Generating Rotational & Translational Fitted Trajectory..."
echo "        (Least-squares fitting on DNA)"
echo "════════════════════════════════════════════════════════════"

# A. Make whole and center
echo -e "DNA\nSystem" | $GMX trjconv \
  -s run/prod/prod.tpr \
  -f run/prod/prod.xtc \
  -o run/prod/prod_whole.xtc \
  -pbc mol -center -quiet

# B. Fit rotation and translation
echo -e "DNA\nSystem" | $GMX trjconv \
  -s run/prod/prod.tpr \
  -f run/prod/prod_whole.xtc \
  -o run/prod/prod_fit.xtc \
  -fit rot+trans -quiet

rm -f run/prod/prod_whole.xtc

# C. Extract DNA-only fitted trajectory
echo "DNA" | $GMX trjconv \
  -s run/prod/prod.tpr \
  -f run/prod/prod_fit.xtc \
  -o analysis/aptamer_clean_fit.xtc -quiet

echo "DNA" | $GMX trjconv \
  -s run/prod/prod.tpr \
  -f run/prod/prod.gro \
  -o analysis/aptamer_ref.pdb \
  -dump 0 -quiet

echo "  -> Created: run/prod/prod_fit.xtc"
echo "  -> Created: analysis/aptamer_clean_fit.xtc"


echo -e "\n════════════════════════════════════════════════════════════"
echo "  [3/3] Post-Processing Completed Successfully!"
echo "════════════════════════════════════════════════════════════"
echo "To visualize the Electric Field response in PyMOL:"
echo "  pymol analysis/view_trajectory.pml"
