#!/usr/bin/env bash
# =============================================================================
#  run_md.sh  –  Automated MD Pipeline for 5'-Anchored Aptamer + Electric Field
#  =============================================================================
#
#  Workflow:
#    Step 1  – Auto-Build  : AMBER OL15 parameterization + 0.15 M NaCl auto-calc
#                            + acpype conversion + 5' restraint generation
#    Step 2  – Minimization: Energy minimization (steepest descent)
#    Step 3  – NVT         : 5 ns NVT equilibration (300 K, 5' DT-1 restrained)
#    Step 4  – NPT         : 5 ns NPT equilibration (1 bar, 5' DT-1 restrained)
#    Step 5  – Production  : 20 ns Production (5' DT-1 restrained + 0.1 V/nm E-field)
#
#  Usage:
#    cd /home/hadi/Project/prot-apt/apt-alone
#    pixi run bash run_md.sh              # Run entire pipeline from start
#    pixi run bash run_md.sh --step 2     # Start from Minimization
#    pixi run bash run_md.sh --step 3     # Start from NVT
#    pixi run bash run_md.sh --step 4     # Start from NPT
#    pixi run bash run_md.sh --step 5     # Start from Production
# =============================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
GMX=/home/hadi/.local/gromacs-2026.2/bin/gmx
INPUT_PDB="file/aptamer.pdb"
SALT_CONC=0.15
BUFFER=20.0
NTHREADS=6
START_STEP=1

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --step)
      START_STEP="$2"
      shift 2
      ;;
    --conc)
      SALT_CONC="$2"
      shift 2
      ;;
    --buffer)
      BUFFER="$2"
      shift 2
      ;;
    --pdb)
      INPUT_PDB="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

MDP=md
PREP=prep
RUN=run

GRN="\033[0;32m"; RED="\033[0;31m"; YEL="\033[0;33m"; BLU="\033[0;34m"; RST="\033[0m"
info()  { echo -e "${GRN}[INFO]${RST}  $*"; }
warn()  { echo -e "${YEL}[WARN]${RST}  $*"; }
die()   { echo -e "${RED}[ERR] ${RST}  $*" >&2; exit 1; }
step_banner() { echo -e "\n${BLU}════════════════════════════════════════════════════════════${RST}\n${GRN}▶ STEP $1:${RST} $2\n${BLU}════════════════════════════════════════════════════════════${RST}\n"; }
should_run()  { [[ "$START_STEP" -le "$1" ]]; }

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 – Automated System Building & Parameterization
# ─────────────────────────────────────────────────────────────────────────────
if should_run 1; then
  step_banner 1 "Auto-Build (Buffer ${BUFFER} Å + ${SALT_CONC} M NaCl + AMBER OL15 + 5' Restraint)"
  [[ -f "$INPUT_PDB" ]] || die "Input PDB not found: $INPUT_PDB"
  
  python prep/auto_build.py --pdb "$INPUT_PDB" --conc "$SALT_CONC" --buffer "$BUFFER"
  info "System preparation and GROMACS topology generation completed."
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 – Energy Minimization
# ─────────────────────────────────────────────────────────────────────────────
if should_run 2; then
  step_banner 2 "Energy Minimization (Steepest Descent)"
  mkdir -p $RUN/em && cd $RUN/em
  
  $GMX grompp -f ../../$MDP/em.mdp \
              -c system_GMX.gro \
              -p topol.top \
              -o em.tpr \
              -maxwarn 2
              
  $GMX mdrun -v -deffnm em -ntmpi 1 -ntomp $NTHREADS
  info "Energy minimization converged."
  cd ../..
fi

# Generate default index groups if missing
mkdir -p $RUN/nvt
if [[ ! -f $RUN/nvt/index.ndx ]]; then
  echo "q" | $GMX make_ndx -f $RUN/em/em.gro -o $RUN/nvt/index.ndx -quiet
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 – NVT Equilibration (5 ns, 5' Restrained, 300 K)
# ─────────────────────────────────────────────────────────────────────────────
if should_run 3; then
  step_banner 3 "NVT Equilibration – 5 ns (300 K, 5' DT-1 Restrained)"
  mkdir -p $RUN/nvt && cd $RUN/nvt
  
  $GMX grompp -f ../../$MDP/nvt_posres.mdp \
              -c ../em/em.gro \
              -r ../em/em.gro \
              -p ../em/topol.top \
              -n index.ndx \
              -o nvt.tpr \
              -maxwarn 2
              
  $GMX mdrun -v -deffnm nvt -ntmpi 1 -ntomp $NTHREADS
  info "NVT equilibration completed."
  cd ../..
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 – NPT Equilibration (5 ns, 5' Restrained, 1 bar)
# ─────────────────────────────────────────────────────────────────────────────
if should_run 4; then
  step_banner 4 "NPT Equilibration – 5 ns (1 bar, 5' DT-1 Restrained)"
  mkdir -p $RUN/npt && cd $RUN/npt
  
  $GMX grompp -f ../../$MDP/npt_posres.mdp \
              -c ../nvt/nvt.gro \
              -r ../em/em.gro \
              -t ../nvt/nvt.cpt \
              -p ../em/topol.top \
              -n ../nvt/index.ndx \
              -o npt.tpr \
              -maxwarn 2
              
  $GMX mdrun -v -deffnm npt -ntmpi 1 -ntomp $NTHREADS
  info "NPT equilibration completed."
  cd ../..
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 – Production Simulation (20 ns + 5' Restrained + Electric Field)
# ─────────────────────────────────────────────────────────────────────────────
if should_run 5; then
  step_banner 5 "Production MD – 100 ns (5' Restrained + DC E-field Z = 0.1 V/nm)"
  mkdir -p $RUN/prod && cd $RUN/prod
  
  $GMX grompp -f ../../$MDP/prod_efield.mdp \
              -c ../npt/npt.gro \
              -r ../em/em.gro \
              -t ../npt/npt.cpt \
              -p ../em/topol.top \
              -n ../nvt/index.ndx \
              -o prod.tpr \
              -maxwarn 2
              
  $GMX mdrun -v -deffnm prod -ntmpi 1 -ntomp $NTHREADS
  info "Production run completed."
  cd ../..
fi

echo -e "\n${GRN}════════════════════════════════════════════════════════════${RST}"
echo -e "${GRN}  Pipeline finished successfully!${RST}"
echo -e "${GRN}════════════════════════════════════════════════════════════${RST}"
echo "  Trajectory : $RUN/prod/prod.xtc"
echo "  Energies   : $RUN/prod/prod.edr"
echo "  Final Gro  : $RUN/prod/prod.gro"
