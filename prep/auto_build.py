#!/usr/bin/env python3
"""
auto_build.py
=============
Fully automated preparation of a DNA aptamer system:
  1. Solvates the PDB in TIP3P (12 Å buffer) with AMBER DNA.OL15.
  2. Measures the box volume and solute net charge automatically.
  3. Calculates the exact number of neutralizing + 0.15 M salt ions.
  4. Generates and executes the final tleap script.
  5. Converts the system to GROMACS format via acpype.
  6. Generates 5' position restraints and patches topol.top.

Usage:
    pixi run python prep/auto_build.py --pdb file/aptamer.pdb --conc 0.15
"""

import argparse
import subprocess
import re
from pathlib import Path
import MDAnalysis as mda

AVOGADRO = 6.02214076e23

def run_cmd(cmd, cwd=None):
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if res.returncode != 0:
        print(f"Error running: {cmd}\n{res.stderr}\n{res.stdout}")
        raise RuntimeError(f"Command failed: {cmd}")
    return res.stdout

def main():
    parser = argparse.ArgumentParser(description="Automated AMBER -> GROMACS builder with exact salt concentration.")
    parser.add_argument("--pdb", default="file/aptamer.pdb", help="Path to input PDB file")
    parser.add_argument("--conc", type=float, default=0.15, help="Target NaCl concentration in M (default: 0.15)")
    parser.add_argument("--buffer", type=float, default=20.0, help="Water buffer distance in Å (default: 20.0)")
    args = parser.parse_args()

    base = Path(__file__).parent.parent.resolve()
    pdb_path = (base / args.pdb).resolve()
    prep_dir = base / "prep"
    run_dir = base / "run" / "em"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/5] Measuring initial system & solvating in tleap (Buffer: {args.buffer} Å)...")
    temp_leap = f"""
source leaprc.DNA.OL15
source leaprc.water.tip3p
mol = loadpdb {pdb_path}
solvateoct mol TIP3PBOX {args.buffer}
check mol
charge mol
quit
"""
    (prep_dir / "temp_solv.in").write_text(temp_leap)
    out = run_cmd(f"tleap -f {prep_dir / 'temp_solv.in'}", cwd=str(base))

    # Extract Volume and Net Charge from tleap log
    vol_match = re.search(r"Volume:\s+([\d\.]+)\s+A\^3", out)
    charge_match = re.search(r"unperturbed charge of the unit \(([-\d\.]+)\)", out)

    if not vol_match or not charge_match:
        raise RuntimeError("Failed to parse box volume or charge from tleap output.")

    vol_a3 = float(vol_match.group(1))
    charge = round(float(charge_match.group(1)))
    vol_liters = vol_a3 * 1e-27

    # Calculate salt pairs: N = C * V * N_A
    salt_pairs = round(args.conc * vol_liters * AVOGADRO)
    n_na = abs(charge) + salt_pairs if charge < 0 else salt_pairs
    n_cl = salt_pairs if charge < 0 else abs(charge) + salt_pairs

    print(f"  Box Volume:     {vol_a3:,.1f} Å³ ({vol_liters*1e21:.2f} × 10⁻²¹ L)")
    print(f"  Solute Charge:  {charge:+d} e")
    print(f"  Target Salt:    {args.conc} M NaCl")
    print(f"  -> Added Ions:  {n_na} Na+ ({abs(charge)} counterions + {salt_pairs} salt) and {n_cl} Cl-")

    print(f"\n[2/5] Running final tleap parameterization...")
    final_leap = f"""
source leaprc.DNA.OL15
source leaprc.water.tip3p
apt = loadpdb {pdb_path}
check apt
solvateoct apt TIP3PBOX {args.buffer}
addions2 apt Na+ {abs(charge)}
addions2 apt Na+ {salt_pairs}
addions2 apt Cl- {salt_pairs}
saveamberparm apt {prep_dir / 'aptamer.prmtop'} {prep_dir / 'aptamer.inpcrd'}
quit
"""
    (prep_dir / "tleap.in").write_text(final_leap)
    run_cmd(f"tleap -f {prep_dir / 'tleap.in'}", cwd=str(base))

    print(f"\n[3/5] Converting AMBER prmtop -> GROMACS with acpype...")
    run_cmd(f"rm -rf system.amb2gmx && acpype -p aptamer.prmtop -x aptamer.inpcrd -b system -d", cwd=str(prep_dir))
    run_cmd(f"cp {prep_dir}/system.amb2gmx/system_GMX.gro {run_dir}/")
    run_cmd(f"cp {prep_dir}/system.amb2gmx/system_GMX.top {run_dir}/")

    print(f"\n[4/5] Generating 5' position restraints...")
    run_cmd(f"python {prep_dir / 'make_posre_5prime.py'}", cwd=str(base))

    print(f"\n[5/5] Patching GROMACS topology with #ifdef POSRES_5PRIME...")
    run_cmd(f"python {prep_dir / 'patch_topology.py'}", cwd=str(base))

    print(f"\n✅ All steps completed automatically!")
    print(f"  GROMACS Structure: {run_dir / 'system_GMX.gro'}")
    print(f"  GROMACS Topology:  {run_dir / 'topol.top'}")
    print(f"  5' Restraints:     {prep_dir / 'posre_DNA5prime.itp'}")

if __name__ == "__main__":
    main()
