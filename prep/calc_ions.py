#!/usr/bin/env python3
"""
calc_ions.py
============
Utility to calculate required counterions and salt ion pairs (e.g., 0.15 M NaCl)
from either:
  1. Box volume (in nm³ or Å³)
  2. Number of water molecules (N_water)
  3. A structure file (.gro or .pdb) using MDAnalysis

Usage examples:
    # 1. Inspect a structure file directly:
    python prep/calc_ions.py --file run/em/system_GMX.gro --conc 0.15 --charge -19

    # 2. Calculate from number of water molecules:
    python prep/calc_ions.py --nwater 5099 --conc 0.15 --charge -19

    # 3. Calculate from box volume in Å³:
    python prep/calc_ions.py --volume 183587 --conc 0.15 --charge -19
"""

import argparse
import sys

AVOGADRO = 6.02214076e23  # mol^-1
WATER_MOLARITY = 55.5      # mol / L

def calc_from_volume(vol_angstrom3: float, conc_molar: float = 0.15, net_charge: int = -19):
    # 1 Å³ = 1e-30 m³ = 1e-27 L
    vol_liters = vol_angstrom3 * 1e-27
    n_pairs = round(conc_molar * vol_liters * AVOGADRO)
    
    if net_charge < 0:
        n_pos = abs(net_charge) + n_pairs
        n_neg = n_pairs
    else:
        n_pos = n_pairs
        n_neg = abs(net_charge) + n_pairs
        
    return n_pairs, n_pos, n_neg, vol_liters

def calc_from_nwater(n_water: int, conc_molar: float = 0.15, net_charge: int = -19):
    # N_salt = N_water * (C_salt / C_water)
    n_pairs = round(n_water * (conc_molar / WATER_MOLARITY))
    
    if net_charge < 0:
        n_pos = abs(net_charge) + n_pairs
        n_neg = n_pairs
    else:
        n_pos = n_pairs
        n_neg = abs(net_charge) + n_pairs
        
    return n_pairs, n_pos, n_neg

def main():
    parser = argparse.ArgumentParser(description="Calculate ion counts for MD simulation.")
    parser.add_argument("--file", "-f", help="Path to .gro or .pdb file")
    parser.add_argument("--nwater", "-w", type=int, help="Number of water molecules")
    parser.add_argument("--volume", "-v", type=float, help="Box volume in Å³")
    parser.add_argument("--conc", "-c", type=float, default=0.15, help="Salt concentration in M (default: 0.15)")
    parser.add_argument("--charge", "-q", type=int, default=-19, help="Solute net charge (default: -19)")
    args = parser.parse_args()

    print(f"============================================================")
    print(f"  Ion Concentration Calculator (Target: {args.conc} M)")
    print(f"============================================================")

    if args.file:
        import MDAnalysis as mda
        u = mda.Universe(args.file)
        vol_angstrom3 = u.dimensions[0] * u.dimensions[1] * u.dimensions[2]  # approximate
        n_water = len(u.select_atoms("resname WAT or resname SOL or resname TIP3")) // 3
        print(f"Loaded structure: {args.file}")
        print(f"Detected waters: {n_water}")
        n_pairs, n_pos, n_neg, vol_l = calc_from_volume(vol_angstrom3, args.conc, args.charge)
    elif args.volume:
        n_pairs, n_pos, n_neg, vol_l = calc_from_volume(args.volume, args.conc, args.charge)
    elif args.nwater:
        n_pairs, n_pos, n_neg = calc_from_nwater(args.nwater, args.conc, args.charge)
    else:
        # Default with our box values
        n_pairs, n_pos, n_neg, vol_l = calc_from_volume(183587, args.conc, args.charge)
        print("Using default box volume for 8TFD aptamer (183,587 Å³):")

    print(f"Solute net charge:        {args.charge:+d} e")
    print(f"Neutralizing counterions: {abs(args.charge)} Na+")
    print(f"Salt pairs ({args.conc} M):       {n_pairs} (Na+ / Cl- pairs)")
    print(f"------------------------------------------------------------")
    print(f"Total Na+ ions needed:    {n_pos}")
    print(f"Total Cl- ions needed:    {n_neg}")
    print(f"============================================================")
    print("\nFor tleap:")
    print(f"  addions2 unit Na+ {abs(args.charge)}   # neutralize")
    print(f"  addions2 unit Na+ {n_pairs}      # salt")
    print(f"  addions2 unit Cl- {n_pairs}      # salt\n")

if __name__ == "__main__":
    main()
