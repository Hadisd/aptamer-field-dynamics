#!/usr/bin/env python3
"""
make_posre_5prime.py
====================
Generate posre_DNA5prime.itp restraining all heavy atoms of the
5' terminal nucleotide (DT residue 1) of the aptamer.

Run from apt-alone/ after acpype conversion:
    pixi run python prep/make_posre_5prime.py
"""

from pathlib import Path
import MDAnalysis as mda

BASE     = Path(__file__).parent.parent
GRO      = BASE / "run" / "em" / "system_GMX.gro"
OUT_ITP  = BASE / "prep" / "posre_DNA5prime.itp"
OUT_NDX  = BASE / "prep" / "index_DNA5prime.ndx"
FC       = 1000   # kJ mol-1 nm-2

u = mda.Universe(str(GRO))

DNA_RESNAMES = {"DA", "DT", "DG", "DC", "DA3", "DT3", "DG3", "DC3",
                "DA5", "DT5", "DG5", "DC5"}

dna = [r for r in u.residues if r.resname.upper() in DNA_RESNAMES]
if not dna:
    raise RuntimeError(
        f"No DNA residues found. Residue names present: {set(u.residues.resnames)}"
    )
dna.sort(key=lambda r: r.resindex)

five_prime = dna[0]
print(f"5' residue: {five_prime.resname} resid={five_prime.resid}  atoms={five_prime.atoms.n_atoms}")

# Heavy atoms only
heavy = five_prime.atoms.select_atoms("not name H*")
print(f"Heavy atoms to restrain: {heavy.n_atoms}")

# For acpype single-molecule output there is only one moleculetype;
# atom indices in the itp are 1-based and match the GRO order directly.
with open(OUT_ITP, "w") as f:
    f.write("; Position restraints – 5' DT-1 heavy atoms of the DNA aptamer\n")
    f.write(f"; Force constant: {FC} kJ mol-1 nm-2 (x, y, z)\n;\n")
    f.write("[ position_restraints ]\n")
    f.write(";  i funct       fcx        fcy        fcz\n")
    for atom in heavy.atoms:
        idx = int(atom.index) + 1   # 1-based
        f.write(f"{idx:5d}    1   {FC:8d}   {FC:8d}   {FC:8d}\n")
        print(f"  {idx:5d}  {atom.name}")

print(f"\nWrote: {OUT_ITP}")

# NDX for convenience
with open(OUT_NDX, "w") as f:
    f.write("[ DNA_5prime ]\n")
    for i, atom in enumerate(heavy.atoms):
        f.write(f"{int(atom.index)+1:6d}")
        if (i + 1) % 15 == 0:
            f.write("\n")
    f.write("\n")
print(f"Wrote: {OUT_NDX}")
