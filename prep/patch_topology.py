#!/usr/bin/env python3
"""
patch_topology.py
=================
Patches the acpype-generated GROMACS topology to insert
  #ifdef POSRES_5PRIME ... #endif
blocks into the DNA moleculetype, just before the [ bonds ] section.

Usage (run from apt-alone/):
    pixi run python prep/patch_topology.py

Input:  run/em/system_GMX.top
Output: run/em/topol.top   (patched, safe to use for all subsequent grompp calls)
        prep/posre_DNA5prime.itp must exist (run make_posre_5prime.py first)
"""

from pathlib import Path
import re

BASE     = Path(__file__).parent.parent
TOP_IN   = BASE / "run" / "em" / "system_GMX.top"
TOP_OUT  = BASE / "run" / "em" / "topol.top"
POSRE    = BASE / "prep" / "posre_DNA5prime.itp"

DNA_RESNAMES = {"DA", "DT", "DG", "DC", "DA3", "DT3", "DG3", "DC3",
                "DA5", "DT5", "DG5", "DC5"}

assert TOP_IN.exists(),  f"Missing: {TOP_IN}"
assert POSRE.exists(),   f"Missing: {POSRE} — run make_posre_5prime.py first"

lines = TOP_IN.read_text().splitlines(keepends=True)

# ── Locate the DNA moleculetype section ────────────────────────────────────
# We look for '[ moleculetype ]' blocks, then check whether the name
# contains any DNA residue name.  Within that block we find '[ bonds ]'
# and insert the ifdef block just before it.

# Build a map: line-index → section header
mol_starts = [i for i, l in enumerate(lines)
              if re.match(r"\[\s*moleculetype\s*\]", l)]

# Identify which moleculetype block is the DNA one by scanning atoms
def is_dna_mol(start_idx):
    """Return True if the moleculetype starting at start_idx is DNA."""
    for j in range(start_idx, min(start_idx + 2000, len(lines))):
        # hit the next [ moleculetype ] → stop
        if j != start_idx and re.match(r"\[\s*moleculetype\s*\]", lines[j]):
            break
        # look for DNA residue names in [ atoms ] section
        parts = lines[j].split()
        if len(parts) >= 4 and parts[3].upper() in DNA_RESNAMES:
            return True
    return False

dna_mol_start = None
for ms in mol_starts:
    if is_dna_mol(ms):
        dna_mol_start = ms
        break

if dna_mol_start is None:
    raise RuntimeError("Could not identify DNA moleculetype block. "
                       "Check residue names in the topology.")

print(f"DNA moleculetype block starts at line {dna_mol_start + 1}")

# ── Find the first '[ bonds ]' after dna_mol_start ────────────────────────
bonds_idx = None
for j in range(dna_mol_start + 1, len(lines)):
    # stop if we've entered the next moleculetype
    if j != dna_mol_start and re.match(r"\[\s*moleculetype\s*\]", lines[j]):
        break
    if re.match(r"\[\s*bonds\s*\]", lines[j]):
        bonds_idx = j
        break

if bonds_idx is None:
    raise RuntimeError("Could not find [ bonds ] inside DNA moleculetype block.")

print(f"Inserting POSRES_5PRIME block before line {bonds_idx + 1}")

# Relative path from run/em/ to prep/
rel_posre = Path("../../prep") / POSRE.name

insert_block = [
    "\n",
    "; ---- 5' end position restraints (apply with -DPOSRES_5PRIME) ----\n",
    "#ifdef POSRES_5PRIME\n",
    f'#include "{rel_posre}"\n',
    "#endif\n",
    "\n",
]

new_lines = lines[:bonds_idx] + insert_block + lines[bonds_idx:]
TOP_OUT.write_text("".join(new_lines))
print(f"Wrote patched topology: {TOP_OUT}")
print(f"  POSRES_5PRIME block added ({len(insert_block)} lines inserted)")
