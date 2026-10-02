# =============================================================================
#  view_trajectory.pml  –  PyMOL Visualization with Real Electric Field Arrow (+Z)
# =============================================================================
#
#  Usage:
#    pymol analysis/view_trajectory.pml
#
# =============================================================================

reinitialize

# 1. Load laboratory-frame reference structure and trajectory
load analysis/aptamer_lab_ref.pdb, aptamer
load_traj analysis/aptamer_labframe.xtc, aptamer

# 2. Visual styling
bg_color white
show cartoon, aptamer
show sticks, aptamer and (resn DT5 or resn DA3)
set stick_radius, 0.25
set cartoon_ring_mode, 3
set cartoon_nucleic_acid_mode, 4

# 3. Highlight 5' Anchor (Red) and 3' Free End (Blue)
color red, aptamer and resi 1
color blue, aptamer and resi 20
color marine, aptamer and not (resi 1 or resi 20)

label aptamer and resi 1 and name C1', "  5' Anchor [Fixed]"
label aptamer and resi 20 and name C1', "  3' Free End"
set label_color, black
set label_size, 18

# 4. Single Cyan 3D Arrow showing the REAL Applied Electric Field (+Z direction)
python
from pymol import cmd
from pymol.cgo import *

# Position arrow next to the molecule (X=5, Y=20, Z=10 to 35 Angstroms)
x0, y0, z0 = 5.0, 20.0, 10.0
z_top = z0 + 22.0

# Cyan Arrow pointing upwards in +Z (Electric Field vector E)
arrow_E = [
    CYLINDER, x0, y0, z0,  x0, y0, z_top,  0.5,  0.0, 0.75, 0.9,  0.0, 0.75, 0.9,
    CONE,     x0, y0, z_top,  x0, y0, z_top + 6.0,  1.1, 0.0,  0.0, 0.75, 0.9,  0.0, 0.75, 0.9, 1.0, 1.0
]
cmd.load_cgo(arrow_E, "Electric_Field_E_plusZ")

python end

# 5. Orient camera so +Z points straight UP and +X points RIGHT
set_view (\
     1.00000,    0.00000,    0.00000,\
     0.00000,    0.00000,    1.00000,\
     0.00000,   -1.00000,    0.00000,\
     0.00000,    0.00000, -120.00000,\
    20.00000,   20.00000,   20.00000,\
    70.00000,  150.00000,  -20.00000 )

zoom aptamer, 10
