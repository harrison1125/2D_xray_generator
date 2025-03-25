"""
Calculate the structure factor for an FCC lattice for the (h,k,l) reflection. This is used as a kernel that is convolved over the activated Reciprocal Lattice Points 

Parameters:
-----------
h, k, l : int
    Miller indices of the reflection.
f : float
    Atomic form factor (default=1.0).

Returns:
--------
float
    Structure factor F(h,k,l) for the given reflection.
"""


# Simple Cubic (SC)
def structure_factor_sc(h, k, l, f):
    return f

# Face-Centered Cubic (FCC)
def structure_factor_fcc(h, k, l, f):
    return f * (1 + (-1)**(h+k) + (-1)**(k+l) + (-1)**(h+l))
# 
# Body-Centered Cubic (BCC)
def structure_factor_bcc(h, k, l, f):
    return f * (1 + (-1)**(h+k+l))

# Diamond Cubic (DC)
def structure_factor_diamond(h, k, l, f):
    return f * (1 + (-1)**(h+k+l) * complex(0, -1)**(h+k+l))

# Hexagonal Close-Packed (HCP)
def structure_factor_hcp(h, k, l, f):
    if (h + 2*k) % 3 == 0:  # Condition for allowed reflections
        return f * (1 + (-1)**l * complex(0, -1)**l)
    else:
        return 0

# Body-Centered Tetragonal (BCT)
def structure_factor_bct(h, k, l, f):
    return f * (1 + (-1)**(h+k+l))

# Face-Centered Tetragonal (FCT)
def structure_factor_fct(h, k, l, f):
    return f * (1 + (-1)**(h+k) + (-1)**(h+l) + (-1)**(k+l))

# Orthorhombic Primitive (Simple Orthorhombic)
def structure_factor_orthorhombic(h, k, l, f):
    return f

# Body-Centered Orthorhombic (BCO)
def structure_factor_bco(h, k, l, f):
    return f * (1 + (-1)**(h+k+l))

# Face-Centered Orthorhombic (FCO)
def structure_factor_fco(h, k, l, f):
    return f * (1 + (-1)**(h+k) + (-1)**(h+l) + (-1)**(k+l))

# Rhombohedral (Hexagonal Setting)
def structure_factor_rhombohedral(h, k, l, f):
    if (h - k) % 3 == 0:  # Condition for allowed reflections
        return f
    else:
        return 0
    
