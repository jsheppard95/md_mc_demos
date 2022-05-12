"""
CH E 210B, HW3, Problem 6

Author: Jackson Sheppard
Last Edit: 5/11/22

Python Script to perform an NVE MD simulation of a medel fluid with a LJ 6-8
pair potential:

u(r) = 10\epsilon[(\sigma/r)^8 - (\sigma/r)^6]

We estimate the compressibility factor and internal energy at T_star = 1.0,
rho_star = 0.8; where standard LJ units are imployed:

T_star = kT/(\epsilon), rho_star = (\rho)(\sigma)^3

Adapted from mathematica_nve_md_demo2.nb and Matlab_MC__MD_programs
"""
import numpy as np
import matplotlib.pyplot as plt

def MD_main(N_part, rho):
    """
    Main function to run NVE MD simulation.
    Parameters:
    -----------
    N_part : int
        Number of particles to simulate, must be multiple of 3
    rho : float
        reduced denisty (LJ units), i.e rho_star = (rho [N/m^3])(sigma^3)
    """
    # Input parameter check
    if N_part % 3 != 0:
        raise ValueError("Number of particles must be a multiple of 3")
    
    generate_lattice(N_part, rho)


def generate_lattice(N_part, rho):
    """
    Initializes particles on cubic lattice with a unit cell size determined
    from the reduced density.
    Parameters:
    -----------
    See MD_main
    """
    l = N_part ** (1/3)  # number of particles per unit cell length
    box1 = l/rho**(1/3)
    box12 = box1/2
    box13 = box1/l

    coords = np.zeros((N_part, 3))  # row = particle, col = x, y, z coordinate

    # place particles - See MatLab code for magic
    for i in range(1, N_part + 1):
        x = np.ceil(i/l**2)
        x1 = x - 1
        y1 = i - x1*l**2
        y = np.ceil(y1/l)
        z = (i % l) + 1

        coords[i - 1, 0] = x
        coords[i - 1, 1] = y
        coords[i - 1, 2] = z
    coords = coords - (l + 1)/2
    coords = coords * box13

    # Plot initial particle positions in lattice
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    ax.set_xlim(-box12, box12)
    ax.set_ylim(-box12, box12)
    ax.set_zlim(-box12, box12)
    plt.show()


MD_main(27, 0.8)
