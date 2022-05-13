"""
CH E 210B, HW3, Problem 6

Author: Jackson Sheppard
Last Edit: 5/12/22

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

def MD_main(N_part, rho, init_vel=3., timestep=0.005, n_iter=200):
    """
    Main function to run NVE MD simulation.
    Parameters:
    -----------
    N_part : int
        Number of particles to simulate, must be multiple of 3
    rho : float
        reduced denisty (LJ units), i.e rho_star = (rho [N/m^3])(sigma^3)
    init_vel : float
        initial velocity, determines initial temperature, default = 3
    timestep : float
        timestep for integrating ODEs, default = 0.005
    n_iter : int
        Number of iterations of Verlet algorithm to perform, default = 200
    """
    # Input parameter check
    if N_part % 3 != 0:
        raise ValueError("Number of particles must be a multiple of 3")
    
    coord, box1 = generate_lattice(N_part, rho)

    # Set random initial velocities from uniform distribution and let the
    # fluid equilibrate. Since the variance of the distribution of velocities
    # is T* in dimensionless units, start with something "hot" to melt the
    # lattice
    randomvel = np.random.uniform(low=-init_vel, high=init_vel, size=(N_part, 3))

    # calculate coordinates at previous timestep
    oldcoord = coord - timestep * randomvel

    # Run Verlet Algorithm
    verlet(coord, oldcoord, timestep, n_iter, box1)


def generate_lattice(N_part, rho):
    """
    Initializes particles on cubic lattice with a unit cell size determined
    from the reduced density.
    Parameters:
    -----------
    See MD_main
    Returns:
    --------
    (coord, box1) : tuple (numpy.array, float)
        Tuple including particle coordinate matrix of shape (N_part, 3);
        rows -> particle#, col-> x,y,z and box unit cell size
    """
    l = N_part ** (1/3)  # number of particles per unit cell length
    box1 = l/rho**(1/3)
    box12 = box1/2
    box13 = box1/l

    coord = np.zeros((N_part, 3))  # row = particle, col = x, y, z coordinate

    # place particles - See MatLab code for magic
    for i in range(1, N_part + 1):
        x = np.ceil(i/l**2)
        x1 = x - 1
        y1 = i - x1*l**2
        y = np.ceil(y1/l)
        z = (i % l) + 1

        coord[i - 1, 0] = x
        coord[i - 1, 1] = y
        coord[i - 1, 2] = z
    coord = coord - (l + 1)/2
    coord = coord * box13

    # Plot initial particle positions in lattice
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(coord[:, 0], coord[:, 1], coord[:, 2])
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")
    ax.set_xlim(-box12, box12)
    ax.set_ylim(-box12, box12)
    ax.set_zlim(-box12, box12)
    plt.show()
    return (coord, box1)


def verlet(coord, oldcoord, timestep, n_iter, box1):
    """
    Runs Verlet algorithm on particle coordinates
    Parameters:
    -----------
    coord : numpy.array
        Particle coordinate matrix at current timestep
    oldcoord : numpy.array
        Particle coordinate matrix at previous timestep
    timestep : float
        See MD_main
    n_iter : int
        See MD_main
    box1 : float
        Box unit cell size, returned by `generate_lattice`
    """
    dtsq = timestep**2
    dt2 = timestep*2

    n_part = len(coord[:, 0])
    print(box1)

    # Initialize arrays for data collection
    tstar = np.zeros(n_iter)
    kstar = np.zeros(n_iter)
    estar = np.zeros(n_iter)
    kstar = np.zeros(n_iter)

    # Outer loop to move particles `n_iter` times
    for k in range(n_iter):
        # Initialize force matrix, potential, virial coefficients, and K.E
        force = np.zeros((n_part, 3))
        pot = 0
        vir = 0

        sumvsq = 0

        # Outer loop of force calculation
        for i in range(n_part - 1):
            rxi = coord[i, 0]
            ryi = coord[i, 1]
            rzi = coord[i, 2]

            fxi = force[i, 0]
            fyi = force[i, 1]
            fzi = force[i, 2]

            # Inner loop of force calculation
            for j in range(1, n_part):
                # Calculate displacements
                rxij = rxi - coord[j, 0]
                ryij = ryi - coord[j, 1]
                rzij = rzi - coord[j, 2]

                # Minimum image convention
                rxij = rxij - round(rxij/box1) * box1


MD_main(27, 0.8)
