"""
Python Script to perform an NVE MD simulation of a medel fluid with a LJ 6-8
pair potential:

u(r) = 10\epsilon[(\sigma/r)^8 - (\sigma/r)^6]

We estimate the compressibility factor and internal energy at T_star = 1.0,
rho_star = 0.8; where standard LJ units are imployed:

T_star = kT/(\epsilon), rho_star = (\rho)(\sigma)^3

Adapted from mathematica_nve_md_demo2.nb
"""

# Define box parameters for simulation
n = 27  # number of particles
boxl = 3.2316520350478255 # box side length
timestep = 0.005

# Calculate density and half box dimension
boxl2 = boxl/2
boxl3 = boxl/(n**(1/3))  # not sure what this is
rho = n/boxl**3
print("Denisty:", rho)  # 0.8

