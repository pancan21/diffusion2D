"""
Solving the two-dimensional diffusion equation

Example acquired from https://scipython.com/book/chapter-7-matplotlib/examples/the-two-dimensional-diffusion-equation/
"""

import numpy as np

from src.output import output_plots

def do_timestep(u_nm1, u, D, dt, dx2, dy2):
    # Propagate with forward-difference in time, central-difference in space
    u[1:-1, 1:-1] = u_nm1[1:-1, 1:-1] + D * dt * (
            (u_nm1[2:, 1:-1] - 2 * u_nm1[1:-1, 1:-1] + u_nm1[:-2, 1:-1]) / dx2
            + (u_nm1[1:-1, 2:] - 2 * u_nm1[1:-1, 1:-1] + u_nm1[1:-1, :-2]) / dy2)

    u_nm1 = u.copy()
    return u_nm1, u

def solver(
    nsteps=101, n_output=[0, 10, 50, 100], 
    w=10., h=10., dx=1.0, dy=1.0,
    D=4., T_cold=300, T_hot=700):
    
    # Number of discrete mesh points in X and Y directions
    nx, ny = int(w / dx), int(h / dy)

    # Computing a stable time step
    dx2, dy2 = dx * dx, dy * dy
    dt = dx2 * dy2 / (2 * D * (dx2 + dy2))

    print("dt = {}".format(dt))
    u0 = T_cold * np.ones((nx, ny))
    u = u0.copy()

    # Initial conditions - circle of radius r centred at (cx,cy) (mm)
    r, cx, cy = 2, 5, 5
    r2 = r ** 2
    for i in range(nx):
        for j in range(ny):
            p2 = (i * dx - cx) ** 2 + (j * dy - cy) ** 2
            if p2 < r2:
                u0[i, j] = T_hot

    # Number of timesteps
    nsteps = 101
    n_output = [0, 10, 50, 100]  # Save data and create figures at these timesteps
    us = []  # List to save data

    # Time loop
    for n in range(nsteps):
        u0, u = do_timestep(u0, u, D, dt, dx2, dy2)
        if n in n_output:
            us.append(u.copy())

    return us, dt, T_cold, T_hot

if __name__ == "__main__":
    # plate size, mm
    w = h = 10.
    # intervals in x-, y- directions, mm
    dx = dy = 0.1
    # Thermal diffusivity of steel, mm^2/s
    D = 4.
    # Initial cold temperature of square domain
    T_cold = 300
    # Initial hot temperature of circular disc at the center
    T_hot = 700

    # Number of timesteps
    nsteps = 101
    n_output = [0, 10, 50, 100] # Save data and create figures at these timesteps

    us, dt, T_cold, T_hot = solver(dx=dx, dy=dy, D=D, nsteps=nsteps, n_output=n_output, T_cold=T_cold, T_hot=T_hot)

    # Plot output figures
    output_plots(us, n_output, dt, T_cold, T_hot)