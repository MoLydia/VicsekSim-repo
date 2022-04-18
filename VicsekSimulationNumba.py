"""Vicsek Simulation optimized by numba"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit

@jit
def numbaUpdateX(N, x_i, v_i, L):
    """Updates the positions of a particles; timestep is set to 1; 1. step of numbaUpdate
        Args.:  N (int) - Number of particles
                x_i (array 2xN) - positions of the particles at timestep i
                v (array 2xN) - calculated velocitys of the particles
                L (double) - length of the box
        Return: x_ip1 (array 2xN) - positions of the partiles at timestep i+1"""
    x_ip1 = x_i + v_i
    for i in range(N):
        for j in range(2):
            while x_ip1[i,j] > L:
                x_ip1[i,j] =  x_ip1[i,j] - L
            while x_ip1[i,j] < 0:
                x_ip1[i,j] = x_ip1[i,j] + L
    return x_ip1

@jit
def numbaUpdateTheta(x_i, N, theta_i, L):
    """Updates the angles of the particles without noise; update radius is set to 1; 2. step of numbaUpdate
        Args.:  N  (int) - number of particles
                x_i (array 2xN) - positions of the particles at timestep i
                theta_i (array 1xN) - angles of the particles at the timestep i
                L (double) - length of the box
        Return: theta_ip1 (array 1xN) - angles of the particles at the timestep i+1 without noise"""
    theta_ip1 = np.zeros(N)
    for i in range(N):
        thetaSin = []
        thetaCos = []
        for j in range(N):
            xr = x_i[i] - x_i[j]
            #periodic boundary conditions
            xr = xr - L/2 * np.array((int(xr[0]/(L/2)),int(xr[1]/(L/2))))
            if np.linalg.norm(xr) <= 1:
                thetaSin.append(np.sin(theta_i[j]))
                thetaCos.append(np.cos(theta_i[j]))
        theta_ip1[i] = np.arctan2(np.mean(thetaSin), np.mean(thetaCos))

    return theta_ip1

@jit
def numbaUpdateV(N, theta_ip1, varT, eta):
    """Updates the velocities of the particles; 3. step of numbaUpdate
        Args.:  N (int) - number of particles
                theta_ip1 (array 1xN) - angles of the particles at timestep i+1 without noise
                varT (double) - given absolute velocity of each particle
                eta (double) - noise of the simulation
        Return: v_ip1 (array 2xN) - velocities of particles at timestep i+1
                theta_ip1 (array 2xN) - angles of the particles at timestep i+1 with noise"""
    v_ip1 = np.zeros(N)
    for i in range(N):
        theta_ip1[i] += np.random.uniform(-eta/2,eta/2)
        v_ip1 = np.array((np.cos(theta_ip1[i]), np.sin(theta_ip1[i]))) * varT
    return v_ip1, theta_ip1

@jit
def numbaUpdate(N, varT, L, x_i, v_i, theta_i, eta):
    """Calculates the simluation after one timestep (i+1) with the following steps:

        1. updates the positions of the particles - x_i+1 = x_i + v_i * t
        2. updates the angles theta - theta_i+1 = arctan(meanSin(theta_i)(x_i)/meanCos(theta_i)(x_i))
        3. updates the velocities v - v_i+1 = (cos(theta_i+1), sin(theta_i+1)) * varT

        Args.:  N (int) - number of the particles 
                varT (double) - given absolute velocity of each particle
                L (double) - length of the box
                x_i (array 2xN) - positions of the particles at timestep i
                v_i (array 2xN) - velocities of the particles at timestep i
                theta_i (array 2xN) - angles of the particles at timestep i
        Return: x_ip1 (array 2xN) - positions of the particles at timestep i+1
                theta_ip1 (array 2xN) - angles of the particles at timestep i+1
                v_ip1 (array 2xN) - velocities of the particles at timestep i+1
                """
    #Step 1
    x_ip1 = numbaUpdateX(x_i, N, v_i, L)
    #Step 2
    theta_ip1 = numbaUpdateTheta(x_i, N, theta_i, L)
    #Step 3 
    v_ip1, theta_ip1 = numbaUpdateV(N, theta_ip1, varT, eta)

    return x_ip1, theta_ip1, v_ip1

def Funcupdate(ax1, N, varT, L, x_i, v_i, theta_i, eta, i):
    """The method FuncAnimation performs for each frame; plots the resent positions of the particles
        Args.:  ax1 (plt.axis) - axis of the plot 
                L (double) - length of the box
                (i (int) - frame number; used in animation.FuncAnimation) """
    #Resetting the axes
    ax1.clear()
    ax1.set_xlim([0,L])
    ax1.set_ylim([0,L])

    #One timestep of the simulation
    x_ip1, theta_ip1, v_ip1 = numbaUpdate(N, varT, L, x_i, v_i, theta_i, eta)
    
    #Plot
    ax1.quiver(x_ip1[:,[0]],x_ip1[:,[1]],v_ip1[:,[0]],v_ip1[:,[1]] )
    
    
def main(N, varT, L, animate, eta, frameMax, name):
    """Simulates the phase transition of active particles after the Vicsek model; main method"""

    #Initialization
    x_i = np.zeros((2,N))
    theta_i = np.zeros((2,N))
    v_i = np.zeros((2,N))
    for i in range(N):
        x_i[i] = np.array((np.random.random() * L, np.random.random() * L))
        theta_i[i] = 2 * np.pi * np.random.random()
        v_i[i] = np.array((np.cos(theta_i), np.sin(theta_i))) * varT

    #Plot of the Initialization
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.quiver(x_i[:,[0]],x_i[:,[1]],v_i[:,[0]],v_i[:,[1]])
    ax1.set_xlim([0,L])
    ax1.set_ylim([0,L])
    plt.show()

    #Animation of the Simulation if animate == True
    if animate:
        simu = animation.FuncAnimation(fig, Funcupdate, frames = np.arange(0,frameMax), interval = 25)
        simu.save(name+".gif", dpi = 80)


