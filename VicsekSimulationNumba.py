"""Vicsek Simulation optimized by numbas @jitn
    creates a simulation of phase transition in a system of self-driven particles after Tamas Vicsek
    
    In the following '_i' labels the variable at the timestep i and '_ip1' labels the variable at the timestep i plus 1"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit
import pandas as pd

@njit
def numbaUpdateX(N, x_i, v_i, L):
    """Updates the positions of a particles; timestep is set to 1; 1 step of numbaUpdate
        Args.:  N (int) - Number of particles
                x_i (array Nx2) - positions of the particles at timestep i
                v_i (array Nx2) - calculated velocitys of the particles
                L (double) - length of the box
        Return: x_ip1 (array Nx2) - positions of the partiles at timestep i+1"""
    #Integration
    x_ip1 = x_i + v_i

    #Putting the particle back in the box
    for i in range(N):
        for j in range(2):
            if x_ip1[i,j] > L:
                x_ip1[i,j] = x_ip1[i,j] - L
            if x_ip1[i,j] < 0:
                x_ip1[i,j] = x_ip1[i,j] + L

    return x_ip1

@njit
def numbaUpdateTheta(x_i, N, theta_i, L):
    """Updates the angles of the particles without noise; update radius is set to 1; 2 step of numbaUpdate
        Args.:  x_i (array 2xN) - positions of the particles at timestep i
                N  (int) - number of particles
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
            for k in range(2):
                if xr[k] > L/2:
                    xr[k] = L - xr[k]
            #xr = xr - L/2 * np.array((int(xr[0]/(L/2)),int(xr[1]/(L/2))))
            #radius of influence set to 1
            if np.linalg.norm(xr) <= 1:
                thetaSin.append(np.sin(theta_i[j]))
                thetaCos.append(np.cos(theta_i[j]))
        #claculation of the new angle for every particle
        thetaSin = np.mean(np.array(thetaSin))
        thetaCos = np.mean(np.array(thetaCos))
        #if thetaCos < 0:
        #    theta = np.arctan(thetaSin/ thetaCos) + np.pi
        #else: theta = np.arctan(thetaSin/ thetaCos)
        theta_ip1[i] = np.arctan2(thetaSin, thetaCos)

    return theta_ip1

@njit
def numbaUpdateV(N, theta_ip1, varT, eta):
    """Updates the velocities of the particles; 3 step of numbaUpdate
        Args.:  N (int) - number of particles
                theta_ip1 (array 1xN) - angles of the particles at timestep i+1 without noise
                varT (double) - given absolute velocity of each particle
                eta (double) - noise added to theta
        Return: v_ip1 (array 2xN) - velocities of particles at timestep i+1
                theta_ip1 (array 2xN) - angles of the particles at timestep i+1 with noise"""
    v_ip1 = np.zeros((N,2))
    for i in range(N):
        #adding noise to the calculated angle theta for the timestep i+1
        theta_ip1[i] = theta_ip1[i] + np.random.uniform(-eta/2,eta/2)
        #calculating the vector v out of the angle with given absolute velocity varT
        v_ip1[i] = np.array((np.cos(theta_ip1[i]), np.sin(theta_ip1[i]))) * varT

    return v_ip1, theta_ip1

@njit
def numbaUpdate(N, varT, L, x_i, v_i, theta_i, eta):
    """Calculates the simluation after one timestep (i+1) with the following steps:

        1. updates the positions of the particles - x_i+1 = x_i + v_i * t
        2. updates the angles theta - theta_i+1 = arctan(meanSin(theta_i)(x_i)/meanCos(theta_i)(x_i))
        3. updates the velocities v - v_i+1 = (cos(theta_i+1), sin(theta_i+1)) * varT

        Args.:  N (int) - number of the particles 
                varT (double) - given absolute velocity of each particle
                L (double) - length of the box
                x_i (array Nx2) - positions of the particles at timestep i
                v_i (array Nx2) - velocities of the particles at timestep i
                theta_i (array Nx2) - angles of the particles at timestep i
                eta (double) - noise added to angle theta
        Return: x_ip1 (array Nx2) - positions of the particles at timestep i+1
                theta_ip1 (array Nx2) - angles of the particles at timestep i+1
                v_ip1 (array Nx2) - velocities of the particles at timestep i+1
                """
    #Step 1
    x_ip1 = numbaUpdateX(N, x_i, v_i, L)
    #Step 2
    theta_ip1 = numbaUpdateTheta(x_i, N, theta_i, L)
    #Step 3 
    v_ip1, theta_ip1 = numbaUpdateV(N, theta_ip1, varT, eta)

    return x_ip1, theta_ip1, v_ip1

@njit
def calculateVa(N, v, varT):
    """Calculates the absolute value of the average normalized velocity v_a of the particles
        Args.:  N (int) - number of the particles
                v (array Nx2) - current velocities of the particles
                varT (double) - given absolute velocity of each particle
        Return: v_a (double) - absolute value of the average normalized velocity"""
    #Calculating the sum of all velocities
    sumV = np.array((np.sum(v[:,0]), np.sum(v[:,1])))
    #Calculating v_a after the formular given in the paper
    v_a = 1/(N * varT) * np.linalg.norm(sumV)
    return v_a


def vaOfEta(N, rho, nNoise, steps, reps, equidistant):
    """Calculates the absolute value of the average normalized velocity v_a of the particles in one system for different noise eta
        Args.:  N (int) - number of particles
                rho (double) - density of  the system; N and rho define the Length L of the box
                nNoise (int) - number of different noises for whose v_a will be calculated
                steps (int) - number of timesteps; v_a of the system will be calculated afterwards
                reps (int) - number of repetitions for a single eta
                equidistant (boolean) - sets if array eta has equidistant steps or not
        Return: v_a (array) - array of all v_a for different noises in the same system
                eta (array) - corresponding noise eta to the v_a values"""
    #Calculation of the length of the box L 
    L = np.sqrt( N/rho )
    v_a = np.array(())
    #Creates the array of eta (equidistant or not)
    if equidistant:
        eta = np.linspace(0,5,nNoise)
    else:
        eta = np.linspace(0,2,3)
        eta = np.append(eta, np.linspace(2.5, 5, nNoise-3))

    
    for i in range(nNoise):
        v_aNoise = np.array(())
        for k in range(reps):
            v_aReps = np.array(())
            #Initialization of one System with the i-th eta
            vi = Vicsek(N, L, eta[i], False)
            #n timesteps of the simulation
            for j in range(100):
                x_ip1, theta_ip1, v_ip1 = numbaUpdate(vi.N, vi.varT, vi.L, vi.x_i, vi.v_i, vi.theta_i, vi.eta)
                vi.x_i, vi.theta_i, vi.v_i = x_ip1, theta_ip1, v_ip1
            #After 100 ts v_a is calculated after every ts 
            for j in range(steps - 100):
                x_ip1, theta_ip1, v_ip1 = numbaUpdate(vi.N, vi.varT, vi.L, vi.x_i, vi.v_i, vi.theta_i, vi.eta)
                vi.x_i, vi.theta_i, vi.v_i = x_ip1, theta_ip1, v_ip1
                v_aReps = np.append(v_aReps, calculateVa(N, vi.v_i, vi.varT))
        
            #Adding the mean of the calculated v_a 
            v_aNoise = np.append(v_aNoise, np.mean(v_aReps))
        v_a = np.append(v_a, np.mean(v_aNoise) )


    #Saving data in csv
    dict = {'v_a': v_a, 'eta' : eta}
    df = pd.DataFrame(dict)
    df.to_csv(f"vaOfEta"+str(N)+"calculated.csv")
    
    #Plot of the data
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim([0,5])
    ax.set_ylim([0,1])
    ax.set_ylabel("v_a")
    ax.set_xlabel("eta")
    ax.scatter(eta, v_a, label =N)
    plt.show()

    return v_a, eta

def vaOfRho(steps, eta, n = 20, L = 20):
    """Calculates the absolute value of the average normalized velocity v_a of the particles in one system for different densities rho
        Args.:  n (int) - number of different densities for whose v_a will be calculated
                steps (int) - number of timesteps; v_a will be calculated afterwars
                eta (doible) - noise of the system; stays the same for all simulations
                L (double) - Length of the box: intial value 20 (value of the paper)
        Return: v_a (array) - array of all v_a for different densities in the same system
                rho (array) - corresponding densities rho to the v_a values"""
    v_a = np.array(())
    rho = np.linspace(0.1,4,n-2)
    rho = np.append(rho, np.linspace(5, 6, 2))
    for i in range(n):
        #Initialization of one system with the i-th rho
        N = int(L**2 * rho[i])
        vi = Vicsek(N, L, eta, False)
        #n timesteps of the simulation 
        for j in range(steps):
            x_ip1, theta_ip1, v_ip1 = numbaUpdate(vi.N, vi.varT, vi.L, vi.x_i, vi.v_i, vi.theta_i, vi.eta)
            vi.x_i, vi.theta_i, vi.v_i = x_ip1, theta_ip1, v_ip1
        #Adding v_a after n timesteps with the density rho[i] to the array
        v_a = np.append(v_a, calculateVa(N, vi.v_i, vi.varT))
    return v_a, rho

def vaOfT(steps, eta, L, N):
    """Calculates the absolute value of the average normalized velocity v_a of the particles in one system for each time step with given noise and density
        Args.:  steps (int) - number of timesteps
                eta (double) - noise of the system
                L (double) - length of the box
                N (int) - number of particles in the system
        Return: v_a (array) -  calculated v_a's for each timestep"""
    v_a = np.array(())
    t = np.arange(0,steps,1)
    vi = Vicsek(N, L, eta, False)
    for i in range(steps):
        v_a = np.append(v_a, calculateVa(N, vi.v_i, vi.varT))
        x_ip1, theta_ip1, v_ip1 = numbaUpdate(vi.N, vi.varT, vi.L, vi.x_i, vi.v_i, vi.theta_i, vi.eta)
        vi.x_i, vi.theta_i, vi.v_i = x_ip1, theta_ip1, v_ip1
    return v_a



class Vicsek:

    def __init__(self, N, L, eta, plot, varT = 0.03):
        """Initialization of Vicsek
            Args.:  N (int) - number of the particles 
                    L (double) - length of the box
                    eta (double) - noise added to angle theta
                    varT (double) - given absolute velocity of each particle; set to varT = 0.03
            Attr.:  N (as above)
                    L (as above)
                    varT (as above)
                    x_i (array Nx2) - positions of the particles at timestep i
                    v_i (array Nx2) - velocities of the particles at timestep i
                    theta_i (array Nx2) - angles of the particles at timestep i
                    fig (plt figure) - figure of the plot
                    ax1 (plt axis) - axis of the plot
                    """
        self.N = N
        self.L = L
        self.eta = eta
        self.varT = varT
        self.x_i = np.zeros((N,2))
        self.theta_i = np.zeros(N)
        self.v_i = np.zeros((N,2))
        for i in range(N):
            #init of x with random positions
            self.x_i[i] = np.array((np.random.random() * L, np.random.random() * L))
            #init of theta with random angles
            self.theta_i[i] = 2 * np.pi * np.random.random()
            #calculating v out of the angle and the absolute velocity varT
            self.v_i[i] = np.array((np.cos(self.theta_i[i]), np.sin(self.theta_i[i]))) * varT

        #Plot of the Initialization
        if plot:
            self.fig = plt.figure()
            self.ax1 = self.fig.add_subplot(1,1,1)
            self.ax1.quiver(self.x_i[:,[0]],self.x_i[:,[1]],self.v_i[:,[0]],self.v_i[:,[1]])
            self.ax1.set_xlim([0,L])
            self.ax1.set_ylim([0,L])

    def funcUpdate(self, i):
        """The method FuncAnimation performed for each frame; plots the resent positions of the particles
            Args.:  (i (int) - frame number; used in animation.FuncAnimation) """
        #Resetting the axes
        self.ax1.clear()
        self.ax1.set_xlim([0,self.L])
        self.ax1.set_ylim([0,self.L])

        #One timestep of the simulation
        x_ip1, theta_ip1, v_ip1 = numbaUpdate(self.N, self.varT, self.L, self.x_i, self.v_i, self.theta_i, self.eta)
        #print(np.linalg.norm(self.x_i-x_ip1))
        self.x_i, self.theta_i, self.v_i = x_ip1, theta_ip1, v_ip1
        #Plot
        self.ax1.quiver(self.x_i[:,[0]],self.x_i[:,[1]],self.v_i[:,[0]],self.v_i[:,[1]])
        

    def animate(self, maxFrames, name):
        """Method to animate the simulation
            Args.:  maxFrames (int) - number of the Frames (and timesteps) for the animation
                    name (string) - name of the saved .gif animation"""
        simu = animation.FuncAnimation(self.fig, self.funcUpdate, frames = np.arange(0,maxFrames), interval = 25)
        simu.save(name+".gif", dpi = 80)