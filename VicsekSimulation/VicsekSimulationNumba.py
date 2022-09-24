"""Vicsek Simulation optimized by numbas @jitn
    creates a simulation of phase transition in a system of self-driven particles after Tamas Vicsek
    
    In the following '_i' labels the variable at the timestep i and '_ip1' labels the variable at the timestep i plus 1"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit
import pandas as pd
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

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
            x_ip1[i,j] = x_ip1[i,j] - L*(x_ip1[i,j]//L)

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
        thetaSin = np.zeros(N)
        thetaCos = np.zeros(N)
        count = 0
        for j in range(N):
            xr = x_i[i] - x_i[j]
            #periodic boundary conditions
            xr = xr - L * np.rint(xr/L)

            #radius of influence set to 1
            if xr[0]**2 + xr[1]**2 <= 1:
                thetaSin[count] = np.sin(theta_i[j])
                thetaCos[count] = np.cos(theta_i[j])
                count+=1
        #claculation of the new angle for every particle
        theta_ip1[i] = np.arctan2(np.mean(thetaSin[:count]), np.mean(thetaCos[:count]))

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


def vaOfEta(N, rho, nNoise, noiseMax, start, tau, reps):
    """Calculates the absolute value of the average normalized velocity v_a of the particles in one system for different noise eta
        Args.:  N (int) - number of particles
                rho (double) or L (int) - density of  the system; N and rho define the Length L of the box or Length of the box L
                nNoise (int) - number of different noises for whose v_a will be calculated
                steps (int) - number of timesteps; v_a of the system will be calculated afterwards
                reps (int) - number of repetitions for a single eta
        Return: v_a (array) - array of all v_a for different noises in the same system
                eta (array) - corresponding noise eta to the v_a values"""
    #Calculation of the length of the box L 
    L = np.sqrt( N/rho )
    v_a = np.zeros(nNoise)
    #Creates the array of eta 
    eta = np.linspace(0,noiseMax,nNoise)
    error = np.zeros(nNoise)
    
    for i in range(nNoise):
        v_aNoise = np.zeros(reps)
        errorReps = np.zeros(reps)
        for k in range(reps):
            v_aReps = np.zeros(tau*10)
            #Initialization of one System with the i-th eta
            vi = Vicsek(N, L, eta[i], False)
            #n timesteps of the simulation
            for j in range(start):
                x_ip1, theta_ip1, v_ip1 = numbaUpdate(vi.N, vi.varT, L, vi.x_i, vi.v_i, vi.theta_i, vi.eta)
                vi.x_i, vi.theta_i, vi.v_i = x_ip1, theta_ip1, v_ip1
            #After start ts v_a is calculated after every ts 
            for j in range(tau*10):
                x_ip1, theta_ip1, v_ip1 = numbaUpdate(vi.N, vi.varT, vi.L, vi.x_i, vi.v_i, vi.theta_i, vi.eta)
                vi.x_i, vi.theta_i, vi.v_i = x_ip1, theta_ip1, v_ip1
                v_aReps[j] = calculateVa(N, vi.v_i, vi.varT)

            #Adding the mean of the calculated v_a 
            errorReps[k] = np.sqrt((np.mean(v_aReps[::tau]**2)-np.mean(v_aReps[::tau])**2)/(10-1))
            v_aNoise[k] = np.mean(v_aReps)
        error[i] = np.sqrt(np.sum(errorReps**2))/reps
        v_a[i] = np.mean(v_aNoise)


    #Saving data in csv
    dict = {'v_a': v_a, 'eta' : eta, 'error' : error}
    df = pd.DataFrame(dict)
    df.to_csv(f"vaOfEta"+str(N)+"calculatedRho"+str(rho*10)+"WithErrors.csv")
    
    #Plot of the data
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim([0,5])
    ax.set_ylim([0,1])
    ax.set_ylabel("v_a")
    ax.set_xlabel("eta")
    ax.errorbar(eta, v_a, yerr = error, label =N, fmt = '.', ecolor = 'red')
    plt.show()

    #return v_a, eta

def vaOfRho(tau, eta, reps, single, val, start, L = 20):
    """Calculates the absolute value of the average normalized velocity v_a of the particles in one system for different densities rho
        Args.:  n (int) - number of different densities for whose v_a will be calculated
                steps (int) - number of timesteps; v_a will be calculated afterwars
                eta (doible) - noise of the system; stays the same for all simulations
                L (double) - Length of the box: intial value 20 (value of the paper)
        Return: v_a (array) - array of all v_a for different densities in the same system
                rho (array) - corresponding densities rho to the v_a values"""
    

    #Init the different values for rho eta is calculated for 
    if single: 
        rho = np.array([val])
    else:
        rho = np.linspace(0.1,1.3,7)
        rho = np.append(rho, np.linspace(1.4,3.1,8))
    
    v_a = np.zeros(len(rho))
    error = np.zeros(len(rho))

    for i in range(len(rho)):
        N = int(L**2 * rho[i])
        v_aDensity = np.zeros(reps)
        errorReps = np.zeros(reps)
        for k in range(reps):
            v_aReps = np.zeros(tau*10)
            #Initialization of one System with the i-th rho
            vi = Vicsek(N, L, eta, False)
            #n timesteps of the simulation
            for j in range(start):
                x_ip1, theta_ip1, v_ip1 = numbaUpdate(vi.N, vi.varT, L, vi.x_i, vi.v_i, vi.theta_i, vi.eta)
                vi.x_i, vi.theta_i, vi.v_i = x_ip1, theta_ip1, v_ip1
            #After start ts v_a is calculated after every ts 
            for j in range(tau*10):
                x_ip1, theta_ip1, v_ip1 = numbaUpdate(vi.N, vi.varT, vi.L, vi.x_i, vi.v_i, vi.theta_i, vi.eta)
                vi.x_i, vi.theta_i, vi.v_i = x_ip1, theta_ip1, v_ip1
                v_aReps[j] = calculateVa(N, vi.v_i, vi.varT)
        
            #Adding the mean of the calculated v_a 
            errorReps[k] = np.sqrt((np.mean(v_aReps[::tau]**2)-np.mean(v_aReps[::tau])**2)/(10-1))
            v_aDensity[k] = np.mean(v_aReps)
        v_a[i] = np.mean(v_aDensity)
        error[i] = np.sqrt(np.sum(errorReps**2))/reps
        print(i)

    #Saving data in csv
    if single:
        dict = {'v_a': v_a, 'rho' : rho, 'error': error}
        df = pd.DataFrame(dict)
        df.to_csv(f"vaOfRho"+str(eta)+"SingleCalculated"+str(val)+"WithError.csv")

    else: 
        dict = {'v_a': v_a, 'rho' : rho, 'error': error}
        df = pd.DataFrame(dict)
        df.to_csv(f"vaOfRho"+str(eta)+"calculatedWithError.csv")
    
    #Plot of the data
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_xlim([0,10])
    ax.set_ylim([0,0.8])
    ax.set_ylabel("v_a")
    ax.set_xlabel("rho")
    ax.errorbar(rho, v_a, label =eta, yerr = error, fmt = '.', ecolor = 'red')
    plt.show()

    return v_a, rho

def vaOfT(steps, eta, L, N, name):
    """Calculates the absolute value of the average normalized velocity v_a of the particles in one system for each time step with given noise and density
        Args.:  steps (int) - number of timesteps
                eta (double) - noise of the system
                L (double) - length of the box
                N (int) - number of particles in the system
        Return: v_a (array) -  calculated v_a's for each timestep"""
    v_a = np.zeros(steps)
    t = np.arange(0,steps,1)
    vi = Vicsek(N, L, eta, False)   
    rho = N/L**2
    for i in range(steps):
        v_a[i] =  calculateVa(N, vi.v_i, vi.varT)
        x_ip1, theta_ip1, v_ip1 = numbaUpdate(vi.N, vi.varT, vi.L, vi.x_i, vi.v_i, vi.theta_i, vi.eta)
        vi.x_i, vi.theta_i, vi.v_i = x_ip1, theta_ip1, v_ip1

    dict = {'v_a(t)': v_a}
    df = pd.DataFrame(dict)
    df.to_csv(f"vaOfT"+name+".csv")

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
            self.fig.set_figwidth(10)
            self.fig.set_figheight(10)
            self.ax1 = self.fig.add_subplot(1,1,1)
            self.ax1.quiver(self.x_i[:,[0]],self.x_i[:,[1]],self.v_i[:,[0]],self.v_i[:,[1]], color = cm.get_cmap('viridis')(self.theta_i/(2*np.pi)))
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
        self.theta_i = np.where(self.theta_i > 0, self.theta_i, self.theta_i + 2*np.pi)
        self.ax1.quiver(self.x_i[:,[0]],self.x_i[:,[1]],self.v_i[:,[0]],self.v_i[:,[1]], color = cm.get_cmap('viridis')(self.theta_i/(2*np.pi)))
        

    def animate(self, maxFrames, name):
        """Method to animate the simulation
            Args.:  maxFrames (int) - number of the Frames (and timesteps) for the animation
                    name (string) - name of the saved .gif animation"""
        simu = animation.FuncAnimation(self.fig, self.funcUpdate, frames = np.arange(0,maxFrames), interval = 25)
        simu.save(name+".gif", dpi = 80)