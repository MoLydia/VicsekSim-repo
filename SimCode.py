#Initialization 
import ParticleClass as par
import CellClass as cell
import numpy as np

def init(L, N, eta, varTB = 0.03):
        """initialized the particles and the cell
                Args.:  L - (int) size of the square shaped cell
                        N - (int) particle number 
                        eta - (double) noise of the system
                        varTB - (double) absolute velocity at the beginning (default value of the paper 0,03)
                Attr.:  rho - (double) density of the system (rho = N/L**2)"""
        rho = N/L**2
        parA = []
        for i in range(N):
                parA.append(par.Particle(L, varTB))

def getXarray(parA):
        """Method to get an array consisting of all the particles positions in parA 
                Args.:  parA - (array) array of N particles
                Return: xA - (array) array of the positions"""
        xA = []
        for par.Particle in parA:
                xA.append(par.Particle.x)
        return np.array(xA)

def getVarray(parA):
        """Method to get an array consisting of all the particles velocities in parA 
                Args.:  parA - (array) array of N particles
                Return: vA - (array) array of the velocities"""
        vA = []
        for par.Particle in parA:
                vA.append(par.Particle.v)
        return np.array(vA)

def update(L, parA, ts = 1):
        """Updates the Simulation: calculates the new positions and velocities after one timestep"""
        for i in parA:
                i.updateX(ts)
        for i in parA:
                thetaR = []
                for j in parA:
                        xr = np.linalg.norm(i.x - j.x)
                        xr = xr - L * np.rint(xr/L)
                        if xr <= 1:
                                thetaR.append([np.sin(j.theta),np.cos(j.theta)])
                thetaR = np.array(thetaR)
                
