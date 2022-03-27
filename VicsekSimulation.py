"""Class Vicsek Simulation"""

import ParticleClass as par
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class VicsekSimulation:

    def __init__(self, L, N, eta, ts, varTB = 0.03):
        """initialized the particles and the cell
                Args.:  L - (int) size of the square shaped cell
                        N - (int) particle number 
                        eta - (double) noise of the system
                        varTB - (double) absolute velocity at the beginning (default value of the paper 0,03)
                Attr.:  rho - (double) density of the system (rho = N/L**2)"""
        self.N = N
        self.L = L
        self.rho = N/L**2
        self.eta = eta
        self.ts = ts
        self.parA = []
        for i in range(N):
                self.parA.append(par.Particle(L, varTB))
        self.animate = animate
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(1,1,1)
        x = self.getXarray()
        v = self.getVarray()
        self.ax1.quiver(x[:,[0]],x[:,[1]],v[:,[0]],v[:,[1]] )
        self.ax1.set_xlim([0,self.L])
        self.ax1.set_ylim([0,self.L])

    def getXarray(self):
        """Method to get an array consisting of all the particles positions in parA 
                Args.:  parA - (array) array of N particles
                Return: xA - (array) array of the positions"""
        xA = []
        for p in self.parA:
                xA.append(p.x)
        return np.array(xA)

    def getVarray(self):
        """Method to get an array consisting of all the particles velocities in parA 
                Args.:  parA - (array) array of N particles
                Return: vA - (array) array of the velocities"""
        vA = []
        for p in self.parA:
                vA.append(p.v)
        return np.array(vA)


    def update(self):
        """Updates the Simulation: calculates the new positions and velocities after one timestep"""
        for i in self.parA:
                i.updateX(self.ts)
        for i in self.parA:
                thetaSin = []
                thetaCos = []
                for j in self.parA:
                        xr = np.linalg.norm(i.x - j.x)
                        xr = xr - self.L * np.rint(xr/self.L) #Boundary Conditions
                        if abs(xr) <= 1:
                                thetaSin.append(np.sin(j.theta))
                                thetaCos.append(np.cos(j.theta))
                
                theta = np.arctan(np.mean(thetaSin)/ np.mean(thetaCos))
                i.nextT = theta
        for i in self.parA:
                i.updateV(self.eta)


    def Funcupdate(self, i):
        self.ax1.clear()
        self.ax1.set_xlim([0,self.L])
        self.ax1.set_ylim([0,self.L])

        self.update()
        x = self.getXarray()
        v = self.getVarray()
        self.ax1.quiver(x[:,[0]],x[:,[1]],v[:,[0]],v[:,[1]] )
    
    
    def AnimateMatplot(self):
        simu = animation.FuncAnimation(self.fig, self.Funcupdate, frames = np.arange(0,100), interval = 25)
        simu.save("animation.gif", dpi = 80)

    
