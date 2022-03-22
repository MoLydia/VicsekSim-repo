""" -- Class Particle -- """

from turtle import clear
import numpy as np

class Particle():
    """A Particle with a vectorial position x, absolute velocity varT and a vectorial direction theta"""

    def __init__(self, L, varT):
        """Constructor of Particle
                Args:   varT - (double) absolute velocity
                        L - (int) length of the cell
                Attr:   same as Args
                        x - (array) position
                        v - (array) velocity
                        theta - (array) angle to x axis
                        nextT - (array) the direction after the next update"""
        
        self.L = L
        self.x = np.array((np.random.random() * L, np.random.random() * L))
        self.varT = varT
        self.theta = 2 * np.pi * np.random.random()
        self.v = np.array((np.cos(self.theta), np.sin(self.theta))) * self.varT
        self.nextT = 0

    @property
    def x(self):
        """Getter of x"""
        return self._x

    @x.setter
    def x(self, value):
        if value[0] > self.L or value[1] > self.L :
            raise ValueError("The Particle is out of the cell")
        else:
            self._x = value

    @property
    def theta(self):
        """Getter of theta"""

        return self._theta

    @theta.setter
    def theta(self, value):
        """setter of theta: the absolute value of theta has to be 1"""

        if np.linalg.norm(value) == 1:
            self._theta = value
        else: raise ValueError("The absolute value of theta has to be 1")

    def updateX(self, ts):
        """updates the position of a particle
            Arg.:   ts - (double) timestep of the update"""
        self.x = self.x + self.v * ts
    
    def updateV(self, eta):
        """updates the velocity of a particle
            Arg.: theta - (array) vector of the mean direction of the particles within the circle"""
        self.theta = self.nextT + np.array((np.random.uniform(-eta/2,eta/2),np.random.uniform(-eta/2,eta/2)))
        self.v = np.array((np.cos(self.theta), np.sin(self.theta))) * self.varT

        



