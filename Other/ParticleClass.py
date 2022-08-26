""" -- Class Particle -- """

import numpy as np

class Particle():
    """A Particle with a vectorial position x, absolute velocity varT and a vectorial direction theta"""

    def __init__(self, L, varT):
        """Constructor of Particle
                Args:   varT - (double) absolute velocity
                        L - (int) length of the cell
                Attr:   same as Args
                        x - (array) position
                        xold - (array) old positions to calculate new theta
                        v - (array) velocity
                        theta - (array) angle to x axis
                        nextT - (array) the direction after the next update"""
        
        self.L = L
        self.x = np.array((np.random.random() * L, np.random.random() * L))
        self.xold = None
        self.varT = varT
        self.theta = 2 * np.pi * np.random.random()
        self.v = np.array((np.cos(self.theta), np.sin(self.theta))) * self.varT
        self.nextT = None

    @property
    def x(self):
        """Getter of x"""
        return self._x

    @x.setter
    def x(self, value):
        if value[0] > self.L or value[1] > self.L or value[0] < 0 or value[1] < 0 :
            raise ValueError("The Particle is out of the box")
        else:
            self._x = value


    def updateX(self, ts):
        """updates the position of a particle
            Arg.:   ts - (double) timestep of the update"""
        self.xold = self.x
        x = self.x + self.v * ts
        for i in range(2):
            if x[i] > self.L:
                x[i] =  x[i] - self.L
            if x[i] < 0:
                x[i] = x[i] + self.L
        self.x = x
    
    def updateV(self, eta):
        """updates the velocity of a particle
            Arg.: theta - (array) vector of the mean direction of the particles within the circle"""
        self.theta = self.nextT + np.random.uniform(-eta/2,eta/2)
        self.v = np.array((np.cos(self.theta), np.sin(self.theta))) * self.varT




