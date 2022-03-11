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
                        v - (array) velocity
                        theta - (array) direction"""

        self.x = np.array((np.random.random() * L, np.random.random() * L))
        self.varT = varT
        self.theta = np.array((np.random.random() * L, np.random.random() * L))
        self.theta = self.theta / abs(self.theta)
        self.v = self.theta * self.varT

    @property
    def theta(self):
        """Getter of theta"""

        return self._theta

    @theta.setter
    def theta(self, value):
        """setter of theta: the absolute value of theta has to be 1"""

        if abs(value) == 1:
            self._theta = value
        else: raise ValueError("The absolute value of theta has to be 1")



