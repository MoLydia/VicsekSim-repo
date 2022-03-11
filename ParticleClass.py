""" -- Class Particle -- """

class Particle():
    """A Particle with a vectorial position x, absolute velocity varT and a vectorial direction theta"""

    def __init__(self, x, varT, theta):
        """Constructor of Particle
                Args:   x - (array) position
                        varT - (double) absolute velocity
                        theta - (array) direction
                Attr:   same as Args
                        v - (array) velocity"""

        self.x = x
        self.varT = varT
        self.theta = theta
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



