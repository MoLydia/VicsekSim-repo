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
        parA.append(par(L, varTB))
