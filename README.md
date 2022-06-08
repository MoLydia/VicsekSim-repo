# VicsekSim-repo


## Examples

### To Run a simulation: 

import VicsekSimulationNumba as vi

v = vi.Viscek(300, 25, 0.1, True) \\this gives a plot of the initial conditions

v.animate(400, 'name') \\this creates an animation and saves it in the directory

### To compute the graphs: 

import VicsekSimulationNumba as vi

vi.vaOfRho(500, 0.1, 10, False)

vi.vaOfEta(40, 25, 20, 500, 100, True)

vi.vaOfT(2000, 0.5, 25, 300)


