# VicsekSim-repo


The full Code to run the simulations is in VicsekSimulationNumba.py.
The used Data for the thesis is in the directory Data. 
The jupyter notebooks were used to produce the plots.


## Examples

### To Run a simulation: 

import VicsekSimulationNumba as vi

v = vi.Viscek(300, 25, 0.1, True) \\this gives a plot of the initial conditions

v.animate(400, 'name') \\this creates an animation and saves it in the directory

### To compute the graphs: 

import VicsekSimulationNumba as vi

vi.vaOfRho(20, 2, 15, False, 100)

vi.vaOfEta(40, 4, 20, 5, 100, 20, 10)

v_a = vi.vaOfT(2000,2,20,40,'DiffN40N')


