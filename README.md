# VicsekSim-repo
Simulation of Vicsek Model for the Bachelor Thesis.
Structure of the code based on "Understanding Molecular Simulation" by Daan Frenkel and Berend Smit chapter 4.
Raw structure:
1. Initialization
2. Loop 
- Determine Forces 
- Integrate equations of motion (vicsek formular)
- sample averages (plot)

## Initialization (from the paper)
1. square shaped cell of linear size L
2. periodic boundary conditions (-> loop)
3. interaction radius r as the unit to measure distances (r = 1) (-> loop)
4. time unit dt = 1 (-> loop)
5. initial conditions:
    + at time t = 0, N particles were randomly distributed in the cell 
    + they have the same absolute velocity varT (vartheta)
    + they have randomly distributed directions Theta 

## Loop for a timestep dt (ts)
1. update the position of every particle -> boundary conditions
2. for i-th particle
    + calculate the sin and cos of every particle in the circle  -> boundary conditions
    + calculate mean of the directions
    + update direction for i-th particle
5. plot 


### Questions: 
