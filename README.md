# VicsekSim-repo
Simulation of Vicsek Model for the Bachelor Thesis.
Structure of the code based on "Understanding Molecular Simulation" by Daan Frenkel and Berend Smit chapter 4.
Raw structure:
1. Initialization
2. Loop 
- Determine Forces 
- Integrate equations of motion 
- sample averages (plot)

## Initialization (from the paper)
1. square shaped cell of linear size L
2. periodic boundary conditions
3. interaction radius r as the unit to measure distances (r = 1)
4. time unit dt = 1
5. initial conditions:
    + at time t = 0, N particles were randomly distributed in the cell 
    + they have the same absolute velocity varT (vartheta)
    + they have randomly distributed directions Theta 




### Questions: 
1. can 2 particles have the same position?
2. Particle class: sometimes the norm is not =1 but =0.9999999 how to fix?