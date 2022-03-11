# VicsekSim-repo
Simulation of Vicsek Model for the Bachelor Thesis.

Structure of the code based on "Understanding Molecular Simulation" by Daan Frenkel and Berend Smit chapter 4.
Raw structure:
1. Initialization
2. Loop 
    2.1 Determine Forces 
    2.2 Integrate equations of motion 
    2.3 sample averages (plot)

##Initialization (from the paper)
-square shaped cell of linear size L
-periodic boundary conditions
-interaction radius r as the unit to measure distances (r = 1)
-time unit dt = 1
-initial conditions:
    + at time t = 0, N particles were randomly distributed in the cell 
    + they have the same absolute velocity varT (vartheta)
    + they have randomly distributed directions Theta 




Questions: 
    -can 2 particles have the same position?
    -Particle class: sometimes the norm is not =1 but =0.9999999 how to fix?