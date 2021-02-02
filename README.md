# HardSphereGas
This is a 2D simulation of a hard sphere gas. The number of "spheres" and their size can be varied, and the temperature and pressure of the system calculated.

Files included(19)
- This README file
- objects.py
- simulation.py
- testing.py
- investigations.py
- plotting.py
- locate_balls.py

following datafiles for optional plotting(12):
- p_T_data_with dev2.txt
- p_T_A_data3[1].txt
- p_T_N_data.txt
- v_dist_data_new.txt
- p_T_A_data3[1].txt
- p_T_A_data2[1].txt
- p_T_A_data[1].txt
- p_T_A_data5.txt
- p_T_data_with dev3.txt
- p_T_data_with dev4.txt
- p_T_data_with dev5.txt
- size_comparison_PTdata2.txt


Instructions:

All the classes (3) required to run the simulation are found in modules objects.py and simulation.py. The script testing.py contains the essential tests to check that the simulation is working as expected, including animation, distance histograms and conservation plots. The script investigations.py can be used to run the different thermodynamics experiments and produce plots of the measurements. However, for some of these, the run might take a long time due to the number of datapoints and collisions required, so plotting.py offers an alternative to produce the same plots of pre-existing datafiles. The script locate_balls is a demonstration for the packing algorithm of balls used in the simulation. It can be also used to test out different scaling factors and find the best packing for a ball size.

Important notes:

Please only run one section at a time in testing.py, investigations.py and plotting.py. Each section is written as an independent script that can be run by uncommenting it.

investigations.py writes the measurement data on files. Remember to change filename if don't want to overwrite

Depending on the ball size and number it may be wise to adjust the scaling in the ball packing algorithm to avoid packing them too tightly. locate_balls can be used to visually test the packing to make sure it is appropriate.
