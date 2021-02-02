"""
Hard spheres simulation
Anni Kauniskangas, 2020

Script testing.py contains the tests that were used to verify that the simulation was working properly.
Uncomment a section to run the test
"""
import numpy as np
import matplotlib.pyplot as plt
from objects import Ball
from objects import Container
from simulation import Simulation

# Settings for plots
SMALL_SIZE = 22
MEDIUM_SIZE = 24
BIGGER_SIZE = 26

plt.rc('font', size=SMALL_SIZE)          # default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#-----------------------------------------------------------------------------------------------------------------------

container1 = Container(R=10)
"""
# (Task 7) A simple animation test for multiple balls

sim1 = Simulation(container=container1,num_balls = 100,radius_balls=0.3)

sim1.run(500,animate=True)

"""
# Histograms of ball positions and separations
"""
sim1 = Simulation(container=container1,num_balls = 25,radius_balls=0.5)

r_magnitudes, separation_magnitudes = sim1.run(3000,r_histograms=True)
r_magnitudes = np.array(r_magnitudes)
separation_magnitudes = np.array(separation_magnitudes)

plt.figure(1)
plt.subplot(211)
plt.hist(r_magnitudes,20)
plt.tight_layout()
plt.title("Ball distance from origin",fontsize=26)
plt.subplot(212)
plt.hist(separation_magnitudes,20)
plt.title("Ball separation",fontsize=26)
plt.xlabel("x")
plt.tight_layout()
plt.show()

"""
# Plots of kinetic energy and momentum to check conservation
"""
num_sampl = 1000
sim1 = Simulation(container=container1, num_balls=50, radius_balls=0.1)
kinetic_energies = sim1.run(num_sampl, kinetic_energy=True)
x_momentums,y_momentums = sim1.run(num_sampl, momentum=True)

x = np.arange(0,num_sampl,1)

plt.figure(1)
plt.subplot(211)
plt.plot(x,kinetic_energies, label="kinetic energy", color="g")
plt.title("Kinetic energy and momentum",fontsize="28")
plt.ylabel("Total kinetic energy")
plt.legend()
plt.subplot(212)
plt.plot(x,x_momentums, label="x-momentum")
plt.plot(x,y_momentums, label="y-momentum")
plt.xlabel("Collision number")
plt.ylabel("Total momentum")
plt.legend()
plt.show()
"""