"""
Hard spheres simulation
Anni Kauniskangas, 2020

Script locate_balls.py is for demonstrating the packing algorithm used for balls in Simulation and for experimenting
with it. For large number of balls, use smaller scaling factor s. For smaller ball radii increase s to spread them more
evenly.
"""

import numpy as np
import pylab as pl
from project_v2 import Ball
from project_v2 import Container


num_balls = 331
radius_balls = 0.1
r_theta = []
r_magnitude = []
s = 3 # Scaling factor for ball spacing. Change this and run to see how best fit wanted number of balls without packing
      # them too tightly

container = Container(R=10)

print(r_magnitude)

counter = 0
for i in range(0,num_balls):

    if counter >= num_balls:
        break

    elif i == 0:
        r_theta.append(0.0)
        r_magnitude.append(0.0)
        counter += 1


    else:
        separation = 2*np.pi/(i*6)
        for j in range (0,i*6):
            r_theta.append(separation*j+1)
            if i * s * radius_balls > container.radius() - radius_balls:
                raise Exception("Cannot fit so many balls in container!")
            else:
                r_magnitude.append(s*i*(radius_balls))

            counter += 1
            print(counter)

r_magnitude = np.array(r_magnitude)
r_theta = np.array(r_theta)
x = r_magnitude*np.cos(r_theta)
y = r_magnitude*np.sin(r_theta)

balls = []
for i in range (0,num_balls):
    balls.append(Ball(r=[x[i],y[i]],R=radius_balls))

f = pl.figure(figsize=[5,5])
ax = pl.axes(xlim=(-10, 10), ylim=(-10, 10))
ax.add_artist(container.patch())
for ball in balls:
    ax.add_patch(ball.patch())
pl.show()