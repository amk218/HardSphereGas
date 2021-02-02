"""
Hard spheres simulation
Anni Kauniskangas, 2020

Module simulation.py contains the Simulation class. It uses the classes Ball and Container from module objects.py.
"""

import numpy as np
import pylab as pl
from objects import Ball
from objects import Container


class Simulation:
    """
    A class to handle the simulation for multiple hard balls in a container
    """
    def __init__(self, num_balls=9, radius_balls=1.0, mass_balls=1.0, container=Container(), temperature=1000, s=3):
        """
        Initialises the simulation with any number of balls. Sets ball positions and velocities and calculates first
        collision
        :param num_balls:
        :param radius_balls:
        :param container:
        :param temperature: In units of kT
        :param s: Scaling factor for initial ball spacing. Should be adjusted depending on ball size and number
        """
        self._num_balls = num_balls
        self._radius_balls = radius_balls
        self._mass_balls = mass_balls
        self._container = container
        self._balls = []
        self._temperature = temperature

        # Set r for balls so that they spawn on concentric circles. To test only this, use locate_balls.py
        r_theta = []
        r_magnitude = []
        counter = 0
        for i in range(0, self._num_balls):

            if counter >= self._num_balls:
                break

            elif i == 0:
                r_theta.append(0.0)
                r_magnitude.append(0.0)
                counter += 1

            else:
                separation = 2 * np.pi / (i*6)
                for j in range(0, i*6):
                    if counter >= self._num_balls:
                        break
                    r_theta.append(separation * j+1)
                    if (i * s * (radius_balls)) > self._container.radius()-(radius_balls):
                        raise Exception("Cannot fit so many balls in container!")
                    else:
                        r_magnitude.append(i * s * (radius_balls))
                    counter += 1

        # Set ball positions
        r_magnitude = np.array(r_magnitude)
        r_theta = np.array(r_theta)
        r_x = r_magnitude * np.cos(r_theta)
        r_y = r_magnitude * np.sin(r_theta)

        # Set ball velocities
        v_theta = np.arange(0, 2*np.pi, 2*np.pi/num_balls)
        speed = np.sqrt(2*self._temperature/self._mass_balls)
        v_magnitude = speed
        v_x = v_magnitude*np.cos(v_theta)
        v_y = v_magnitude*np.sign(v_theta)

        # Initialise all balls
        for i in range (0,num_balls):
            self._balls.append(Ball(r=[r_x[i],r_y[i]], v=[v_x[i],v_y[i]], R=radius_balls))

        # Calculate all dt's to find the first collision

        self.dt_ball_list = []  # A list to keep a list of each ball pair and their corresponding dt
        for i in range(0, len(self._balls)):  # Find shortest time to collision between balls

            for j in range(i + 1, len(self._balls)):
                dt_ball = self._balls[i].t_to_collision(self._balls[j])

                if dt_ball is not None:
                    self.dt_ball_list.append([dt_ball, self._balls[i], self._balls[j]])

            dt_cont = self._container.t_to_collision(self._balls[i])  # Check t to collision between container and ball
            if dt_cont is not None:
                self.dt_ball_list.append([dt_cont, self._container, self._balls[i]])

        dt_item = min(self.dt_ball_list)  # Get sublist with minimum dt
        self.dt = dt_item[0]

        for ball in self._balls:
            ball.move(self.dt)

        dt_item[1].collide(dt_item[2])

    def __repr__(self):
        return "Simulation(container={0}, num_balls={1})".format(self._container, self._num_balls)

    def mass_balls(self):
        return self._mass_balls

    def radius_balls(self):
        return self._radius_balls

    def num_balls(self):
        return self._num_balls

    def given_temperature(self):
        """
        :return: Temperature based on the speed given to balls in initialisation
        """
        return self._temperature

    def measured_temperature(self):
        """
        Calculates temperature in units of kT based of the rms velocity of balls
        :return: Temperature in units of kT
        """
        v_list = []
        for ball in self._balls:
            v_list.append((ball.magnitude(ball.velocity()))**2)
        ms_v = np.average(v_list)
        temp = self._mass_balls*ms_v/(2)
        return temp


    def r_between_balls(self, ball1, ball2):
        """
        :param ball1:
        :param ball2:
        :return: magnitude of the separation between two balls
        """
        r1 = ball1.position()
        r2 = ball2.position()
        separation = np.linalg.norm(r2-r1)
        return separation


    def next_collision (self):
        """
        Finds the time to the next collision, moves balls to that moment and collides the balls/container
        Optimised to only recalculate pairs of objects that have just collided
        :return: change of momentum for balls in the collision
        """

        dt_item = min(self.dt_ball_list) # sublist with minimum dt
        self.dt = dt_item[0]
        dt_objects = [dt_item[1],dt_item[2]]

        # Find items involving two collided object
        items_to_delete = []
        for index, item in enumerate(self.dt_ball_list):
            for object in item:
                if object in dt_objects:
                    items_to_delete.append(index)
        items_to_delete = list(set(items_to_delete)) # remove duplicates


        # Delete items involving collided objects. For the rest, simply substract dt
        for item in sorted(items_to_delete, reverse=True):
            del self.dt_ball_list[item]
        for item in self.dt_ball_list:
            item[0] -= self.dt


        # Recalculate dt for collided objects. If statements to avoid calculating same pair twice
        for item in dt_objects:

            for ball in self._balls:
                if ball == item:
                    continue
                elif ball == dt_objects[0]:
                    continue
                elif item == self._container and ball in dt_objects:
                    continue
                else:
                    dt_ball = item.t_to_collision(ball)
                    if dt_ball is not None:
                        self.dt_ball_list.append([dt_ball,item,ball])

            if item != self._container:
                dt_cont = self._container.t_to_collision(item) # Check time to collision between container and ball
                if dt_cont is not None:
                    self.dt_ball_list.append([dt_cont, self._container, item])

        dt_item = min(self.dt_ball_list) # Get sublist with minimum dt
        self.dt = dt_item[0]

        for ball in self._balls:
            ball.move(self.dt)

        dt_item[1].collide(dt_item[2])

        # For calculating pressure, return change in momentum in the case of a collision with container
        if isinstance(dt_item[2], Container):
            momentum_change = dt_item[1].mass()*2*dt_item[1].magnitude(dt_item[1].velocity())
            return momentum_change
        elif isinstance(dt_item[1],Container):
            momentum_change = dt_item[2].mass()*2*dt_item[2].magnitude(dt_item[2].velocity())
            return momentum_change
        else:
            return 0


    def run(self, num_frames, animate=False, r_histograms=False, kinetic_energy=False, momentum=False,
                pressure=False,temperature=False,velocity=False):
        """
        Method to run the simulation and to calculate various quantities
        :param num_frames: number of iterations to be calculated
        :param animate: set to True to show animation
        :param r_histograms: set to True to calculate ball positions and separations
        :param kinetic_energy: set to True to calculate total kinetic energy
        :param momentum: set to True to calculate total momentum
        :param pressure: set to True to calculate pressure on container
        :param velocity: set to True to get the distribution of speeds for all balls
        :return:
        """
        if animate:
            f = pl.figure(figsize=[5,5])
            ax = pl.axes(xlim=(-10, 10), ylim=(-10, 10))
            ax.add_artist(self._container.patch())
            for ball in self._balls:
                ax.add_patch(ball.patch())

        r_magnitudes = []
        separation_magnitudes = []
        kinetic_energies = []
        x_momenta = []
        y_momenta = []
        temps = []
        velocities = []
        momentum_change = 0.0
        elapsed_time = 0.0

        for frame in range(num_frames):
            dp = self.next_collision()

            if animate:
                pl.pause(0.05)

            if momentum:
                total_px = 0
                total_py = 0
                for ball in self._balls:
                    total_px += ball.x_momentum()
                    total_py += ball.y_momentum()

                x_momenta.append(total_px)
                y_momenta.append(total_py)

            if kinetic_energy:
                total_ke = 0
                for ball in self._balls:
                    total_ke += ball.kinetic_energy()
                kinetic_energies.append(total_ke)

            if r_histograms:
                for i1, ball1 in enumerate(self._balls):
                    r_magnitudes.append(ball1.magnitude(ball1.position()))
                    for i2 in range(i1+1,len(self._balls)):
                        separation_magnitudes.append((self.r_between_balls(ball1,self._balls[i2])))

            if pressure:
                momentum_change += dp
                elapsed_time += self.dt

            if temperature:
                temps.append(self.measured_temperature())


        if animate:
            pl.show()

        if r_histograms:
            return (r_magnitudes, separation_magnitudes)

        if kinetic_energy:
            return kinetic_energies

        if momentum:
            return x_momenta,y_momenta

        if temperature and pressure:
            pressure = momentum_change / (elapsed_time * 2 * np.pi * self._container.radius())
            mean_temp = np.average(temps)
            dev_temp = np.std(temps)
            return mean_temp, dev_temp, pressure

        if pressure:
            pressure = momentum_change / (elapsed_time * 2 * np.pi * self._container.radius())
            print("Pressure was " + str(pressure))
            return pressure

        if velocity and temperature:
            mean_temp = np.average(temps)
            dev_temp = np.std(temps)
            for ball in self._balls:
                v = ball.magnitude(ball.velocity())
                velocities.append(v)
            return mean_temp, dev_temp, velocities

        if temperature:
            mean_temp = np.average(temps)
            dev_temp = np.std(temps)
            print("Average temperature over run was "+str(mean_temp)+" with standard deviation of "+str(dev_temp))
            return mean_temp,dev_temp



