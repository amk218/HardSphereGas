"""
Hard spheres simulation
Anni Kauniskangas, 2020

Module objects.py contains the classes Ball and Container
"""

import numpy as np
import pylab as pl

class Ball:

    """
    A class for creating ball objects to use in Simulation. All physical attributes can be given as input
    """
    def __init__(self, m=1, R=1, v=[1.0,1.0], r=[1.0,1.0]):
        self._m = m
        self._R = R
        self._v = np.array(v)
        self._r = np.array(r)
        self._patch = pl.Circle((self._r[0],self._r[1]), self._R , fc='g')

    def __repr__(self):
        return "Ball(r={0}, v={1})".format(self._r, self._v)

    def magnitude(self,quantity):
        return np.sqrt(np.dot(quantity,quantity))

    def radius(self):
        return self._R

    def mass(self):
        return self._m

    def position(self):
        return self._r

    def velocity(self):
        return self._v

    def patch(self):
        return self._patch

    def move(self, dt):
        self._r = self._r + self._v*dt
        self._patch.center = self._r.tolist()

    def change_v(self, new_v):
        self._v = new_v

    def kinetic_energy(self):
        return 1/2 * self._m*(self.magnitude(self._v)**2)

    def x_momentum(self):
        px = self._m*self._v[0]
        return px

    def y_momentum(self):
        py = self._m * self._v[1]
        return py

    def t_to_collision(self,other):
        """
        Calculates the times to collision and chooses the smallest positive solution
        :param other: other object to collide with
        :return: time to collision between self and other
        """
        r = self._r - other.position()
        v = self._v - other.velocity()
        R = self._R + other.radius()

        dt1 = (-np.dot(r,v) + np.sqrt((np.dot(r,v))**2-np.dot(v,v)*(np.dot(r,r)-R**2)))/(np.dot(v,v))
        dt2 = (-np.dot(r,v) - np.sqrt((np.dot(r,v))**2-np.dot(v,v)*(np.dot(r,r)-R**2)))/(np.dot(v,v))

        dts = [dt1, dt2]
        pos_dts = [dt for dt in dts if dt > 10 ** (-8)]

        # check determinant
        if (((np.dot(r, v)) ** 2 - np.dot(v, v) * (np.dot(r, r) - R ** 2))) < 0:
            return None
        # Smallest positive root (excluding really small ones to avoid balls stuck together)
        elif len(pos_dts) > 0:
            return min(pos_dts)
        # No valid roots
        else:
            return None


    def collide(self,other):
        """
        Changes the velocities of the two colliding Ball objects upon collision
        :param other: object to collide with
        :return:
        """
        if isinstance(other, Container):
            raise Exception("Warning! Use the Container class' method for collisions with container instead.")

        u1 = self._v
        u2 = other.velocity()
        r1 = self._r
        r2 = other.position()
        m1 = self._m
        m2 = other.mass()
        r = self._r - other.position()

        v1 = u1 - 2*m2/(m1+m2) * np.dot((u1-u2),(r1-r2))/np.dot(r,r) * (r1-r2)
        v2 = u2 - 2*m1/(m1+m2) * np.dot((u2-u1),(r2-r1))/np.dot(r,r) * (r2-r1)

        self.change_v(v1)
        other.change_v(v2)



class Container (Ball):
    """
    A class for the container. Inherits from Ball but adjusted to incorporate infinite mass and internal collisions.
    """
    def __init__(self, m=1, R=10, v=[0.0, 0.0], r=[0.0, 0.0]):
        Ball.__init__(self,m,R,v,r)
        self._patch = pl.Circle((self._r[0], self._r[1]), self._R, fc='b', fill=False)

    def __repr__(self):
        return "Container(r={0}, v={1})".format(self._r, self._v)

    def area(self):
        area = np.pi*self._R**2
        return area


    def t_to_collision(self, other):
        """
        Finds the time to next collision with ball. Overrides method in Ball to accommodate for internal collision
        :param other: object to collide with
        :return: time to collision between self and other
        """

        r = self._r - other.position()
        v = self._v - other.velocity()
        R = self._R - other.radius()

        dt1 = (-np.dot(r,v) + np.sqrt((np.dot(r,v))**2-np.dot(v,v)*(np.dot(r,r)-R**2)))/(np.dot(v,v))
        dt2 = (-np.dot(r,v) - np.sqrt((np.dot(r,v))**2-np.dot(v,v)*(np.dot(r,r)-R**2)))/(np.dot(v,v))

        dts = [dt1, dt2]
        pos_dts = [dt for dt in dts if dt > 10 ** (-8)]

        #check determinant
        if np.dot(r,v)**2-np.dot(v,v)*(np.dot(r,r)-R**2) < 0:
            return None
        # Smallest positive root (excluding really small ones)
        elif len(pos_dts) > 0:
            return min(pos_dts)
        # No valid roots
        else:
            return None

    def collide(self, other):
        """
        Performs collision of ball with container, in the limit of container infinite mass
        :param other: object to collide with
        :return:
        """
        if isinstance(other,Container):
            raise Exception("Warning! Should not collide container with container!")

        u1 = self._v
        u2 = other.velocity()
        r1 = self._r
        r2 = other.position()
        r = self._r - other.position()

        v1 = u1
        v2 = u2 - 2 * np.dot((u2 - u1), (r2 - r1)) / np.dot(r, r) * (r2 - r1)

        self.change_v(v1)
        other.change_v(v2)
