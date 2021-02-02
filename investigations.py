"""
Hard spheres simulation
Anni Kauniskangas, 2020

Script investigations.py contains the thermodynamics investigations used to study the simulation.
Each part also produces the appropriate plots
Uncomment a section to run and plot

Note:Running some parts of the simulation will take a long time due to the size of the datasets taken and number of
collision run! For quick plotting, please use module plotting.py instead
"""

import numpy as np
import matplotlib.pyplot as plt
from objects import Ball
from objects import Container
from simulation import Simulation
from scipy import optimize

# Settings for plots
SMALL_SIZE = 19
MEDIUM_SIZE = 24
BIGGER_SIZE = 26

plt.rc('font', size=SMALL_SIZE)          # default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#----------------------------------------------------------------------------------------------------------------------

container1 = Container(R=10)

# Simply calculate pressure
"""
sim1 = Simulation(container=container1, num_balls=100, radius_balls=0.5)
sim1.run(1000,pressure=True)
"""
# Pressure stabilisation. Demonstrates how pressure fluctuates if the number of collisions is not large enough
"""
sim1 = Simulation(container=container1, num_balls=50, radius_balls=0.5)
frames = np.arange(1,15000,250)
pressure=[]
for f in frames:
    P = sim1.run(num_frames=f,pressure=True)
    pressure.append(P)
plt.plot(frames,pressure,color="red",markersize=16)
plt.title("Pressure stabilisation",fontsize=28)
plt.ylabel("Pressure")
plt.xlabel("Number of collisions")
plt.show()
"""
# Dependence of pressure on temperature
"""
T_in = np.arange(1, 200, 10)
T_meas = []
T_devs = []
P1 = []
P2 = []
P3 = []
P = []
P_dev = []

for temp in T_in:
    sim2 = Simulation(container=container1,num_balls=100, radius_balls=0.01, mass_balls=1, temperature=temp)
    t,t_dev,p1 = sim2.run(10000, pressure=True,temperature=True)
    P1.append(p1)
    T_meas.append(t)
    T_devs.append(t_dev)
    t,t_dev,p2 = sim2.run(10000, pressure=True,temperature=True)
    P2.append(p2)
    t,t_dev,p3= sim2.run(10000, pressure=True,temperature=True)
    P3.append(p3)

file = open("p_T_data_newfile.txt", "w+")

sim2 = Simulation(container=container1,num_balls=100, radius_balls=0.01, mass_balls=1)
file.write("Container area: "+str(container1.area())+"\n")
file.write("Number of balls: "+str(sim2.num_balls())+"\n")
file.write("Mass "+str(sim2.mass_balls())+"\n")
file.write("Radius "+str(sim2.radius_balls())+"\n")
file.write("T,    T_dev,     P1,    P2,    P3\n")

for i in range (len(T_meas)):
    file.write(str(T_meas[i])+","+str(T_devs[i])+","+str(P1[i])+","+str(P2[i])+","+str(P3[i])+"\n")
file.close()

for i in range (len(P1)):
    P.append(np.average([P1[i],P2[i],P3[i]]))
    P_dev.append(np.std([P1[i],P2[i],P3[i]]))
    
fit1,cov1=np.polyfit(T_meas,P,1,w=P_dev,cov=True)
#print(np.sqrt(cov1[0][0]))
p1=np.poly1d(fit1)
x = np.linspace(0,300,1000)
plt.plot(x,p1(x),color="g",label="Best fit line y="+str(p1))
plt.errorbar(T_meas, P, xerr=T_devs, yerr=P_dev,fmt='rx',markersize='16',label=("Simulation data A="+str(container1.area())
            +", N="+str(sim2.num_balls())))
            
plt.title("Pressure as a function of temperature", fontsize=28)
plt.xlabel("Temperature")
plt.ylabel("Pressure")
plt.legend()
plt.show()

"""
# Temperature and pressure change with area
"""
T_meas = []
T_devs = []
P1 = []
P2 = []
P3 = []
P = []
P_dev = []

cont_radii = np.arange(7,20,0.5)

for r in cont_radii:
    container = Container(R=r)
    sim2 = Simulation(container=container,num_balls=100, radius_balls=0.1, mass_balls=1)
    t,t_dev,p1 = sim2.run(11000, pressure=True,temperature=True)
    P1.append(p1)
    T_meas.append(t)
    T_devs.append(t_dev)

    t,t_dev,p2 = sim2.run(11000, pressure=True,temperature=True)
    P2.append(p2)
    t,t_dev,p3= sim2.run(11000, pressure=True,temperature=True)
    P3.append(p3)

file = open("p_T_A_data_newfile.txt", "a+")
container1 = Container()
sim2 = Simulation(container=container1,num_balls=100, radius_balls=0.1, mass_balls=1)
file.write("Number of balls: "+str(sim2.num_balls())+"\n")
file.write("Mass "+str(sim2.mass_balls())+"\n")
file.write("Radius "+str(sim2.radius_balls())+"\n")
file.write("A,     T,    T_dev,     P1,    P2,    P3\n")

A = np.pi*(cont_radii)**2
for i in range (len(T_meas)):
    file.write(str(A[i])+","+str(T_meas[i])+","+str(T_devs[i])+","+str(P1[i])+","+str(P2[i])+","+str(P3[i])+"\n")
file.close()

for i in range (len(P1)):
    P.append(np.average([P1[i],P2[i],P3[i]]))
    P_dev.append(np.std([P1[i],P2[i],P3[i]]))


fit1,cov1=np.polyfit(A,T_meas,1,w=T_devs,cov=True)
p1=np.poly1d(fit1)
x = np.linspace(40,1200,1000)

def inverse(x,a):
    y=a/x
    return y

popt,pcov = optimize.curve_fit(inverse, A, P)
plt.plot(x,inverse(x,*popt),color="g",label="Best fit curve y=28101/x")

plt.plot(x,p1(x),color="orange",label="Best fit line y="+str(p1))
print(*popt)
plt.errorbar(A, P, yerr=P_dev,fmt='rx',markersize='18',label=("Pressure. N="+str(sim2.num_balls())))
plt.errorbar(A, T_meas, yerr=T_devs,fmt='bx',markersize='18',label=("Temperature. R balls ="+str(sim2.radius_balls())))
plt.title("Pressure and temperature as a function of container area", fontsize=28)
plt.xlabel("Container area")
plt.ylabel("Pressure/Temperature")
plt.legend()
plt.show()

"""
# Temperature and pressure change with number of balls
"""
T_meas = []
T_devs = []
P1 = []
P2 = []
P3 = []
P = []
P_dev = []

N_balls = np.arange(1,500,10)

for N in N_balls:
    container = Container(R=10)
    sim2 = Simulation(container=container,num_balls=N, radius_balls=0.01, mass_balls=1)
    t,t_dev,p1 = sim2.run(110000, pressure=True, temperature=True)
    P1.append(p1)
    T_meas.append(t)
    T_devs.append(t_dev)

    t,t_dev,p2 = sim2.run(11000, pressure=True, temperature=True)
    P2.append(p2)
    t,t_dev,p3= sim2.run(11000, pressure=True, temperature=True)
    P3.append(p3)

A = np.pi*(container.radius())**2

file = open("p_T_N_data_newfile.txt", "a+")
container1 = Container()
sim2 = Simulation(container=container1,num_balls=100, radius_balls=0.1, mass_balls=1)
file.write("Area of container: "+str(A)+"\n")
file.write("Mass "+str(sim2.mass_balls())+"\n")
file.write("Radius "+str(sim2.radius_balls())+"\n")
file.write("N,     T,    T_dev,     P1,    P2,    P3\n")


for i in range (len(T_meas)):
    file.write(str(N_balls[i])+","+str(T_meas[i])+","+str(T_devs[i])+","+str(P1[i])+","+str(P2[i])+","+str(P3[i])+"\n")
file.close()

for i in range(len(P1)):
    P.append(np.average([P1[i],P2[i],P3[i]]))
    P_dev.append(np.std([P1[i],P2[i],P3[i]]))

fit1,cov1=np.polyfit(N_balls,P,1,w=P_dev,cov=True)
fit2,cov2=np.polyfit(N_balls,T_meas,1,w=T_devs,cov=True)
p2=np.poly1d(fit2)
p1=np.poly1d(fit1)
x = np.linspace(0,550,1000)

plt.plot(x,p1(x),color="orange",label="Best fit line y="+str(p1))
plt.plot(x,p2(x),color="magenta",label="Best fit line y="+str(p2))

plt.errorbar(N_balls, P, yerr=P_dev,fmt='rx',markersize='18',label="Pressure. Radius of balls="+str(sim2.radius_balls()))
plt.errorbar(N_balls, T_meas, yerr=T_devs,fmt='bx',markersize='18',label="Temperature")
plt.title("Pressure and temperature as a function of number of balls", fontsize=28)
plt.xlabel("Number of balls")
plt.ylabel("Pressure/Temperature")
plt.legend()
plt.show()

"""
# Maxwell-Boltzmann distribution
""""
container = Container(R=10)

def max_boltz (v,m,T):
    P = (m*v)/T*np.exp(-m*v**2/(2*T))
    return P

x=np.linspace(0,200,1000000)

sim2 = Simulation(container=container,num_balls=850, radius_balls=0.01, mass_balls=1)
t,t_dev,v = sim2.run(num_frames=11000, velocity=True, temperature=True)

A = np.pi*(container.radius())**2

file = open("v_dist_data_newfile.txt", "w+")
file.write("Number of balls: 1000\n")
file.write("Area of container: "+str(A)+"\n")
file.write("Mass "+str(sim2.mass_balls())+"\n")
file.write("Radius "+str(sim2.radius_balls())+"\n")
file.write("T,     T_dev,     v\n")

for i in range(len(v)):
    file.write(str(t)+","+str(t_dev)+","+str(v[i])+"\n")
file.close()

x=np.linspace(0,200,1000000)
bins=plt.hist(v,label="Simulation data, "+str(sim2.num_balls()),bins=30)
scaling_factor = (bins[1][1]-bins[1][0])*len(v)


plt.plot(x,(max_boltz(x,sim2.mass_balls(),t)*scaling_factor),label="Theoretical Maxwell-Boltzmann distribution",color="r")
plt.title("Velocity distribution of the balls",fontsize=30)
plt.xlabel("Velocity")
plt.ylabel("Number of balls")
plt.legend()
plt.show()

"""
# Ideal gas comparison with only varying ball radii
"""
ball_radiis=np.arange(0.005,1,0.005)
pres=[]
temp=[]
dev_temp=[]

for r in ball_radiis:
    sim1 = Simulation(container=container1, num_balls=37, radius_balls=r)
    T,dev_T,P=sim1.run(10000,pressure=True,temperature=True)
    pres.append(P)
    temp.append(T)
    dev_temp.append(dev_T)
pres=np.asarray(pres)
temp=np.asarray(temp)
y=(pres*np.pi*100)/(37*temp)

file = open("size_comparison_newfile.txt", "w+")
file.write("Number of balls: 37 \n")
file.write("P,     T,       Ball radius \n")
for i in range(len(ball_radiis)):
    file.write(str(pres[i])+","+str(temp[i])+","+str(ball_radiis[i])+"\n")

file.close()

plt.plot(ball_radiis,y,"x",markersize=18,color="red")
plt.hlines(1,0,1,color="blue",label="Ideal gas law")
plt.title("Deviation from ideal gas law with ball radius",fontsize=28)
plt.xlabel("Ball radius")
plt.ylabel("PA/NkT")
plt.show()
"""