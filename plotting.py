"""
Hard spheres simulation
Anni Kauniskangas, 2020

Script plotting.py can be used to reproduce plots from existing data without having to run the investigations.py script
Uncomment a section to run and create the plot
"""

import numpy as np
import matplotlib.pyplot as plt
from objects import Ball
from objects import Container
from simulation import Simulation
from scipy import optimize

# Settings for plots
SMALL_SIZE = 20
MEDIUM_SIZE = 26
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

container1 = Container(R=10)
sim2 = Simulation(container=container1,num_balls=100, radius_balls=0.1, mass_balls=1)
#-----------------------------------------------------------------------------------------------------------------------

# P with T
"""
T,T_devs,P1,P2,P3 = np.loadtxt("p_T_data_with dev2.txt", delimiter=",",skiprows=5, unpack=True)

P = []
P_dev = []

for i in range(len(P1)):
    P.append(np.average([P1[i],P2[i],P3[i]]))
    P_dev.append(np.std([P1[i],P2[i],P3[i]]))

fit1,cov1=np.polyfit(T,P,1,w=P_dev,cov=True)
#print(np.sqrt(cov1[0][0]))
p1=np.poly1d(fit1)
x = np.linspace(0,300,1000)
plt.plot(x,p1(x),color="g",label="Best fit line y="+str(p1))
plt.errorbar(T, P, xerr=T_devs, yerr=P_dev,fmt='rx',markersize='16',label=("Simulation data A="+str(container1.area())
   +", N="+str(sim2.num_balls())))
plt.title("Pressure as a function of temperature", fontsize=28)
plt.xlabel("Temperature")
plt.ylabel("Pressure")
plt.legend()
plt.show()
"""
# P and T with A
"""
container1 = Container(R=10)
sim2 = Simulation(container=container1,num_balls=50, radius_balls=0.01, mass_balls=1)
A,T,T_devs,P1,P2,P3 = np.loadtxt("p_T_A_data3[1].txt", delimiter=",",skiprows=5, unpack=True)

P = []
P_dev = []

for i in range(len(P1)):
    P.append(np.average([P1[i],P2[i],P3[i]]))
    P_dev.append(np.std([P1[i],P2[i],P3[i]]))

fit1,cov1=np.polyfit(A,T,1,w=T_devs,cov=True)
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
plt.errorbar(A, T, yerr=T_devs,fmt='bx',markersize='18',label=("Temperature. R balls ="+str(sim2.radius_balls())))
plt.title("Pressure and temperature as a function of container area", fontsize=28)
plt.xlabel("Container area")
plt.ylabel("Pressure/Temperature")
plt.legend()
plt.show()

"""
# P and T with N
"""
container1 = Container(R=10)
sim2 = Simulation(container=container1,num_balls=100, radius_balls=0.1, mass_balls=1)
N,T,T_devs,P1,P2,P3 = np.loadtxt("p_T_N_data.txt", delimiter=",",skiprows=5, unpack=True)

P = []
P_dev = []

for i in range(len(P1)):
    P.append(np.average([P1[i],P2[i],P3[i]]))
    P_dev.append(np.std([P1[i],P2[i],P3[i]]))

fit1,cov1=np.polyfit(N,P,1,w=P_dev,cov=True)
fit2,cov2=np.polyfit(N,T,1,w=T_devs,cov=True)
p2=np.poly1d(fit2)
p1=np.poly1d(fit1)
x = np.linspace(0,550,1000)

plt.plot(x,p1(x),color="orange",label="Best fit line y="+str(p1))
plt.plot(x,p2(x),color="magenta",label="Best fit line y="+str(p2))

plt.errorbar(N, P, yerr=P_dev,fmt='rx',markersize='18',label="Pressure. Radius of balls 0.1")
plt.errorbar(N, T, yerr=T_devs,fmt='bx',markersize='18',label="Temperature")
plt.title("Pressure and temperature as a function of number of balls", fontsize=28)
plt.xlabel("Number of balls")
plt.ylabel("Pressure/Temperature")
plt.legend()
plt.show()

"""
# Maxwell Boltzmann distribution
"""

container = Container(R=10)

t,t_dev,v=np.loadtxt("v_dist_data_new.txt",unpack=True,skiprows=5,delimiter=",")
t1=t[0]


def max_boltz (v,m,T):
    P = (m*v)/T*np.exp(-m*v**2/(2*T))
    return P

x=np.linspace(0,200,1000000)
A = np.pi*(container.radius())**2


bins=plt.hist(v,label="Simulation data, N=850",bins=20)

scaling_factor = (bins[1][1]-bins[1][0])*len(v)
print("scaling factor "+str(scaling_factor))

plt.plot(x,(max_boltz(x,sim2.mass_balls(),t1)*scaling_factor),label="Theoretical Maxwell-Boltzmann distribution",
   color="r",linewidth=3.5)
plt.title("Velocity distribution of the balls",fontsize=30)
plt.xlabel("Velocity")
plt.ylabel("Number of balls")
plt.legend()
plt.show()

"""
# Ideal gas law comparison P-T-A data (Not used in report)
"""
# N=50 Rad=0.01
A1,T1,T1_devs,P11,P12,P13 = np.loadtxt("p_T_A_data3[1].txt", delimiter=",",skiprows=4, unpack=True)

P1 = []
P1_dev = []

for i in range(len(A1)):
    P1.append(np.average([P11[i],P12[i],P13[i]]))
    P1_dev.append(np.std([P11[i],P12[i],P13[i]]))

P1=np.asarray(P1)

# N=50, Rad=0.05
A2, T2, T2_devs, P21, P22, P23 = np.loadtxt("p_T_A_data2[1].txt", delimiter=",", skiprows=4, unpack=True)

P2 = []
P2_dev = []

for i in range(len(A2)):
    P2.append(np.average([P21[i], P22[i], P23[i]]))
    P2_dev.append(np.std([P21[i], P22[i], P23[i]]))
P2=np.asarray(P2)

# N=100, Rad=0.1
A3, T3, T3_devs, P31, P32, P33 = np.loadtxt("p_T_A_data[1].txt", delimiter=",", skiprows=4, unpack=True)

P3 = []
P3_dev = []

for i in range(len(A3)):
    P3.append(np.average([P31[i], P32[i], P33[i]]))
    P3_dev.append(np.std([P31[i], P32[i], P33[i]]))

P3=np.asarray(P3)

# N=50, Rad=0.5

A4, T4, T4_devs, P41, P42, P43 = np.loadtxt("p_T_A_data5.txt", delimiter=",", skiprows=4, unpack=True)

P4 = []
P4_dev = []

for i in range(len(A4)):
    P4.append(np.average([P41[i], P42[i], P43[i]]))
    P4_dev.append(np.std([P41[i], P42[i], P43[i]]))

P4=np.asarray(P4)

# Plots
x=np.linspace(0,1,1000)

y1=P1/T1
x1=50/A1
fit1,cov1=np.polyfit(x1,y1,1,cov=True)
p1=np.poly1d(fit1)

plt.plot(x,p1(x),color="orange",label="R=0.01")
plt.errorbar(x1, y1,fmt='rx',markersize='12')


y2=P2/T2
x2=50/A2
fit2,cov2=np.polyfit(x2,y2,1,cov=True)
p2=np.poly1d(fit2)

plt.plot(x,p2(x),color="black",label="R=0.05")
plt.errorbar(x2, y2,fmt='bx',markersize='12')

y3=P3/T3
x3=100/A3
fit3,cov3=np.polyfit(x3,y3,1,cov=True)
p3=np.poly1d(fit3)

plt.plot(x,p3(x),color="green",label="R=0.1")
plt.errorbar(x3, y3,fmt='yx',markersize='12')

y4=P4/T4
x4=50/A4
fit4,cov4=np.polyfit(x4,y4,1,cov=True)
p4=np.poly1d(fit4)

plt.plot(x,p4(x),color="magenta",label="R=0.5")
plt.errorbar(x4, y4,fmt='cx',markersize='12')
plt.plot(x,x,label="Ideal gas law",color="red")
plt.title("Deviation from ideal gas law with ball radius",fontsize="28")
plt.xlabel("N/A")
plt.ylabel("P/kT")

plt.legend()
plt.show()

"""
# Ideal gas law comparison P-T data
"""
#N=100, Rad=0.1
T1,T1_devs,P11,P12,P13 = np.loadtxt("p_T_data_with dev2[1].txt", delimiter=",",skiprows=5, unpack=True)

P1 = []
P1_dev = []

for i in range(len(T1)):
    P1.append(np.average([P11[i],P12[i],P13[i]]))
    P1_dev.append(np.std([P11[i],P12[i],P13[i]]))

P1=np.asarray(P1)

# N=37, Rad=0.8
T2, T2_devs, P21, P22, P23 = np.loadtxt("p_T_data_with dev3.txt", delimiter=",", skiprows=5, unpack=True)

P2 = []
P2_dev = []

for i in range(len(T2)):
    P2.append(np.average([P21[i], P22[i], P23[i]]))
    P2_dev.append(np.std([P21[i], P22[i], P23[i]]))
P2=np.asarray(P2)

# N=37, Rad=1
T3, T3_devs, P31, P32, P33 = np.loadtxt("p_T_data_with dev4.txt", delimiter=",", skiprows=5, unpack=True)

P3 = []
P3_dev = []

for i in range(len(T3)):
    P3.append(np.average([P31[i], P32[i], P33[i]]))
    P3_dev.append(np.std([P31[i], P32[i], P33[i]]))

P3=np.asarray(P3)

# N=19, Rad=1.5

T4, T4_devs, P41, P42, P43 = np.loadtxt("p_T_data_with dev5.txt", delimiter=",", skiprows=5, unpack=True)

P4 = []
P4_dev = []

for i in range(len(T4)):
    P4.append(np.average([P41[i], P42[i], P43[i]]))
    P4_dev.append(np.std([P41[i], P42[i], P43[i]]))

P4=np.asarray(P4)

# Plots
x1=P1
y1=(P1*np.pi*100)/(100*T1)
fit1,cov1=np.polyfit(x1,y1,1,cov=True)
p1=np.poly1d(fit1)

plt.plot(x1,p1(x1),color="orange",label="R=0.1")
plt.errorbar(x1, y1, xerr=P1_dev,fmt='rx',markersize='18')

x2=P2
y2=(P2*np.pi*100)/(37*T2)
fit2,cov2=np.polyfit(x2,y2,1,cov=True)
p2=np.poly1d(fit2)

plt.plot(x2,p2(x2),color="black",label="R=0.8")
plt.errorbar(x2, y2, xerr=P2_dev,fmt='bx',markersize='18')

x3=P3
y3=(P3*np.pi*37)/(100*T3)
fit3,cov3=np.polyfit(x3,y3,1,cov=True)
p3=np.poly1d(fit3)

plt.plot(x3,p3(x3),color="green",label="R=1")
plt.errorbar(x3, y3, xerr=P3_dev,fmt='yx',markersize='18')

x4=P4
y4=(P4*np.pi*100)/(19*T4)
fit4,cov4=np.polyfit(x4,y4,1,cov=True)
p4=np.poly1d(fit4)

plt.plot(x4,p4(x4),color="magenta",label="R=1.5")
plt.errorbar(x4, y4, xerr=P4_dev,fmt='cx',markersize='18')
plt.title("Pressure-Temperature data comparison to ideal gas law", fontsize="30")
plt.xlabel("P")
plt.ylabel("PA/NkT")
plt.hlines(1,0,150,label="Ideal gas law",color="brown")
plt.legend()
plt.show()
"""
# Ideal gas law and ball size with Van der Waals correction
"""
P,T,R = np.loadtxt("size_comparison_PTdata2.txt", delimiter=",",skiprows=2, unpack=True)
N=37
k=1.72

y = (P*np.pi*100)/(N*T)
y= np.delete(y,[0,1])
y_corr = P*((np.pi*100)-N*k*np.pi*R**2)/(N*T)
y_corr = np.delete(y_corr,[0,1])
R = np.delete(R,[0,1])

plt.plot(R,y,"x",markersize=18,color="r",label="Uncorrected data, b=0")
plt.plot(R,y_corr,"x",markersize=18,color="g",label="Corrected data, b=1.72 x ball area")
plt.title("van der Waals correction",fontsize=28)
plt.hlines(1,0,1,color="blue",label="van der Waals law (Ideal gas law for b=0))",linewidth=3.5)
plt.legend()
plt.xlabel("Ball radius")
plt.ylabel("P(A-Nb)/NkT")
plt.show()
"""



