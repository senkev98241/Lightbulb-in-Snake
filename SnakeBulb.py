#FILE: SnakeBulb.py
#LANGUAGE: python 3.11
#DESCRIPTION: 

#################################################
# TO-DO
#################################################
# (1) Establish Constants
# (2) Establish Differential Equation
# (3) Establish
###################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import trapz
import csv
# import math

# need to rework model1: dv/dt = -g+(v_0k-\beta v^2)/(M_0-kt)

# Set constants and other parameter values
VOLT = 10.0             # Voltage (units here)
RESIST = 10.8           # Resistance (units here) 
MASSBEAKER = 0.11851    # Mass of beaker (kg)
HEATCAPBEAKER = 830     # Heat Capacity of Beaker
MASSWATER = 0.09979     # Mass of water (kg)
HEATCAPWATER = 4186     # Heat Capacity of Water
OUTSIDETEMP = 294.95    # Temperature of Outside Air
C = OUTSIDETEMP         # Guess of Constant of Integration

# Mass and Heat Capacity
PHO = (MASSBEAKER * HEATCAPBEAKER + MASSWATER * HEATCAPWATER) ** -1

# Define the rate of change in temperature differential equation model
def theoryModel(y, t, a, b): 
    # a' = b
    # b' = 
    t, temp = y
    # EPOWER = VOLT ** 2 / RESIST
    # CONDUCT = a * (temp - OUTSIDETEMP)
    # RADIATE = b * (temp ** 4 - OUTSIDETEMP ** 4)
    dydt = [temp, PHO * ( (VOLT ** 2 / RESIST) - ( (a * (temp - OUTSIDETEMP) ) + (b * (temp ** 4 - OUTSIDETEMP ** 4) ) ) ) ]
    return dydt

a = 0.35 # arbitraryConduct
b = 0.0000000001 # arbitraryRadiate

#Set initial conditions, [t=0,T(0)=OUTSIDETEMP]
y0=[0, OUTSIDETEMP]

# Create timesteps from zero up to 61 minutes
t_totalDuration = 3660
t = np.linspace(0, t_totalDuration, 61 + 1) #Currently for every minutes, may change 3660 seconds to miliseconds for increased accuracy 
print("Delta Time=", t,"s")

# Solve model given initial conditions and parameter values
theoryGrated, infodist = odeint(theoryModel, y0, t, args=(a, b), full_output=True )
print("Theory Grated --------------------")
print(theoryGrated)

# Plot solution
plt.plot(t, theoryGrated[:, 1], 'b')
plt.xlabel('t(s)')
plt.ylabel('T(K)')
# plt.title('Velocity vs. Time (burn phase)') # Rename
plt.grid()
# plt.text(1, 10, r"$\frac{dv}{dt}=-g+(v_0k-\beta v^2)/(M_0-kt)$")
# plt.text(1, 9,"$t_{burnout} = %s s$" %round(t_burnout,2))
# plt.text(1, 8,"$v_{burnout} = %s m/s$" %round(v_burnout,2))
plt.show()

################################################
# THRUST PHASE: MAXIMUM HEIGHT REACHED BY t_burnout
################################################

# Numerically integrate v(t) from sol1 w.r.t. time t from t=0s to t=t_burnout
# to get total displacement (change in height) using trapezoidal rule
x_burnout = trapz(t,theoryGrated[:,1])
print("x(t_burnout)=",x_burnout,"m")

#################################################
# THRUST PHASE: POSITION vs. TIME from t=0s to t_burnout
#################################################

# Define a funtion to numerically integrate v(t) from
# t=0s to each value of t_i in the list t. This gives the
# position x(t_i) integrated from t=0s up to and including t_i
def f1(i):
    t2 = [item for item in t if item <= t[i]]
    sol2 = odeint(model1, y0, t2, args=(k,v0,b,M0))
    a2 =trapz(t2,sol2[:,1])
    return(a2)

# Apply f1(i) to every element in t
h = list(map(lambda i:f1(i),range(0,len(t))))

#Plot position vs time x(t) up to t_burnout
plt.plot(t,h,'b')
plt.xlabel('t(s)')
plt.ylabel('x(m)')
#plt.title('Position vs. Time (burn phase)')
plt.grid()
plt.text(1, 18, "$x(t)=\int_0^t v(t)$")
plt.text(1, 25,"$t_{burnout} = %s s$" %round(t_burnout,2))
plt.text(1, 22,"$x_{burnout} = %s m$" %round(x_burnout,2))
plt.show()