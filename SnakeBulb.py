#FILE: SnakeBulb.py
#LANGUAGE: python 3.11
#DESCRIPTION: Quick python script for determining velocity vs. time of a 
# 1D rocket with constant gravity and simple v^2 wind resistance acting, 
# and with constant propellant ejection rate.

#################################################
# TO-DO
#################################################
# (1) ADD COAST PHASE: BURNOUT-TO-APEX
# Add code to calculate apex height and time-to-apex after t_burnout

# (2) ADD COAST PHASE: POST-APEX
# Add code for free-fall after reaching apex, t > t_apex

# (3)ADD GENERAL THRUST FUNCTIONS
# Generalize the code to take general exhaust velocity
# and exhaust mass rate functions v0(t), dm(t)/dt (or general
# time-dependent thrust functions, either functional or discrete).
# Such code will ultimately be needed to model with the actual
# measured motor thrust characteristics


###################################################
# THRUST PHASE: VELOCITY VS. TIME
###################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import trapz

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
arbitraryConduct = 0.0
arbitraryRadiate = 0.0

# Define the rate of change in temperature differential equation model
def theoryModel(T, t): #y,t,k,v0,b,M0))
    # t, v = y
    PHO = (MASSBEAKER * HEATCAPBEAKER + MASSWATER * HEATCAPWATER) ** -1
    EPOWER = VOLT ** 2 / RESIST
    CONDUCT = arbitraryConduct * (T - OUTSIDETEMP ** 4)
    RADIATE = arbitraryRadiate * (T ** 4 - OUTSIDETEMP ** 4)
    dTdt = [t, PHO * (EPOWER - (CONDUCT + RADIATE) ) ]
    return dTdt

#Set initial conditions, [t=0,v(0)=0]
y0=[0,OUTSIDETEMP]

# Create timesteps from zero up to 61 minutes
t_totalDuration = 60 * 61
t = np.linspace(0, t_totalDuration, 50) #replace 
print("Burnout time=",t_burnout,"s")

# Solve model given initial conditions and parameter values
sol1 = odeint(model1, y0, t, args=(k,v0,b,M0))
#print(sol1)
#Velocity at t_burnout
v_burnout = max(sol1[:,1])

# Plot solution
plt.plot(t, sol1[:, 1], 'b')
plt.xlabel('t(s)')
plt.ylabel('v(m/s)')
#plt.title('Velocity vs. Time (burn phase)')
plt.grid()
plt.text(1, 10, r"$\frac{dv}{dt}=-g+(v_0k-\beta v^2)/(M_0-kt)$")
plt.text(1, 9,"$t_{burnout} = %s s$" %round(t_burnout,2))
plt.text(1, 8,"$v_{burnout} = %s m/s$" %round(v_burnout,2))
plt.show()

################################################
# THRUST PHASE: MAXIMUM HEIGHT REACHED BY t_burnout
################################################

# Numerically integrate v(t) from sol1 w.r.t. time t from t=0s to t=t_burnout
# to get total displacement (change in height) using trapezoidal rule
x_burnout = trapz(t,sol1[:,1])
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