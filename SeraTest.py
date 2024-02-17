import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import trapz

#Constants

V = 10 #Voltage
R = 10.8 #Resistance
M_b = 0.11851 #Mass of Beaker
C_b = 830 #Specific Heat Capacity of Beaker
M_w = 0.09979 #Mass of Water
C_w = 4186 #Specific Heat Capacity of Water
T0 = 294.95 #Initial Temp of Water
S = 5.670374419**(-8) #Stefan-Boltzmann Constant
W = 0.002799 #Width of Beaker Wall
E = 0.9 #Emissivity of Beaker?
K = 1.2 #Conductivity of Beaker
A = 0.0145582188966 #Surface of Beaker



def model1(y,t,V,R,M_b,C_b,M_w,C_w,T0,S,E,K,A,W):
    t, T = y
    dydt = [t, (1/(M_b*C_b+M_w*C_w))*((V**2)/R-((K*A)/W))*(y-T0)-S*A*E*(y**4-T0**4)]
    return dydt



y0=[0, 294.95]

t = np.linspace(0,80,3600)

sol1 = odeint(model1, y0, t, args=(V,R,M_b,C_b,M_w,C_w,T0,S,E,K,A,W))

plt.plot(t, sol1[:, 1], 'b')