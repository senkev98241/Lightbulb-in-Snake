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
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.integrate import trapz
from lmfit import conf_interval, report_ci, Parameters, Model
from sympy import Symbol, integrate, lambdify, pprint
import csv
# import data
from data import y_data, x_data
# import math

x = np.asarray(x_data)
y = np.asarray(y_data)
# combined_data = combined_data

# Set constants and other parameter values
VOLT = 10.0             # Voltage (units here)
RESIST = 10.8           # Resistance (units here) 
MASSBEAKER = 0.11851    # Mass of beaker (kg)
HEATCAPBEAKER = 830     # Heat Capacity of Beaker
MASSWATER = 0.09979     # Mass of water (kg)
HEATCAPWATER = 4186     # Heat Capacity of Water
OUTSIDETEMP = 294.95    # Temperature of Outside Air
C = 294.65        # Guess of Constant of Integration

#Constants used for starting point

S = 5.670374419 * 10**(-8)  #Stefan-Boltzmann Constant
W = 0.002799  #Width of Beaker Wall
E = 0.9  #Emissivity of Beaker?
K = 1.2  #Conductivity of Beaker
A = 0.0145582188966  #Surface of Beaker

m = (K * A) / W
n = S * A * E

# Mass and Heat Capacity
PHO = (MASSBEAKER * HEATCAPBEAKER + MASSWATER * HEATCAPWATER) ** -1

# Define the rate of change in temperature differential equation model
def diffModel(): # Used y, t, a, b and theoryModel()
    # a' = b
    # b' = 
    # t, temp = y
    # EPOWER = VOLT ** 2 / RESIST
    # CONDUCT = a * (temp - OUTSIDETEMP)
    # RADIATE = b * (temp ** 4 - OUTSIDETEMP ** 4)
    x, y, a, b, o, p, v, r, c = (Symbol("x"), Symbol("y"), Symbol("a"), Symbol("b"), Symbol("o"), Symbol("p"), Symbol("v"), Symbol("r"), Symbol("c"))
    # diffFunc = PHO * ( (VOLT ** 2 / RESIST) - ( (a * (temp - OUTSIDETEMP) ) + (b * (temp ** 4 - OUTSIDETEMP ** 4) ) ) )
    diffFunc = p * ( (v ** 2 / r) - ( (a * (y - o) ) + (b * (y ** 4 - o ** 4) ) ) )
    integratedFunc = integrate(diffFunc, x) + c
    print(integratedFunc)
    return lambdify([x, y, a, b, o, p, v, r, c], expr=integratedFunc, modules="scipy", cse=True, docstring_limit=None)

# Coefficients of Optimization
# a = 0 # arbitraryConduct
# b = 0.000000000 # arbitraryRadiate

#Set initial conditions, [t=0,T(0)=OUTSIDETEMP]
# y0=[0, OUTSIDETEMP]

# Create timesteps from zero up to 61 minutes
# t_totalDuration = 3660
# t = np.linspace(0, t_totalDuration, 61 + 1) #Currently for every minutes, may change 3660 seconds to miliseconds for increased accuracy 
# print("Delta Time=", t,"s")

# Solve model given initial conditions and parameter values
# def requestIntegral(a, b): 
#     theoryGrated, infodist = odeint(theoryModel, y0, t, args=(a, b), full_output=True )
#     print("Theory Grated --------------------")
#     print(theoryGrated)
#     return theoryGrated

df = pd.DataFrame({'x': x_data, 'y': y_data})

# Create lmfit Parameters object and set initial values
paramse = Parameters()
paramse.add('a', value=n, min=0, max=100)
paramse.add('b', value=m, min=0, max=0.00001)

paramse.add('c', value=294.65, vary=False)  #Integration constant, assumed to be intitial temp of water
paramse.add('o', value=OUTSIDETEMP, vary=False)
paramse.add('p', value=PHO, vary=False)
paramse.add('v', value=VOLT, vary=False)
paramse.add('r', value=RESIST, vary=False)

# modelTheory = lmfit.Model(requestIntegral, param_names = paramse)
model = Model(diffModel(), independent_vars=['x', 'y'])#, paramse=paramse)

#Curve Fitting

fit = model.fit(df['y'], x=df['x'], y=df['y'], params=paramse)

sigma_levels = [1, 2, 3]
ci = conf_interval(fit, fit, sigmas=sigma_levels)

##Printing Coefficient Data & CI

print(fit.fit_report())
report_ci(ci)

#print(fit.best_fit)
############################################################################################

##Plotting Graphs

plt.plot(x_data, y_data, '.', label="Lab Data", color="teal")
plt.plot(x, fit.best_fit, '-', label='Best Fit', color="orange")
plt.ylabel("Temperature (Kelvin)")
plt.xlabel("Time (Minutes)")
plt.title(
    "Curve Fit of Thermal Losses Compared to Lab Data \n Temperature (k) vs. Time (Min)"
)
plt.xlim(-10, 3670)
plt.legend()
plt.show()

# result = modelTheory.minimize(modelTheory, paramse)

# result = modelTheory.fit(combined_data, params = paramse)

# Get optimized values of 'a' and 'b'
# optimized_a = result.paramse['a'].value
# optimized_b = result.paramse['b'].value

# Create a lmfit Model using the theoryModel function
# lm_model = lmfit.Model(requestIntegral)

# Set initial parameter values for optimization
# lm_params = lm_model.make_params(a=0, b=0.000000000)

# Perform the optimization
# print("hello")
# print(lm_model.param_names)
# print(lm_model.independent_vars)
# result = lm_model.fit(combined_data)

# Print the result
# print(result.fit_report())

# # Get optimized values of 'a' and 'b'
# optimized_a = result.params['a'].value
# optimized_b = result.params['b'].value

# print("Optimized 'a':", optimized_a)
# print("Optimized 'b':", optimized_b)

# # Define the objective function for lmfit
# def objective(params, y0, t):
#     a = params['a']
#     b = params['b']
#     theoryGrated, _ = odeint(theoryModel, y0, t, args=(a, b), full_output=True)
#     return theoryGrated[:, 0]  # Return only the temperature values

# # Create lmfit Parameters object and set initial values
# params = Parameters()
# params.add('a', value=0.0)
# params.add('b', value=0.0)

# # Perform the optimization
# result = lmfit.minimize(objective, params, args=(y0, t))

# # Extract optimized values
# optimized_a = result.params['a'].value
# optimized_b = result.params['b'].value

# # Print optimized values
# print(f"Optimized 'a': {optimized_a}")
# print(f"Optimized 'b': {optimized_b}")


# Plot solution
# plt.plot(x_data, y_data, '.', label="Lab Data", color="black")
# plt.plot(t, requestIntegral(optimized_a, optimized_b)[:, 1], 'b')
# plt.xlabel('t(s)')
# plt.ylabel('T(K)')
# plt.title('Temperature (K) vs. Time (Seconds)')
# plt.grid()
# plt.text(1, 10, r"$\frac{dv}{dt}=-g+(v_0k-\beta v^2)/(M_0-kt)$")
# plt.text(1, 9,"$t_{burnout} = %s s$" %round(t_burnout,2))
# plt.text(1, 8,"$v_{burnout} = %s m/s$" %round(v_burnout,2))
# plt.show()