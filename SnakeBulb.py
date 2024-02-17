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
import lmfit
from lmfit import minimize, Parameters
import csv
import data
# import math

x_data = data.x_data
y_data = data.y_data
combined_data = data.combined_data

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
    dydt = [temp, t * PHO * ( (VOLT ** 2 / RESIST) - ( (a * (temp - OUTSIDETEMP) ) + (b * (temp ** 4 - OUTSIDETEMP ** 4) ) ) ) ]
    return dydt

# Coefficients of Optimization
# a = 0 # arbitraryConduct
# b = 0.000000000 # arbitraryRadiate

#Set initial conditions, [t=0,T(0)=OUTSIDETEMP]
y0=[0, OUTSIDETEMP]

# Create timesteps from zero up to 61 minutes
t_totalDuration = 3660
t = np.linspace(0, t_totalDuration, 61 + 1) #Currently for every minutes, may change 3660 seconds to miliseconds for increased accuracy 
print("Delta Time=", t,"s")

# Solve model given initial conditions and parameter values
def requestIntegral(a, b): 
    theoryGrated, infodist = odeint(theoryModel, y0, t, args=(a, b), full_output=True )
    print("Theory Grated --------------------")
    print(theoryGrated)
    return theoryGrated

# Create lmfit Parameters object and set initial values
paramse = Parameters()
paramse.add('a', value=0.1)
paramse.add('b', value=0.0000000001)

# modelTheory = lmfit.Model(requestIntegral, param_names = paramse)
modelTheory = lmfit.Minimizer(requestIntegral, params=paramse)

result = modelTheory.minimize(modelTheory, paramse)

# result = modelTheory.fit(combined_data, params = paramse)

# Get optimized values of 'a' and 'b'
optimized_a = result.paramse['a'].value
optimized_b = result.paramse['b'].value

# Create a lmfit Model using the theoryModel function
lm_model = lmfit.Model(requestIntegral)

# Set initial parameter values for optimization
lm_params = lm_model.make_params(a=0, b=0.000000000)

# Perform the optimization
print("hello")
print(lm_model.param_names)
print(lm_model.independent_vars)
result = lm_model.fit(combined_data)

# Print the result
print(result.fit_report())

# Get optimized values of 'a' and 'b'
optimized_a = result.params['a'].value
optimized_b = result.params['b'].value

print("Optimized 'a':", optimized_a)
print("Optimized 'b':", optimized_b)

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
plt.plot(x_data, y_data, '.', label="Lab Data", color="black")
plt.plot(t, requestIntegral(optimized_a, optimized_b)[:, 1], 'b')
plt.xlabel('t(s)')
plt.ylabel('T(K)')
plt.title('Temperature (K) vs. Time (Seconds)')
plt.grid()
# plt.text(1, 10, r"$\frac{dv}{dt}=-g+(v_0k-\beta v^2)/(M_0-kt)$")
# plt.text(1, 9,"$t_{burnout} = %s s$" %round(t_burnout,2))
# plt.text(1, 8,"$v_{burnout} = %s m/s$" %round(v_burnout,2))
plt.show()