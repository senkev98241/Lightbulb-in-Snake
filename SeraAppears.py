from sympy import Symbol, integrate, lambdify, pprint
from data import y_data, rawX_data, x_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lmfit import conf_interval, report_ci, Parameters, Model

############################################################################################

##Constants

VOLT = 10.0  # Voltage (v)
RESIST = 10.8  # Resistance (ohms)
MASSBEAKER = 0.11851  # Mass of beaker (kg)
HEATCAPBEAKER = 830  # Heat Capacity of Beaker
MASSWATER = 0.09979  # Mass of water (kg)
HEATCAPWATER = 4186  # Heat Capacity of Water
OUTSIDETEMP = 294.65  # Temperature of Outside Air (k)

#Constants used for starting point

S = 5.670374419 * 10**(-8)  #Stefan-Boltzmann Constant
W = 0.002799  #Width of Beaker Wall
E = 0.9  #Emissivity of Beaker?
K = 1.2  #Conductivity of Beaker
A = 0.0145582188966  #Surface of Beaker

m = (K * A) / W
n = S * A * E

# Mass and Heat Capacity
PHO = (MASSBEAKER * HEATCAPBEAKER + MASSWATER * HEATCAPWATER)

############################################################################################


##integral solved for y, but y is not isolated
def funY():
  x, y, a, b, o, p, v, r, c = (Symbol("x"), Symbol("y"), Symbol("a"),
                               Symbol("b"), Symbol("o"), Symbol("p"),
                               Symbol("v"), Symbol("r"), Symbol("c"))
  ##fun1 is the equation= that will be integrated
  fun1 = (p**(-1)) * (v**2 / r - (a * ((y**4) - (o**4))) - b * (y - o))
  ##fun2 is the losses integrated in respect to temperature or y
  fun2 = integrate(fun1, x) + c
  ##Prints out ascii equation in console
  pprint(fun2)
  return lambdify([x, y, a, b, o, p, v, r, c],
                  expr=fun2,
                  modules="scipy",
                  cse=True,
                  docstring_limit=None)


############################################################################################

##Assigning Lab Data to Independent Variables

df = pd.DataFrame({
    'x': x_data,
    'y': y_data,
})

y = np.asarray(y_data)
x = np.asarray(x_data)

##Paramater Declaration

paramse = Parameters()
paramse.add('a', value=n, min=0, max=0.00001)
paramse.add('b', value=m, min=0, max=500)

paramse.add(
    'c', value=OUTSIDETEMP,
    vary=False)  #Integration constant, assumed to be intitial temp of water
paramse.add('o', value=OUTSIDETEMP, vary=False)
paramse.add('p', value=PHO, vary=False)
paramse.add('v', value=VOLT, vary=False)
paramse.add('r', value=RESIST, vary=False)

##Creating function used for curve fit

model = Model(funY(), independent_vars=["x", "y"], paramse=paramse)

##Curve Fitting

fit = model.fit(df['y'],
                x=df['x'],
                y=df['y'],
                method="differential_evolution",
                params=paramse)

##Calculating 3 standard deviations of confidence intervals

# sigma_levels = [1, 2, 3]
# ci = conf_interval(fit, fit, sigmas=sigma_levels)

##Printing Coefficient Data & CI

print(fit.fit_report())
# report_ci(ci)

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
