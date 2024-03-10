# from sympy import Symbol, integrate, lambdify, pprint
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from lmfit import conf_interval, report_ci, Parameters, Model

############################################################################################

## Key Data
import timeStamps
time = timeStamps.timeStampsPoints

import rawTemps
rawTempData = np.array(rawTemps.rawTempPoints)

import rawTempTime
rawTempTimeData = rawTempTime.rawTempTimePoints

import pseudoTrapTemps
pseudoTrapTempData = np.array(pseudoTrapTemps.pseudoTempPoints)

import midEulerTemps
midEulerTempData = np.array(midEulerTemps.midEulerTempPoints)

import OrderTwoRKTemps
OrderTwoRKTempData = np.array(OrderTwoRKTemps.O2RGK)

import OrderFourRKTemps
OrderFourRKTempData = np.array(OrderFourRKTemps.O4RGK)
############################################################################################

##Plotting Graphs

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Major ticks every 20==360, minor ticks every 60
major_ticks = np.arange(0, 3720, 360)
minor_ticks = np.arange(0, 3720, 60)

ax.set_xticks(major_ticks)
ax.xaxis.grid(True, which='major')

ax.set_xticks(minor_ticks, minor=True)
ax.xaxis.grid(False, which='minor', linewidth=0.5, color='lightgrey')

major_ticks = np.arange(293.15, 323.15, 5)
ax.set_yticks(major_ticks)
ax.yaxis.grid(True, which='major')

# And a corresponding grid
ax.grid(which='both')
ax.grid(which='minor', linewidth=0.5)

# # Or if you want different settings for the grids:
# ax.grid(which='minor', alpha=0.2)
# ax.grid(which='major', alpha=0.5)

plt.plot(time, pseudoTrapTempData, '.', label="Pseudo Trapezoidal", color="red", markersize=8)
plt.plot(time, midEulerTempData, '.', label="Midpoint Euler", color="purple")
plt.plot(time, OrderTwoRKTempData, '.', label="2nd Order Runge-Kutta", color="yellow", markersize=4)
plt.plot(time, OrderTwoRKTempData, '.', label="4th Order Runge-Kutta", color="black", markersize=1)
plt.plot(rawTempTimeData, rawTempData, '.', label="Raw Temps", color="teal", markersize=10)

plt.ylabel("Temperature (Kelvin)")
plt.xlabel("Time (Seconds)")
plt.title("Comparison of Different Numerical Methods Optimized to Raw Lab Data \n Temperature (K) vs. Time (s)")
plt.xlim(-60, 3720)
plt.ylim(293.15,323.15)

plt.grid()
plt.legend()
plt.show()
