# from sympy import Symbol, integrate, lambdify, pprint
# import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
# from lmfit import conf_interval, report_ci, Parameters, Model

############################################################################################

## Key Data
import timeStamps
time = timeStamps.timeStampsPoints

import rawTemps
rawTempData = rawTemps.rawTempPoints

rawTempTime = [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960, 1020, 1080, 1140, 1200, 1260, 1320, 1380, 1440, 1500, 1560, 1620, 1680, 1740, 1800, 1860, 1920, 1980, 2040, 2100, 2160, 2220, 2280, 2340, 2400, 2460, 2520, 2580, 2640, 2700, 2760, 2820, 2880, 2940, 3000, 3060, 3120, 3180, 3240, 3300, 3360, 3420, 3480, 3540, 3600, 3660]

import pseudoTrapTemps
pseudoTrapTempData = pseudoTrapTemps.pseudoTempPoints

import midEulerTemps
midEulerTempData = midEulerTemps.midEulerTempPoints

import OrderTwoRKTemps
OrderTwoRKTempData = OrderTwoRKTemps.O2RGK

import OrderFourRKTemps
OrderFourRKTempData = OrderFourRKTemps.O4RGK
############################################################################################

##Plotting Graphs

# plt.plot(rawTempTime, rawTemps, '.', label="Raw Temps", color="teal")
plt.plot(time, pseudoTrapTemps, '.', label="Pseudo Trapezoidal", color="red")
plt.plot(time, midEulerTemps, '.', label="Midpoint Euler", color="purple")
plt.plot(time, OrderTwoRKTemps, '.', label="2nd Order Runge-Kutta", color="yellow")
plt.plot(time, OrderTwoRKTemps, '.', label="4th Order Runge-Kutta", color="black")


plt.ylabel("Temperature (Kelvin)")
plt.xlabel("Time (Seconds)")
plt.title(
    "Curve Fit of Thermal Losses Compared to Lab Data \n Temperature (k) vs. Time (Min)"
)
plt.xlim(-60, 3720)
plt.ylim(293.15,323.15)
plt.legend()
plt.show()
