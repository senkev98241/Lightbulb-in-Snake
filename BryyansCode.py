import numpy as np
import time
import os

# Get the current time
timestamp = time.strftime("Year_%Y_Month_%m_Day_%d_Hour_%H_Minute_%M_Second_%S")

# Get the path to the current user's Downloads folder
downloads_folder = os.path.expanduser("~\\Downloads")

# Use the timestamp in the filename
filename = os.path.join(downloads_folder, f'Water_Temperatures_{timestamp}.txt')

# Constants
C_water = 4.1816
T_air = 294.95
E_water = 0.96
E_beaker = 0.9
A_water = 0.003264745270927
A_beaker = 0.01971223
m_beaker = 118.51
m_water = 99.79
C_beaker = 0.83
V = 10
R = 10.8
A_sidesofwater = 0.007187649832
o_stefan = 0.00000005670367
k_beaker = 1.2
d_thickness = 0.002799

# Initial conditions
T0 = 294.65  # initial temperature
t0 = 0.0  # initial time
dt = 1  # time step
t_final = 3660  # final time

# Time array
t = np.arange(t0, t_final, dt)

# Initialize temperature array
T = np.zeros_like(t)
T[0] = T0

with open(filename, 'w') as f:
    # Write the initial temperature
    f.write(str(T[0]) + '\n')
    for i in range(1, len(t)):
        dTdt = (1/(C_water*m_water + C_beaker*m_beaker)) * ((V**2)/R 
                                                          - A_beaker*E_beaker*o_stefan*T[i-1]**4 
                                                          - A_water*E_water*o_stefan*T[i-1]**4 
                                                          - k_beaker*A_sidesofwater*(T[i-1]-T_air)/d_thickness)
        # dTdt = (1/(C_water*m_water + C_beaker*m_beaker)) * ((V**2)/R 
        #                                                   - 0.0000000000000000000000001*T[i-1]**4 
        #                                                   - 0.0000000000000000000000001*T[i-1]**4 
        #                                                   - .33855*(T[i-1]-T_air))
        T[i] = T[i-1] + dTdt*dt
        # Write every 60th step, 1 minute
        if i % 60 == 0:
            f.write(str(T[i]) + '\n')