import numpy as np
rawX_data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61]
x_data = np.array(rawX_data) * 60
y_data = [294.65, 294.55, 294.95, 295.45, 296.15, 296.75, 297.65, 298.35, 299.15, 299.95, 300.75, 301.55, 302.25, 303.05, 303.75, 304.45, 305.15, 305.95, 306.35, 307.15, 307.55, 308.05, 308.55, 309.15, 309.45, 310.15, 310.45, 311.05, 311.45, 311.85, 312.15, 312.55, 313.05, 313.45, 313.65, 313.95, 314.25, 314.35, 314.55, 314.85, 315.15, 315.35, 315.45, 315.55, 315.75, 315.95, 316.15, 316.35, 316.45, 316.55, 316.55, 316.85, 316.75, 316.85, 317.05, 317.25, 317.45, 317.65, 317.55, 317.45, 317.65, 317.65]

# Combine the two arrays into two columns
combined_data = np.column_stack((x_data, y_data))

# Print the result
print(combined_data)