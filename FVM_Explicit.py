"""
This code uses four library which needs to be preinstalled and can be done by

'pip install numpy matplotlib colorama pandas'

After installing the modules change the value of parameters if needed
or just run the code.
"""

# importing the required libraries
import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style
import pandas as pd

# defining the parameters
T1 = 100
T2 = 200
T0 = 150
rho = 7750
cp = 500.0
k = 16.2
L = 1
W = 1
Nx = 80
Ny = 80
convergence_factor = 1e-6
alpha = k / (rho*cp)
max_iter = 100000

dx = L/Nx
dy = W/Ny
dt = (0.1 * dx**2)/alpha





# -----------------Defining the required functions-----------------

# for analytical solution, taking infinity as 100 by default
def analytical_temperature(x, y, T1, T2, L, W, terms=50):
    T = T1 + (T2 - T1) * (2 / np.pi) * sum(
        [( (-1)**(n + 1) + 1 ) / n * np.sin(n * np.pi * x / L) * 
         np.sinh(n * np.pi * y / L) / np.sinh(n * np.pi * W / L)
        for n in range(1, terms + 1)]
    )
    return T

# defining the boundary conditions function
def boundary_condition(T_r, T_b):
    return (T_r + 2 * (T_b - T_r))

# defining BC function for T_n
def BCforArray(T1, T2, T_n):
    Nx = T_n.shape[1] - 2
    Ny = T_n.shape[0] - 2
    
    # changing y
    for i in range(0, Ny+1):
        # for left wall
        T_r = T_n[i, 1]
        T_n[i, 0] = boundary_condition(T_r, T1)
        
        # for right wall
        T_r = T_n[i, Nx]
        T_n[i, Nx+1] = boundary_condition(T_r, T1)

    for j in range(0, Nx+1):
        # for top wall
        T_r = T_n[1, j]
        T_n[0, j] = boundary_condition(T_r, T2)
        
        # for bottom wall
        T_r = T_n[Nx, j]
        T_n[Ny+1, j] = boundary_condition(T_r, T1)
    
    # returning the updated values of T_n
    return T_n




# -----------------Main coding part-----------------

# creating arrays and initialising with zero value assuming m as n+1
T_n = np.zeros((Ny+2, Nx+2))
T_m = np.zeros((Ny+2, Nx+2))

# applying boundary conditions
T_n = BCforArray(T1, T2, T_n)


# defining trace for time passed as required to plot at center
time = 0
time_passed = []
T_center = []


# updating T_n+1
n = 0
while(n < max_iter):
    for i in range(1, Ny+1):
        for j in range(1, Nx+1):
            T_m[i, j] = (T_n[i, j] +
                         alpha * dt * (
                            (T_n[i+1, j] - 2*T_n[i, j] + T_n[i-1, j])/dy**2 +
                            (T_n[i, j+1] - 2*T_n[i, j] + T_n[i, j-1])/dx**2
                        ))
    T_m = BCforArray(T1, T2, T_m)
        
    Rms = 0
    for i in range(1, Ny):
        for j in range(1, Nx):
            Rms = Rms + (T_m[i, j] - T_n[i, j])**2
    
    Rms = (Rms/(Nx*Ny))**0.5
    T_n = T_m.copy()
    if (Rms/dt) < convergence_factor:
        # print(T_m)
        print(f"Converged after {n}th iterations.")
        
        # for minimum drop in steady state convergence
        print(Fore.RED + f'The minimum drop is: {Rms}')
        break
    else:
        print(f"No. of iterations: {n}")
        time_passed.append(n*dt)
        i_center = int(Ny/2 - 1)
        j_center = int(Nx/2 - 1)
        T_center.append(T_n[i_center, j_center])
        n = n+1

# defining temperature without fictious cell
T = np.zeros((Ny, Nx))
for i in range(0, Ny):
    for j in range(0, Nx):
        T[i, j] = T_m[i+1, j+1]
print(Fore.GREEN) 
df = pd.DataFrame(T)
# printing the final temperature
print("The final temperature along plate is......\n\n")
print(df)




# -----------------Plotting the results-----------------

# comparing analytical and numerical result along x = 0.5
"""
numerical_T_x05 = T[:, int(Nx/2)] this value needed to be reversed as y=0 is not j = 0
also the value needed to be calculated on face of cells,
therefore, used average value of left and right cell
"""
# reversed value for x05is given by below code
numerical_T_x05 = (T[::-1, (int(Nx/2)-1)] + T[::-1, int(Nx/2)])/2
numerical_T_y05 = (T[(int(Ny/2)-1), :] + T[int(Ny/2), :])/2
y_values = np.linspace(0, W, len(numerical_T_x05))
x_values = np.linspace(0, L, len(numerical_T_y05))
analytical_T_x05 = [analytical_temperature(0.5, y, T1, T2, L, W, 100) for y in y_values]
analytical_T_y05 = [analytical_temperature(x, 0.5, T1, T2, L, W, 100) for x in x_values]

# Plot comparison along x=0.5
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(y_values, numerical_T_x05, label="Numerical", marker="o")
plt.plot(y_values, analytical_T_x05, label="Analytical", linestyle="--")
plt.xlabel("y along x = 0.5")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Profile along x = 0.5")
plt.legend()

# Plot comparison along y=0.5
plt.subplot(1, 2, 2)
plt.plot(x_values, numerical_T_y05, label="Numerical", marker="o")
plt.plot(x_values, analytical_T_y05, label="Analytical", linestyle="--")
plt.xlabel("x along y = 0.5")
plt.ylabel("Temperature (°C)")
plt.title("Temperature Profile along y = 0.5")
plt.legend()


# Calculate heat flux for each boundary
q_left = sum([-k * (T[i, 1] - T[i, 0]) / (-dx) for i in range(Ny)])
q_right = sum([-k * (T[i, Nx-1] - T[i, Nx-2]) / dx for i in range(Ny)])
q_top = sum([-k * (T[0, j] - T[1, j]) / dy for j in range(Nx)])
q_bottom = sum([-k * (T[Ny-1, j] - T[Ny-2, j]) / dy for j in range(Nx)])
# print(q_left, q_right, q_top, q_bottom)

# defining tolerance as 10% of max q value
tolerance = abs(0.1*q_top)

# Total heat flux
total_heat_flux = q_left + q_right + q_bottom + q_top

# Check if conservation holds
if abs(total_heat_flux) < tolerance:
    print(Fore.YELLOW + "Global conservation is approximately obeyed.")
    print(Style.RESET_ALL)
else:
    print(Fore.YELLOW + "Global conservation is not obeyed.")
    print(Style.RESET_ALL)
    # print(total_heat_flux)


# Plot the array as a heatmap
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
cax = ax1.imshow(T, cmap='viridis', interpolation='nearest')
fig.colorbar(cax, ax=ax1)  # Add colorbar to the heatmap
ax1.set_title("Heatmap of Temperature")
ax2.plot(time_passed, T_center)
ax2.set_title("Temperature at Center vs. Time")
ax2.set_xlabel("Time")
ax2.set_ylabel("Temperature")
plt.tight_layout()
plt.show()