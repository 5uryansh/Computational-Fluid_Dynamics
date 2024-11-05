import numpy as np
import matplotlib.pyplot as plt

# defining the parameters
T1 = 100
T2 = 200
T0 = 150
rho = 7750
cp = 500.0
k = 16.2
L = 1
W = 1
Nx = 20
Ny = 20
convergence_factor = 1e-6
alpha = k / (rho*cp)
max_iter = 10000

dx = L/Nx
dy = W/Ny
dt = (0.1 * dx**2)/alpha



# defining the boundary conditions function
def boundary_condition(T_r, T_b):
    return (T_r + 2 * (T_b - T_r))

# defining BC function for T_n
def BCforArray(T1, T2, T_n):
    Nx = T_n.shape[0] - 2
    Ny = T_n.shape[1] - 2
    
    for i in range(0, Nx+1):
        # for left wall
        T_r = T_n[i, 1]
        T_n[i, 0] = boundary_condition(T_r, T1)
        
        # for right wall
        T_r = T_n[i, Nx]
        T_n[i, Nx+1] = boundary_condition(T_r, T1)

    for j in range(0, Ny+1):
        # for top wall
        T_r = T_n[1, j]
        T_n[0, j] = boundary_condition(T_r, T2)
        
        # for bottom wall
        T_r = T_n[Nx, j]
        T_n[Ny+1, j] = boundary_condition(T_r, T1)
    
    # returning the updated values of T_n
    return T_n



# creating arrays and initialising with zero value assuming m as n+1
T_n = np.zeros((Nx+2, Ny+2))
T_m = np.zeros((Nx+2, Ny+2))

# applying boundary conditions
T_n = BCforArray(T1, T2, T_n)

# defining time as needed to plot at center
time = 0
time_passed = []
T_center = []



# updating T_n+1
n = 0
while(n < max_iter):
    for i in range(1, Nx+1):
        for j in range(1, Ny+1):
            T_m[i, j] = (T_n[i, j] +
                         alpha * dt * (
                            (T_n[i+1, j] - 2*T_n[i, j] + T_n[i-1, j])/dy**2 +
                            (T_n[i, j+1] - 2*T_n[i, j] + T_n[i, j-1])/dx**2
                        ))
        T_m = BCforArray(T1, T2, T_m)
        
    Rms = 0
    for i in range(1, Nx):
        for j in range(1, Ny):
            Rms = Rms + (T_m[i, j] - T_n[i, j])**2
    
    Rms = (Rms/(Nx*Ny))**0.5
    T_n = T_m.copy()
    if Rms < convergence_factor:
        # print(T_m)
        print(f"Converged after {n}th iterations.")
        break
    else:
        print(f"No. of iterations: {n}")
        time_passed.append(n*dt)
        x = int(Nx/2 - 1)
        y = int(Ny/2 - 1)
        T_center.append(T_n[x, y])
        n = n+1

# defining temperature without fictious cell
T = np.zeros((Nx, Ny))
for i in range(0, Ny):
    for j in range(0, Nx):
        T[i, j] = T_m[i+1, j+1]


# Plot the array as a heatmap
plt.imshow(T, cmap='viridis', interpolation='nearest')
# plt.plot(time_passed, T_center)
plt.colorbar()
plt.show()