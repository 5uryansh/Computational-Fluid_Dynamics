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
Nx = 10
Ny = 10

dx = L/Nx
dy = W/Ny
dt = 0.00001



# creating arrays and initialising with zero value assuming m as n+1
T_n = np.zeros((Nx+2, Ny+2))
T_m = np.zeros((Nx+2, Ny+2))



# defining the boundary conditions function
def boundary_condition(T_r, T_b):
    """returns the value of fictious cell
    corresponding to its real cell

    Args:
        T_r (flat): real cell temperature
        T_b (float): wall temperature

    Returns:
        float: fictious cell temperature
    """
    return (T_r + 2 * (T_b - T_r))

# defining BC function for T_n
def BCforArray(T1, T2, T_n):
    Nx = T_n.shape[0] - 2
    Ny = T_n.shape[1] - 2
    
    for i in range(0, Ny+1):
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




# applying boundary conditions
T_n = BCforArray(T1, T2, T_n)
# print(T_n)

# Plot the array as a heatmap
plt.imshow(T_n, cmap='viridis', interpolation='nearest')
plt.colorbar()
plt.show()