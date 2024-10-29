"""
Instructions to run the code.

The code requries two libraries numpy and colorama.
This can be installed by simply running below command in terminal

`pip install numpy colorama`

After running the code, it will ask for three values
1. n = number of rows and column of the matrix
2. w = relaxation factor
3. d = diagonal element of the matrix, for example choosing 2 will make all diagonal element as 2.

If the solution does not converge, it will give very large values in order of e+100, or just give `nan` if the program can't handle the values.
In both cases the solution did not converge.
"""


# importing numpy for mathematical operations
import numpy as np

# import colorama to get visually good output (not just white texts)
from colorama import Style, Fore

# taking the value of n from user, n is size of matrix and w is relaxation factor
n = int(input("Please enter the value of n you want to solve for: "))
w = float(input("Please enter the value of w you want to solve with: "))
d = int(input("Please choose your diagonal element, for example 2, 4, 6: "))


# Defining A in AX = B
A = np.zeros((n,n))
for i in range(0, n):
    
    # for first row of matrix A
    if i==0:
        A[0][0]=d
        A[0][1]=-1
        A[0][-1]=1
        continue
    
    # for last row of matrix A
    if i==n-1:
        A[n-1][0]=1
        A[n-1][-1]=d
        A[n-1][-2]=-1
        continue
    
    # for remaining rows of matrix A
    for j in range (0, n):
        if i==j:
            A[i][j]=d
        elif i==j+1 or i==j-1:
            A[i][j]=-1
            
print(Fore.GREEN + f"The matrix A is:\n {A}" + Style.RESET_ALL)

# initialising the X in AX = B with random values
X = np.random.rand(n, 1)
print(Fore.RED + f"The inital guess values of X is\n {X}" + Style.RESET_ALL)

# defining b in AX = b
B = np.zeros((n, 1))
B[-1][0] = 1
print(Fore.YELLOW + f"The value of b is\n {B}" + Style.RESET_ALL)


# function for solver
def jacobi_guass(A, X, B, w=1.5, max_iterator=2):
    n = len(A)
    
    # initialising the R_rms
    R_rms = 0.0
    
    # defining R and R_rms for very first step so that the value don't overshoot to infinity
    R = B - np.dot(A,X)
    R_rms = R_rms + R**2 
    iteration = 0
    
    # setting the stopping value R_rms > 1e-6
    while(np.min(R_rms) > 1e-6 and iteration<=4):
        R = B - np.dot(A,X)
        R_rms = R_rms + R**2
        
        # iterating for all value of x
        for i in range(0, n):
            # print(f'The parameters are:\n\n')
            # print(X[i], w, R[i], A[i][i])
            X[i] = X[i] + w*(R[i]/A[i][i])
        R_rms = (R_rms/n)**0.5
        iteration+=1
    # print(X)
    return X

import time
start = time.time()
print(Fore.BLUE + f'\n\n\nThe solution is:\n')
print(jacobi_guass(A,X,B,w, 1000))
print(Style.RESET_ALL)

end = time.time()
# print the difference between start 
# and end time in milli. secs
print("The time of execution of above program is :",
      (end-start) * 10**3, "ms")