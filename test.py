from scipy.optimize import minimize
import numpy as np

# Define the objective function to be minimized
def objective(x):
    # return the objective function value
    return (x[0] - 3) ** 2 + (x[1] - 2) ** 2 + (x[2] - 2) ** 2 + (x[3] - 0) ** 2

# Define the equality constraint function
def eq_constraint1(x):
    # return the value of the constraint function
    return (x[0] ** 2) + (x[1] ** 2) - 1

def eq_constraint2(x):
    # return the value of the constraint function
    return x[2]

def eq_constraint3(x):
    # return the value of the constraint function
    return x[3]

def psi_eq_constraint(x):
    angle = np.arctan2(x[1], x[0])
    print(f"hello {angle}")
    if angle < 0:
        angle += 2 * np.pi
    angle += np.pi / 2
    return angle % (2 * np.pi)

# Define the initial guess
x0 = [(1 / 2 ** 0.5), (1/2**0.5), 0, 0.1]

print(psi_eq_constraint(np.array([1, 0])) * 180 / np.pi )
