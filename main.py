import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import odeint
import rk4

Ks = np.diag([1.5, 1.5, 1.5, 1.5])
K = np.diag([1.6, 1.4, 1.6, 1.8])


class Path:
    def __init__(self, cur_pos, cur_velocity, xy_eq_constraint, h_eq_constraint, psi_eq_constraint):
        self.cur_pos = cur_pos
        self.cur_velocity = cur_velocity
        self.xy_eq_constraint = xy_eq_constraint
        self.h_eq_constraint = h_eq_constraint
        self.psi_eq_constraint = psi_eq_constraint

def controller(x, t, path):

    theta = np.arctan2(4*x[1], 8*x[0])
    if theta < 0:
        theta += 2*np.pi
    xr_app = np.array([8 * np.cos(theta), 8 * np.sin(theta), 0, theta + (np.pi / 2)])

    objective = lambda x_p : (x_p[0] - x[0]) ** 2 + (x_p[1] - x[1]) ** 2 + (x_p[2] - x[2]) ** 2
    xy_eq_constraint = path.xy_eq_constraint
    h_eq_constraint = path.h_eq_constraint
    psi_eq_constraint = path.psi_eq_constraint

    x0 = xr_app

    bounds = ((None, None), (None, None), (None, None), (None, None))

    eq_con1 = {'type': 'eq', 'fun': xy_eq_constraint}
    eq_con2 = {'type': 'eq', 'fun': h_eq_constraint}
    eq_con3 = {'type': 'eq', 'fun': psi_eq_constraint}

    xr = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=[eq_con1, eq_con2, eq_con3]).x

    # theta = np.arctan2(x[1], x[0])
    # if theta < 0:
    #     theta += 2*np.pi
    # xr = np.array([np.cos(theta), np.sin(theta), 0, theta + (np.pi / 2)])
    
    x_tilde = xr - x

    vd = path.cur_velocity(t)

    J = np.array([[np.cos(x[3]), -1*np.sin(x[3]), 0, 0],
                        [np.sin(x[3]), np.cos(x[3]), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    u = Ks.dot(np.tanh(K.dot(x_tilde))) + vd
    u = np.linalg.inv(J).dot(u)
    # print(x_tilde)
    return u



def dynamics(x, t, path):
    J = np.array([[np.cos(x[3]), -1*np.sin(x[3]), 0, 0],
                        [np.sin(x[3]), np.cos(x[3]), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    u = controller(x, t, path)
    return J.dot(u)





if __name__ == "__main__":
    x0 = np.array([9, 9, 2, np.pi / 3])
    a = 8
    b = 4
    # Path properties
    def cur_pos(t):
        return np.array([a*np.cos(t), b*np.sin(t), 1, (np.pi / 2) + (t % (2*np.pi))])

    def cur_velocity(t):
        return np.array([-1 * a * np.sin(t), b*np.cos(t), 0, 1])

    def xy_eq_constraint(x):
        return (x[0] / a)**2 + (x[1] / b)**2 - 1
    
    def h_eq_constraint(x):
        return x[2] - 1

    def psi_eq_constraint(x):
        angle = np.arctan2(b*x[1], a*x[0])
        if angle < 0:
            angle += 2 * np.pi
        angle += np.pi / 2
        return (x[3] - angle)

    path = Path(cur_pos, cur_velocity, xy_eq_constraint, h_eq_constraint, psi_eq_constraint)

    def rk4_func(t, x):
        return dynamics(x, t, path)

    time = np.linspace(0, 30, 1500)

    result = rk4.rk4method(rk4_func, x0, time)

    plt.plot(result[:, 0], result[:, 1])
    plt.show()
