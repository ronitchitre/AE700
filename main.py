import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.integrate import odeint
import rk4

Ks = np.diag([1.5, 1.5, 1.5, 1.5])
K = np.diag([1.6, 1.4, 1.6, 1.8])
v_max = 1.5
ksc = 2
kc = 0.8

class Path:
    def __init__(self, cur_pos, cur_velocity, xy_eq_constraint, h_eq_constraint, psi_eq_constraint, gamma):
        self.cur_pos = cur_pos
        self.cur_velocity = cur_velocity
        self.xy_eq_constraint = xy_eq_constraint
        self.h_eq_constraint = h_eq_constraint
        self.psi_eq_constraint = psi_eq_constraint
        self.gamma = gamma

def controller(x, path):

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

    des_theta = np.arctan2(2 * xr[1], xr[0])

    # theta = np.arctan2(x[1], x[0])
    # if theta < 0:
    #     theta += 2*np.pi
    # xr = np.array([np.cos(theta), np.sin(theta), 0, theta + (np.pi / 2)])
    
    x_tilde = xr - x

    vd_norm = v_max / (1 + ksc * np.tanh(kc * path.gamma(des_theta)))

    vd = path.cur_velocity(des_theta)
    old_vd_norm = np.linalg.norm(vd)
    vd = vd * vd_norm / old_vd_norm

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
    u = controller(x, path)
    return J.dot(u)





if __name__ == "__main__":
    x0 = np.array([6, 7, 2, np.pi / 3])
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

    def gamma(t):
        r_2d = np.array([-1*a*np.cos(t), -1*b*np.sin(t)])
        r_1d = np.array([-1*a*np.sin(t), b*np.cos(t)])
        return np.linalg.norm(np.cross(r_2d, r_1d)) / np.linalg.norm(r_1d)**3


    path = Path(cur_pos, cur_velocity, xy_eq_constraint, h_eq_constraint, psi_eq_constraint, gamma)

    def rk4_func(t, x):
        return dynamics(x, t, path)

    time = np.linspace(0, 50, 2000)

    result = rk4.rk4method(rk4_func, x0, time)

    control_vect = np.zeros_like(result)
    i = 0
    for x in result:
        control_vect[i] = controller(x, path)
        i += 1

    plt.plot(result[:, 0], result[:, 1])
    plt.title("xy plane")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

    plt.plot(time, result[:, 2])
    plt.title("height vs time")
    plt.xlabel("time")
    plt.ylabel("height")
    plt.show()

    plt.plot(time, (result[:, 3] % (2 * np.pi)))
    plt.title("yaw vs time")
    plt.xlabel("time")
    plt.ylabel("yaw")
    plt.show()

    plt.plot(time, control_vect[:, 0], label="x velocity")
    plt.plot(time, control_vect[:, 1], label="y velocity")
    plt.plot(time, control_vect[:, 2], label="z velocity")
    plt.title("control velocity vs time")
    plt.xlabel("time")
    plt.ylabel("control input")
    plt.legend()
    plt.show()

    plt.plot(time, control_vect[:, 3])
    plt.title("control angular velocity vs time")
    plt.xlabel("time")
    plt.ylabel("control angular velocity")
    plt.show()

