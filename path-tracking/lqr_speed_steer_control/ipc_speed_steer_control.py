import time
import rpyc
import math
import numpy as np
import scipy.linalg as la
from utils.angle import angle_mod

from CubicSpline import cubic_spline_planner

# === Parameters =====

# LQR parameter
lqr_Q = np.eye(5)
lqr_R = np.eye(2)
dt = 0.1  # time tick[s]
L = 0.5  # Wheel base of the vehicle [m]
max_steer = np.deg2rad(45.0)  # maximum steering angle[rad]

show_animation = True

# === static methods =====
def update(state, a, delta):
    if delta >= max_steer:
        delta = max_steer
    if delta <= - max_steer:
        delta = - max_steer

    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / L * math.tan(delta) * dt
    state.v = state.v + a * dt

    return state


def pi_2_pi(angle):
    return angle_mod(angle)


def solve_dare(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE)
    """
    x = Q
    x_next = Q
    max_iter = 150
    eps = 0.01

    for i in range(max_iter):
        x_next = A.T @ x @ A - A.T @ x @ B @ \
                 la.inv(R + B.T @ x @ B) @ B.T @ x @ A + Q
        if (abs(x_next - x)).max() < eps:
            break
        x = x_next

    return x_next


def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    X = solve_dare(A, B, Q, R)

    # compute the LQR gain
    K = la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    eig_result = la.eig(A - B @ K)

    return K, X, eig_result[0]


def lqr_speed_steering_control(state, cx, cy, cyaw, ck, pe, pth_e, sp, Q, R):
    ind, e = calc_nearest_index(state, cx, cy, cyaw)

    tv = sp[ind]

    k = ck[ind]
    v = state.v
    th_e = pi_2_pi(state.yaw - cyaw[ind])

    # A = [1.0, dt, 0.0, 0.0, 0.0
    #      0.0, 0.0, v, 0.0, 0.0]
    #      0.0, 0.0, 1.0, dt, 0.0]
    #      0.0, 0.0, 0.0, 0.0, 0.0]
    #      0.0, 0.0, 0.0, 0.0, 1.0]
    A = np.zeros((5, 5))
    A[0, 0] = 1.0
    A[0, 1] = dt
    A[1, 2] = v
    A[2, 2] = 1.0
    A[2, 3] = dt
    A[4, 4] = 1.0

    # B = [0.0, 0.0
    #     0.0, 0.0
    #     0.0, 0.0
    #     v/L, 0.0
    #     0.0, dt]
    B = np.zeros((5, 2))
    B[3, 0] = v / L
    B[4, 1] = dt

    K, _, _ = dlqr(A, B, Q, R)

    # state vector
    # x = [e, dot_e, th_e, dot_th_e, delta_v]
    # e: lateral distance to the path
    # dot_e: derivative of e
    # th_e: angle difference to the path
    # dot_th_e: derivative of th_e
    # delta_v: difference between current speed and target speed
    x = np.zeros((5, 1))
    x[0, 0] = e
    x[1, 0] = (e - pe) / dt
    x[2, 0] = th_e
    x[3, 0] = (th_e - pth_e) / dt
    x[4, 0] = v - tv

    # input vector
    # u = [delta, accel]
    # delta: steering angle
    # accel: acceleration
    ustar = -K @ x

    # calc steering input
    ff = math.atan2(L * k, 1)  # feedforward steering angle
    fb = pi_2_pi(ustar[0, 0])  # feedback steering angle
    delta = ff + fb

    # calc accel input
    accel = ustar[1, 0]

    return delta, ind, e, th_e, accel


def calc_nearest_index(state, cx, cy, cyaw):
    dx = [state.x - icx for icx in cx]
    dy = [state.y - icy for icy in cy]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind)

    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind

def calc_speed_profile(cyaw, target_speed):
    speed_profile = [target_speed] * len(cyaw)

    direction = 1.0

    # Set stop point
    for i in range(len(cyaw) - 1):
        dyaw = abs(cyaw[i + 1] - cyaw[i])
        switch = math.pi / 4.0 <= dyaw < math.pi / 2.0

        if switch:
            direction *= -1

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

        if switch:
            speed_profile[i] = 0.0

    # speed down
    for i in range(int(len(cyaw) * 0.25)):
        speed_profile[-i] = target_speed / (((len(cyaw) // 2 + 10) - i) + 1)
        if speed_profile[-i] <= 1.0 / 3.6:
            speed_profile[-i] = 1.0 / 3.6

    return speed_profile

def reimann(x):
    reimann_sum = []
    for i in range(1, len(accel)):
        reimann_sum.append(np.trapz(accel[:i]) * 0.1 * 2.23)
    return reimann_sum

class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v


class SteeringControl:

    def __init__(self):
        self.goal_tolerance = 0.1 # meters
        self.stop_speed = 0.05 # m/s
        self.init_state = State(x=-0.0, y=-0.0, yaw=0.0, v=0.0)
        self.x_history = [self.init_state.x]
        self.y_history = [self.init_state.y]
        self.yaw_history = [self.init_state.yaw]
        self.v_history = [self.init_state.v]
        self.init_e = 0.0
        self.init_e_th = 0.0
        self.speed_target = 10.0 / 2.237 # divide by 2.237 to convert from mph to m/s
        self.ax = [0.0, 2.0, -2.0, 0.0]
        self.ay = [0.0, 5.0, 10.0, 15.0]
        self.goal = [self.ax[-1], self.ay[-1]]
        self.cx, self.cy, self.cyaw, self.ck, self.s = cubic_spline_planner.calc_spline_course(self.ax, self.ay, ds=0.1)
        self.speed_profile = calc_speed_profile(self.cyaw, self.speed_target)
        self.accel_control_history = []
        self.steering_control_history = []
        self.velocity_control_history = []

    def time_step(self, curr_state, e, e_th):
        dl, target_ind, e, e_th, ai = lqr_speed_steering_control(
            curr_state, self.cx, self.cy, self.cyaw, self.ck, e, e_th, self.speed_profile, lqr_Q, lqr_R)

        state = update(curr_state, ai, dl)

        # if abs(state.v) <= stop_speed:
        #     target_ind += 1

        # time = time + dt

        # check goal
        dx = state.x - self.goal[0]
        dy = state.y - self.goal[1]
        if math.hypot(dx, dy) <= self.goal_tolerance:
            print("Goal")

        self.x_history.append(state.x)
        self.y_history.append(state.y)
        self.yaw_history.append(state.yaw)
        self.v_history.append(state.v)

        # plt.cla()
        # # for stopping simulation with the esc key.
        # plt.gcf().canvas.mpl_connect('key_release_event',
        #                              lambda event: [exit(0) if event.key == 'escape' else None])
        # plt.plot(cx, cy, "-r", label="course")
        # plt.plot(x, y, "ob", label="trajectory")
        # plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
        # plt.axis("equal")
        # plt.grid(True)
        # plt.title("speed[km/h]:" + str(round(state.v * 3.6, 2))
        #           + ",target index:" + str(target_ind))
        # plt.pause(0.0001)

        return state, e, e_th, ai, dl

    def set_trajectory(self, ax, ay):
        self.ax = ax
        self.ay = ay
        self.goal = [ax[-1], ay[-1]]
        self.cx, self.cy, self.cyaw, self.ck, self.s = cubic_spline_planner.calc_spline_course(self.ax, self.ay, ds=0.1)
        self.speed_profile = calc_speed_profile(self.cyaw, self.speed_target)

    def do_simulation(self):
        state = self.init_state
        e = self.init_e
        e_th = self.init_e_th
        for i in range(len(self.cx)):
            state, e, e_th, ai, dl = self.time_step(state, e, e_th)
            self.accel_control_history.append(ai)
            self.steering_control_history.append(dl)
            self.velocity_control_history.append(state.v)
            if math.hypot(state.x - self.goal[0], state.y - self.goal[1]) <= self.goal_tolerance:
                break
        self.velocity_control_history = reimann(self.accel_control_history)
        self.velocity_control_history.insert(0, 0.0)

class Talker(rpyc.service):

    def __init__(self):
        self.steering_control = SteeringControl()
        self.kart_control_conn = rpyc.connect('localhost', 18864)

    def on_connect(self):
        print("connected")

    def on_disconnect(self):
        print("disconnected")


    def exposed_set_trajectory(self, ax, ay):
        self.steering_control.set_trajectory(ax, ay)
        self.steering_control.do_simulation()

    def exposed_control_kart_with_trajectory(self):
        steer_values = self.steering_control.steering_control_history
        velocity_values = self.steering_control.velocity_control_history
        for i in range(len(steer_values)):
            self.kart_control_conn.root.update_velocity(velocity_values[i])
            self.kart_control_conn.root.update_steering(steer_values[i])
            self.kart_control_conn.root.write_serial()
            time.sleep(0.1)
