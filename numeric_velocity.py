import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from common import print_info, print_error, print_ok, ndarray

def quad_fit_functional(n, dt):
    t_history = np.arange(-n + 1, 1)
    A = ndarray(np.vstack([t_history ** 2, t_history, np.ones(n)]).T)
    A_op = np.linalg.inv(A.T @ A) @ A.T
    def quad_fit(x_history):
        x_history = ndarray(x_history).reshape((-1, n))
        return x_history @ A_op[1] / dt
    return quad_fit

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print_info('Usage: python analyze_log.py [log file path]')
        sys.exit(0)
    log_name = sys.argv[1]
    log_header, log_entries = pickle.load(open(log_name, 'rb'))

    # Comparing different methods of obtaining linear velocity.
    t = log_entries[:, log_header['t']]
    t -= t[0]
    x = log_entries[:, log_header['x']]
    y = log_entries[:, log_header['y']]
    z = log_entries[:, log_header['z']]
    vx = log_entries[:, log_header['x_dot']]
    vy = log_entries[:, log_header['y_dot']]
    vz = log_entries[:, log_header['z_dot']]
    t_diff = t[1:] - t[:-1]
    print_info('t mean:', np.mean(t_diff), 't max:', np.max(t_diff), 't min:', np.min(t_diff))
    assert np.isclose(np.mean(t_diff), 0.01)

    # Plot positions.
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(t, x, color='tab:red', label='x')
    ax.plot(t, y, color='tab:green', label='y')
    ax.plot(t, z, color='tab:blue', label='z')
    ax.set_xlabel('t (s)')
    ax.set_ylabel('position (m)')
    ax.legend()
    ax.grid(True)
    plt.show()

    # Velocity estimation.
    dt = 0.01

    # Sanity check.
    random_quad = np.random.normal(size=3)
    random_x = []
    n = 3
    for i in range(n):
        random_t = i * dt
        random_x.append(random_quad[0] * random_t * random_t + random_quad[1] * random_t + random_quad[2])
    grad = quad_fit_functional(n, dt)(random_x)
    assert np.isclose(grad, 2 * random_quad[0] * (n - 1) * dt + random_quad[1])

    # Now compare the performance with differet n.
    fig = plt.figure()
    ax = fig.add_subplot()

    # Intentionally perturbing x, y, z and see how sensitive the velocity estimation is.
    x_perturb = np.copy(x)
    y_perturb = np.copy(y)
    z_perturb = np.copy(z)
    for i in range(len(t)):
        x_perturb[i] += np.random.uniform(-1e-3, 1e-3)
        y_perturb[i] += np.random.uniform(-1e-3, 1e-3)
        z_perturb[i] += np.random.uniform(-1e-3, 1e-3)

    for n, mk in zip([3, 5, 7], ['o', '+', '*']):
        quad_fit = quad_fit_functional(n, dt)
        vx_n = [0] * (n - 1)
        vy_n = [0] * (n - 1)
        vz_n = [0] * (n - 1)
        vx_n_perturb = [0] * (n - 1)
        vy_n_perturb = [0] * (n - 1)
        vz_n_perturb = [0] * (n - 1)
        for i in range(n - 1, len(t)):
            vx_n.append(quad_fit(x[i - n + 1:i + 1]))
            vy_n.append(quad_fit(y[i - n + 1:i + 1]))
            vz_n.append(quad_fit(z[i - n + 1:i + 1]))
            vx_n_perturb.append(quad_fit(x_perturb[i - n + 1:i + 1]))
            vy_n_perturb.append(quad_fit(y_perturb[i - n + 1:i + 1]))
            vz_n_perturb.append(quad_fit(z_perturb[i - n + 1:i + 1]))
        # Report the relative difference between vx_n and vx_n_perturb.
        vx_diff = np.abs(ndarray(vx_n_perturb) - ndarray(vx_n))
        vy_diff = np.abs(ndarray(vy_n_perturb) - ndarray(vy_n))
        vz_diff = np.abs(ndarray(vz_n_perturb) - ndarray(vz_n))
        vx_mean_diff = np.mean(vx_diff) / np.mean(np.abs(ndarray(vx_n)))
        vy_mean_diff = np.mean(vy_diff) / np.mean(np.abs(ndarray(vy_n)))
        vz_mean_diff = np.mean(vz_diff) / np.mean(np.abs(ndarray(vz_n)))
        print_info('rel err: {}%, {}%, {}%'.format(vx_mean_diff * 100, vy_mean_diff * 100, vz_mean_diff * 100))

        ax.plot(t, vx_n, color='tab:red', marker=mk, markersize=4, label='vx_{}'.format(n))
        ax.plot(t, vy_n, color='tab:green', marker=mk, markersize=4, label='vy_{}'.format(n))
        ax.plot(t, vz_n, color='tab:blue', marker=mk, markersize=4, label='vz_{}'.format(n))
    ax.set_xlabel('t (s)')
    ax.set_ylabel('velocity (m/s)')
    ax.legend()
    ax.grid(True)
    plt.show()
