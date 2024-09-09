import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

a = 1
l = 1
h0 = 0.01
nx = round(l / h0) + 1  # количество элементов
x_axis, h0 = np.linspace(0, l, nx, retstep=True)
c = 0.6
alpha_l = 1
beta_l = -1
alpha_r = 0
beta_r = 1


def phi(x):  # начальное условие
    return x


def f_(x, t): # неоднородность
    return (x + 2 * t**2 * np.tanh(x * t)) / (np.cosh(x * t) ** 2)


def mu_right(t):
    return 1 + t / (np.cosh(t) ** 2)


def mu_left(t):
    return 1 + t


def exact(x, t):
    return x + np.tanh(x * t)


def step_1st_order(u1, x_axis_, h, tau, mu_l, mu_r, f, time):
    u2 = np.zeros(len(x_axis_))
    u2[1:-1] = u1[1:-1] + a * tau / h**2 * (u1[2:] - 2*u1[1:-1] + u1[:-2]) + tau * f(x_axis_[1:-1], time + tau/2)
    u2[0] = (mu_l(time + tau) + beta_l/h * u2[1]) / (alpha_l + beta_l/h)
    u2[-1] = (mu_r(time + tau) + beta_r/h * u2[-2]) / (alpha_r + beta_r/h)
    return u2


def wave_1st_order(t1, phi, mu_l, mu_r, f, x_axis_, h):
    t_cur = 0
    tau1 = c * h ** 2 / (2*a)
    u0 = phi(x_axis_)
    while t_cur <= t1:
        t_cur += tau1
        u = step_1st_order(u0, x_axis_, h, tau1, mu_l, mu_r, f, t_cur)
        u0 = u
    return u0


err_h0 = max(abs(wave_1st_order(1, phi, mu_left, mu_right, f_, x_axis, h0) - exact(x_axis, 1)))
h1 = 0.1
nx1 = round(l / h1) + 1
x_axis1, h1 = np.linspace(0, l, nx1, retstep=True)
err_h1 = max(abs(wave_1st_order(1, phi, mu_left, mu_right, f_, x_axis1, h1) - exact(x_axis1, 1)))

print('Отношение максимальной ошибки для h1 = 0.1 к максимальной ошибке для h0 = 0.01:\n', err_h1 / err_h0)


sigma = 0.5


def C0_(h):
    return 3 * beta_l / (2 * h * alpha_l - beta_l)


def B_(h, tau, x_axis_):
    B = np.zeros(len(x_axis_))
    B[0] = -4 * beta_l / (2 * h * alpha_l - beta_l)
    B[1] = tau * sigma * (C0_(h) + 1) / (tau * sigma * (2 - B[0]) + h ** 2)
    for j in range(2, len(x_axis_) - 1):
        B[j] = tau * sigma / (tau * sigma * (2 - B[j - 1]) + h ** 2)
    return B


def A_(h, tau, x, t, idx, u):
    A = np.zeros(len(x))
    B = B_(h, tau, x)
    A[0] = 2 * h * mu_left(t[int(idx)]) / (2 * h * alpha_l - beta_l)
    for j in range(1, len(x) - 1):
        A[j] = (tau * h ** 2 / (tau * sigma * (2 - B[j - 1]) + h ** 2)) * (
                    f_(x[j], t[int(idx)] - tau / 2) + (1 - sigma) * (u[j + 1] - 2 * u[j] + u[j - 1])
                    / h ** 2 + u[j] / tau + A[j - 1] * sigma / h ** 2)
    A[len(x) - 1] = (h * mu_right(t[int(idx)]) + 2 * beta_r * A[len(x) - 2] - 0.5 * beta_r * (
                                A[len(x) - 3] + B[len(x) - 3] * A[len(x) - 2])) / (
                                h * alpha_r + 1.5 * beta_r - 2 * beta_r * B[len(x) - 2] + 0.5 * beta_r * B[len(x) - 2]
                                * B[len(x) - 3])
    return A


def Krank_Nicholson(h, tau, x, t, idx_t):
    u_tmp = np.zeros(len(x))
    u_current = phi(x)
    for n in range(0, int(idx_t)):
        A = A_(h, tau, x, t, n + 1, u_current)
        B = B_(h, tau, x)
        u_tmp[len(x) - 1] = A[len(x) - 1]
        for j in range(len(x) - 2, 0, -1):
            u_tmp[j] = A[j] + B[j] * u_tmp[j + 1]
        u_tmp[0] = A[0] + B[0] * u_tmp[1] + C0_(h) * u_tmp[2]
        u_current = u_tmp
    return u_current


def animation_diff(t1):
    return wave_1st_order(t1, phi, mu_left, mu_right, f_, x_axis, h0)


t, tau = np.linspace(0, 1, 100, retstep=True)


def animation_diff2(t1):
    return Krank_Nicholson(h0, tau, x_axis, t, t1/tau)


def animation_exact(t1):
    return exact(x_axis, t1)


err_h0 = max(abs(Krank_Nicholson(h0, tau, x_axis, t, 1/tau) - exact(x_axis, 1)))
h1 = 0.001
nx1 = round(l / h1) + 1
x_axis1, h1 = np.linspace(0, l, nx1, retstep=True)
t1, tau1 = np.linspace(0, 1, nx1, retstep=True)
err_h1 = max(abs(Krank_Nicholson(h1, tau1, x_axis1, t1, 1/tau1) - exact(x_axis1, 1)))

print()
print('Отношение максимальной ошибки для h0 = 0.01 к максимальной ошибке для h1 = 0.001:\n', err_h0 / err_h1)

fig, ax = plt.subplots(figsize=(14, 5))
plt.axis([0, l, 0, 2.5])
plt.grid()

numerical_line, = ax.plot(x_axis, [0] * x_axis, color='lightskyblue', lw=1.5, mec='r', mew=2, ms=12,  label=f'Numerical 1st order')
numerical_line2, = ax.plot(x_axis, [0] * x_axis, ls='--', c='coral', lw=1.5, mec='r', mew=2, ms=12,  label=f'Numerical 2nd order')
exact_line, = ax.plot(x_axis, exact(0, x_axis), ls='-.', color='darkblue', lw=1.5, mec='r', mew=2, ms=12, label='Exact')
ax.legend()


dt = 0.005

def update(frame):

    u = animation_diff(frame * dt)
    u_ = animation_diff2(frame * dt)
    exact_line.set_ydata(exact(x_axis, frame * dt))
    numerical_line.set_ydata(u)
    numerical_line2.set_ydata(u_)
    ax.set_title(f'Схема, C = {c}, Time: {frame * dt:.2f}')


# Запуск анимации
animation = FuncAnimation(fig, update, frames=200, interval=1, repeat=True)
animation.save("anim5.gif", fps=60)

