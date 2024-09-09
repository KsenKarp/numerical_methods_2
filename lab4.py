import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

a = 1
x0 = 3
e = 4
l = 5
h0 = 0.01
nx = round(l / h0) + 1  # количество элементов
x_axis, h0 = np.linspace(0, l, nx, retstep=True)
t0 = 0
c = 0.5
amplitude = 1
t_s = 2
alpha_l = 1
beta_l = 0
alpha_r = 1
beta_r = 0


def ksi(x):
    return np.abs((x - x0) / e)


def phi1(x):  # начальное условие
    return np.heaviside(1 - ksi(x), 1)


def phi2(x):
    return phi1(x) * (1 - ksi(x) ** 2)


def phi3(x):
    return phi1(x) * np.exp(- ksi(x) ** 2 / np.abs(1 - ksi(x) ** 2))


def phi4(x):
    return phi1(x) * ((np.cos(np.pi * ksi(x) / 2)) ** 3)


def f_(x, t):
    return 0


def mu_right(t):
    return 0


def mu_left(t):
    return amplitude * np.heaviside(t_s - t, 0) * np.sin(np.pi * t / t_s) ** 3


def step_1st_order(u1, u2, x_axis_, h, tau, mu_l, mu_r, f, time):
    u3 = np.zeros(len(x_axis_))
    u3[1:-1] = 2 * u2[1:-1] - u1[1:-1] + a**2 * tau**2 / h**2 * (u2[2:] - 2*u2[1:-1] + u2[:-2]) + tau**2 * f(x_axis_, time)
    u3[0] = (mu_l(time) + beta_l/h * u3[1]) / (alpha_l + beta_l/h)
    u3[-1] = (mu_r(time) + beta_r/h * u3[-2]) / (alpha_r + beta_r/h)
    return u3


def wave_1st_order(t1, phi, psi, mu_l, mu_r, f, x_axis_, h):
    t_cur = 0
    tau1 = c * h / a
    u0 = np.array([phi(i) for i in x_axis_])
    u1 = np.array([phi(i) for i in x_axis_]) + tau1 * np.array([psi(i) for i in x_axis_])
    while t_cur <= t1:
        u = step_1st_order(u0, u1, x_axis_, h, tau1, mu_l, mu_r, f, t_cur)
        u0 = u1
        u1 = u
        t_cur += tau1
    return u1


def mu_(t):
    return mu_left(t) if t > 0 else 0


def exact(t1, phi, x_axis_): #для нулевых начальных
    u = np.zeros_like(x_axis_)
    sum = 0
    for j in range(len(x_axis_)):
        for i in range(100):
            sum = sum + mu_(t1 - 2*i*l/a - x_axis_[j]/a) - mu_(t1 - 2*(i+1)*l/a + x_axis_[j]/a)
        u[j] = sum
        sum = 0
    return u


def step_2nd_order(u1, u2, x_axis_, h, tau, mu_l, mu_r, f, time):
    u3 = np.zeros(len(x_axis_))
    u3[1:-1] = 2 * u2[1:-1] - u1[1:-1] + a**2 * tau**2 / h**2 * (u2[2:] - 2*u2[1:-1] + u2[:-2]) + tau**2 * f(x_axis_, time)
    u3[0] = (mu_l(time + tau) + beta_l / h * u3[1]) / (alpha_l + beta_l / h)
    u3[-1] = (mu_r(time + tau) + beta_r / h * u3[-2]) / (alpha_r + beta_r / h)
    return u3


def wave_2nd_order(t1, phi, psi, mu_l, mu_r, f, x_axis_, h):
    t_cur = 0
    tau1 = c * h / a
    phi_xx = np.zeros_like(x_axis_)
    phi_xx[1:-1] = (phi(x_axis_[2:]) - 2 * phi(x_axis_[1:-1]) + phi(x_axis_[:-2])) / h**2
    phi_xx[0] = (phi(x_axis_[1]) - phi(x_axis_[0]) + phi(x_axis_[0] - h)) / h**2
    phi_xx[-1] = (phi(x_axis_[-2]) - phi(x_axis_[-1]) + phi(x_axis_[-1] + h)) / h ** 2
    u0 = np.array([phi(i) for i in x_axis_])
    u1 = (np.array([phi(i) for i in x_axis_]) + tau1 * np.array([psi(i) for i in x_axis_]) +
          (tau1 ** 2) / 2 * (a ** 2 * phi_xx + f(x_axis_, 0)))
    while t_cur <= t1:
        u = step_2nd_order(u0, u1, x_axis_, h, tau1, mu_l, mu_r, f, t_cur)
        u0 = u1
        u1 = u
        t_cur += tau1
    return u1


def animation_diff(t1):
    return wave_1st_order(t1, mu_right, mu_right, mu_left, mu_right, f_, x_axis, h0)


def animation_diff2(t1):
    return wave_2nd_order(t1, mu_right, mu_right, mu_left, mu_right, f_, x_axis, h0)


def animation_exact(t1):
    return exact(t1, phi1, x_axis)


fig, axs = plt.subplots(2, 2)
ax1 = axs[0, 0]
ax2 = axs[1, 0]
ax3 = axs[0, 1]
ax4 = axs[1, 1]
line1 = ax1.plot(x_axis, animation_diff(1), color='pink', lw=2)
line2 = ax1.plot(x_axis, animation_diff2(1), "--", color='plum', lw=2)
line21 = ax1.plot(x_axis, animation_exact(1), "-.", color='black', lw=2)
ax1.set_xlabel('x', rotation=0)
ax1.set_ylabel('u', rotation=0, labelpad=10)
ax1.set_title('Схема "with corrections", t = 2 с')

line3 = ax2.plot(x_axis, animation_diff(10), color='pink', lw=2)
line4 = ax2.plot(x_axis, animation_diff2(10), "--", color='plum', lw=2)
line23 = ax2.plot(x_axis, animation_exact(10), "-.", color='black', lw=2)
ax2.set_xlabel('x', rotation=0)
ax2.set_ylabel('u', rotation=0, labelpad=10)
ax2.set_title('Схема "with corrections", t = 10 с')

line5 = ax3.plot(x_axis, animation_diff(12), color='pink', lw=2)
line6 = ax3.plot(x_axis, animation_diff2(12), "--", color='plum', lw=2)
line24 = ax3.plot(x_axis, animation_exact(12), "-.", color='black', lw=2)
ax3.set_xlabel('x', rotation=0)
ax3.set_ylabel('u', rotation=0, labelpad=10)
ax3.set_title('Схема "with corrections", t = 12 с')

line7 = ax4.plot(x_axis, animation_diff(25), color='pink', lw=2)
line8 = ax4.plot(x_axis, animation_diff2(25), "--", color='plum', lw=2)
line26 = ax4.plot(x_axis, animation_exact(25), "-.", color='black', lw=2)
ax4.set_xlabel('x', rotation=0)
ax4.set_ylabel('u', rotation=0, labelpad=10)
ax4.set_title('Схема "with corrections", t = 25 с')

plt.show()

fig, ax = plt.subplots(figsize=(14, 3))
plt.axis([0, l, -1.5, 1.5])
plt.grid()

numerical_line, = ax.plot(x_axis, [0] * x_axis, color='lightskyblue', lw=1.5, mec='r', mew=2, ms=12,  label=f'Numerical 1st order')
numerical_line2, = ax.plot(x_axis, [0] * x_axis, ls='--', c='coral', lw=1.5, mec='r', mew=2, ms=12,  label=f'Numerical 2nd order')
exact_line, = ax.plot(x_axis, exact(0, mu_right, x_axis), ls='-.', color='black', lw=1.5, mec='r', mew=2, ms=12, label='Exact' )
ax.legend()


tau1 = c * h0 / a
u_0 = np.array([0 for i in x_axis])
u_1 = np.array([0 for i in x_axis]) + tau1 * np.array([0 for i in x_axis])

u_00 = np.array([0 for i in x_axis])
u_11 = np.array([0 for i in x_axis]) + tau1 * np.array([0 for i in x_axis])

def update(frame):

    global u_0, u_1, u_00, u_11
    u = step_1st_order(u_0.copy(), u_1.copy(), x_axis, h0, tau1, mu_left, mu_right, f_, frame * tau1)
    u_0 = u_1
    u_1 = u
    u_ = step_2nd_order(u_00.copy(), u_11.copy(), x_axis, h0, tau1, mu_left, mu_right, f_, frame * tau1)
    u_00 = u_11
    u_11 = u_
    exact_line.set_ydata(exact(frame * tau1, mu_right, x_axis))
    numerical_line.set_ydata(u_1)
    numerical_line2.set_ydata(u_11)
    ax.set_title(f'Схема, C = {c}, Time: {frame * tau1:.2f}')


# Запуск анимации
#u_ = mu_right(x_axis)
animation = FuncAnimation(fig, update, frames=4000, interval=0.1, repeat=False)
animation.save("anim_reflection_fixed.mp4", writer="ffmpeg", fps=60)
