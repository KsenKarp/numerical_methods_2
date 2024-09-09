import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

a = np.sqrt(0.5)
l = 1
h0 = 0.001
nx = round(l / h0) + 1  # количество элементов
x_axis, h0 = np.linspace(0, l, nx, retstep=True)
t0 = 0.0
c = 0.4
alpha_l = 1
beta_l = 1
alpha_r = 1
beta_r = 0


def phi(x):  # начальное условие
    return 2 * np.sin(x)


def psi(x):  # начальное условие
    return 2 * np.cos(x)


def f_(x, t): # неоднородность
    return -np.sin(x + t)


def mu_right(t):
    return 2 * np.sin(1 + t)


def mu_left(t):
    return 2 * (np.sin(t) - np.cos(t))


def step_1st_order(u1, u2, x_axis_, h, tau, mu_l, mu_r, f, time):
    u3 = np.zeros(len(x_axis_))
    u3[1:-1] = (2 * u2[1:-1] - u1[1:-1] + a**2 * tau**2 / h**2 * (u2[2:] - 2*u2[1:-1] + u2[:-2]) + tau**2 *
                f(x_axis_[1:-1], time))
    u3[0] = (mu_l(time + tau) + beta_l/h * u3[1]) / (alpha_l + beta_l/h)
    u3[-1] = (mu_r(time + tau) + beta_r/h * u3[-2]) / (alpha_r + beta_r/h)
    return u3


def wave_1st_order(t1, phi, psi, mu_l, mu_r, f, x_axis_, h):
    t_cur = 0
    tau1 = c * h / a
    u0 = phi(x_axis_)
    u1 = phi(x_axis_) + tau1 * psi(x_axis_)
    t_cur = t_cur + tau1
    while t_cur <= t1:
        u = step_1st_order(u0, u1, x_axis_, h, tau1, mu_l, mu_r, f, t_cur)
        u0 = u1
        u1 = u
        t_cur += tau1
    return u1


def exact(t1, x_axis_):
    return 2 * np.sin(x_axis_ + t1)


def step_2nd_order(u1, u2, x_axis_, h, tau, mu_l, mu_r, f, time):
    u3 = np.zeros(len(x_axis_))
    u3[1:-1] = (2 * u2[1:-1] - u1[1:-1] + a ** 2 * tau ** 2 / h ** 2 * (u2[2:] - 2 * u2[1:-1] + u2[:-2]) + tau ** 2 *
                f(x_axis_[1:-1], time))
    u3[0] = (mu_l(time + tau) + beta_l / (2 * h) * (-u3[2] + 4 * u3[1])) / (alpha_l + 3 * beta_l / (2 * h))
    u3[-1] = (mu_r(time + tau) - beta_r / (2 * h) * (-4 * u3[-2] + u3[-3])) / (alpha_r + 3 * beta_r / (2 * h))
    return u3


def wave_2nd_order(t1, phi, psi, mu_l, mu_r, f, x_axis_, h):
    t_cur = 0
    tau1 = c * h / a
    u0 = phi(x_axis_)
    u1 = phi(x_axis_) + tau1 * psi(x_axis_) + (tau1 ** 2) / 2 * (a ** 2 * (-phi(x_axis_)) + f(x_axis_, 0))
    t_cur = t_cur + tau1
    while t_cur <= t1:
        u = step_2nd_order(u0, u1, x_axis_, h, tau1, mu_l, mu_r, f, t_cur)
        u0 = u1
        u1 = u
        t_cur += tau1
    return u1

tt = 56
u1_h1 = wave_1st_order(tt, phi, psi, mu_left, mu_right, f_, x_axis, h0)
u2_h1 = wave_2nd_order(tt, phi, psi, mu_left, mu_right, f_, x_axis, h0)

h1 = 0.01
nx1 = round(l / h1) + 1
x_axis1, h1 = np.linspace(0, l, nx1, retstep=True)
u1_h2 = wave_1st_order(tt, phi, psi, mu_left, mu_right, f_, x_axis1, h1)
u2_h2 = wave_2nd_order(tt, phi, psi, mu_left, mu_right, f_, x_axis1, h1)

u_ex_h1 = exact(tt, x_axis)
u_ex_h2 = exact(tt, x_axis1)

err1 = max(abs(u_ex_h1 - u1_h1)) #h = 0.001
err11 = max(abs(u_ex_h2 - u1_h2)) #h = 0.01
print(err11/err1, err11, err1)

err2 = max(abs(u_ex_h1 - u2_h1)) #h = 0.001
err21 = max(abs(u_ex_h2 - u2_h2)) #h = 0.01
print(err21/err2, err21, err2)


def animation_diff(t1):
    return wave_1st_order(t1, phi, psi, mu_left, mu_right, f_, x_axis, h0)


def animation_diff2(t1):
    return wave_2nd_order(t1, phi, psi, mu_left, mu_right, f_, x_axis, h0)


def animation_exact(t1):
    return exact(t1, x_axis)


fig, axs = plt.subplots(2, 2)
ax1 = axs[0, 0]
ax2 = axs[1, 0]
ax3 = axs[0, 1]
ax4 = axs[1, 1]
line1 = ax1.plot(x_axis, animation_diff(1), color='pink', lw=2, label='1st order')
line2 = ax1.plot(x_axis, animation_diff2(1), "--", color='plum', lw=2, label='2nd order')
line21 = ax1.plot(x_axis, animation_exact(1), "-.", color='black', lw=2)
ax1.legend()
ax1.set_xlabel('x', rotation=0)
ax1.set_ylabel('u', rotation=0, labelpad=10)
ax1.set_title('Схема "with corrections", t = 1 с')

line3 = ax2.plot(x_axis, animation_diff(10), color='pink', lw=2, label='1st order')
line4 = ax2.plot(x_axis, animation_diff2(10), "--", color='plum', lw=2, label='2nd order')
line23 = ax2.plot(x_axis, animation_exact(10), "-.", color='black', lw=2)
ax2.legend()
ax2.set_xlabel('x', rotation=0)
ax2.set_ylabel('u', rotation=0, labelpad=10)
ax2.set_title('Схема "with corrections", t = 10 с')

line5 = ax3.plot(x_axis, animation_diff(20), color='pink', lw=2, label='1st order')
line6 = ax3.plot(x_axis, animation_diff2(20), "--", color='plum', lw=2, label='2nd order')
line24 = ax3.plot(x_axis, animation_exact(20), "-.", color='black', lw=2)
ax3.legend()
ax3.set_xlabel('x', rotation=0)
ax3.set_ylabel('u', rotation=0, labelpad=10)
ax3.set_title('Схема "with corrections", t = 20 с')

line7 = ax4.plot(x_axis, animation_diff(tt), color='pink', lw=2, label='1st order')
line8 = ax4.plot(x_axis, animation_diff2(tt), "--", color='plum', lw=2, label='2nd order')
line26 = ax4.plot(x_axis, animation_exact(tt), "-.", color='black', lw=2)
ax4.legend()
ax4.set_xlabel('x', rotation=0)
ax4.set_ylabel('u', rotation=0, labelpad=10)
ax4.set_title('Схема "with corrections", t = 57 с')

plt.show()

fig, ax = plt.subplots(figsize=(14, 3))
plt.axis([0, l, -3, 3])
plt.grid()

numerical_line, = ax.plot(x_axis, [0] * x_axis, color='lightskyblue', lw=1.5, mec='r', mew=2, ms=12,  label=f'Numerical 1st order')
numerical_line2, = ax.plot(x_axis, [0] * x_axis, ls='--', c='coral', lw=1.5, mec='r', mew=2, ms=12,  label=f'Numerical 2nd order')
exact_line, = ax.plot(x_axis, exact(0, x_axis), ls='-.', color='darkblue', lw=1.5, mec='r', mew=2, ms=12, label='Exact')
ax.legend()


tau1 = 0.05
u_0 = phi(x_axis)
u_1 = phi(x_axis) + tau1 * psi(x_axis)

u_00 = phi(x_axis)
u_11 = phi(x_axis) + tau1 * psi(x_axis) + tau1**2 / 2 * (-phi(x_axis) * a**2 + f_(x_axis, 0))

def update(frame):

    global u_0, u_1, u_00, u_11
    u = animation_exact(frame * tau1)
    u_ = animation_diff2(frame * tau1)
    exact_line.set_ydata(exact(frame * tau1, x_axis))
    numerical_line.set_ydata(u)
    numerical_line2.set_ydata(u_)
    ax.set_title(f'Схема, C = {c}, Time: {frame * tau1:.2f}')


# Запуск анимации
animation = FuncAnimation(fig, update, frames=200, interval=0.1, repeat=False)
animation.save("anim1.mp4", writer="ffmpeg", fps=60)
