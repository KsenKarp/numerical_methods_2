import numpy as np
from matplotlib import pyplot as plt

a0 = 25
x0 = 0.5
e = 0.5
l = 4
h0 = 0.02
nx = round(l / h0) + 1  # количество элементов
x_axis, h0 = np.linspace(0, l, nx, retstep=True)
t0 = 0
c = 0.5


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


def a(x):     # задаём скорость
    #return a0 * np.tanh(x / l)
    return a0


def a_der(x):
    #return a0 / (l * (np.cosh(x / l)) ** 2)
    return 0


#В данном случае тау можно передать, в более общем - считаем каждый раз по новой и возвращаем тау из "шага"
def step(u1, x_axis_, h, tau):
    u2 = np.zeros(len(x_axis_))
    u2[1:-1] = u1[1:-1] - a_der(x_axis_[1:-1]) * tau * u1[1:-1] - tau * a(x_axis_[1:-1]) / h * (0.5 * (u1[2:] - u1[:-2]) - tau / (2 * h) *
    (a(x_axis_[1:-1] + h / 2) * (u1[2:] - u1[1:-1]) - a(x_axis_[1:-1] - h / 2) * (u1[1:-1] - u1[:-2])
    + a_der(x_axis_[1:-1] + h / 2) * 0.5 * h * (u1[1:-1] + u1[2:]) - a_der(x_axis_[1:-1] - h / 2) * 0.5 * h * (u1[1:-1] + u1[:-2])))

    u2[0] = u1[0] - tau * a(x_axis_[0]) / h * (u1[1] - u1[0]) + tau * a_der(x_axis_[0]) * u1[0]
    u2[-1] = u1[-1] - tau * a(x_axis_[-1]) / h * (u1[-1] - u1[-2]) + tau * a_der(x_axis_[-1]) * u1[-1]
    return u2


def lax(t1, phi, x_axis_, h):
    t_cur = t0
    tau1 = c * h / a(l)
    previous_layer = np.array([phi(i) for i in x_axis_])
    while t_cur <= t1:
        previous_layer = step(previous_layer, x_axis_, h, tau1)
        t_cur += tau1
    return previous_layer


def exact(t1, phi, x_axis_):
    u = (np.cosh(x_axis_ / l) / np.sqrt(np.sinh(x_axis_ / l)**2 + np.exp(2 * a0 * t1 / l)) *
         phi(l * np.arcsinh(np.sinh(x_axis_ / l) * np.exp(-a0 * t1 / l))))
    return u


def animation_diff(t1):
    return lax(t1, phi1, x_axis, h0)


def animation_exact(t1):
    return exact(t1, phi1, x_axis)


fig, axs = plt.subplots(2, 2)
ax1 = axs[0, 0]
ax2 = axs[1, 0]
ax3 = axs[0, 1]
ax4 = axs[1, 1]
line1 = ax1.plot(x_axis, animation_diff(0.01), color='pink', lw=2)
#line2 = ax1.plot(x_axis, animation_exact(0.01), "--", color='black', lw=2)
ax1.set_xlabel('x', rotation=0)
ax1.set_ylabel('u', rotation=0, labelpad=10)
ax1.set_title('Схема "lax", t = 0.01 с')

line3 = ax2.plot(x_axis, animation_diff(0.02), color='pink', lw=2)
#line4 = ax2.plot(x_axis, animation_exact(0.02), "--", color='black', lw=2)
ax2.set_xlabel('x', rotation=0)
ax2.set_ylabel('u', rotation=0, labelpad=10)
ax2.set_title('Схема "lax", t = 0.02 с')

line5 = ax3.plot(x_axis, animation_diff(0.05), color='pink', lw=2)
#line6 = ax3.plot(x_axis, animation_exact(0.05), "--", color='black', lw=2)
ax3.set_xlabel('x', rotation=0)
ax3.set_ylabel('u', rotation=0, labelpad=10)
ax3.set_title('Схема "lax", t = 0.05 с')

line7 = ax4.plot(x_axis, animation_diff(0.095), color='pink', lw=2)
#line8 = ax4.plot(x_axis, animation_exact(0.095), "--", color='black', lw=2)
ax4.set_xlabel('x', rotation=0)
ax4.set_ylabel('u', rotation=0, labelpad=10)
ax4.set_title('Схема "lax", t = 0.095 с')

plt.show()
