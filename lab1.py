import numpy as np
from matplotlib import pyplot as plt

a = 25
x0 = 0.5
e = 0.5
x1 = 0
x2 = 4
h0 = 0.002
nx = round((x2 - x1) / h0) + 1  # количество элементов
x_axis = np.linspace(x1, x2, nx)
t0 = 0


def ksi(x):
    return np.abs((x - x0) / e)


def phi1(x):  # начальное условие
    return np.heaviside(1 - ksi(x), 1)


def phi2(x):
    #return phi1(x) * (1 - ksi(x) ** 2)
    return 0


def phi3(x):
    return phi1(x) * np.exp(- ksi(x) ** 2 / np.abs(1 - ksi(x) ** 2))


def phi4(x):
    return phi1(x) * ((np.cos(np.pi * ksi(x) / 2)) ** 3)


def psi(x, t):  # неоднородность
    return 0


def gamma(t):  # граничное условие
    return phi2(t)


def step(u1, x_axis_, h, tau, gamma_, psi_, t_cur):
    u2 = np.zeros(len(x_axis_))
    for i in range(1, len(x_axis_)):
        u2[i] = ((psi_(x_axis_[i], t_cur) - a * (u1[i] - u1[i - 1]) / h) *
                         tau + u1[i])
    u2[0] = gamma_(t_cur)
    return u2


def lax(c_, t1, phi, psi_, gamma_, x_axis_, h):
    t_cur = t0
    tau1 = c_ * h / a
    previous_layer = np.array([phi(i) for i in x_axis_])
    #next_layer = np.zeros(len(x_axis_))
    while t_cur <= t1:
        previous_layer = step(previous_layer, x_axis_, h, tau1, gamma_, psi_, t_cur)
        t_cur += tau1
    return previous_layer



def diff_triangle_right(c_, t1, phi, psi_, gamma_, x_axis_, h):
    tau = c_ * h / a
    nt = round((t1 - t0) / tau) + 1
    t_axis = np.linspace(t0, t1, nt)

    previous_layer = np.array([phi(i) for i in x_axis_])
    next_layer = np.zeros(len(x_axis_))

    for n in range(nt):
        for i in range(1, len(x_axis_)):
            next_layer[i] = ((psi_(x_axis_[i], t_axis[n]) - a * (previous_layer[i] - previous_layer[i - 1]) / h) *
                            tau + previous_layer[i])
        next_layer[0] = gamma_(t_axis[n])
        previous_layer, next_layer = next_layer, previous_layer

    return previous_layer


def exact(t1, phi, gamma_, x_axis_):
    u = np.zeros(len(x_axis_))

    for j in range(len(x_axis_)):
        if x_axis_[j] >= a * t1:
            u[j] = phi(x_axis_[j] - a * t1)
        else:
            u[j] = gamma_(t1 - x_axis_[j] / a)

    return u


def animation_diff(t1):
    #return diff_triangle_right(0.5, t1, phi1, psi, phi2, x_axis, h0)
    return lax(0.5, t1, phi1, psi, phi2, x_axis, h0)


def animation_exact(t1):
    return exact(t1, phi1, phi2, x_axis)


fig, axs = plt.subplots(2, 2)
ax1 = axs[0, 0]
ax2 = axs[1, 0]
ax3 = axs[0, 1]
ax4 = axs[1, 1]
line1 = ax1.plot(x_axis, animation_diff(0.01), color='pink', lw=2)
line2 = ax1.plot(x_axis, animation_exact(0.01), "--", color='black', lw=2)
ax1.set_xlabel('x', rotation=0)
ax1.set_ylabel('u', rotation=0, labelpad=10)
ax1.set_title('Схема "уголок", t = 0.01 с')

line3 = ax2.plot(x_axis, animation_diff(0.02), color='pink', lw=2)
line4 = ax2.plot(x_axis, animation_exact(0.02), "--", color='black', lw=2)
ax2.set_xlabel('x', rotation=0)
ax2.set_ylabel('u', rotation=0, labelpad=10)
ax2.set_title('Схема "уголок", t = 0.02 с')

line5 = ax3.plot(x_axis, animation_diff(0.05), color='pink', lw=2)
line6 = ax3.plot(x_axis, animation_exact(0.05), "--", color='black', lw=2)
ax3.set_xlabel('x', rotation=0)
ax3.set_ylabel('u', rotation=0, labelpad=10)
ax3.set_title('Схема "уголок", t = 0.05 с')

line7 = ax4.plot(x_axis, animation_diff(0.095), color='pink', lw=2)
line8 = ax4.plot(x_axis, animation_exact(0.095), "--", color='black', lw=2)
ax4.set_xlabel('x', rotation=0)
ax4.set_ylabel('u', rotation=0, labelpad=10)
ax4.set_title('Схема "уголок", t = 0.095 с')

plt.show()


#Демонстрация порядка точности на достаточно гладких решениях (phi4)

h_ = [0.05, 0.025, 0.01, 0.0025]
delta_ = []
t = 0.5
for h in h_:
    nx = round((x2 - x1) / h) + 1  # количество элементов
    x_axis = np.linspace(x1, x2, nx)
    u_ = diff_triangle_right(0.5, t, phi4, psi, gamma, x_axis, h)
    u_ex = exact(t, phi4, gamma, x_axis)
    delta_.append(np.max([abs(u_ - u_ex)]))

h_ln = np.log(h_)
delta_ln = np.log(delta_)

fig1, ax1 = plt.subplots()
line = ax1.plot(h_ln, delta_ln, color='crimson', lw=2, marker='.')
ax1.set_xlabel('h', rotation=0)
ax1.set_ylabel('delta', rotation=0, labelpad=10)
ax1.set_title('Зависимость логарифма ошибки от логарифма шага')
fig1.subplots_adjust(left=0.1, bottom=0.1)

plt.grid(True)
plt.show()

h0 = 0.01
nx = round((x2 - x1) / h0) + 1  # количество элементов
x_axis = np.linspace(x1, x2, nx)


def concentration(u, h):
    conc = u.sum() * h
    return conc


print('Концентрация вещества в начале: ', concentration(diff_triangle_right(0.5, 10, phi1, psi, phi2, x_axis, h0), x_axis, h0),
      'Концентрация вещества в t = 50: ', concentration(diff_triangle_right(0.5, 50, phi1, psi, phi2, x_axis, h0), x_axis, h0))
