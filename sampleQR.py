import numpy as np
import matplotlib.pyplot as plt

Nfreq = 5
Nmol = 3

P = np.empty((Nfreq, Nmol))
Q = np.empty((Nmol, Nfreq, Nfreq))
QN = np.empty((Nmol, Nfreq, Nfreq-Nmol+1))
H = np.empty((Nmol, Nfreq))
R = np.empty((Nmol, Nfreq, Nmol-1))
Z = np.empty((Nmol, Nmol))
Y = np.empty((Nmol, Nmol))

x = np.linspace(0, np.pi, Nfreq)
env = np.exp(-(x - np.pi/2)**2 / (2*.6**2))
env1 = np.exp(-(x - np.pi/2)**2 / (2*1**2))

for i in range(Nmol):
    # P[:, i] = (np.sin((2.0 + .25*i)*x) + np.cos(1 + 2*i))*env
    P[:, i] = np.exp(- (x - (np.pi/2 + i*np.pi/16))**2 / (2*.5**2))

for i in range(Nmol):
    Q[i], R[i] = np.linalg.qr(np.delete(P, i, 1), mode='complete')
    QN[i] = np.asarray([q * np.dot(q, env1) for q in Q[i, :, Nmol - 1:].T]).T
    # H[i] = sum(q * np.dot(q, env1) for q in Q[i, :, Nmol - 1:].T) + np.random.uniform(-0.05, 0.05, (1000,))
    H[i] = sum(q * np.dot(q, env1) for q in Q[i, :, Nmol - 1:].T)

fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True)
fig1, ax1 = plt.subplots(nrows=3, ncols=2, sharex=True)
for i in range(Nmol):
    for j in range(Nmol):
        ax1[i, 0].plot(x, P[:, i], 'r')
        ax1[i, 0].plot(x, H[i], 'b')
        ax1[i, 1].plot(x, x*0, 'k-')
        if i==j:
            ax1[i, 1].plot(x, H[i]*P[:, j], 'k')
        else:
            ax1[i, 1].plot(x, H[i]*P[:, j], 'g', alpha=0.5)
    print((H[i]*P[:, 0]).sum(), (H[i]*P[:, 1]).sum(), (H[i]*P[:, 2]).sum())


for i in range(2):
    ax[0, i+1].get_shared_y_axes().join(ax[0, 0], ax[0, i+1])
    ax[1, i+1].get_shared_y_axes().join(ax[1, 0], ax[1, i+1])
for i in range(Nmol):
    ax[0, i].plot(x, P[:, i])
    ax[1, i].plot(x, QN[i])
    ax[1, i].plot(x, H[i])
    ax[0, i].grid()
    ax[1, i].grid()


for i in range(Nmol):
    for j in range(Nmol):
        Z[i, j] = H[i].dot(P[:, j])
        Y[i, j] = P[:, i].dot(P[:, j])

print(Z, np.linalg.det(Z))
print(Y, np.linalg.det(Y))

plt.show()