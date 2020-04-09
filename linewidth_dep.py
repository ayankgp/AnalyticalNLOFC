import pickle
import numpy as np
import matplotlib.pyplot as plt
from functions import render_axis

timeFACTOR = 2.418884e-5

with open("Pickle/pol3max_vector_atomic.pickle", "rb") as f:
    data_atm = pickle.load(f)['pol3max']
with open("Pickle/pol3max_vector_molecular.pickle", "rb") as f:
    data_mol = pickle.load(f)['pol3max']

N = 50
widths = np.logspace(2, -5, N)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 5))
ax.loglog(widths * 1e12, data_atm, 'r*-')
ax.loglog(widths * 1e12, data_mol, 'b*-')
ax.set_xlim(2e14, 5e6)
render_axis(ax, labelSIZE='x-large', gridLINE='')
ax.set_xlabel('Comb linewidth', fontsize='x-large')
# ax.axhline(1., color='k', linestyle='--', linewidth=0.8)
ax.set_ylabel('Max[$P^{(3)} \omega$] \n (in arb. units)', fontsize='x-large')
labels = ['0.1 PHz', '10 THz', '1 THz', '0.1 THz', '10 GHz', '1 GHz', '0.1 GHz', '10 MHz']
ax.set_xticks([1e14, 1e13, 1e12, 1e11, 1e10, 1e9, 1e8, 1e7])
ax.set_xticklabels(labels)
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
for tick in ax.get_yticklabels():
    tick.set_rotation(45)
fig.subplots_adjust(left=0.15, bottom=0.2)

with open("Pickle/pol3M1M2_vector.pickle", "rb") as f:
    dataM1M2 = pickle.load(f)['pol3max']

N = 100
M1axis = np.linspace(2, 9, N, endpoint=True)
figM1, axM1 = plt.subplots(nrows=1, ncols=1)
figM1.suptitle('Comb Spacing ($\Delta \omega$) = ' + "{0:0.2f}".format(11 * timeFACTOR * 0.1 * 1e6) + ' MHz', fontsize='x-large')
dataM1M2 /= dataM1M2.min()
axM1.plot(M1axis * timeFACTOR * 0.1 * 1e6, dataM1M2, 'r*-')
axM1.set_xlim(5., 13.)
axM1.axhline(dataM1M2.min(), color='k', linestyle='--', linewidth=0.8)
axM1.set_xlabel('$\omega_{M1} = \Delta \omega - \omega_{M2}$ (in MHz)', fontsize='x-large')
axM1.set_ylabel('Max[$P^{(3)}(\omega)$] (in rel. units)', fontsize='x-large')
render_axis(axM1)
figM1.subplots_adjust(bottom=0.15)
plt.savefig('ResultPlots/M1M2_dep.png')
plt.show()
