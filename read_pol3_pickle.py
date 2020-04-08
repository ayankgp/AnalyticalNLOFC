import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import pickle
from functions import *

with open("Pickle/pol3DIST5000_2_4lvls.pickle", "rb") as f:
    data = pickle.load(f)
with open("Pickle/polARGS5000_8.pickle", "rb") as fARGS:
    dataARGS = pickle.load(fARGS)
timeFACTOR = 2.418884e-5 / dataARGS['freqDEL']

bNUM = 2
bINDX = int(bNUM/2)
bRNG = np.arange(-bINDX, bINDX)
molNUM = 1
pol3EMPTY = np.zeros((molNUM, data['pol3DIST_EMPTY'].shape[-1]), dtype=np.complex)
pol3FIELD = np.zeros((molNUM, data['pol3DIST_FIELD'].shape[-1]), dtype=np.complex)
con_a = np.zeros(bNUM, dtype=np.complex)
con_b = np.zeros(bNUM, dtype=np.complex)
con_c = np.zeros(bNUM, dtype=np.complex)

min = 0
max = 1
title = ''
for num in range(molNUM):
    pol3EMPTY *= 0.0
    pol3FIELD *= 0.0
    for a in range(bNUM):
        con_a[a] = np.random.uniform(min, max) + 1j*np.random.uniform(min, max)
        title += str("{:2.2f}".format(con_a[a])) + '  '
    title += ' | '
    for b in range(bNUM):
        con_b[b] = np.random.uniform(min, max) + 1j * np.random.uniform(min, max)
        title += str("{:2.2f}".format(con_b[b])) + '  '
    title += ' | '
    for c in range(bNUM):
        con_c[c] = np.random.uniform(min, max) + 1j * np.random.uniform(min, max)
        title += str("{:2.2f}".format(con_c[c])) + '  '

    for I_ in bRNG:
        for J_ in bRNG:
            for K_ in bRNG:
                c = con_a[I_+bINDX] * con_b[J_+bINDX] * con_c[K_+bINDX]
                c = 1
                for i in range(molNUM):
                    pol3EMPTY[i] += data['pol3DIST_EMPTY'][I_][J_][K_][i] * c
                    pol3FIELD[i] += data['pol3DIST_FIELD'][I_][J_][K_][i] * c

    fieldFREQ1 = dataARGS['field1FREQ']
    fieldFREQ2 = dataARGS['field2FREQ']
    field1 = dataARGS['field1']
    field2 = dataARGS['field2']
    freq12 = dataARGS['freq12']
    freq21 = dataARGS['freq21']
    field1 /= field1.max() / np.abs(pol3FIELD).max()
    field2 /= field2.max() / np.abs(pol3FIELD).max()

    print(np.abs(pol3FIELD).max())
    freq = dataARGS['frequency']
    flen = len(freq)
    f1 = int(flen*1)
    f2 = int(flen*1)
    if molNUM > 1:
        fig, ax = plt.subplots(nrows=molNUM, ncols=2, sharex=True, sharey=True, figsize=(11, 7))
        for i in range(molNUM):
            ax[i, 0].plot(freq, pol3EMPTY[i].real/np.abs(pol3EMPTY[i]).max())
            ax[i, 0].plot(freq, pol3FIELD[i].real/np.abs(pol3FIELD[i]).max())
            ax[i, 1].plot(freq, pol3EMPTY[i].imag/np.abs(pol3EMPTY[i]).max())
            ax[i, 1].plot(freq, pol3FIELD[i].imag/np.abs(pol3FIELD[i]).max())
            ax[i, 0].grid()
            ax[i, 1].grid()
            render_axis(ax[i, 0], labelSIZE='x-large')
            render_axis(ax[i, 1], labelSIZE='x-large')
    else:
        fig, ax = plt.subplots(nrows=2, ncols=1, sharey=True, figsize=(11, 7))
        axins0 = inset_axes(ax[0], width="35%", height="30%", loc=9, borderpad=1.7)
        axins1 = inset_axes(ax[1], width="35%", height="30%", loc=9, borderpad=1.7)
        ax[0].set_title("Real Response", fontsize='x-large')
        ax[0].plot(fieldFREQ1 / (timeFACTOR * 2 * np.pi), field1, 'r', alpha=0.25)
        ax[0].plot(fieldFREQ2 / (timeFACTOR * 2 * np.pi), field2, 'b', alpha=0.25)
        ax[0].plot(freq / (timeFACTOR * 2 * np.pi), pol3FIELD[molNUM-1].real, 'r', linewidth=2., label='Real $P^{(3)}(\omega)$ field overlap')
        ax[0].plot(freq / (timeFACTOR * 2 * np.pi), pol3EMPTY[molNUM-1].real, 'b', linewidth=2., label='Real $P^{(3)}(\omega)$ field free')
        axins0.plot(fieldFREQ1 / (timeFACTOR * 2 * np.pi), field1, 'r', alpha=0.25)
        axins0.plot(fieldFREQ2 / (timeFACTOR * 2 * np.pi), field2, 'b', alpha=0.25)
        axins0.plot(freq / (timeFACTOR * 2 * np.pi), pol3FIELD[molNUM-1].real, 'r', linewidth=2., label='Real $P^{(3)}(\omega)$ field overlap')
        axins0.plot(freq / (timeFACTOR * 2 * np.pi), pol3EMPTY[molNUM-1].real, 'b', linewidth=2., label='Real $P^{(3)}(\omega)$ field free')

        ax[1].set_title("Imaginary Response", fontsize='x-large')
        ax[1].plot(fieldFREQ1 / (timeFACTOR * 2 * np.pi), field1, 'r', alpha=0.25)
        ax[1].plot(fieldFREQ2 / (timeFACTOR * 2 * np.pi), field2, 'b', alpha=0.25)
        ax[1].plot(freq / (timeFACTOR * 2 * np.pi), pol3FIELD[molNUM-1].imag, 'r', linewidth=2., label='Imag. $P^{(3)}(\omega)$ field overlap')
        ax[1].plot(freq / (timeFACTOR * 2 * np.pi), pol3EMPTY[molNUM-1].imag, 'b', linewidth=2., label='Imag. $P^{(3)}(\omega)$ field free')
        axins1.plot(fieldFREQ1 / (timeFACTOR * 2 * np.pi), field1, 'r', alpha=0.25)
        axins1.plot(fieldFREQ2 / (timeFACTOR * 2 * np.pi), field2, 'b', alpha=0.25)
        axins1.plot(freq / (timeFACTOR * 2 * np.pi), pol3FIELD[molNUM-1].imag, 'r', linewidth=2., label='Imag. $P^{(3)}(\omega)$ field overlap')
        axins1.plot(freq / (timeFACTOR * 2 * np.pi), pol3EMPTY[molNUM-1].imag, 'b', linewidth=2., label='Imag. $P^{(3)}(\omega)$ field free')
        # ax[0].set_title(title)
        for i in range(2):
            render_axis(ax[i], labelSIZE='xx-large')
            # ax[i].legend(loc=3, prop={'size': 8, 'weight':'normal'})
            ax[i].set_ylabel("$P^{(3)}(\omega)$ \n Normalized units", fontsize='x-large')
            ax[i].set_xlim(-1400, 2000)
            ax[i].set_xlim(-50, 700)
        ax[0].set_xticks([])
        axins0.set_yticks([])
        axins1.set_yticks([])
        axins0.set_xlim(-48, -47.4)
        axins0.set_xlim(491.6, 492.45)
        axins1.set_ylim(1.05 * pol3FIELD.real.min(), 1.05 * pol3FIELD.real.max())
        axins1.set_xlim(-48, -47.4)
        axins1.set_xlim(491.6, 492.45)
        axins1.set_ylim(1.05 * pol3FIELD.imag.min(), 1.05 * pol3FIELD.imag.max())
        render_axis(axins0, labelSIZE='x-large')
        render_axis(axins1, labelSIZE='x-large')

        ax[1].set_xlabel("$\omega$ \n (in THz)", fontsize='x-large')
        fig.subplots_adjust(hspace=0.2, wspace=0.2, bottom=0.15)

plt.show()