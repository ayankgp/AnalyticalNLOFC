import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

fMAX = 30.
freq = np.linspace(0., fMAX, 10000)
basisNUM = 4
freqN = 3
tau = 0.05
r = np.random.uniform(0.3, 0.7, 4)

ofc_freq = np.asarray([[0.5 + (fMAX / (basisNUM * freqN))*i + (fMAX / basisNUM)*k for i in range(freqN)] for k in range(basisNUM)])

combs = [r[i]*tau / (tau**2 + (freq[:, np.newaxis] - ofc_freq[i][np.newaxis, :])**2) for i in range(basisNUM)]
tau = tau / 1.2
response_b = [(tau + np.abs(np.random.normal(0, 0.002/tau))) / (tau**2 + (freq[:, np.newaxis] - ofc_freq[i][np.newaxis, :])**2) for i in range(basisNUM)]
response_f1 = [(tau + np.abs(np.random.normal(0, 0.002/tau))) / (tau**2 + (freq[:, np.newaxis] - ofc_freq[i][np.newaxis, :] - 0.7)**2) for i in range(basisNUM)]
response_f2 = [(tau + np.abs(np.random.normal(0, 0.002/tau))) / (tau**2 + (freq[:, np.newaxis] - ofc_freq[i][np.newaxis, :] + 0.7)**2) for i in range(basisNUM)]
gaussian = np.exp(-(np.linspace(0, fMAX, basisNUM*freqN) - fMAX/2)**2).reshape(4, 3)
plt.figure()
for i in range(basisNUM):
    plt.plot(freq, combs[i], 'b', linewidth=2., alpha=0.5)
    plt.plot(freq, 0.05*response_b[i], color='k', linewidth=2.)
    plt.plot(freq, 0.025*response_f1[i], color='r', linewidth=2., alpha=0.8)
    plt.plot(freq, 0.025*response_f2[i], color='r', linewidth=2., alpha=0.8)
    plt.hlines(r[i]/tau/1.2, (fMAX / basisNUM)*i, (fMAX / basisNUM)*(i+1) - fMAX / (2*basisNUM*freqN), linestyles='dashed', linewidth=1)
    plt.vlines((fMAX / basisNUM)*i, 0., r[i]/tau/1.2, linestyles='dashed', linewidth=0.5)
    plt.vlines((fMAX / basisNUM)*(i+1) - fMAX / (2*basisNUM*freqN), 0., r[i]/tau/1.2, linestyles='dashed', linewidth=0.5)

plt.xticks([], [])
plt.yticks([], [])
plt.xlabel('Frequency (arb. units)', fontsize='x-large')
plt.ylabel('OFC comb lines \n response \'combs\' (arb. units)', fontsize='x-large')
plt.ylim(-1., 16)
plt.show()