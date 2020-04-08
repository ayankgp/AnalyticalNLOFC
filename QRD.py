#!/usr/bin/env python

"""
CalculateSpectra.py:

Class containing C calls for spectra calculation and discriminating OFC-pulse generation.
Plots results obtained from C calls.
"""

__author__ = "Ayan Chattopadhyay"
__affiliation__ = "Princeton University"


# ---------------------------------------------------------------------------- #
#                      LOADING PYTHON LIBRARIES AND FILES                      #
# ---------------------------------------------------------------------------- #

from types import MethodType, FunctionType
import numpy as np
from FP_QRdiscrimination import ADict
import matplotlib.cm as cm


class QRD():
    """
    Calculates the QR decomposition to calculate orthogonal heterodyne fields in OFC experiment
    """

    def __init__(self, params, **kwargs):
        """
        __init__ function call to initialize variables from the keyword args for the class instance
         provided in __main__ and add new variables for use in other functions in this class, with
         data from SystemVars.
         :type SystemVars: object
        """

        for name, value in kwargs.items():
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self))
            else:
                setattr(self, name, value)

        self.molNUM = params.molNUM
        self.combNUM = params.combNUM
        self.freqNUM = params.freqNUM
        self.resolutionNUM = params.resolutionNUM
        self.omegaM1 = params.omegaM1
        self.omegaM2 = params.omegaM2
        self.combGAMMA = params.combGAMMA
        self.freqDEL = params.freqDEL
        self.termsNUM = params.termsNUM
        self.frequency = params.frequency
        self.field1FREQ = params.field1FREQ
        self.field2FREQ = params.field2FREQ
        self.field1 = params.field1 / params.field1.max()
        self.field2 = params.field2 / params.field2.max()
        self.round = params.round
        self.basisNUM_CB = params.basisNUM_CB
        self.basisNUM_FB = params.basisNUM_FB
        self.basisENVwidth_CB = params.basisENVwidth_CB
        self.basisENVwidth_FB = params.basisENVwidth_FB
        self.pol3_EMPTY.real /= np.abs(self.pol3_EMPTY.real).max()
        self.pol3_EMPTY.imag /= np.abs(self.pol3_EMPTY.imag).max()
        self.pol3basisMATRIX = np.empty((self.basisNUM_FB, self.freqNUM))
        self.pol3combMATRIX = np.empty((self.basisNUM_CB, self.combNUM))
        self.freq2combMATRIX = np.empty((self.freqNUM, self.combNUM))
        self.freq2basisMATRIX = np.zeros((self.basisNUM_FB, self.freqNUM))
        self.comb2basisMATRIX = np.zeros((self.combNUM, self.basisNUM_CB))

    def freq2basis_func(self):
        arrayCOMB_FB = np.arange(self.freqNUM)
        arrayBASIS_FB = np.linspace(0, self.combNUM, self.basisNUM_FB * 2 + 1, endpoint=True)[1:-1:2] * self.resolutionNUM * 2

        for i in range(self.freqNUM):
            for j in range(self.basisNUM_FB):
                if np.abs(arrayCOMB_FB[i] - arrayBASIS_FB[j]) <= self.basisENVwidth_FB:
                    self.freq2basisMATRIX[j, i] = (self.basisENVwidth_FB ** self.round - (arrayCOMB_FB[i] - arrayBASIS_FB[j])**self.round) ** (1./self.round)


    def basis_transform(self):
        # ------------------------------------------------------------------------------------------------------------ #
        #             BASIS TRANSFORMATION MATRICES: F->frequency C->comb B->basis (any newly devised basis            #
        # ------------------------------------------------------------------------------------------------------------ #

        arrayFREQ_FC = self.frequency[:, np.newaxis] * self.freqDEL
        arrayCOMB_FC = (self.freqDEL * np.arange(self.combNUM))[np.newaxis, :]

        arrayCOMB_CB = np.arange(self.combNUM)[:, np.newaxis]
        arrayBASIS_CB = np.linspace(0, self.combNUM, self.basisNUM_CB + 6, endpoint=True)[3:-3][np.newaxis, :]

        arrayCOMB_FB = np.arange((self.combNUM + 1) * self.resolutionNUM * 2)[:, np.newaxis]
        arrayBASIS_FB = np.linspace(0, self.combNUM, self.basisNUM_FB + 2, endpoint=True)[1:-1][np.newaxis, :] * self.resolutionNUM * 2

        self.freq2combMATRIX = self.combGAMMA / ((arrayFREQ_FC - 2 * self.omegaM2 + self.omegaM1 - arrayCOMB_FC) ** 2 + self.combGAMMA ** 2) + \
                    self.combGAMMA / ((arrayFREQ_FC - 2 * self.omegaM1 + self.omegaM2 - arrayCOMB_FC) ** 2 + self.combGAMMA ** 2)

        self.freq2basis_func()
        # plt.figure()
        # plt.plot(self.freq2basisMATRIX.T)
        # plt.plot(self.freq2basisMATRIX.T.sum(axis=1))

        for i in range(self.basisNUM_FB):
            self.freq2basisMATRIX[i] *= self.freq2combMATRIX.sum(axis=1)

        self.pol3combMATRIX = self.comb2basisMATRIX.T.dot(self.freq2combMATRIX.T.dot(self.pol3_EMPTY.T))
        self.pol3combMATRIX /= np.abs(self.pol3combMATRIX).max()
        self.pol3basisMATRIX = self.freq2combMATRIX.T.dot(self.pol3_EMPTY.T)
        self.pol3basisMATRIX /= np.abs(self.pol3basisMATRIX).max()

        self.pol3basisMATRIX = self.freq2basisMATRIX.dot(self.pol3_EMPTY.T)
        self.pol3basisMATRIX /= np.abs(self.pol3basisMATRIX).max()

        fig1, axes1 = plt.subplots(nrows=3, ncols=2, sharex=True)
        basis_axis_CB = np.linspace(0, self.basisNUM_CB, self.basisNUM_CB)*self.combNUM/self.basisNUM_CB
        basis_axis_FB = np.linspace(0, self.basisNUM_FB, self.basisNUM_FB)*self.combNUM/self.basisNUM_FB
        for molINDX in range(self.molNUM):
            axes1[molINDX, 0].plot(self.field1FREQ, self.field1.real, 'g', alpha=0.5)
            axes1[molINDX, 0].plot(self.field2FREQ, self.field2.real, 'y', alpha=0.5)
            axes1[molINDX, 0].plot(self.frequency, self.pol3_EMPTY[molINDX].real, 'r')
            axes1[molINDX, 1].plot(self.field1FREQ, self.field1.real, 'g', alpha=0.5)
            axes1[molINDX, 1].plot(self.field2FREQ, self.field2.real, 'y', alpha=0.5)
            axes1[molINDX, 1].plot(self.frequency, self.pol3_EMPTY[molINDX].imag, 'r')
            axes1[molINDX, 0].plot(basis_axis_CB, self.pol3combMATRIX.T[molINDX].real, 'k')
            axes1[molINDX, 0].plot(basis_axis_FB, self.pol3basisMATRIX.T[molINDX].real, 'b', linewidth=2.)
            axes1[molINDX, 1].plot(basis_axis_CB, self.pol3combMATRIX.T[molINDX].imag, 'k')
            axes1[molINDX, 1].plot(basis_axis_FB, self.pol3basisMATRIX.T[molINDX].imag, 'b', linewidth=2.)

        return

    def calculate_heterodyne(self):

        np.set_printoptions(precision=3, suppress=True)
        Q_mat = np.empty((self.molNUM, self.basisNUM_FB, self.basisNUM_FB), dtype=np.complex)
        heterodyne = np.empty((self.molNUM, self.basisNUM_FB), dtype=np.complex)
        R_mat = np.empty((self.molNUM, self.basisNUM_FB, self.molNUM-1), dtype=np.complex)
        ImatBASIS = np.empty((self.molNUM, self.molNUM), dtype=np.complex)
        ImatFREQ = np.empty((self.molNUM, self.molNUM), dtype=np.complex)

        ncols = 4
        basisAXIS = np.arange(self.basisNUM_FB)
        envelopeBASIS = (np.exp(-(basisAXIS - 0.5 * self.basisNUM_FB)**2 / (2 * (self.basisNUM_FB/20) ** 2)))
        # envelopeBASIS *= np.sin(2*np.pi*basisAXIS/max(basisAXIS))
        print(envelopeBASIS)

        fig1, axes1 = plt.subplots(nrows=3, ncols=ncols)
        for molINDX in range(self.molNUM):
            for colINDX in range(ncols):
                axes1[molINDX, colINDX].plot(self.field1FREQ, self.field1.real, 'g', alpha=0.5)
                axes1[molINDX, colINDX].plot(self.field2FREQ, self.field2.real, 'y', alpha=0.5)
            axes1[molINDX, 0].plot(self.frequency, self.pol3_EMPTY[molINDX].real)
            axes1[molINDX, 1].plot(self.frequency, self.pol3_EMPTY[molINDX].imag)

            print(envelopeBASIS.shape)
            print(Q_mat[molINDX, :, self.molNUM-1:].T.shape)
            Q_mat[molINDX], R_mat[molINDX] = np.linalg.qr(np.delete(self.pol3basisMATRIX, molINDX, 1), mode='complete')
            heterodyne[molINDX] = sum(q * np.vdot(q, envelopeBASIS) for q in Q_mat[molINDX, :, self.molNUM-1:].T)

            for j in range(self.molNUM):
                ImatBASIS[molINDX, j] = np.vdot(heterodyne[molINDX], self.pol3basisMATRIX[:, j])
                ImatFREQ[molINDX, j] = np.vdot(heterodyne[molINDX].dot(self.freq2basisMATRIX), self.pol3_EMPTY[j])

            axes1[molINDX, 2].plot(self.frequency, heterodyne[molINDX].dot(self.freq2basisMATRIX).real.T)
            axes1[molINDX, 3].plot(self.frequency, heterodyne[molINDX].dot(self.freq2basisMATRIX).imag.T)

        plt.figure()
        plt.imshow(np.abs(Q_mat[2]), cmap=plt.cm.rainbow)
        cax = plt.axes([0.85, 0.1, 0.075, 0.8])
        plt.colorbar(cax=cax)

        print(ImatFREQ)
        print(ImatBASIS)

        fig2, axes2 = plt.subplots(nrows=1, ncols=2, sharex=True)
        for molINDX in range(self.molNUM):
            axes2[0].plot(heterodyne[molINDX].real, '-')
            axes2[1].plot(heterodyne[molINDX].imag, '-')

        print(np.linalg.cond(ImatFREQ))
        print(np.linalg.cond(ImatBASIS))

    def calculate_heterodyne_COMB(self):

        np.set_printoptions(precision=3, suppress=True)
        Q_mat = np.empty((self.molNUM, self.combNUM, self.combNUM), dtype=np.complex)
        R_mat = np.empty((self.molNUM, self.combNUM, self.molNUM - 1), dtype=np.complex)

        ncols = 4
        envelopeCOMB = (np.exp(-(np.arange(self.combNUM - self.molNUM + 1) - 0.5 * self.combNUM)**2 / (2 * (self.combNUM/6) ** 2)))
        envelopeCOMB = np.ones_like(envelopeCOMB)
        print(envelopeCOMB)
        fig1, axes1 = plt.subplots(nrows=3, ncols=ncols)
        for molINDX in range(self.molNUM):
            for colINDX in range(ncols):
                axes1[molINDX, colINDX].plot(self.field1FREQ, self.field1.real, 'g', alpha=0.5)
                axes1[molINDX, colINDX].plot(self.field2FREQ, self.field2.real, 'y', alpha=0.5)
            axes1[molINDX, 0].plot(self.frequency, self.pol3_EMPTY[molINDX].real)
            axes1[molINDX, 1].plot(self.frequency, self.pol3_EMPTY[molINDX].imag)

            Q_mat[molINDX], R_mat[molINDX] = np.linalg.qr(np.delete(self.freq2combMATRIX.T.dot(self.pol3_EMPTY.T), molINDX, 1), mode='complete')

            Q_mat[molINDX, :, self.molNUM - 1:] *= envelopeCOMB
            axes1[molINDX, 2].plot(self.frequency,
                                   (Q_mat[molINDX, :, self.molNUM - 1:].sum(axis=1)).dot(self.freq2combMATRIX.T).real)
            axes1[molINDX, 3].plot(self.frequency,
                                   np.abs((Q_mat[molINDX, :, self.molNUM - 1:].sum(axis=1)).dot(self.freq2combMATRIX.T)))


if __name__ == '__main__':
    import pickle
    import matplotlib.pyplot as plt
    import time

    with open("polARGS500_1500.pickle", "rb") as f_args:
        args = pickle.load(f_args)
    molNUM = args['molNUM']
    freqNUM = args['freqNUM']
    freqDEL = args['freqDEL']
    frequency = args['frequency']

    pol3_FIELD = np.empty((molNUM, freqNUM), dtype=np.complex)
    pol3_EMPTY = np.empty((molNUM, freqNUM), dtype=np.complex)

    for i in range(molNUM):
        with open("pol3DIST500_1500" + ".pickle", "rb") as f_data:
            data = pickle.load(f_data)

            pol3_EMPTY[i] = data['pol3DIST'].sum(axis=(0, 1, 2))[i]
            # pol3_FIELD[i] = data['polFIELD']

    SystemVars = ADict(
        molNUM=args['molNUM'],
        combNUM=args['combNUM'],
        freqNUM=freqNUM,
        resolutionNUM=args['resolutionNUM'],
        omegaM1=args['omegaM1'],
        omegaM2=args['omegaM2'],
        combGAMMA=args['combGAMMA'],
        freqDEL=freqDEL,
        termsNUM=args['termsNUM'],
        frequency=args['frequency'],
        field1FREQ=args['field1FREQ'],
        field2FREQ=args['field2FREQ'],
        field1=args['field1'],
        field2=args['field2'],
        round=6,
        basisNUM_CB=50,
        basisENVwidth_CB=500,
        basisNUM_FB=50,
        basisENVwidth_FB=500
    )

    SystemArgs = dict(
        pol3_EMPTY=pol3_EMPTY,
        pol3_FIELD=pol3_FIELD
    )

    start = time.time()
    system = QRD(SystemVars, **SystemArgs)
    system.basis_transform()
    system.calculate_heterodyne()
    print("Time elapsed: ", time.time() - start)

    plt.show()
