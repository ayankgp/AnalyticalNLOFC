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

from multiprocessing import cpu_count
from types import MethodType, FunctionType
import time
from itertools import product, permutations
from functions import *
from wrapper import *
import pickle


class ADict(dict):
    """
    Appended Dictionary: where keys can be accessed as attributes: A['*'] --> A.*
    """

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            dict.__getattribute__(self, item)


class Spectra:
    """
    Calculates the linear absorption spectra and fits molecular parameters in the process
    """

    def __init__(self, spectra_variables, **kwargs):
        """
        __init__ function call to initialize variables from the keyword args for the class instance
         provided in __main__ and add new variables for use in other functions in this class, with
         data from SystemVars.
        """

        for name, value in kwargs.items():
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self))
            else:
                setattr(self, name, value)

        self.rho_0 = np.ascontiguousarray(spectra_variables.rho_0)
        self.rho = np.ascontiguousarray([spectra_variables.rho_0.copy() for _ in range(spectra_variables.molNUM)])
        self.spectra_time = np.linspace(0, spectra_variables.spectra_timeAMP, spectra_variables.spectra_timeDIM)
        self.spectra_field = np.zeros(spectra_variables.spectra_timeDIM, dtype=np.complex)
        self.gammaMATRIXpopd = np.ascontiguousarray(self.gammaMATRIXpopd)
        self.gammaMATRIXdephasing = np.ascontiguousarray(self.gammaMATRIXdephasing)
        self.muMATRIX = np.ascontiguousarray(self.muMATRIX)
        self.energies = np.ascontiguousarray(self.energies)
        self.levelsNUM = spectra_variables.levelsNUM
        self.spectra_absTOTAL = np.ascontiguousarray(np.zeros((spectra_variables.molNUM, len(self.spectra_frequencies[0]))))
        self.spectra_absDIST = np.ascontiguousarray(np.empty((spectra_variables.molNUM, spectra_variables.ensembleNUM, len(self.spectra_frequencies[0]))))
        # ------------------------------------------------------------------------------------------------------------ #
        #                       DECLARE NEW SET OF VARIABLES FOR ALL N MOLECULES IN ENSEMBLE                           #
        # ------------------------------------------------------------------------------------------------------------ #

    def create_molecule(self, spectra_molecule, indices):
        spectra_molecule.levelsNUM = self.levelsNUM
        spectra_molecule.energies = self.energies[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.gammaMATRIXpopd = self.gammaMATRIXpopd[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.gammaMATRIXdephasing = self.gammaMATRIXdephasing[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.spectra_frequencies = self.spectra_frequencies[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.spectra_freqDIM = len(self.spectra_frequencies[indices])
        spectra_molecule.muMATRIX = self.muMATRIX[indices].ctypes.data_as(POINTER(c_complex))
        spectra_molecule.spectra_field = self.spectra_field.ctypes.data_as(POINTER(c_complex))
        spectra_molecule.rho = self.rho[indices].ctypes.data_as(POINTER(c_complex))
        spectra_molecule.spectra_absTOTAL = self.spectra_absTOTAL[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.spectra_absDIST = self.spectra_absDIST[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.spectra_absREF = self.spectra_absREF[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.levelsVIBR = self.levelsVIBR[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.levels = self.levels[indices].ctypes.data_as(POINTER(c_double))
        spectra_molecule.probabilities = self.probabilities[indices].ctypes.data_as(POINTER(c_double))
        return

    def create_parameters(self, spectra_parameters, variables):
        spectra_parameters.rho_0 = self.rho_0.ctypes.data_as(POINTER(c_complex))
        spectra_parameters.levelsNUM = self.levelsNUM
        spectra_parameters.excitedNUM = variables.excitedNUM
        spectra_parameters.spectra_time = self.spectra_time.ctypes.data_as(POINTER(c_double))
        spectra_parameters.spectra_timeAMP = variables.spectra_timeAMP
        spectra_parameters.spectra_timeDIM = len(self.spectra_time)
        spectra_parameters.spectra_fieldAMP = variables.spectra_fieldAMP
        spectra_parameters.threadNUM = variables.threadNUM
        spectra_parameters.ensembleNUM = variables.ensembleNUM
        spectra_parameters.guessLOWER = variables.guessLOWER.ctypes.data_as(POINTER(c_double))
        spectra_parameters.guessUPPER = variables.guessUPPER.ctypes.data_as(POINTER(c_double))
        spectra_parameters.iterMAX = variables.iterMAX
        return

    def fit_spectra(self, variables):
        spectra_parameters = SpectraParameters()
        self.create_parameters(spectra_parameters, variables)

        molENSEMBLE = [SpectraMolecule() for _ in range(variables.molNUM)]
        for molINDX in range(variables.molNUM):
            self.create_molecule(molENSEMBLE[molINDX], molINDX)
            CalculateSpectra(molENSEMBLE[molINDX], spectra_parameters)


class OFC:
    """
    Calculates the ofc response of the molecule
    """

    def __init__(self, ofc_variables, **kwargs):
        """
        __init__ function call to initialize variables from the keyword args for the class instance
         provided in __main__ and add new variables for use in other functions in this class, with
         data from SystemVars.
        """

        for name, value in kwargs.items():
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self))
            else:
                setattr(self, name, value)

        self.muMATRIX = np.ascontiguousarray(self.muMATRIX)

        self.energies = np.ascontiguousarray(self.energies)
        self.levelsNUM = ofc_variables.levelsNUM
        self.frequency, self.freq12, self.freq21, self.field1FREQ, self.field2FREQ = nonuniform_frequency_range_3(ofc_variables, -1.6, 2.4)
        self.omega = self.frequency
        self.polarizationEMPTY = np.zeros((ofc_variables.molNUM, ofc_variables.ensembleNUM, self.frequency.size), dtype=np.complex)
        self.polarizationFIELD = np.zeros((ofc_variables.molNUM, ofc_variables.ensembleNUM, self.frequency.size), dtype=np.complex)
        self.polarizationINDEX = np.zeros((ofc_variables.molNUM, self.frequency.size), dtype=np.complex)
        self.polarizationLINEAR = np.zeros((ofc_variables.molNUM, self.frequency.size), dtype=np.complex)
        self.polarizationLINEARMOLECULE = np.zeros((ofc_variables.molNUM, ofc_variables.ensembleNUM, self.frequency.size), dtype=np.complex)
        self.polarizationLINEARTOTAL = np.zeros((ofc_variables.molNUM, self.frequency.size), dtype=np.complex)
        self.polarizationMOLECULE = np.zeros((ofc_variables.molNUM, ofc_variables.ensembleNUM, self.frequency.size), dtype=np.complex)
        self.polarizationTOTALEMPTY = np.zeros((ofc_variables.molNUM, self.frequency.size), dtype=np.complex)
        self.polarizationTOTALFIELD = np.zeros((ofc_variables.molNUM, self.frequency.size), dtype=np.complex)
        self.polarizationTOTALEMPTY_DIST = np.zeros((ofc_variables.basisNUM, ofc_variables.basisNUM,
                                                     ofc_variables.basisNUM, ofc_variables.molNUM, self.frequency.size), dtype=np.complex)
        self.polarizationTOTALFIELD_DIST = np.zeros((ofc_variables.basisNUM, ofc_variables.basisNUM,
                                                     ofc_variables.basisNUM, ofc_variables.molNUM, self.frequency.size),
                                                    dtype=np.complex)
        self.polINDX = np.empty((ofc_variables.basisNUM, ofc_variables.basisNUM, ofc_variables.basisNUM))
        self.basisINDX = np.empty(3, dtype=int)
        self.indices = np.empty(3, dtype=int)


        # ------------------------------------------------------------------------------------------------------------ #
        #                       DECLARE NEW SET OF VARIABLES FOR ALL N MOLECULES IN ENSEMBLE                           #
        # ------------------------------------------------------------------------------------------------------------ #


    def create_ofc_molecule(self, ofc_molecule, indices):
        ofc_molecule.levelsNUM = self.levelsNUM
        ofc_molecule.energies = self.energies[indices].ctypes.data_as(POINTER(c_double))
        ofc_molecule.levelsVIBR = self.levelsVIBR[indices].ctypes.data_as(POINTER(c_double))
        ofc_molecule.levels = self.levels[indices].ctypes.data_as(POINTER(c_double))
        ofc_molecule.gammaMATRIX = np.ascontiguousarray(self.gammaMATRIX).ctypes.data_as(POINTER(c_double))
        ofc_molecule.muMATRIX = self.muMATRIX[indices].ctypes.data_as(POINTER(c_complex))
        ofc_molecule.polarizationINDEX = self.polarizationINDEX[indices].ctypes.data_as(POINTER(c_complex))
        ofc_molecule.polarizationMOLECULE = self.polarizationMOLECULE[indices].ctypes.data_as(POINTER(c_complex))
        ofc_molecule.polarizationLINEAR = self.polarizationLINEAR[indices].ctypes.data_as(POINTER(c_complex))
        ofc_molecule.polarizationLINEARMOLECULE = self.polarizationLINEARMOLECULE[indices].ctypes.data_as(POINTER(c_complex))
        ofc_molecule.probabilities = self.probabilities[indices].ctypes.data_as(POINTER(c_double))
        return

    def create_ofc_parameters(self, ofc_parameters, ofc_variables):
        ofc_parameters.excitedNUM = ofc_variables.excitedNUM
        ofc_parameters.ensembleNUM = ofc_variables.ensembleNUM
        ofc_parameters.freqNUM = len(self.frequency)
        ofc_parameters.combNUM = ofc_variables.combNUM
        ofc_parameters.resolutionNUM = ofc_variables.resolutionNUM
        ofc_parameters.basisNUM = ofc_variables.basisNUM
        ofc_parameters.frequency = self.frequency.ctypes.data_as(POINTER(c_double))
        ofc_parameters.omega = self.omega.ctypes.data_as(POINTER(c_double))
        ofc_parameters.combGAMMA = ofc_variables.combGAMMA
        ofc_parameters.freqDEL = ofc_variables.freqDEL
        ofc_parameters.termsNUM = ofc_variables.termsNUM
        ofc_parameters.indices = self.indices.ctypes.data_as(POINTER(c_long))
        ofc_parameters.basisINDX = self.basisINDX.ctypes.data_as(POINTER(c_long))
        ofc_parameters.modulations = np.zeros(3, dtype=int).ctypes.data_as(POINTER(c_double))
        ofc_parameters.envelopeWIDTH = ofc_variables.envelopeWIDTH
        ofc_parameters.envelopeCENTER = ofc_variables.envelopeCENTER
        return

    def calculate_ofc_system(self, ofc_variables):
        ofc_parameters = OFCParameters()
        self.create_ofc_parameters(ofc_parameters, ofc_variables)
        molENSEMBLE = [OFCMolecule() for _ in range(ofc_variables.molNUM)]

        basisRNG = int(ofc_parameters.basisNUM / 2)
        for I_ in range(-basisRNG, basisRNG):
            ofc_parameters.basisINDX[0] = I_
            self.polINDX[I_] = I_
            for J_ in range(-basisRNG, basisRNG):
                ofc_parameters.basisINDX[1] = J_
                self.polINDX[J_] = J_
                for K_ in range(-basisRNG, basisRNG):
                    ofc_parameters.basisINDX[2] = K_
                    self.polINDX[K_] = K_
                    print(I_, J_, K_)
                    fig, ax = plt.subplots(nrows=ofc_variables.molNUM, ncols=2, sharex=True, sharey=True, figsize=(22, 11))
                    for molINDX in range(ofc_variables.molNUM):
                        self.create_ofc_molecule(molENSEMBLE[molINDX], molINDX)
                        for i, modulations in enumerate(list(product(*(3 * [[ofc_variables.omegaM1, ofc_variables.omegaM2]])))):
                            if i in range(8):
                                for mINDX, nINDX, vINDX in ofc_variables.modulationINDXlist:
                                    ofc_parameters.indices[0] = mINDX
                                    ofc_parameters.indices[1] = nINDX
                                    ofc_parameters.indices[2] = vINDX
                                    ofc_parameters.modulations = np.asarray(modulations).ctypes.data_as(POINTER(c_double))
                                    mu_product = self.muMATRIX[molINDX][0, mINDX] * self.muMATRIX[molINDX][mINDX, nINDX] * \
                                                 self.muMATRIX[molINDX][nINDX, vINDX] * self.muMATRIX[molINDX][vINDX, 0]
                                    self.polarizationMOLECULE[molINDX][:] = 0.
                                    CalculateNLResponse(molENSEMBLE[molINDX], ofc_parameters)
                                    self.polarizationMOLECULE[molINDX] *= mu_product
                                    for ensembleINDX in range(ofc_variables.ensembleNUM):
                                        if (i == 1) or (i == 6):
                                            self.polarizationEMPTY[molINDX][ensembleINDX] += self.polarizationMOLECULE[molINDX][
                                                ensembleINDX]
                                        else:
                                            self.polarizationFIELD[molINDX][ensembleINDX] += self.polarizationMOLECULE[molINDX][
                                                ensembleINDX]

                        for ensembleINDX in range(ofc_variables.ensembleNUM):
                            self.polarizationTOTALEMPTY[molINDX] += (self.polarizationEMPTY[molINDX])[ensembleINDX]*self.probabilities[molINDX][ensembleINDX]
                            self.polarizationTOTALFIELD[molINDX] += (self.polarizationFIELD[molINDX])[ensembleINDX]*self.probabilities[molINDX][ensembleINDX]

                        self.polarizationTOTALEMPTY_DIST[I_][J_][K_][molINDX] = self.polarizationTOTALEMPTY[molINDX]
                        self.polarizationTOTALFIELD_DIST[I_][J_][K_][molINDX] = self.polarizationTOTALFIELD[molINDX]

                        if ofc_variables.molNUM > 1:
                            ax[molINDX, 0].plot(system.frequency / (timeFACTOR * 2 * np.pi), self.polarizationTOTALEMPTY[molINDX].real, 'r')
                            ax[molINDX, 0].plot(system.frequency / (timeFACTOR * 2 * np.pi), self.polarizationTOTALFIELD[molINDX].real, 'k')
                            ax[molINDX, 1].plot(system.frequency / (timeFACTOR * 2 * np.pi), self.polarizationTOTALEMPTY[molINDX].imag, 'r')
                            ax[molINDX, 1].plot(system.frequency / (timeFACTOR * 2 * np.pi), self.polarizationTOTALFIELD[molINDX].imag, 'k')
                            ax[molINDX, 0].grid()
                            ax[molINDX, 1].grid()
                        else:
                            ax[0].plot(system.frequency / (timeFACTOR * 2 * np.pi), self.polarizationTOTALEMPTY[molINDX].real, 'r')
                            ax[0].plot(system.frequency / (timeFACTOR * 2 * np.pi), self.polarizationTOTALFIELD[molINDX].real, 'k')
                            ax[1].plot(system.frequency / (timeFACTOR * 2 * np.pi), self.polarizationTOTALEMPTY[molINDX].imag, 'r')
                            ax[1].plot(system.frequency / (timeFACTOR * 2 * np.pi), self.polarizationTOTALFIELD[molINDX].imag, 'k')
                            ax[0].grid()
                            ax[1].grid()

                        # if np.abs(self.polarizationTOTALEMPTY).max() > 1e1:
                        #     plt.savefig("Plots4/" + str(I_) + str(J_) + str(K_) + "_500lines.png", dpi=fig.dpi)

                        self.polarizationFIELD *= 0.
                        self.polarizationEMPTY *= 0.
                        self.polarizationTOTALEMPTY[molINDX] *= 0.
                        self.polarizationTOTALFIELD[molINDX] *= 0.
                    plt.close(fig)


if __name__ == '__main__':


    # --------------------------------------------------------- #
    #                       LIST OF CONSTANTS                   #
    # --------------------------------------------------------- #

    energyFACTOR = 1./27.211385
    timeFACTOR = 2.418884e-5
    wavelength2freqFACTOR = 1239.84
    cm_inv2evFACTOR = 1.23984e-4

    # ------------------------------------------------------------------------------------------ #
    #                       MOLECULAR CONSTANTS, VARIABLES, VECTORS & MATRICES                   #
    # ------------------------------------------------------------------------------------------ #

    molNUM = 1
    levelsNUM = 4
    ensembleNUM = 1
    groundNUM = 2
    excitedNUM = levelsNUM - groundNUM

    # ------------------ MOLECULAR ENERGY LEVEL STRUCTURE ------------------ #

    energies = np.empty((molNUM, levelsNUM))
    levelMIN = [555, 558, 558]
    levelMAX = [725, 735, 785]
    wavelengthMIN = [539, 551, 551]
    wavelengthMAX = [772, 819, 799]
    levels = [
        np.asarray(wavelength2freqFACTOR * energyFACTOR / np.linspace(levelMIN[i], levelMAX[i], excitedNUM * ensembleNUM)[::-1])
        for i in range(molNUM)]

    vibrations = [1600, 1610, 1590]
    levelsVIBR = [np.asarray([0, vibrations[i]]) * energyFACTOR * cm_inv2evFACTOR for i in range(molNUM)]

    # ------------------------ INITIAL DENSITY MATRIX ---------------------- #

    rho_0 = np.zeros((levelsNUM, levelsNUM), dtype=np.complex)
    rho_0[0, 0] = 1 + 0j

    # ------------------ TRANSITION DIPOLE MOMENT AND DECAY PARAMETERS ------------------ #

    MU = [2., 2.1, 2.2]
    MUvibr = [0.5, 0.55, 0.6]

    gammaPOPD = [2.418884e-8, 2.518884e-8, 2.618884e-8]
    gammaELEC = [1.5 * 2.418884e-6, 1.7 * 2.518884e-4, 1.9 * 2.618884e-4]
    gammaVIBR = [2.5e-6, 2.e-6, 3e-6]

    muMATRIX = [MUvibr[i] * np.ones((levelsNUM, levelsNUM), dtype=np.complex) for i in range(molNUM)]
    [np.fill_diagonal(muMATRIX[i], 0j) for i in range(molNUM)]
    for i in range(molNUM):
        np.fill_diagonal(muMATRIX[i], 0j)
        for j in range(groundNUM):
            for k in range(groundNUM, levelsNUM):
                muMATRIX[i][j, k] = MU[i]
                muMATRIX[i][k, j] = MU[i]

    gammaMATRIXpopd = [np.ones((levelsNUM, levelsNUM)) * gammaPOPD[i] for i in range(molNUM)]
    gammaMATRIXdephasing = [np.ones((levelsNUM, levelsNUM)) * gammaVIBR[i] for i in range(molNUM)]
    for i in range(molNUM):
        np.fill_diagonal(gammaMATRIXpopd[i], 0.0)
        gammaMATRIXpopd[i] = np.tril(gammaMATRIXpopd[i]).T
        np.fill_diagonal(gammaMATRIXdephasing[i], 0.0)
        for j in range(groundNUM):
            for k in range(groundNUM, levelsNUM):
                gammaMATRIXdephasing[i][j, k] = gammaELEC[i]
                gammaMATRIXdephasing[i][k, j] = gammaELEC[i]
    gammaMATRIX = gammaMATRIXdephasing
    for k in range(molNUM):
        for n in range(levelsNUM):
            for m in range(levelsNUM):
                for i in range(levelsNUM):
                    gammaMATRIX[k][n][m] += 0.5 * (gammaMATRIXpopd[k][n][i] + gammaMATRIXpopd[k][m][i])


    # ------------------ SPECTRA FITTING PROBABILITIES  ------------------ #

    probabilities = np.asarray(
        [
            [0.0559383, 0.0405529, 0.0496727, 0.0502181, 0.0505829, 0.0540158, 0.0698249, 0.0561983, 0.0872398,
             0.0899126, 0.0889188, 0.125873, 0.17377, 0.164667, 0.131627, 0.167373, 0.235067, 0.272492, 0.291075,
             0.314825, 0.344628, 0.382892, 0.454013, 0.449995, 0.503389, 0.543675, 0.59588, 0.633565, 0.648047,
             0.701027, 0.825524, 0.979807, 0.998223, 0.999991, 0.999955, 0.9996, 0.966293, 0.903151, 0.624991,
             0.362229, 0.245723, 0.13213, 0.072025, 0.0118298],
            [0.0293968, 0.0338536, 0.0383105, 0.040928, 0.0434535, 0.0481712, 0.0634281, 0.0460764, 0.079831, 0.057043,
             0.0757835, 0.0893302, 0.10581, 0.126468, 0.148797, 0.16973, 0.189966, 0.222053, 0.261433, 0.292574,
             0.317536, 0.355447, 0.405128, 0.456216, 0.50885, 0.559217, 0.606564, 0.660551, 0.72533, 0.802389, 0.976003,
             0.994937, 0.997694, 0.997776, 0.989299, 0.909013, 0.800151, 0.680393, 0.495251, 0.319516, 0.233149,
             0.147702, 0.0806267, 0.0135515],
            [0.00679299, 0.00727662, 0.0130347, 0.0137522, 0.0143939, 0.0188319, 0.023665, 0.0301884, 0.0370206,
             0.0362768, 0.0337105, 0.048696, 0.0690665, 0.0735955, 0.071801, 0.0826089, 0.0996923, 0.118934, 0.139534,
             0.152036, 0.19888, 0.219307, 0.250201, 0.279543, 0.303425, 0.333486, 0.361688, 0.432412, 0.449959,
             0.516771, 0.59716, 0.687907, 0.955484, 0.972082, 0.906222, 0.882957, 0.667665, 0.457709, 0.265371,
             0.0926473, 0.0708056, 0.0489661, 0.0277671, 0.00653886]
        ]
    )

    guessLOWER = np.zeros(ensembleNUM)
    guessUPPER = np.ones(ensembleNUM)

    # ---------------------------------------------------------------------------------------------------------------- #
    #              READ csv-DATA FILES INTO WAVELENGTH & ABSORPTION MATRICES: (SIZE) N x wavelengthNUM                 #
    # ---------------------------------------------------------------------------------------------------------------- #

    data_protein_files_list = ['DataFP/mSCARLET.csv', 'DataFP/FusionRED.csv', 'DataFP/mCHERRY.csv']
    protein_plot_colors = ['r', 'b', 'k']

    wavelengthNUM = 100
    wavelengths = np.empty([molNUM, wavelengthNUM])
    absorptions = np.empty_like(wavelengths)
    frequencies = np.empty_like(wavelengths)
    for i in range(molNUM):
        wavelengths[i], absorptions[i] = get_experimental_spectra(data_protein_files_list[i], wavelengthMIN[i], wavelengthMAX[i], wavelengthNUM)
        frequencies[i] = wavelength2freqFACTOR * energyFACTOR / wavelengths[i]

    # -------------------------------------------#
    #              OFC PARAMETERS                #
    # -------------------------------------------#

    combNUM = 5000
    resolutionNUM = 3
    omegaM1 = 3e-2 * timeFACTOR * 10
    omegaM2 = 8e-2 * timeFACTOR * 10
    freqDEL = 11e-2 * timeFACTOR * 10
    combGAMMA = 1e-10 * timeFACTOR
    termsNUM = 5
    envelopeWIDTH = 100000
    envelopeCENTER = 0

    SystemArgs = dict(
        gammaMATRIXpopd=gammaMATRIXpopd,
        gammaMATRIXdephasing=gammaMATRIXdephasing,
        gammaMATRIX=gammaMATRIX,
        muMATRIX=muMATRIX,
        energies=energies,
        levelsVIBR=levelsVIBR,
        levels=levels,
        probabilities=probabilities,
        spectra_wavelengths=np.ascontiguousarray(wavelengths),
        spectra_frequencies=np.ascontiguousarray(frequencies),
        spectra_absREF=np.ascontiguousarray(absorptions),
    )

    SystemVars = ADict(
        molNUM=molNUM,
        levelsNUM=levelsNUM,
        excitedNUM=excitedNUM,
        ensembleNUM=ensembleNUM,
        threadNUM=cpu_count(),
        rho_0=rho_0,
        spectra_timeAMP=10000,
        spectra_timeDIM=1000,
        spectra_fieldAMP=8e-6,
        guessLOWER=guessLOWER,
        guessUPPER=guessUPPER,
        iterMAX=1,
        combNUM=combNUM,
        basisNUM=2,
        resolutionNUM=resolutionNUM,
        omegaM1=omegaM1,
        omegaM2=omegaM2,
        combGAMMA=combGAMMA,
        freqDEL=freqDEL,
        termsNUM=termsNUM,
        envelopeWIDTH=envelopeWIDTH,
        envelopeCENTER=envelopeCENTER,
        modulationINDXlist=[(3, 1, 2)]
        # modulationINDXlist=[(1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1),]
    )

    start = time.time()

    system = OFC(SystemVars, **SystemArgs)
    system.calculate_ofc_system(SystemVars)
    with open("Pickle/pol3DIST5000_" + str(SystemVars.basisNUM) + "_4lvls.pickle", "wb") as dist_file:
        pickle.dump(
            {
                "pol3DIST_EMPTY": system.polarizationTOTALEMPTY_DIST,
                "pol3DIST_FIELD": system.polarizationTOTALFIELD_DIST,
                "pol3INDX": system.polINDX,
            },
            dist_file
        )
    print('TIME ELAPSED FOR OFC RESPONSE CALCULATION:', time.time() - start, 'seconds')

    field1, field2 = plot_field_pol_params(system, SystemVars)
    with open("Pickle/polARGS5000_8.pickle", "wb") as args_file:
        pickle.dump(
            {
                "freqNUM": len(system.frequency),
                "combNUM": SystemVars.combNUM,
                "resolutionNUM": SystemVars.resolutionNUM,
                "omegaM1": SystemVars.omegaM1,
                "omegaM2": SystemVars.omegaM2,
                "combGAMMA": SystemVars.combGAMMA,
                "freqDEL": SystemVars.freqDEL,
                "termsNUM": SystemVars.termsNUM,
                "basisNUM": SystemVars.basisNUM,
                "envelopeWIDTH": SystemVars.envelopeWIDTH,
                "envelopeCENTER": SystemVars.envelopeCENTER,
                "field1FREQ": system.field1FREQ / SystemVars.freqDEL,
                "field2FREQ": system.field2FREQ / SystemVars.freqDEL,
                "freq12": system.freq12 / SystemVars.freqDEL,
                "freq21": system.freq21 / SystemVars.freqDEL,
                "field1": field1,
                "field2": field2,
                "frequency": system.frequency / SystemVars.freqDEL,
            },
            args_file
        )

    # del system

    # ---------------------------------------------------------------------------------------------------------------- #
    #           Calculations for log-log plot of polarization dependence on comb linewidths                            #
    # ---------------------------------------------------------------------------------------------------------------- #

    def linewidth_dep():
        N = 50
        widths = np.logspace(2, -5, N)
        pol3max = np.empty_like(widths)
        for i in range(N):
            SystemVars.combGAMMA = widths[i] * timeFACTOR
            print(i, SystemVars.combGAMMA / timeFACTOR)
            system = OFC(SystemVars, **SystemArgs)
            system.calculate_ofc_system(SystemVars)
            pol3max[i] = system.polarizationTOTALEMPTY_DIST.real.max()
            del system

        print(pol3max)
        with open("Pickle/pol3max_vector_atomic.pickle", "wb") as pol3_file:
            pickle.dump(
                {
                    "pol3max": pol3max
                },
                pol3_file
            )
        plt.figure()
        plt.loglog(widths * 1e12, pol3max, 'r*-')
        plt.xlabel('frequency (in Hz)')
        plt.ylabel('max[$P^{(3)}(\omega)$] (in arb. units)')
        plt.show()

    # ---------------------------------------------------------------------------------------------------------------- #
    #           Calculations for polarization dependence on choice of omegaM1 and omegaM2                              #
    # ---------------------------------------------------------------------------------------------------------------- #

    def omega_M1_M2_dep():
        N = 100
        M1axis = np.linspace(2, 9, N, endpoint=True)
        pol3M1M2 = np.empty_like(M1axis, dtype=np.complex)
        for i in range(N):
            SystemVars.omegaM1 = M1axis[i] * timeFACTOR * 0.1
            SystemVars.omegaM2 = SystemVars.freqDEL - SystemVars.omegaM1
            print(i, SystemVars.omegaM1)
            system = OFC(SystemVars, **SystemArgs)
            system.calculate_ofc_system(SystemVars)
            pol3M1M2[i] = system.polarizationTOTALEMPTY_DIST.real.max()
            del system

        with open("Pickle/pol3M1M2_vector.pickle", "wb") as pol3_file:
            pickle.dump(
                {
                    "pol3max": pol3M1M2
                },
                pol3_file
            )
        plt.figure()
        plt.plot(M1axis, pol3M1M2, 'r*-')
        plt.xlabel('frequency (in Hz)')
        plt.ylabel('max[$P^{(3)}(\omega)$] (in arb. units)')
        plt.show()

    linewidth_dep()