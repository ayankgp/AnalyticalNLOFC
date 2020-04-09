import os
import ctypes
from ctypes import c_int, c_long, c_double, POINTER, Structure
import subprocess

__doc__ = """
Python wrapper for linear_response.c
Compile with:
gcc -O3 -shared -o linear_response.so linear_response.c -lm -fopenmp -lnlopt -fPIC
"""

subprocess.run(["gcc", "-O3", "-shared", "-o", "linear_response.so", "linear_response.c", "-lm", "-lnlopt", "-fopenmp", "-fPIC"])


class c_complex(Structure):
    """
    Complex double ctypes
    """
    _fields_ = [
        ('real', c_double),
        ('imag', c_double)
    ]


class SpectraParameters(Structure):
    """
    SpectraParameters structure ctypes
    """
    _fields_ = [
        ('rho_0', POINTER(c_complex)),
        ('levelsNUM', c_int),
        ('excitedNUM', c_int),
        ('spectra_time', POINTER(c_double)),
        ('spectra_timeAMP', c_double),
        ('spectra_timeDIM', c_int),
        ('spectra_fieldAMP', c_double),
        ('threadNUM', c_int),
        ('ensembleNUM', c_int),
        ('guessLOWER', POINTER(c_double)),
        ('guessUPPER', POINTER(c_double)),
        ('iterMAX', c_int),
    ]


class SpectraMolecule(Structure):
    """
    Parameters structure ctypes
    """
    _fields_ = [
        ('levelsNUM', c_int),
        ('energies', POINTER(c_double)),
        ('gammaMATRIXpopd', POINTER(c_double)),
        ('gammaMATRIXdephasing', POINTER(c_double)),
        ('spectra_frequencies', POINTER(c_double)),
        ('spectra_freqDIM', c_int),
        ('muMATRIX', POINTER(c_complex)),
        ('spectra_field', POINTER(c_complex)),
        ('rho', POINTER(c_complex)),
        ('spectra_absTOTAL', POINTER(c_double)),
        ('spectra_absDIST', POINTER(c_double)),
        ('spectra_absREF', POINTER(c_double)),
        ('levelsVIBR', POINTER(c_double)),
        ('levels', POINTER(c_double)),
        ('probabilities', POINTER(c_double))
    ]


class OFCParameters(Structure):
    """
    SpectraParameters structure ctypes
    """
    _fields_ = [
        ('excitedNUM', c_int),
        ('ensembleNUM', c_int),
        ('freqNUM', c_int),
        ('combNUM', c_int),
        ('resolutionNUM', c_int),
        ('basisNUM', c_int),
        ('frequency', POINTER(c_double)),
        ('omega', POINTER(c_double)),
        ('combGAMMA', c_double),
        ('freqDEL', c_double),
        ('termsNUM', c_int),
        ('indices', POINTER(c_long)),
        ('basisINDX', POINTER(c_long)),
        ('modulations', POINTER(c_double)),
        ('envelopeWIDTH', c_double),
        ('envelopeCENTER', c_double)
    ]


class OFCMolecule(Structure):
    """
    Parameters structure ctypes
    """
    _fields_ = [
        ('levelsNUM', c_int),
        ('energies', POINTER(c_double)),
        ('levels', POINTER(c_double)),
        ('levelsVIBR', POINTER(c_double)),
        ('gammaMATRIX', POINTER(c_double)),
        ('muMATRIX', POINTER(c_complex)),
        ('polarizationINDEX', POINTER(c_complex)),
        ('polarizationMOLECULE', POINTER(c_complex)),
        ('polarizationLINEAR', POINTER(c_complex)),
        ('polarizationLINEARMOLECULE', POINTER(c_complex)),
        ('probabilities', POINTER(c_double))
    ]


try:
    lib1 = ctypes.cdll.LoadLibrary(os.getcwd() + "/linear_response.so")
except OSError:
    raise NotImplementedError(
        """
        The library is absent. You must compile the C shared library using the commands:
        gcc -O3 -shared -o linear_response.so linear_response.c -lm -lnlopt -fopenmp -fPIC
        """
    )

lib1.CalculateLinearResponse.argtypes = (
    POINTER(SpectraMolecule),
    POINTER(SpectraParameters),
)
lib1.CalculateLinearResponse.restype = None

lib1.CalculateOFCResponse.argtypes = (
    POINTER(OFCMolecule),
    POINTER(OFCParameters),
)
lib1.CalculateOFCResponse.restype = None

lib1.CalculateOFCLinearResponse.argtypes = (
    POINTER(OFCMolecule),
    POINTER(OFCParameters),
)
lib1.CalculateOFCLinearResponse.restype = None



def CalculateSpectra(spectra_mol, spectra_params):
    return lib1.CalculateLinearResponse(
        spectra_mol,
        spectra_params
    )


def CalculateNLResponse(ofc_mol, ofc_params):
    return lib1.CalculateOFCResponse(
        ofc_mol,
        ofc_params
    )


def CalculateLinearResponse(ofc_mol, ofc_params):
    return lib1.CalculateOFCLinearResponse(
        ofc_mol,
        ofc_params
    )