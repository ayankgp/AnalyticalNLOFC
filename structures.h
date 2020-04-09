typedef struct spectra_parameters{
    cmplx* rho_0;
    int levelsNUM;
    int excitedNUM;
    double* spectra_time;
    double spectra_timeAMP;
    int spectra_timeDIM;
    double spectra_fieldAMP;
    int threadNUM;
    int ensembleNUM;
    double* guessLOWER;
    double* guessUPPER;
    int iterMAX;
} spectra_parameters;

typedef struct spectra_molecule{
    int levelsNUM;
    double* energies;
    double* gammaMATRIXpopd;
    double* gammaMATRIXdephasing;
    double* spectra_frequencies;
    int spectra_freqDIM;
    cmplx* muMATRIX;
    cmplx* spectra_field;
    cmplx* rho;
    double* spectra_absTOTAL;
    double* spectra_absDIST;
    double* spectra_absREF;
    double* levelsVIBR;
    double* levels;
    double* probabilities;
} spectra_molecule;

typedef struct mol_system{
    spectra_molecule** ensemble;
    spectra_molecule* original;
    spectra_parameters* spectra_params;
    int* count;
} mol_system;

typedef struct ofc_parameters{
    int excitedNUM;
    int ensembleNUM;
    int freqNUM;
    int combNUM;
    int resolutionNUM;
    int basisNUM;
    double* frequency;
    double* omega;
    double combGAMMA;
    double freqDEL;
    int termsNUM;
    long* indices;
    long* basisINDX;
    double* modulations;
    double envelopeWIDTH;
    double envelopeCENTER;
} ofc_parameters;

typedef struct ofc_molecule{
    int levelsNUM;
    double* energies;
    double* levels;
    double* levelsVIBR;
    double* gammaMATRIX;
    cmplx* muMATRIX;
    cmplx* polarizationINDEX;
    cmplx* polarizationMOLECULE;
    cmplx* polarizationLINEAR;
    cmplx* polarizationLINEARMOLECULE;
    double* probabilities;
} ofc_molecule;
