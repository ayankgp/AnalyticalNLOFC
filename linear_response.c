//===================================================================//
//                                                                   //
//              Linear Spectra ----> Molecular Parameters            //
//   Molecular Spectra + Field Parameters ----> Nonlinear Response   //
//      Optimal Nonlinear Response ----> Optimal Field parameters    //
//                                                                   //
//                @author  A. Chattopadhyay                          //
//    @affiliation Princeton University, Dept. of Chemistry          //
//           @version Updated last on Dec 14 2018                    //
//                                                                   //
//===================================================================//

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <nlopt.h>
#include <omp.h>
#include <time.h>
#include "OFCintegral.h"
#define ENERGY_FACTOR 1. / 27.211385
#define ERROR_BOUND 1.0E-8
#define WAVELENGTH2FREQ 1239.84
#define NLOPT_XTOL 1.0E-6


void LinearSpectraField(spectra_molecule* spectra_mol, const spectra_parameters *const spectra_params, const int k)
//--------------------------------------------------------------------------------//
//     RETURNS THE FIELD FOR CALCULATION OF LINEAR SPECTRA AS A FUNCTION OF TIME  //
//             k ----> index corresponding to spectral wavelength                 //
//--------------------------------------------------------------------------------//
{
    int i;
    int levelsNUM = spectra_params->levelsNUM;
    double spectra_timeAMP = spectra_params->spectra_timeAMP;

    double* t = spectra_params->spectra_time;
    const double dt = spectra_params->spectra_timeAMP / (spectra_params->spectra_timeDIM - 1);
    double A = spectra_params->spectra_fieldAMP;

    for(i=0; i<spectra_params->spectra_timeDIM; i++)
    {
        const double spectra_time = t[i] + 0.5 * dt;
        spectra_mol->spectra_field[i] = A * pow(cos(M_PI/2 - M_PI*spectra_time/(spectra_timeAMP)), 2) * cos(spectra_mol->spectra_frequencies[k] * spectra_time);
    }

}


//====================================================================================================================//
//                                                                                                                    //
//                                CALCULATION OF OPEN QUANTUM SYSTEM DYNAMICS                                         //
//                                                                                                                    //
//====================================================================================================================//


void DynamicsOperatorL(cmplx* Q_MAT, const cmplx spectra_field_ti, spectra_molecule* spectra_mol, const spectra_parameters *const spectra_params)
//----------------------------------------------------//
// 	    RETURNS Q <-- L[Q] AT A PARTICULAR TIME (t)   //
//----------------------------------------------------//
{
    const int levelsNUM = spectra_mol->levelsNUM;
    double* gammaMATRIXdephasing = spectra_mol->gammaMATRIXdephasing;
    double* gammaMATRIXpopd = spectra_mol->gammaMATRIXpopd;
    cmplx* muMATRIX = spectra_mol->muMATRIX;
    double* energies = spectra_mol->energies;

    cmplx* L_MAT = (cmplx*)calloc(levelsNUM * levelsNUM,  sizeof(cmplx));

    for(int m = 0; m < levelsNUM; m++)
        {
        for(int n = 0; n < levelsNUM; n++)
            {
                L_MAT[m * levelsNUM + n] -= I * (energies[m] - energies[n]) * Q_MAT[m * levelsNUM + n];
                for(int k = 0; k < levelsNUM; k++)
                {
                    L_MAT[m * levelsNUM + n] += I * spectra_field_ti * (muMATRIX[m * levelsNUM + k] * Q_MAT[k * levelsNUM + n] - Q_MAT[m * levelsNUM + k] * muMATRIX[k * levelsNUM + n]);

                    L_MAT[m * levelsNUM + n] -= 0.5 * (gammaMATRIXpopd[k * levelsNUM + n] + gammaMATRIXpopd[k * levelsNUM + m]) * Q_MAT[m * levelsNUM + n];
                    L_MAT[m * levelsNUM + n] += gammaMATRIXpopd[m * levelsNUM + k] * Q_MAT[k * levelsNUM + k];
                }

                L_MAT[m * levelsNUM + n] -= gammaMATRIXdephasing[m * levelsNUM + n] * Q_MAT[m * levelsNUM + n];
            }
        }

    for(int m = 0; m < levelsNUM; m++)
        {
        for(int n = 0; n < levelsNUM; n++)
            {
                Q_MAT[m * levelsNUM + n] = L_MAT[m * levelsNUM + n];
            }
        }
    free(L_MAT);

}

//====================================================================================================================//
//                                                                                                                    //
//                                  PROPAGATION STEP FOR A GIVEN WAVELENGTH                                           //
//                                                                                                                    //
//====================================================================================================================//


void PropagateLinear(spectra_molecule* spectra_mol, const spectra_parameters *const spectra_params, const int index)
//--------------------------------------------------------------------------------------------------------------------//
// 	 	 		       CALCULATES FULL LINDBLAD DYNAMICS  DUE TO THE CONTROL FIELD FROM TIME 0 to T               	  //
//                            indx gives index of the specific wavelength in the spectra                              //
//--------------------------------------------------------------------------------------------------------------------//
{

    int t_i, convINDX;
    const int levelsNUM = spectra_params->levelsNUM;

    cmplx *rho_0 = spectra_params->rho_0;
    double *spectra_time = spectra_params->spectra_time;
    double spectra_dt = spectra_params->spectra_timeAMP / (spectra_params->spectra_timeDIM - 1);

    cmplx* spectra_field = spectra_mol->spectra_field;

    cmplx* LrhoMAT = (cmplx*)calloc(levelsNUM * levelsNUM, sizeof(cmplx));
    copy_complex_mat(rho_0, LrhoMAT, levelsNUM);
    copy_complex_mat(rho_0, spectra_mol->rho, levelsNUM);

    for(t_i=0; t_i<spectra_params->spectra_timeDIM; t_i++)
    {
        convINDX=1;
        do
        {
            DynamicsOperatorL(LrhoMAT, spectra_field[t_i], spectra_mol, spectra_params);
            scale_complex_mat(LrhoMAT, spectra_dt/convINDX, levelsNUM);
            add_complex_mat(LrhoMAT, spectra_mol->rho, levelsNUM, levelsNUM);
            convINDX+=1;
        }while(max_complex_mat(LrhoMAT, levelsNUM) > ERROR_BOUND);

        copy_complex_mat(spectra_mol->rho, LrhoMAT, levelsNUM);
    }

    for(int j=1; j<=spectra_params->excitedNUM; j++)
    {
        spectra_mol->spectra_absTOTAL[index] += spectra_mol->rho[(levelsNUM-j)*levelsNUM + (levelsNUM-j)];
    }
    free(LrhoMAT);
}


void copy_molecule(spectra_molecule* original, spectra_molecule* copy, spectra_parameters* spectra_params)
//-------------------------------------------------------------------//
//    MAKING A DEEP COPY OF AN INSTANCE OF THE MOLECULE STRUCTURE    //
//-------------------------------------------------------------------//
{
    int ensembleNUM = spectra_params->ensembleNUM;
    int levelsNUM = spectra_params->levelsNUM;

    copy->levelsNUM = original->levelsNUM;
    copy->energies = (double*)malloc(levelsNUM*sizeof(double));
    copy->gammaMATRIXpopd = (double*)malloc(levelsNUM*levelsNUM*sizeof(double));
    copy->gammaMATRIXdephasing = (double*)malloc(levelsNUM*levelsNUM*sizeof(double));
    copy->spectra_frequencies = (double*)malloc(original->spectra_freqDIM*sizeof(double));
    copy->spectra_freqDIM = original->spectra_freqDIM;
    copy->muMATRIX = (cmplx*)malloc(levelsNUM*levelsNUM*sizeof(cmplx));
    copy->spectra_field = (cmplx*)malloc(spectra_params->spectra_timeDIM*sizeof(cmplx));
    copy->rho = (cmplx*)malloc(levelsNUM*levelsNUM*sizeof(cmplx));
    copy->spectra_absTOTAL = (double*)malloc(original->spectra_freqDIM*sizeof(double));
    copy->spectra_absDIST = (double*)malloc(spectra_params->ensembleNUM*original->spectra_freqDIM*sizeof(double));
    copy->spectra_absREF = (double*)malloc(original->spectra_freqDIM*sizeof(double));
    copy->levelsVIBR = (double*)malloc((levelsNUM - spectra_params->excitedNUM)*sizeof(double));
    copy->levels = (double*)malloc(spectra_params->excitedNUM*spectra_params->ensembleNUM*sizeof(double));
    copy->probabilities = (double*)malloc(ensembleNUM*sizeof(double));

    memset(copy->energies, 0, spectra_params->levelsNUM*sizeof(double));
    memcpy(copy->gammaMATRIXpopd, original->gammaMATRIXpopd, levelsNUM*levelsNUM*sizeof(double));
    memcpy(copy->gammaMATRIXdephasing, original->gammaMATRIXdephasing, levelsNUM*levelsNUM*sizeof(double));
    memcpy(copy->spectra_frequencies, original->spectra_frequencies, original->spectra_freqDIM*sizeof(double));
    memcpy(copy->muMATRIX, original->muMATRIX, levelsNUM*levelsNUM*sizeof(cmplx));
    memcpy(copy->spectra_field, original->spectra_field, spectra_params->spectra_timeDIM*sizeof(cmplx));
    memcpy(copy->rho, original->rho, levelsNUM*levelsNUM*sizeof(cmplx));
    memcpy(copy->spectra_absTOTAL, original->spectra_absTOTAL, original->spectra_freqDIM*sizeof(double));
    memcpy(copy->spectra_absDIST, original->spectra_absDIST, spectra_params->ensembleNUM*original->spectra_freqDIM*sizeof(double));
    memcpy(copy->spectra_absREF, original->spectra_absREF, original->spectra_freqDIM*sizeof(double));
    memcpy(copy->levelsVIBR, original->levelsVIBR, (levelsNUM - spectra_params->excitedNUM)*sizeof(double));
    memcpy(copy->levels, original->levels, spectra_params->excitedNUM*spectra_params->ensembleNUM*sizeof(double));
    memcpy(copy->probabilities, original->probabilities, ensembleNUM*sizeof(double));
}


double nloptJ_spectra(unsigned N, const double *optimizeSPECTRA_params, double *grad_J, void *nloptJ_spectra_params)
{
    mol_system* system = (mol_system*)nloptJ_spectra_params;

    spectra_parameters* spectra_params = system->spectra_params;
    spectra_molecule** ensemble = system->ensemble;
    spectra_molecule* spectra_mol = system->original;
    int* count = system->count;

    memset(spectra_mol->spectra_absTOTAL, 0, spectra_mol->spectra_freqDIM*sizeof(double));

    #pragma omp parallel for
    for(int j=0; j<spectra_params->ensembleNUM; j++)
    {
        memset(ensemble[j]->spectra_absTOTAL, 0, spectra_mol->spectra_freqDIM*sizeof(double));

        for(int i=0; i<spectra_mol->spectra_freqDIM; i++)
        {
            LinearSpectraField(ensemble[j], spectra_params, i);
            LinearSpectraField(spectra_mol, spectra_params, i);
            PropagateLinear(ensemble[j], spectra_params, i);
        }
        scale_double_vec(ensemble[j]->spectra_absTOTAL, optimizeSPECTRA_params[j], spectra_mol->spectra_freqDIM);
        add_double_vec(ensemble[j]->spectra_absTOTAL, spectra_mol->spectra_absTOTAL, spectra_mol->spectra_freqDIM);
    }

    for(int j=0; j<spectra_params->ensembleNUM; j++)
    {
        for(int k=0; k<spectra_mol->spectra_freqDIM; k++)
        {
            spectra_mol->spectra_absDIST[j*spectra_mol->spectra_freqDIM + k] = 100. * ensemble[j]->spectra_absTOTAL[k] / max_double_vec(spectra_mol->spectra_absTOTAL, spectra_mol->spectra_freqDIM);
        }

    }
    scale_double_vec(spectra_mol->spectra_absTOTAL, 100./max_double_vec(spectra_mol->spectra_absTOTAL, spectra_mol->spectra_freqDIM), spectra_mol->spectra_freqDIM);
    double J;
    J = diffnorm_double_vec(spectra_mol->spectra_absREF, spectra_mol->spectra_absTOTAL, spectra_mol->spectra_freqDIM);

    *count = *count + 1;
    printf("%d | ", *count);
//    for(int i=0; i<spectra_params->ensembleNUM; i++)
//    {
//        printf("%3.2lf ", optimizeSPECTRA_params[i]);
//    }
//    printf(")  fit = %3.5lf \n", J);
    printf("fit = %3.5lf \n", J);
    return J;
}


void CalculateLinearResponse(spectra_molecule* spectra_mol, spectra_parameters* spectra_params)
//------------------------------------------------------------//
//          CALCULATING SPECTRAL FIT FOR A MOLECULE           //
//------------------------------------------------------------//
{

    // ---------------------------------------------------------------------- //
    //      UPDATING THE PURE DEPHASING MATRIX & ENERGIES FOR MOLECULE        //
    // ---------------------------------------------------------------------- //

    int vibrNUM = spectra_mol->levelsNUM - spectra_params->excitedNUM;

    spectra_molecule** ensemble = (spectra_molecule**)malloc(spectra_params->ensembleNUM * sizeof(spectra_molecule*));
    for(int i=0; i<spectra_params->ensembleNUM; i++)
    {
        ensemble[i] = (spectra_molecule*)malloc(sizeof(spectra_molecule));
        copy_molecule(spectra_mol, ensemble[i], spectra_params);
        for(int j=0; j<vibrNUM; j++)
        {
            ensemble[i]->energies[j] = spectra_mol->levelsVIBR[j];
        }

        for(int j=0; j<spectra_params->excitedNUM; j++)
        {
            ensemble[i]->energies[vibrNUM + j] = spectra_mol->levels[spectra_params->excitedNUM*i+j];
        }
    }

    // ---------------------------------------------------------------------- //
    //                   CREATING THE ENSEMBLE OF MOLECULES                   //
    // ---------------------------------------------------------------------- //

    mol_system system;
    system.ensemble = ensemble;
    system.original = (spectra_molecule*)malloc(sizeof(spectra_molecule));
    memcpy(system.original, spectra_mol, sizeof(spectra_molecule));
    system.spectra_params = spectra_params;
    system.count = (int*)malloc(sizeof(int));
    *system.count = 0;

    // ---------------------------------------------------------------------- //
    //              INITIALIZING NLOPT CLASS AND PARAMETERS                   //
    // ---------------------------------------------------------------------- //

    nlopt_opt optimizeSPECTRA;

    double *guessLOWER = spectra_params->guessLOWER;
    double *guessUPPER = spectra_params->guessUPPER;

    optimizeSPECTRA = nlopt_create(NLOPT_LN_COBYLA, spectra_params->ensembleNUM);  // Local no-derivative optimization algorithm
//    optimizeSPECTRA = nlopt_create(NLOPT_GN_DIRECT, spectra_params->ensembleNUM);  // Global no-derivative optimization algorithm
    nlopt_set_lower_bounds(optimizeSPECTRA, guessLOWER);
    nlopt_set_upper_bounds(optimizeSPECTRA, guessUPPER);
    nlopt_set_min_objective(optimizeSPECTRA, nloptJ_spectra, (void*)&system);
    nlopt_set_xtol_rel(optimizeSPECTRA, NLOPT_XTOL);
    nlopt_set_maxeval(optimizeSPECTRA, spectra_params->iterMAX);

    double *x = system.original->probabilities;
    double minf;

    if (nlopt_optimize(optimizeSPECTRA, x, &minf) < 0) {
        printf("nlopt failed!\n");
    }
    else {
        printf("found minimum at ( ");

        for(int m=0; m<spectra_params->ensembleNUM; m++)
        {
            printf(" %g,", x[m]);
        }
        printf(") = %g\n", minf);
    }

    nlopt_destroy(optimizeSPECTRA);

}


void copy_ofc_molecule(ofc_molecule* original, ofc_molecule* copy, ofc_parameters* ofc_params)
//-------------------------------------------------------------------//
//    MAKING A DEEP COPY OF AN INSTANCE OF THE MOLECULE STRUCTURE    //
//-------------------------------------------------------------------//
{
    int ensembleNUM = ofc_params->ensembleNUM;
    int levelsNUM = original->levelsNUM;
    int freqNUM = ofc_params->freqNUM;

    copy->levelsNUM = original->levelsNUM;
    copy->energies = (double*)malloc(levelsNUM*sizeof(double));
    copy->gammaMATRIX = (double*)malloc(levelsNUM*levelsNUM*sizeof(double));
    copy->muMATRIX = (cmplx*)malloc(levelsNUM*levelsNUM*sizeof(cmplx));
    copy->polarizationINDEX = (cmplx*)malloc(freqNUM*sizeof(cmplx));
    copy->polarizationMOLECULE = (cmplx*)malloc(ensembleNUM*freqNUM*sizeof(cmplx));
    copy->polarizationLINEAR = (cmplx*)malloc(freqNUM*sizeof(cmplx));
    copy->polarizationLINEARMOLECULE = (cmplx*)malloc(ensembleNUM*freqNUM*sizeof(cmplx));
    copy->probabilities = (double*)malloc(ensembleNUM*sizeof(double));

    memset(copy->energies, 0, original->levelsNUM*sizeof(double));
    memcpy(copy->gammaMATRIX, original->gammaMATRIX, levelsNUM*levelsNUM*sizeof(double));
    memcpy(copy->muMATRIX, original->muMATRIX, levelsNUM*levelsNUM*sizeof(cmplx));
    memcpy(copy->polarizationINDEX, original->polarizationINDEX, freqNUM*sizeof(cmplx));
    memcpy(copy->polarizationMOLECULE, original->polarizationMOLECULE, ensembleNUM*freqNUM*sizeof(cmplx));
    memcpy(copy->polarizationLINEAR, original->polarizationLINEAR, freqNUM*sizeof(cmplx));
    memcpy(copy->polarizationLINEARMOLECULE, original->polarizationLINEARMOLECULE, ensembleNUM*freqNUM*sizeof(cmplx));
    memcpy(copy->probabilities, original->probabilities, ensembleNUM*sizeof(double));
}


void free_ofc_molecule(ofc_molecule* mol)
//-------------------------------------------------------------------//
//    MAKING A DEEP COPY OF AN INSTANCE OF THE MOLECULE STRUCTURE    //
//-------------------------------------------------------------------//
{
    free(mol->energies);
    free(mol->gammaMATRIX);
    free(mol->muMATRIX);
    free(mol->polarizationINDEX);
    free(mol->polarizationMOLECULE);
    free(mol->polarizationLINEAR);
    free(mol->polarizationLINEARMOLECULE);
    free(mol->probabilities);
    free(mol);
}


void CalculateOFCResponse(ofc_molecule* ofc_mol, ofc_parameters* ofc_params)
//------------------------------------------------------------//
//          CALCULATING OFC RESPONSE FOR MOLECULE             //
//------------------------------------------------------------//
{
    // ---------------------------------------------------------------------- //
    //      UPDATING THE PURE DEPHASING MATRIX & ENERGIES FOR MOLECULE        //
    // ---------------------------------------------------------------------- //


    int vibrNUM = ofc_mol->levelsNUM - ofc_params->excitedNUM;
    ofc_molecule** ensemble = (ofc_molecule**)malloc(ofc_params->ensembleNUM * sizeof(ofc_molecule*));
    for(int i=0; i<ofc_params->ensembleNUM; i++)
    {
        ensemble[i] = (ofc_molecule*)malloc(sizeof(ofc_molecule));
        copy_ofc_molecule(ofc_mol, ensemble[i], ofc_params);
        for(int j=0; j<vibrNUM; j++)
        {
            ensemble[i]->energies[j] = ofc_mol->levelsVIBR[j];
        }

        for(int j=0; j<ofc_params->excitedNUM; j++)
        {
            ensemble[i]->energies[vibrNUM + j] = ofc_mol->levels[ofc_params->excitedNUM*i+j];
        }
    }

    // ---------------------------------------------------------------------- //
    //                   CREATING THE ENSEMBLE OF MOLECULES                   //
    // ---------------------------------------------------------------------- //

    for(int j=0; j<ofc_params->ensembleNUM; j++)
    {
        CalculateResponse(ensemble[j], ofc_params);
        for(int i=0; i<ofc_params->freqNUM; i++)
        {
            ofc_mol->polarizationMOLECULE[j * ofc_params->freqNUM + i] = ensemble[j]->polarizationINDEX[i];
        }
        free_ofc_molecule(ensemble[j]);
    }

}


void CalculateOFCLinearResponse(ofc_molecule* ofc_mol, ofc_parameters* ofc_params)
//------------------------------------------------------------//
//          CALCULATING OFC RESPONSE FOR MOLECULE             //
//------------------------------------------------------------//
{
    // ---------------------------------------------------------------------- //
    //      UPDATING THE PURE DEPHASING MATRIX & ENERGIES FOR MOLECULE        //
    // ---------------------------------------------------------------------- //


    int vibrNUM = ofc_mol->levelsNUM - ofc_params->excitedNUM;
    ofc_molecule** ensemble = (ofc_molecule**)malloc(ofc_params->ensembleNUM * sizeof(ofc_molecule*));
    for(int i=0; i<ofc_params->ensembleNUM; i++)
    {
        ensemble[i] = (ofc_molecule*)malloc(sizeof(ofc_molecule));
        copy_ofc_molecule(ofc_mol, ensemble[i], ofc_params);
        for(int j=0; j<vibrNUM; j++)
        {
            ensemble[i]->energies[j] = ofc_mol->levelsVIBR[j];
        }

        for(int j=0; j<ofc_params->excitedNUM; j++)
        {
            ensemble[i]->energies[vibrNUM + j] = ofc_mol->levels[ofc_params->excitedNUM*i+j];
        }
    }

    // ---------------------------------------------------------------------- //
    //                   CREATING THE ENSEMBLE OF MOLECULES                   //
    // ---------------------------------------------------------------------- //

    for(int j=0; j<ofc_params->ensembleNUM; j++)
    {
        CalculateResponseLinear(ensemble[j], ofc_params);

        for(int i=0; i<ofc_params->freqNUM; i++)
        {
            ofc_mol->polarizationLINEARMOLECULE[j * ofc_params->freqNUM + i] = ensemble[j]->polarizationLINEAR[i];
        }
        free_ofc_molecule(ensemble[j]);
    }

}