#include "auxiliary.h"
#include "structures.h"

//====================================================================================================================//
//                                                                                                                    //
//                                              INTEGRALS OF SPECTROSCOPIC TERMS                                      //
//   ---------------------------------------------------------------------------------------------------------------  //
//    Given a set of modulations (\omegaM1; \omegaM2; \omegaM3) and permuted indices (m, n, v) calculate the          //
//    non-linear OFC spectroscopic integrals for each spectroscopic term occurring in the susceptibility function     //
//    \chi_3 using analytical forms developed via solving the Cauchy integral with frequencies in the upper z-plane   //
//====================================================================================================================//


//====================================================================================================================//
//                                                                                                                    //
//                                                  INTEGRAL OF TYPE A-1                                              //
//   ---------------------------------------------------------------------------------------------------------------  //
//      I1 = 1/(ABC) + 1/(ABD) + 1/(BCE) - 1/(ADE*)                                                                   //
//      where:                                                                                                        //
//      A -> {\omega} + \omegaM_i + m_i(\Delta \omega) + \Omega_b + i\tau                                            //
//      B -> \omegaM_k + m_k(\Delta \omega) + \Omega_a + i\tau                                                       //
//      C -> \omegaM_k + \omegaM_j + (m_k + m_j)(\Delta \omega) + \Omega_b + 2i\tau                                 //
//      D -> {\omega} + \omegaM_i - \omegaM_j + (m_i - m_j)(\Delta \omega) + \Omega_a + 2i\tau                      //
//      E -> -{\omega} + \omegaM_k + \omegaM_j - \omegaM_i + (m_k + m_j - m_i)(\Delta \omega) + 3i\tau             //
//                                                                                                                    //
//====================================================================================================================//

void pol3(ofc_molecule* ofc_mol, ofc_parameters* ofc_params, const cmplx wg_c, const cmplx wg_b, const cmplx wg_a, const int sign)
{
    int I_, J_, K_;
    I_ = ofc_params->basisINDX[0];
    J_ = ofc_params->basisINDX[1];
    K_ = ofc_params->basisINDX[2];

    double freqDEL = ofc_params->freqDEL;
    double blockDEL = freqDEL * ofc_params->combNUM / ofc_params->basisNUM;
    int combRNG = (int) ofc_params->combNUM / ofc_params->basisNUM;
    int termsNUM = ofc_params->termsNUM;
    double combGAMMA = ofc_params->combGAMMA;

    double omegaM_k = ofc_params->modulations[0];
    double omegaM_j = ofc_params->modulations[1];
    double omegaM_i = ofc_params->modulations[2];

    int m_k_0 = ceil((- omegaM_k - crealf(wg_a) - blockDEL*K_)/freqDEL);
    int m_j_0 = ceil((- omegaM_k - omegaM_j - crealf(wg_b) - blockDEL*(J_ + K_))/freqDEL) - m_k_0;
    double D = 0.;

    if (m_k_0 >= -combRNG && m_k_0 <= combRNG - 1 && m_j_0 >= -combRNG && m_j_0 <= combRNG - 1)
    {
        #pragma omp parallel for
        for(int out_i = 0; out_i < ofc_params->freqNUM; out_i++)
        {
            const double omega = ofc_params->frequency[out_i] - blockDEL*(- K_ - J_ + I_);
            int m_i_0 = m_k_0 + m_j_0 - ceil((omega - omegaM_k - omegaM_j + omegaM_i + blockDEL*(- K_ - J_ + I_))/freqDEL);
            cmplx result = 0. + 0. * I;
            if (m_i_0 >= -combRNG && m_i_0 <= combRNG - 1)
            {
                for(int m_i = m_i_0 - termsNUM; m_i < m_i_0 + termsNUM; m_i++)
                {
                    const cmplx term_A = omega + omegaM_i + m_i * freqDEL + I_ * blockDEL + wg_b + combGAMMA * I;
                    for(int m_j = m_j_0 - termsNUM; m_j < m_j_0 + termsNUM; m_j++)
                    {
                        const cmplx term_D = omega + omegaM_i - omegaM_j + (m_i - m_j) * freqDEL + (I_ - J_) * blockDEL + wg_a + 2 * I * combGAMMA;
                        for(int m_k = m_k_0 - termsNUM; m_k < m_k_0 + termsNUM; m_k++)
                        {
                            const cmplx term_B = omegaM_k + m_k * freqDEL + K_ * blockDEL + wg_a + combGAMMA * I;
                            const cmplx term_C = omegaM_k + omegaM_j + (m_k + m_j) * freqDEL + (K_ + J_) * blockDEL +  wg_b + 2 * I * combGAMMA;
                            const cmplx term_E = omega - (omegaM_k + omegaM_j - omegaM_i) - (m_k + m_j - m_i) * freqDEL - (K_ + J_ - I_) * blockDEL + 3 * I * combGAMMA;
                            const cmplx term_E_star = - omega + (omegaM_k + omegaM_j - omegaM_i) + (m_k + m_j - m_i) * freqDEL + (K_ + J_ - I_) * blockDEL + 3 * I * combGAMMA;
                            result += (1./(term_A * term_D * term_E_star) + (1./(term_B * term_C * term_E)));
                        }

                    }
                }

                ofc_mol->polarizationINDEX[out_i] += M_PI*M_PI*I*sign*result/(omega + wg_c);
            }

        }
    }

}

void pol3linear(ofc_molecule* ofc_mol, ofc_parameters* ofc_params, const cmplx wg_a)
{
    double freqDEL = ofc_params->freqDEL;
    int termsNUM = ofc_params->termsNUM;
    int combNUM = ofc_params->combNUM;
    double combGAMMA = ofc_params->combGAMMA;
    double omega_M = 15E-2 * 2.418884e-5 * 8;

    #pragma omp parallel for
    for(int out_i = 0; out_i < ofc_params->freqNUM; out_i++)
    {
        const double omega = ofc_params->frequency[out_i];
        cmplx result = 0. + 0. * I;
        for(int m = - combNUM; m < combNUM; m++)
        {
            result += (combGAMMA / (pow((omega - omega_M - m * freqDEL), 2) + pow(combGAMMA, 2)));
        }
        result = ((1./(conj(wg_a)- omega)) + (1./(wg_a + omega)));
        ofc_mol->polarizationLINEAR[out_i] += result;
    }
}

void CalculateResponse(ofc_molecule* ofc_mol, ofc_parameters* ofc_params)
{
    int m, n, v, l, levelsNUM;

    levelsNUM = ofc_mol->levelsNUM;
    l = 0;
    m = ofc_params->indices[0];
    n = ofc_params->indices[1];
    v = ofc_params->indices[2];

    cmplx wg_nl = ofc_mol->energies[n] - ofc_mol->energies[l] + I * ofc_mol->gammaMATRIX[n * levelsNUM + l];
    cmplx wg_vl = ofc_mol->energies[v] - ofc_mol->energies[l] + I * ofc_mol->gammaMATRIX[v * levelsNUM + l];
    cmplx wg_ml = ofc_mol->energies[m] - ofc_mol->energies[l] + I * ofc_mol->gammaMATRIX[m * levelsNUM + l];
    cmplx wg_nv = ofc_mol->energies[n] - ofc_mol->energies[v] + I * ofc_mol->gammaMATRIX[n * levelsNUM + v];
    cmplx wg_mv = ofc_mol->energies[m] - ofc_mol->energies[v] + I * ofc_mol->gammaMATRIX[m * levelsNUM + v];
    cmplx wg_vm = ofc_mol->energies[v] - ofc_mol->energies[m] + I * ofc_mol->gammaMATRIX[v * levelsNUM + m];
    cmplx wg_vn = ofc_mol->energies[v] - ofc_mol->energies[n] + I * ofc_mol->gammaMATRIX[v * levelsNUM + n];
    cmplx wg_mn = ofc_mol->energies[m] - ofc_mol->energies[n] + I * ofc_mol->gammaMATRIX[m * levelsNUM + n];
    cmplx wg_nm = ofc_mol->energies[n] - ofc_mol->energies[m] + I * ofc_mol->gammaMATRIX[n * levelsNUM + m];

    //==========================================================================================//
    //  THE FOLLOWING 8 CALLS ARE FOR THE 8 SPECTROSCOPIC TERMS: (a1), (a2), ...., (d1), (d2)   //                                                                         //
    //==========================================================================================//

    pol3(ofc_mol, ofc_params, -conj(wg_vl), -conj(wg_nl), -conj(wg_ml), -1);
    pol3(ofc_mol, ofc_params, -conj(wg_nv), -conj(wg_mv), wg_vl, 1);
    pol3(ofc_mol, ofc_params, -conj(wg_nv), wg_vm, -conj(wg_ml), 1);
    pol3(ofc_mol, ofc_params, -conj(wg_mn), wg_nl, wg_vl, -1);
    pol3(ofc_mol, ofc_params, wg_vn, -conj(wg_nl), -conj(wg_ml), 1);
    pol3(ofc_mol, ofc_params, wg_nm, -conj(wg_mv), wg_vl, -1);
    pol3(ofc_mol, ofc_params, wg_nm, wg_vm, -conj(wg_ml), -1);
    pol3(ofc_mol, ofc_params, wg_ml, wg_nl, wg_vl, 1);

}

void CalculateResponseLinear(ofc_molecule* ofc_mol, ofc_parameters* ofc_params)
{
    int m, n, v, l, levelsNUM;

    levelsNUM = ofc_mol->levelsNUM;
    l = 0;
    m = 1;
    n = 2;
    v = 3;

    cmplx wg_nl = ofc_mol->energies[n] - ofc_mol->energies[l] + I * ofc_mol->gammaMATRIX[n * levelsNUM + l];
    cmplx wg_vl = ofc_mol->energies[v] - ofc_mol->energies[l] + I * ofc_mol->gammaMATRIX[v * levelsNUM + l];
    cmplx wg_ml = ofc_mol->energies[m] - ofc_mol->energies[l] + I * ofc_mol->gammaMATRIX[m * levelsNUM + l];

    pol3linear(ofc_mol, ofc_params, wg_vl);
    pol3linear(ofc_mol, ofc_params, wg_nl);
    pol3linear(ofc_mol, ofc_params, wg_ml);

}