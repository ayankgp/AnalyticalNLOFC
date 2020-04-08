typedef double complex cmplx;
//====================================================================================================================//
//                                                                                                                    //
//                                        AUXILIARY FUNCTIONS FOR MATRIX OPERATIONS                                   //
//   ---------------------------------------------------------------------------------------------------------------  //
//    Given a matrix or vector and their dimensionality, these routines perform the operations of printing, adding,   //
//    scaling, copying etc. The function calls are of the form: OPERATION_(COMPLEX/DOUBLE)_(MAT/VEC)                  //
//                                                                                                                    //
//====================================================================================================================//


void print_complex_mat(const cmplx *A, const int nDIM)
//----------------------------------------------------//
// 	          PRINTS A COMPLEX MATRIX                 //
//----------------------------------------------------//
{
	int i,j;
	for(i=0; i<nDIM; i++)
	{
		for(j=0; j<nDIM; j++)
		{
			printf("%3.3e + %3.3eJ  ", creal(A[i * nDIM + j]), cimag(A[i * nDIM + j]));
		}
	    printf("\n");
	}
	printf("\n\n");
}

void print_complex_vec(const cmplx *A, const int vecDIM)
//----------------------------------------------------//
// 	          PRINTS A COMPLEX VECTOR                 //
//----------------------------------------------------//
{
	int i;
	for(i=0; i<vecDIM; i++)
	{
		printf("%3.3e + %3.3eJ  ", creal(A[i]), cimag(A[i]));
	}
	printf("\n");
}

void print_double_mat(const double *A, const int nDIM)
//----------------------------------------------------//
// 	            PRINTS A REAL MATRIX                  //
//----------------------------------------------------//
{
	int i,j;
	for(i=0; i<nDIM; i++)
	{
		for(j=0; j<nDIM; j++)
		{
			printf("%3.3e  ", A[i * nDIM + j]);
		}
	    printf("\n");
	}
	printf("\n\n");
}

void print_double_vec(const double *A, const int vecDIM)
//----------------------------------------------------//
// 	          PRINTS A REAL VECTOR                    //
//----------------------------------------------------//
{
	int i;
	for(i=0; i<vecDIM; i++)
	{
		printf("%3.3e  ", A[i]);
	}
	printf("\n");
}


void copy_complex_mat(const cmplx *A, cmplx *B, const int nDIM)
//----------------------------------------------------//
// 	        COPIES MATRIX A ----> MATRIX B            //
//----------------------------------------------------//
{
    int i, j = 0;
    for(i=0; i<nDIM; i++)
    {
        for(j=0; j<nDIM; j++)
        {
            B[i * nDIM + j] = A[i * nDIM + j];
        }
    }
}


void copy_complex_vec(const cmplx *A, cmplx *B, const int nDIM)
//----------------------------------------------------//
// 	        COPIES VECTOR A ----> VECTOR B            //
//----------------------------------------------------//
{
    int i, j = 0;
    for(i=0; i<nDIM; i++)
    {
        B[i] = A[i];
    }
}


void add_complex_mat(const cmplx *A, cmplx *B, const int nDIM1, const int nDIM2)
//----------------------------------------------------//
// 	        ADDS A to B ----> MATRIX B = A + B        //
//----------------------------------------------------//
{
    int i, j = 0;
    for(i=0; i<nDIM1; i++)
    {
        for(j=0; j<nDIM2; j++)
        {
            B[i * nDIM2 + j] += A[i * nDIM2 + j];
        }
    }
}


void add_double_vec(const double *A, double *B, const int nDIM)
//----------------------------------------------------//
// 	        ADDS A to B ----> MATRIX B = A + B        //
//----------------------------------------------------//
{
    for(int i=0; i<nDIM; i++)
    {
        B[i] += A[i];
    }
}

void scale_complex_mat(cmplx *A, const double factor, const int nDIM)
//----------------------------------------------------//
// 	     SCALES A BY factor ----> MATRIX B = A + B    //
//----------------------------------------------------//
{
    for(int i=0; i<nDIM; i++)
    {
        for(int j=0; j<nDIM; j++)
        {
            A[i * nDIM + j] *= factor;
        }
    }
}


void scale_double_mat(double *A, const double factor, const int nDIM)
//----------------------------------------------------//
// 	     SCALES A BY factor ----> MATRIX B = A + B    //
//----------------------------------------------------//
{
    for(int i=0; i<nDIM; i++)
    {
        for(int j=0; j<nDIM; j++)
        {
            A[i * nDIM + j] *= factor;
        }
    }
}


void scale_double_vec(double *A, const double factor, const int nDIM)
//----------------------------------------------------//
// 	     SCALES A BY factor ----> VECTOR B = A + B    //
//----------------------------------------------------//
{
    for(int i=0; i<nDIM; i++)
    {
        A[i] *= factor;
    }
}


cmplx trace_complex_mat(const cmplx *A, const int nDIM)
//----------------------------------------------------//
// 	                 RETURNS TRACE[A]                 //
//----------------------------------------------------//
{
    cmplx trace = 0.0 + I * 0.0;
    for(int i=0; i<nDIM; i++)
    {
        trace += A[i*nDIM + i];
    }
    printf("Trace = %3.3e + %3.3eJ  \n", creal(trace), cimag(trace));

    return trace;
}


double abs_complex(cmplx z)
//----------------------------------------------------//
// 	            RETURNS ABSOLUTE VALUE OF Z           //
//----------------------------------------------------//
{

    return sqrt((creal(z)*creal(z) + cimag(z)*cimag(z)));
}


double max_complex_mat(const cmplx *A, const int nDIM)
//----------------------------------------------------//
// 	   RETURNS ELEMENT WITH MAX ABSOLUTE VALUE        //
//----------------------------------------------------//
{
    double max_el = A[0];
    int i, j = 0;
    for(i=0; i<nDIM; i++)
    {
        for(j=0; j<nDIM; j++)
        {
            if(abs_complex(A[i * nDIM + j]) > max_el)
            {
                max_el = abs_complex(A[i * nDIM + j]);
            }
        }
    }
    return max_el;
}


double max_double_vec(const double *const A, const int nDIM)
//----------------------------------------------------//
// 	   RETURNS ELEMENT WITH MAX ABSOLUTE VALUE        //
//----------------------------------------------------//
{
    double max_el = A[0];
    for(int i=0; i<nDIM; i++)
    {
        if(A[i] > max_el)
        {
            max_el = A[i];
        }
    }
    return max_el;
}


double elsum_double_vec(const double *const A, const int nDIM)
//----------------------------------------------------//
// 	            RETURNS SUM OF VECTOR ELEMENTS        //
//----------------------------------------------------//
{
    double sum = 0.0;
    for(int i=0; i<nDIM; i++)
    {
        sum += A[i];
    }
    return sum;
}


double diffnorm_double_vec(const double *const A, const double *const B, const int nDIM)
//----------------------------------------------------//
// 	   RETURNS L-1 NORM OF VECTOR DIFFERENCE          //
//----------------------------------------------------//
{
    int nfrac = (int)(nDIM/2.2);
    double norm = 0.0;
    double norm_long_wavelength = 0.0;
    for(int i=0; i<nfrac; i++)
    {
        norm_long_wavelength += fabs(A[i]-B[i]);
    }

    for(int i=0; i<nDIM; i++)
    {
        norm += fabs(A[i]-B[i]);
    }
    printf("norm = %g, norm1 = %g \n", norm, norm_long_wavelength);
    return norm;
}