

#ifndef lib_utils_hpp
#define lib_utils_hpp

#ifndef DISBALE_CUDA

#include <type_traits>
#include"../objects/ell_matrix.hpp"

template<typename type_t>
struct cusolver_args{

	cusolver_args(): lwork(0),
                syevj_work(nullptr),
                syevj_info(nullptr),
                syevj_params(NULL),
                cusolverH(NULL) {}
                int lwork;
                type_t* syevj_work;
                int* syevj_info;
                syevjInfo_t syevj_params;
                cusolverDnHandle_t cusolverH;
	} ;


/////////////////////////////////////////////////////
void mm_cublas(         const double my_scalar,
                        const double other_scalar,
                        Dense_matrix<double> &other1,
                        Dense_matrix<double> &other2,
			Dense_matrix<double> &result,
                        cublasHandle_t cublasH)
{

       		CUBLAS_CHECK(
                        cublasDgemm(    cublasH,
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        other1.n_rows(),
                                        other2.n_cols(),
                                        other2.n_rows(),
                                        &other_scalar,
                                        other1.data(),
                                        other1.n_rows(),
                                        other2.data(),
                                        other2.n_rows(),
                                        &my_scalar,
                                        result.data(),
                                        result.n_rows()) );
}
void mm_cublas(         const float my_scalar,
                        const float other_scalar,
                        Dense_matrix<float> &other1,
                        Dense_matrix<float> &other2,
                        Dense_matrix<float> &result,
                        cublasHandle_t cublasH)
{

                CUBLAS_CHECK(
                        cublasSgemm(    cublasH,
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        other1.n_rows(),
                                        other2.n_cols(),
                                        other2.n_rows(),
                                        &other_scalar,
                                        other1.data(),
                                        other1.n_rows(),
                                        other2.data(),
                                        other2.n_rows(),
                                        &my_scalar,
                                        result.data(),
                                        result.n_rows()) );
}
/////////////////////////////////////////////////////



void mm_tt_cublas(      Dense_matrix<float> &T,
                        Dense_matrix<float> &result,
                        cublasHandle_t cublasH)
{
	float my_scalar = 0., other_scalar = 1.;

                CUBLAS_CHECK(
                        cublasSgemm(    cublasH,
                                        CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        T.n_cols(),
                                        T.n_cols(),
                                        T.n_rows(),
                                        &other_scalar,
                                        T.data(),
                                        T.n_rows(),
                                        T.data(),
                                        T.n_rows(),
                                        &my_scalar,
                                        result.data(),
                                        result.n_rows()) );
}
void mm_tt_cublas(      Dense_matrix<double> &T,
                        Dense_matrix<double> &result,
                        cublasHandle_t cublasH)
{
        double my_scalar = 0., other_scalar = 1.;

                CUBLAS_CHECK(
                        cublasDgemm(    cublasH,
                                        CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        T.n_cols(),
                                        T.n_cols(),
                                        T.n_rows(),
                                        &other_scalar,
                                        T.data(),
                                        T.n_rows(),
                                        T.data(),
                                        T.n_rows(),
                                        &my_scalar,
                                        result.data(),
                                        result.n_rows()) );
}
/////////////////////////////////////////////////////////

void mm_tt2_cublas(      Dense_matrix<float> &T1,
			Dense_matrix<float> &T2,
                        Dense_matrix<float> &result,
                        cublasHandle_t cublasH)
{
	float my_scalar = 0., other_scalar = 1.;
                CUBLAS_CHECK(
                        cublasSgemm(    cublasH,
                                        CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        T1.n_cols(),
                                        T1.n_cols(),
                                        T1.n_rows(),
                                        &other_scalar,
                                        T1.data(),
                                        T1.n_rows(),
                                        T2.data(),
                                        T1.n_rows(),
                                        &my_scalar,
                                        result.data(),
                                        result.n_rows()) );
	my_scalar = 0.5, other_scalar = 0.5;
	 CUBLAS_CHECK(
                        cublasSgemm(    cublasH,
                                        CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        T1.n_cols(),
                                        T1.n_cols(),
                                        T1.n_rows(),
                                        &other_scalar,
                                        T2.data(),
                                        T1.n_rows(),
                                        T1.data(),
                                        T1.n_rows(),
                                        &my_scalar,
                                        result.data(),
                                        result.n_rows()) );
}
void mm_tt2_cublas(      Dense_matrix<double> &T1,
                        Dense_matrix<double> &T2,
                        Dense_matrix<double> &result,
                        cublasHandle_t cublasH)
{
	double my_scalar = 0., other_scalar = 1.;

                CUBLAS_CHECK(
                        cublasDgemm(    cublasH,
                                        CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        T1.n_cols(),
                                        T1.n_cols(),
                                        T1.n_rows(),
                                        &other_scalar,
                                        T1.data(),
                                        T1.n_rows(),
                                        T2.data(),
                                        T1.n_rows(),
                                        &my_scalar,
                                        result.data(),
                                        result.n_rows()) );
	my_scalar = 0.5, other_scalar = 0.5;
         CUBLAS_CHECK(
                        cublasDgemm(    cublasH,
                                        CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        T1.n_cols(),
                                        T1.n_cols(),
                                        T1.n_rows(),
                                        &other_scalar,
                                        T2.data(),
                                        T1.n_rows(),
                                        T1.data(),
                                        T1.n_rows(),
                                        &my_scalar,
                                        result.data(),
                                        result.n_rows()) );
}
/////////////////////////////////////////////////////////

template< typename type_t>
__global__ void divide_diagonal_by_two( unsigned int n_rows,
                                type_t *mat)
{
        const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
        if ( tid < n_rows )
        {
                mat[tid + n_rows * tid] /= 0.5;
        }
}


void mm_tt2_cublas_v2(	Dense_matrix<double> &T1,
			Dense_matrix<double> &T2,
			Dense_matrix<double> &temp,
			Dense_matrix<double> &result,
			cublasHandle_t cublasH)

{
	double my_scalar = 0, other_scalar = 0.5;
        CUBLAS_CHECK(
                        cublasDsyr2k(   cublasH,
                                        CUBLAS_FILL_MODE_LOWER,
                                        CUBLAS_OP_T,
                                        T1.n_cols(),
                                        T1.n_rows(),
                                        &other_scalar,
                                        T1.data(),
                                        T1.n_rows(),
                                        T2.data(),
                                        T1.n_rows(),
                                        &my_scalar,
                                        temp.data(),
                                        T1.n_cols()));
	my_scalar = 1., other_scalar = 1.;
        CUBLAS_CHECK(
                        cublasDgeam(    cublasH,
                                        CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        temp.n_rows(),
                                        temp.n_rows(),
                                        &other_scalar,
                                        temp.data(),
                                        temp.n_rows(),
                                        &my_scalar,
                                        temp.data(),
                                        temp.n_rows(),
                                        result.data(),
                                        temp.n_rows()) );

        divide_diagonal_by_two<double><<<CEIL_DIV(result.n_rows(),32), 32>>>
                                        (result.n_rows(),
                                         result.data());
}


void mm_tt2_cublas_v2(     Dense_matrix<float> &T1,
                        Dense_matrix<float> &T2,
			Dense_matrix<float> &temp,
                        Dense_matrix<float> &result,
                        cublasHandle_t cublasH)

{
	float my_scalar = 0, other_scalar = 0.5;
        CUBLAS_CHECK(
                        cublasSsyr2k(   cublasH,
                                        CUBLAS_FILL_MODE_LOWER,
                                        CUBLAS_OP_T,
                                        T1.n_cols(),
                                        T1.n_rows(),
                                        &other_scalar,
                                        T1.data(),
                                        T1.n_rows(),
                                        T2.data(),
                                        T1.n_rows(),
                                        &my_scalar,
                                        temp.data(),
                                        T1.n_cols()));
	my_scalar = 1., other_scalar = 1.;
        CUBLAS_CHECK(
                        cublasSgeam(    cublasH,
                                        CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        temp.n_rows(),
                                        temp.n_rows(),
                                        &other_scalar,
                                        temp.data(),
                                        temp.n_rows(),
                                        &my_scalar,
                                        temp.data(),
                                        temp.n_rows(),
                                        result.data(),
                                        temp.n_rows()) );

        divide_diagonal_by_two<float><<<CEIL_DIV(result.n_rows(),32), 32>>>
                                        (result.n_rows(),
                                         result.data());
}
///////////////////////////////////////////////////
void mm_tt_cublas_v2(     
                        Dense_matrix<double> &T,
			Dense_matrix<double> &temp,
                        Dense_matrix<double> &result,
                        cublasHandle_t cublasH)
{
	double my_scalar = 0., other_scalar = 1.;
        CUBLAS_CHECK(
                        cublasDsyrkx(   cublasH,
                                        CUBLAS_FILL_MODE_LOWER,
                                        CUBLAS_OP_T,
                                        T.n_cols(),
                                        T.n_rows(),
                                        &other_scalar,
                                        T.data(),
                                        T.n_rows(),
                                        T.data(),
                                        T.n_rows(),
                                        &my_scalar,
                                        temp.data(),
                                        T.n_cols()));
        my_scalar = 1., other_scalar = 1.;
        CUBLAS_CHECK(
                        cublasDgeam(    cublasH,
                                        CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        temp.n_rows(),
                                        temp.n_rows(),
                                        &other_scalar,
                                        temp.data(),
                                        temp.n_rows(),
                                        &my_scalar,
                                        temp.data(),
                                        temp.n_rows(),
                                        result.data(),
                                        temp.n_rows()) );

        divide_diagonal_by_two<double><<<CEIL_DIV(result.n_rows(),32), 32>>>
                                        (result.n_rows(),
                                         result.data());
}
void mm_tt_cublas_v2(     
                        Dense_matrix<float> &T,
			Dense_matrix<float> &temp,
                        Dense_matrix<float> &result,
                        cublasHandle_t cublasH)
{
	float my_scalar = 0., other_scalar = 1.;
        CUBLAS_CHECK(
                        cublasSsyrkx( 	cublasH,
                                        CUBLAS_FILL_MODE_LOWER,
                                        CUBLAS_OP_T,
                                        T.n_cols(),
                                        T.n_rows(),
                                        &other_scalar,
                                        T.data(),
                                        T.n_rows(),
                                        T.data(),
                                        T.n_rows(),
                                        &my_scalar,
                                        temp.data(),
                                        T.n_cols()));
	my_scalar = 1., other_scalar = 1.;
        CUBLAS_CHECK(
                        cublasSgeam(    cublasH,
                                        CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        temp.n_rows(),
                                        temp.n_rows(),
                                        &other_scalar,
                                        temp.data(),
                                        temp.n_rows(),
                                        &my_scalar,
                                        temp.data(),
                                        temp.n_rows(),
                                        result.data(),
                                        temp.n_rows()) );

        divide_diagonal_by_two<float><<<CEIL_DIV(result.n_rows(),32), 32>>>
                                        (result.n_rows(),
                                         result.data());
}
/////////////////////////////////////////////////////

void vm_cublas(	Dense_matrix<float> &mat,
		Vector<float> &vec,
		Vector<float> &result,
		cublasHandle_t cublasH)
{
	float my_scalar = 0, other_scalar = 1;
	CUBLAS_CHECK(
		cublasSgemv(
				cublasH, 
				CUBLAS_OP_T,
 				mat.n_rows(), 
				mat.n_cols(),
 				&other_scalar,
 				mat.data(), 
				vec.size(),
 				vec.data(), 
				1,
 				&my_scalar,
 				result.data(), 
				1) );
}
void vm_cublas(        Dense_matrix<double> &mat,
                Vector<double> &vec,
                Vector<double> &result,
                cublasHandle_t cublasH)
{
        double my_scalar = 0, other_scalar = 1;
        CUBLAS_CHECK(
                cublasDgemv(
                                cublasH,
                                CUBLAS_OP_T,
                                mat.n_rows(),
                                mat.n_cols(),
                                &other_scalar,
                                mat.data(),
                                mat.n_rows(),
                                vec.data(),
                                1,
                                &my_scalar,
                                result.data(),
                                1) );
}
////////////////////////////////////////////////////////
void vec_add_cublas(	
			const double other_scalar,
			Vector<double> &vec1,
			Vector<double> &vec2,
			cublasHandle_t cublasH)
{
	CUBLAS_CHECK(
                        cublasDaxpy(    cublasH,
                                        vec1.size(),
                                        &other_scalar,
                                        vec1.data(),
                                        1,
                                        vec2.data(),
                                        1 ) );	
}
void vec_add_cublas(   
                        const float other_scalar,
                        Vector<float> &vec1,
                        Vector<float> &vec2,
                        cublasHandle_t cublasH)
{
        CUBLAS_CHECK(
                        cublasSaxpy(    cublasH,
                                        vec1.size(),
                                        &other_scalar,
                                        vec1.data(),
                                        1,
                                        vec2.data(),
                                        1 ) );
}
/////////////////////////////////////////////////////
void dot_cublas(        Vector<double> &vec1,
                        Vector<double> &vec2,
                        double *result,
                        cublasHandle_t cublasH)
{
        CUBLAS_CHECK(
                        cublasDdot(    cublasH,
                                        vec1.size(),
                                        vec1.data(),
                                        1,
                                        vec2.data(),
                                        1,
                                        result));
}

void dot_cublas(	Vector<float> &vec1,
			Vector<float> &vec2,
			float *result,
			cublasHandle_t cublasH)
{
	CUBLAS_CHECK(
                        cublasSdot(    cublasH,
                                        vec1.size(),
                                        vec1.data(),
                                        1,
                                        vec2.data(),
                                        1,
                                        result));
}
/////////////////////////////////////////////////////

void l2_norm_cublas(    Vector<double> &vec,
                        double *result,
                        cublasHandle_t cublasH)
{
         CUBLAS_CHECK(
                        cublasDnrm2(    cublasH,
                                        vec.size(),
                                        vec.data(),
                                        1,
                                        result));
}

void l2_norm_cublas(	Vector<float> &vec,
			float *result,
			cublasHandle_t cublasH)
{
	 CUBLAS_CHECK(
                        cublasSnrm2(    cublasH,
                                        vec.size(),
                                        vec.data(),
                                        1,
                                        result));
}
/////////////////////////////////////////////////////
void mult_scalar_cublas(        Vector<double> &vec,
                                const double scalar,
                                cublasHandle_t cublasH)
{
                CUBLAS_CHECK(
                        cublasDscal(    cublasH,
                                        vec.size(),
                                        &scalar,
                                        vec.data(),
                                        1));
}
void mult_scalar_cublas(	Vector<float> &vec,
				const float scalar,
				cublasHandle_t cublasH)
{
		CUBLAS_CHECK(
                        cublasSscal(    cublasH,
                                        vec.size(),
                                        &scalar,
                                        vec.data(),
                                        1));
}
/////////////////////////////////////////////////////////////////////////////////
//	<<<	This is used to obrain expm(T)	>>>

void expm_cusolver( Dense_matrix<double> &T) 
{
	cusolverDnHandle_t cusolverH = NULL;	
  	const int m = T.n_rows();
 	Dense_matrix<double> eigen_vec(T);
	Vector<double> eigen_val(m,T.memory_space());
    	int *info_cuda = nullptr;
    	int info = 0;
    	int lwork = 0;     	       	/* size of workspace */
    	double* work_cuda = nullptr; 	/* device workspace*/
    	CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    	AssertCuda(cudaMalloc(reinterpret_cast<void **>(&info_cuda), sizeof(int)));
    	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; 
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
   	 	CUSOLVER_CHECK(
			cusolverDnDsyevd_bufferSize(	cusolverH, 
							jobz, 
							uplo, 
							m, 
							eigen_vec.data(), 
							m, 
							eigen_val.data(), 
							&lwork)	);
    		AssertCuda(
		cudaMalloc(reinterpret_cast<void **>(&work_cuda), sizeof(double) * lwork));
   		CUSOLVER_CHECK(
       			cusolverDnDsyevd(	cusolverH, 
						jobz, 
						uplo, 
						m, 
						eigen_vec.data(), 
						m, 
						eigen_val.data(), 
						work_cuda, 
						lwork, 
						info_cuda) );
    	AssertCuda(
		cudaMemcpy(&info, info_cuda, sizeof(int), cudaMemcpyDeviceToHost));
    
    	if (0 > info) 
	{
        	std::printf("%d-th parameter is wrong \n", -info);
        	exit(1);
    	}
    	AssertCuda(cudaFree(info_cuda));
    	AssertCuda(cudaFree(work_cuda));
    	CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));	
    	T.custom_mult(eigen_val, eigen_vec);
}

//std::pair< Vector<float>, Dense_matrix<float> > 
void expm_cusolver( Dense_matrix<float> &T) 
{
	cusolverDnHandle_t cusolverH = NULL;	
  	const int m = T.n_rows();
 	Dense_matrix<float> eigen_vec(T);
	Vector<float> eigen_val(m,T.memory_space());
    	int *info_cuda = nullptr;
    	int info = 0;
    	int lwork = 0;     	       	
    	float* work_cuda = nullptr; 
    	CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    	AssertCuda(cudaMalloc(reinterpret_cast<void **>(&info_cuda), sizeof(int)));
    	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; 
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
   	 	CUSOLVER_CHECK(
			cusolverDnSsyevd_bufferSize(	cusolverH, 
							jobz, 
							uplo, 
							m, 
							eigen_vec.data(), 
							m, 
							eigen_val.data(), 
							&lwork)	);
	
    		AssertCuda(
		cudaMalloc(reinterpret_cast<void **>(&work_cuda), sizeof(float) * lwork));
   		CUSOLVER_CHECK(
       			cusolverDnSsyevd(	cusolverH, 
						jobz, 
						uplo, 
						m, 
						eigen_vec.data(), 
						m, 
						eigen_val.data(), 
						work_cuda, 
						lwork,
						info_cuda) ); 
    	AssertCuda(
		cudaMemcpy(&info, info_cuda, sizeof(int), cudaMemcpyDeviceToHost));
    
    	if (0 > info) 
	{
        	std::printf("%d-th parameter is wrong \n", -info);
        	exit(1);
    	}



    	AssertCuda(cudaFree(info_cuda));
    	AssertCuda(cudaFree(work_cuda));
    	CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));	

	T.custom_mult(eigen_val, eigen_vec); 
}

///////////////////////////////////////////////////////////////////////////////////
template <typename type_t>
__global__ void custom_mult2(	const type_t *eigenval,
				type_t *beta,
				type_t *beta_inv)
{
	const unsigned int row = threadIdx.x % N_COL ;
        const unsigned int col = threadIdx.x / N_COL ;
	__shared__ type_t vec[N_COL][N_COL+16];// N_COL is a compiler parameter
	__shared__ type_t val[N_COL];
	type_t sum1 = 0;
	type_t sum2 = 0;
	if ( threadIdx.x < N_COL*N_COL )
        {

		vec[col][row] = beta[row + N_COL *col];
	}
	if ( row < N_COL )
	{
		val[row] = eigenval[row];
	}
	__syncthreads();
	type_t fragment_row[N_COL];
	type_t fragment_col[N_COL];
	type_t fragment_val[N_COL];
	if ( threadIdx.x < N_COL*N_COL )
	{
		for (unsigned int i = 0; i < N_COL; ++ i)
		{
			fragment_row[i] = vec[i][row];
			fragment_col[i] = vec[i][col];
			fragment_val[i] = val[i];
		}
                for ( unsigned int i = 0; i < N_COL; ++i )
                {
			sum1 += fragment_row[i] * 
				std::sqrt(std::abs( fragment_val[i]) ) * 
				fragment_col[i];

			sum2 += fragment_row[i] *
                                1./(std::sqrt( std::abs( fragment_val[i] ) ) )*
                                fragment_col[i];
                }
		beta[row + N_COL * col] = sum1;
		beta_inv[row + N_COL * col] = sum2;
        }
}
/////////////////////////////////////////////////////////////////////////////////////
void sqrtm_cusolver(	Vector<float> &eigen_val,
			Dense_matrix<float> &beta,
			Dense_matrix<float> &beta_inv,
			cusolver_args<float> &args )
{
	CUSOLVER_CHECK(
                        cusolverDnSsyevjBatched(
                                                args.cusolverH,
                                                CUSOLVER_EIG_MODE_VECTOR,
                                                CUBLAS_FILL_MODE_LOWER,
                                                beta.n_cols(),
                                                beta.data(),
                                                beta.n_cols(),
                                                eigen_val.data(),
                                                args.syevj_work,
                                                args.lwork,
                                                args.syevj_info,
                                                args.syevj_params,
                                                1) );	
	//eigen_val.print();
	//beta.print();
	custom_mult2<float><<<1, beta.size()>>>
                (eigen_val.data(), beta.data(), beta_inv.data());
	
}
void sqrtm_cusolver(    Vector<double> &eigen_val,
			Dense_matrix<double> &beta,
			Dense_matrix<double> &beta_inv,
			cusolver_args<double> &args)
{

        CUSOLVER_CHECK(
                        cusolverDnDsyevjBatched(
                                                args.cusolverH,
                                                CUSOLVER_EIG_MODE_VECTOR,
                                                CUBLAS_FILL_MODE_LOWER,
                                                beta.n_cols(),
                                                beta.data(),
                                                beta.n_cols(),
                                                eigen_val.data(),
                                                args.syevj_work,
                                                args.lwork,
                                                args.syevj_info,
                                                args.syevj_params,
                                                1) );
	
        custom_mult2<double><<<1, beta.size()>>>
                (eigen_val.data(), beta.data(), beta_inv.data());

}
/////////////////////////////////////////////////////////////////////////////////////
void initiate_cusolver( cusolver_args<float> &args, 
			Dense_matrix<float> &eigen_vec,
			Vector<float> &eigen_val)
{
	const int max_sweeps = 7;
	CUSOLVER_CHECK(
                        cusolverDnCreateSyevjInfo(&args.syevj_params)        );
	CUSOLVER_CHECK(cusolverDnXsyevjSetMaxSweeps(args.syevj_params, max_sweeps));

        CUSOLVER_CHECK(
                        cusolverDnCreate(&args.cusolverH)	);

        AssertCuda(
			cudaMalloc(reinterpret_cast<void **>(&args.syevj_info), sizeof(int)) );

        CUSOLVER_CHECK( 
			cusolverDnSsyevjBatched_bufferSize(
        				args.cusolverH, 
					CUSOLVER_EIG_MODE_VECTOR, 
					CUBLAS_FILL_MODE_LOWER, 
					eigen_vec.n_cols(),
        				eigen_vec.data(), 
					eigen_vec.n_cols(),
					eigen_val.data(),
					&args.lwork,
					args.syevj_params,
                                        1       )	);
	AssertCuda(
                cudaMalloc(reinterpret_cast<void **>(&args.syevj_work), sizeof(float) * args.lwork));
}
void initiate_cusolver( cusolver_args<double> &args,
                        Dense_matrix<double> &eigen_vec,
                        Vector<double> &eigen_val)
{
	const int max_sweeps = 15;
        CUSOLVER_CHECK(
                        cusolverDnCreateSyevjInfo(&args.syevj_params)        );
	CUSOLVER_CHECK(cusolverDnXsyevjSetMaxSweeps(args.syevj_params, max_sweeps));
        CUSOLVER_CHECK(
                        cusolverDnCreate(&args.cusolverH)       );
	
        AssertCuda(
                        cudaMalloc(reinterpret_cast<void **>(&args.syevj_info), sizeof(int)) );

        CUSOLVER_CHECK(
                        cusolverDnDsyevjBatched_bufferSize(
                                        args.cusolverH,
                                        CUSOLVER_EIG_MODE_VECTOR,
                                        CUBLAS_FILL_MODE_LOWER,
                                        eigen_vec.n_cols(),
                                        eigen_vec.data(),
                                        eigen_vec.n_cols(),
                                        eigen_val.data(),
                                        &args.lwork,
                                        args.syevj_params,
                                        1       )       );
        AssertCuda(
                cudaMalloc(reinterpret_cast<void **>(&args.syevj_work), sizeof(double) * args.lwork));
}

/////////////////////////////////////////////////////////////////////////////////////
double sqrtm_cusolver_time( Dense_matrix<float> &T) 
{
	cusolverDnHandle_t cusolverH = NULL;	
 	//cudaStream_t stream = NULL;
 
 	//Dense_matrix<float> eigen_vec(T);
	Vector<float> eigen_val(T.n_rows(),T.memory_space());
	Dense_matrix<float> beta(T);
	Dense_matrix<float> beta_inv(T);

    	int *syevj_info = nullptr;
    	int info = 0;
    	int lwork = 0;     	      
    	float *d_work = nullptr; 	
	//const double tol = 1.e-7;
    	//const int max_sweeps = 150;
    	//const int sort_eig = 0;
	syevjInfo_t syevj_params = NULL;
	CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&syevj_params));
	//CUSOLVER_CHECK(cusolverDnXsyevjSetTolerance(syevj_params, tol));
    	//CUSOLVER_CHECK(cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps));
    	//CUSOLVER_CHECK(cusolverDnXsyevjSetSortEig(syevj_params, sort_eig));
    	CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

	//AssertCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    	//CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

	cudaDeviceSynchronize();
        auto t1 = std::chrono::steady_clock::now();

    	AssertCuda(cudaMalloc(reinterpret_cast<void **>(&syevj_info), sizeof(int)));
    	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; 
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
		CUSOLVER_CHECK(
			cusolverDnSsyevjBatched_bufferSize(
    					cusolverH,
    					jobz,
    					uplo,
 					beta.n_rows(),
 					beta.data(),
 					beta.n_rows(),
 					eigen_val.data(),
 					&lwork,
    					syevj_params,
 					1	)
    							);
	
    		AssertCuda(
		cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(float) * lwork));

   		CUSOLVER_CHECK(
       			cusolverDnSsyevjBatched(	
						cusolverH, 
						jobz, 
						uplo, 
						beta.n_rows(),		 
						beta.data(), 	
						beta.n_rows(), 		
						eigen_val.data(), 	
						d_work, 
						lwork,
						syevj_info,
						syevj_params,
						1) );
	
 	custom_mult2<float><<<1, T.size()>>>
		(eigen_val.data(), beta.data(), beta_inv.data());

	cudaDeviceSynchronize();
        double time = std::chrono::duration_cast<std::chrono::duration<double>>(
                                        std::chrono::steady_clock::now() - t1)
                                        .count();

    	AssertCuda(
		cudaMemcpy(&info, syevj_info, sizeof(int), cudaMemcpyDeviceToHost));
    	std::printf("after syevd: info = %d\n", info);
    	if (0 > info) 
	{
        	std::printf("%d-th parameter is wrong \n", -info);
        	exit(1);
    	}
	CUSOLVER_CHECK(cusolverDnDestroySyevjInfo(syevj_params));
    	AssertCuda(cudaFree(syevj_info));
    	AssertCuda(cudaFree(d_work));
    	CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));	
    	//return std::make_pair(eigen_val, eigen_vec);
    	return time;
}
/////////////////////////////////////////////////////////////////////////////////////
#endif


#endif
