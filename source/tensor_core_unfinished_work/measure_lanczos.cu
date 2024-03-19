



#define N_COL 18

//#define USE_DOUBLE

#include "utils/common.hpp"
#include "utils/lib_utils.hpp"
#include "methods/vector_lanczos.hpp"
#include "methods/block_lanczos.hpp"
#include "methods/fdtd.hpp"

#include "matrix_a/build_A_ell.hpp"
#include "objects/tridiagonal_matrix.hpp"


// needed only when running block lanczos with the libearies

template<typename type_t>
void test_VectorLanczos(const unsigned int N, std::ofstream& file, bool USE_BLAS)
{
	

	MemorySpace mem_cuda = MemorySpace::CUDA;  
	steady_clock time = steady_clock();
	// Assemble matrix A
	// ---------	
	auto info = Matrix_A<type_t>(N,N,N);
	Ell_matrix<type_t> D_host = info.first;
	Ell_matrix<type_t> W_host = info.second;
	
	const unsigned int n_rows = D_host.n_rows();
	// Assembale vector b	
	Vector<type_t> b_host = vector_b<type_t>( N, n_rows);

	/*
		- D = D*W this is just to make it symmetric
		- W is a diagonal matrix here
	*/
	D_host.mult_diagonal(W_host); 
	/* convert to new order to enable vectorized copy */  
        D_host.change_order(4);  
	/*We define A = D*W as the ell matrix*/
        Ell_matrix<type_t> A = D_host.copy_to_device(); 
        Vector<type_t> b = b_host.copy_to_device();
	
	///////////////////////////////////////////////////////////////////////////////////////////
	//  <<<  Prepare the memory required for vector Lanczos  >>>
        // 	 This is all the memory consumed by the algorithm
	
	/*	 location of interest or reciver location =>
                 instead of storing krylov subspaces we store only
                 the row demoted by the index lc in every iteration	*/
	unsigned int m = 2;	// number of iteration
	unsigned int lc = 0; 	// solution's location of interest

	Vector<type_t> q(m,mem_cuda);	// here we store the krylov element of index lc
	Vector<type_t> q0(b);		// q0 and q1 are krylovs and w is a helper vector
        Vector<type_t> q1(b);
        Vector<type_t> w(b);
	/*	Here we store the 'm' alpha and 'm-1' beta blocks
         	beta[0] stores dot(b,b) that is needed later for the solution	*/
	type_t *alpha = new type_t[m];	
        type_t *beta = new type_t[m];

	std::cout << " start Lanczos " << std::endl;
        double best = 1e10;
	if ( USE_BLAS )
	{
		cublasHandle_t cublasH;  // in case run the whole thing on cublas
        	CUBLAS_CHECK( cublasCreate(&cublasH) );
		
		for (unsigned int t = 0; t< 20; ++t)
		{
			cudaDeviceSynchronize();
	        	time.start();

			vector_lanczos_blas<type_t>(A, b, m, lc,  q, alpha, beta,
                                q0, q1, w, cublasH);

			cudaDeviceSynchronize();
			time.end();
			best = std::min(best, time.duration());
		}	
		
		CUBLAS_CHECK( cublasDestroy ( cublasH ) );
	}
	else
	{
		cudaDeviceSynchronize();
		time.start();
		
                vector_lanczos<type_t>(A, b, m, lc,  q, alpha, beta,
                                q0, q1, w);

                cudaDeviceSynchronize();
		time.end();
	}	
	double BW_mult_scalar = 1e-9*( (2*b.memory_consumption() + sizeof(type_t) ));
        double GFLOPS_mult_scalar = 1e-9*( b.size() ) ;

        double BW_dot = 1e-9*( (b.memory_consumption()*2 ) + sizeof(type_t));
        double GFLOPS_dot = 1e-9*( 2*b.size() );

        double BW_add = 1e-9*( b.memory_consumption()*3 );
        double GFLOPS_add = 1e-9*( b.size()  );

        double BW_spmv =  1e-9*( (A.memory_consumption() + 2*b.memory_consumption() ));
        double GFLOPS_spmv = 1e-9*( ( 2*A.n_rows()*A.width()  )  );

	double BW_copy_element = 1e-9* sizeof(type_t);
	double BW_memcpy = 1e-9* (2*b.memory_consumption());

	std::cout << "n_rows " << A.n_rows() << std::endl;
	std::cout << "elapsed time: " << std::setw(11) << best << std::endl;
	std::cout << "Bw: " << (2*( 2*BW_dot + BW_mult_scalar + BW_spmv + 
					BW_memcpy + BW_copy_element) + 3*BW_add)/best << std::endl;
	std::cout << "GFLOPS: " <<  (2*( 2*GFLOPS_dot + GFLOPS_mult_scalar + GFLOPS_spmv ) 
						+ 3*GFLOPS_add)/best << std::endl;

	
	file  << A.n_rows() << "," << best << "," <<
		(2*( 2*BW_dot + BW_mult_scalar + BW_spmv + BW_memcpy + BW_copy_element) + 3*BW_add)/best
		<< "," <<  (2*( 2*GFLOPS_dot + GFLOPS_mult_scalar + GFLOPS_spmv )+ 3*GFLOPS_add)/best
		<< std::endl;
	

	/////////////////////////////////////////////////////////////////////////////////////
	
}



template <typename type_t>
void test_BlockLanczos(const unsigned int N, std::ofstream& file, bool USE_BLAS)
{	
	MemorySpace mem_cuda = MemorySpace::CUDA;
	steady_clock time = steady_clock();

	auto info = Matrix_A<type_t>(N,N,N);
	Ell_matrix<type_t> D_host = info.first; // D = [0 Dh; De 0]
	Ell_matrix<type_t> W_host = info.second;
	
	const unsigned int n_rows = D_host.n_rows();

	/* Matrix B is besically agrogomation of columns each represent
		a three dimensional gaussian in x direction with different shift
		 this cause I was making simulation in matlab with it
		so could compare	 
	*/	
	Dense_matrix<type_t> B_host = matrix_B<type_t>( N, n_rows);

	//////////////////////////////////////////////////////////////////////////////////	
	// <<< approperiate/prepare both A & B to match my device kernels mm_tt, mm_tt2, mm_ts >>>
	 unsigned int n_blocks, n_loads, pads;
	
		int deviceId;
        	AssertCuda( cudaGetDevice(&deviceId) );
        	cudaDeviceProp props{};
        	AssertCuda( cudaGetDeviceProperties(&props, deviceId) );		
		/*
	 	- the kernels involving tall skinny matrix multiplication use following parameters
	   		warp_size = 32, float4 loading = 4, number of warps = 6
	 	- n_blocks: number of blocks to use, n_loads: number of iteration,
	   		pads: number of padding rows
	 	- This for float. for double might not be safe I need to revisit register and
			shared memory usage
		*/ 
		if (n_rows < (6 * 4 * 32*props.multiProcessorCount) )
		{
			n_loads = 1;
			pads = 6 * 4 * 32;
			n_blocks = CEIL_DIV(n_rows,pads) ;
			B_host.padding(pads);	
		}
		else
		{
			n_blocks = props.multiProcessorCount;
			pads = (6 * 4 * 32 * props.multiProcessorCount);
			n_loads =  CEIL_DIV(n_rows,pads);
			B_host.padding(pads);
		}
	
	/*	D = D*W this is just to make it symmetric	
		W is a diagonal matrix here	*/
	D_host.mult_diagonal(W_host); 	
	/*	convert to new order to enable vectorized copy	*/
        D_host.change_order(4);	
	/*	We define A = D*W as the ell matrix	*/
	Ell_matrix<type_t> A = D_host.copy_to_device();	
	Dense_matrix<type_t> B = B_host.copy_to_device();

	//////////////////////////////////////////////////////////////////////////////
	//  <<<  Prepare the memory required for block Lanczos  >>>
	// 	-> This is all the memory consumed by the algorithm
	
	/*	instead of storing krylov subspaces we store only
                the row demoted by the index lc in every iteration	*/
	unsigned int m = 2;	// number of iteration
	unsigned int lc = 0; 	// location of interest or reciver location
				
	Vector<type_t> q(m*N_COL, mem_cuda);	// here we store the lc rows
	Dense_matrix<type_t> Q0(B);		// Q0 and Q1 are krylovs and W is a helper matrix
        Dense_matrix<type_t> Q1(B);		
        Dense_matrix<type_t> W(B);

	

	// beta[0] used to store sqrtm(B*B') that is used down in the end
	// beta[m] used to store inv(sqrtm(Qi*Qi')) in every iteration
	Dense_matrix<type_t> *alpha = new Dense_matrix<type_t>[m ];
	Dense_matrix<type_t> *beta = new Dense_matrix<type_t>[m + 1];

	for (unsigned int  i = 0 ; i < m ; ++i)
	{
		alpha[i] = Dense_matrix<type_t>(N_COL,N_COL, mem_cuda);
		beta[i] = Dense_matrix<type_t>(N_COL,N_COL, mem_cuda);	
	}
	beta[m] = Dense_matrix<type_t>(N_COL,N_COL, mem_cuda);
	cublasHandle_t cublasH;  // for matrix-matrix multiplication down
        CUBLAS_CHECK( cublasCreate(&cublasH) );
	
	///////////////////////////////////////////////////////////////////////////////
	//	<<<	Block Lanczos method	>>>	
	
	// there are so many arguments cause I dont wanna time the construction phase
	std::cout << " start Lanczos " << std::endl;
	double best = 1e10;
	if ( USE_BLAS )
	{
		cusolver_args<type_t> args = cusolver_args<type_t>();
                Vector<type_t> eigen_val(N_COL, mem_cuda);
                initiate_cusolver(args, beta[0], eigen_val);

		for (unsigned int t = 0; t<20; ++t)
		{
			cudaDeviceSynchronize();
	        	time.start();

			block_lanczos_blas<type_t>(A, B, m, lc, q, alpha, beta,
                         	Q0, Q1, W,
                        	args, eigen_val, cublasH,
				n_blocks, n_loads);

			cudaDeviceSynchronize();
			time.end();
			best = std::min(best, time.duration());	
		}

		CUSOLVER_CHECK( cusolverDnDestroySyevjInfo(args.syevj_params) );
                CUSOLVER_CHECK( cusolverDnDestroy(args.cusolverH) );
	}
	else
	{
		cudaDeviceSynchronize();
                time.start();

		block_lanczos<type_t>(A, B, m, lc, q, alpha, beta,
				 Q0, Q1, W,
				n_blocks, n_loads);
		
		cudaDeviceSynchronize();
        	time.end();
	}
	
	double BW_mm_ts = 1e-9*( (B.memory_consumption()*2 + beta[0].memory_consumption() ));
	double GFLOPS_mm_ts = 1e-9*( ( 2*B.n_rows()*N_COL*N_COL  )  ) ;
	double BW_mm_tt = 1e-9*( (B.memory_consumption()*2 + beta[0].memory_consumption() ));
	double GFLOPS_mm_tt = 1e-9*( ( 2*B.n_rows()*N_COL*N_COL  )  );
	double BW_mm_tt2 = 1e-9*( (B.memory_consumption()*2 + beta[0].memory_consumption() )*2);
	double GFLOPS_mm_tt2 = 1e-9*( ( 4*B.n_rows()*N_COL*N_COL + N_COL*N_COL )  );
	double BW_spmm =  1e-9*( (A.memory_consumption() + 2*B.memory_consumption() ));
	double GFLOPS_spmm = 1e-9*( ( 2*A.n_rows()*A.width()*N_COL  )  );
	// in sqrtm also account for inv(sqrtm)
	double BW_sqrtm = 1e-9*(5*beta[0].memory_consumption() + 2*N_COL*sizeof(type_t));
	double GFLOPS_sqrtm = 1e-9*( 7 * (6*std::pow(N_COL,3) + 
					81*std::pow(N_COL,2) - 87*N_COL) + // jacobi
					2*(std::pow(N_COL,5)) ); 	// Op(U*v*U') twice
									
	double BW_memcopy = 1e-9* 2*B.memory_consumption();
	double BW_copy_row = 1e-9* 2*N_COL*sizeof(type_t);

	std::cout << " n_col " << N_COL << std::endl;
	std::cout << "n_rows " << A.n_rows() << std::endl;
	std::cout << "elapsed time: " << std::setw(11) << time.duration() << std::endl;
	std::cout << "BW: " << 2*( 2*BW_mm_ts +  BW_mm_tt +  BW_mm_tt2 + BW_spmm +  BW_sqrtm +
				BW_memcopy +  0.5*BW_copy_row)/best << std::endl;

	std::cout << "GFLOPS:  " << 2*( 2*GFLOPS_mm_ts + GFLOPS_mm_tt + GFLOPS_mm_tt2 +
				GFLOPS_spmm + GFLOPS_sqrtm )/best << std::endl; 



	file << N_COL <<"," << A.n_rows() << "," << best << "," <<
	2*( 2*BW_mm_ts +  BW_mm_tt +  BW_mm_tt2 + BW_spmm +  BW_sqrtm +
                                BW_memcopy +  0.5*BW_copy_row)/best << "," <<
	2*( 2*GFLOPS_mm_ts + GFLOPS_mm_tt + GFLOPS_mm_tt2 +
                                GFLOPS_spmm + GFLOPS_sqrtm )/best << std::endl;
	
        ///////////////////////////////////////////////////////////////////////////////
	
	CUBLAS_CHECK( cublasDestroy ( cublasH ) );			
			
}

////////////////////////////////////////////////////////////////////////////////////////////////////


int main(int argc, char **argv)
{

	bool use_blas = false;
	std::string option = argv[1];
	if (option == "-blas")	use_blas =  std::atoi(argv[2]);
	else std::cout << " unknown option " << std::endl;
		
	//CudaDeviceInfo();	

	std::ofstream file;
	file.open("file_VL.csv");

	//const unsigned int N = 160;
	//test_VectorLanczos<float, 0>(N, file);
	//test_BlockLanczos<float>(N, file, use_blas);
	

	for ( unsigned int n = 10; n < 200; n = (1 + n * 1.1) )
	{
		n = (n + 4) / 5 * 5;
		test_VectorLanczos<float>(n, file, use_blas);
	//	 test_BlockLanczos<float>(n, file, use_blas);
	}
	

	


	file.close();
	return 0;

}
