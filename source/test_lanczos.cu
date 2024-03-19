



#define N_COL 4
#define USE_BLAS true

#include "utils/common.hpp"
#include "utils/lib_utils.hpp"
#include "methods/vector_lanczos.hpp"
#include "methods/block_lanczos.hpp"
#include "methods/fdtd.hpp"

#include "matrix_a/build_A_ell.hpp"
#include "objects/tridiagonal_matrix.hpp"


// needed only when running block lanczos with the libearies

template<typename type_t>
void test_VectorLanczos( 	unsigned int N, 
				unsigned int m,
				double T_end,
                        	unsigned int fdtd_steps,
                        	unsigned int lc)
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
	Vector<type_t> b_host = gaussian_vector_b<type_t>( N, n_rows);
	//Vector<type_t> b_host = random_vector_b<type_t>( n_rows);
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
	
	Vector<type_t> q(m,mem_cuda);	// here we store the krylov element of index lc
	Vector<type_t> q0(b);		// q0 and q1 are krylovs and w is a helper vector
        Vector<type_t> q1(b);
        Vector<type_t> w(b);
	/*	Here we store the 'm' alpha and 'm-1' beta blocks
         	beta[0] stores dot(b,b) that is needed later for the solution	*/
	type_t *alpha = new type_t[m];	
        type_t *beta = new type_t[m];

	cublasHandle_t cublasH;
        CUBLAS_CHECK( cublasCreate(&cublasH) );

#ifdef USE_BLAS		
		
		cudaDeviceSynchronize();
	        time.start();

		vector_lanczos_blas<type_t>(A, b, m, lc,  q, alpha, beta,
                                q0, q1, w, cublasH);

		cudaDeviceSynchronize();
		time.end();
		
#else
		cudaDeviceSynchronize();
		time.start();
		
                vector_lanczos<type_t>(A, b, m, lc,  q, alpha, beta,
                                q0, q1, w);

                cudaDeviceSynchronize();
		time.end();

#endif	
	std::cout << "elapsed time: " << std::setw(11) << time.duration() << std::endl;
	//file << n_rows << "," << m << "," << time << std::endl;

	/////////////////////////////////////////////////////////////////////////////////////
	// <<<  This part is irrelevent to the measuremnts of Lanczos and 
	//		is about computing the finalapproximate solution using T and Q >>>
	
	/*	compute the expm	*/
	Dense_matrix<type_t> T(m, m, mem_cuda);
	T.mult_scalar(T_end);
	expm_cusolver(T);

	/*	explain	*/
	Vector<type_t> e1(m, mem_cuda); 
	copy_column_to_vector<type_t>(T, e1, 0);	
	type_t solution = e1.dot(q);
	solution = beta[0]*solution;

	std::cout << "The solution for vector_lanczos " << std::endl;
	std::cout << solution << std::endl;


	/////////////////////////////////////////////////////////////////////////////
	
	type_t fdtd_solution =  fdtd_vector(A, b, 100000., 1, lc);
	std::cout << "Solution from fdtd " << fdtd_solution << std::endl;
	
	std::cout << "Relative error for block lanczos is "<< 
			std::abs(solution - fdtd_solution)/std::abs(fdtd_solution) 
			<< std::endl;

	CUBLAS_CHECK( cublasDestroy ( cublasH ) );
	
}



template <typename type_t>
void test_BlockLanczos( unsigned int N, 
			unsigned int m, 
			double T_end, 
			unsigned int fdtd_steps, 
			unsigned int lc)
{	
	MemorySpace mem_cuda = MemorySpace::CUDA;

	steady_clock time = steady_clock();

	auto info = Matrix_A<type_t>(N,N,N);
	Ell_matrix<type_t> D_host = info.first; // D = [0 Dh; De 0]
	Ell_matrix<type_t> W_host = info.second;
	
	const unsigned int n_rows = D_host.n_rows();
	
	std::cout << " the size of the problem is " << std::endl; print(n_rows);

	/* Matrix B is besically agrogomation of columns each represent
		a three dimensional gaussian in x direction with different shift
		 this cause I was making simulation in matlab with it
		so could compare	 
	*/
	
	//Dense_matrix<type_t> B_host = gaussian_matrix_B<type_t>( N, n_rows);
	Dense_matrix<type_t> B_host = random_matrix_B<type_t>( n_rows);
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
	
#ifdef USE_BLAS
		cusolver_args<type_t> args = cusolver_args<type_t>();
                Vector<type_t> eigen_val(N_COL, mem_cuda);
                initiate_cusolver(args, beta[0], eigen_val);

	
		cudaDeviceSynchronize();
	        time.start();

		block_lanczos_blas<type_t>(A, B, m, lc, q, alpha, beta,
                         	Q0, Q1, W,
                        	args, eigen_val, cublasH,
				n_blocks, n_loads);

		cudaDeviceSynchronize();
		time.end();
			

		CUSOLVER_CHECK( cusolverDnDestroySyevjInfo(args.syevj_params) );
                CUSOLVER_CHECK( cusolverDnDestroy(args.cusolverH) );
#else
		cudaDeviceSynchronize();
                time.start();

		block_lanczos(A, B, m, lc, q, alpha, beta,
				 Q0, Q1, W,
				n_blocks, n_loads);
		
		cudaDeviceSynchronize();
        	time.end();
#endif

	std::cout << " end Lanczos " << std::endl;
        ///////////////////////////////////////////////////////////////////////////////
	// <<< This part is irreelevent to the measurment of Lanczos and 
	//	is about computing the approximate solution using T and Q >>>

	//	compute expm(T)	

	Dense_matrix<type_t> T = Assemble_T(m,alpha,beta);	
	T.mult_scalar(T_end);
	expm_cusolver(T);
	
	//	explain		
	Dense_matrix<type_t> F1(m*N_COL, N_COL, mem_cuda);
	copy_columns_to_matrix<type_t>(T, F1, N_COL);

	// F1 = expm(T)*F1*sqrtm(B*B')
	mm_cublas(0,1.,F1,beta[0],F1, cublasH);
	
	Vector<type_t> solution(N_COL, mem_cuda);

	// -->
	vm_cublas(F1, q, solution, cublasH);
	//F1.mv(q, solution);		// solution = q*expm(T)*F1*sqrtm(B*B')
	std::cout << "Solution for block lanczos";
	solution.print();

	/////////////////////////////////////////////////////////////////	
	std::cout << " start fdtd " << std::endl;

	Vector<type_t> fdtd_solution =  ftdt_block<type_t>(A, B, fdtd_steps, T_end, lc);
	std::cout << "Solution from fdtd ";
	fdtd_solution.print();

	//	compute the relative error	
	solution.sadd(1,-1, fdtd_solution);
	type_t relative_error = solution.l2_norm()/fdtd_solution.l2_norm();
	std::cout << "Relative error for block lanczos is "<< relative_error << std::endl;	
	
	CUBLAS_CHECK( cublasDestroy ( cublasH ) );			
			
}

////////////////////////////////////////////////////////////////////////////////////////////////////


int main(int argc, char **argv)
{
	if (argc % 2 == 0)
	{
		std::cout << "Error, expected odd number of common line arguments"
                << std::endl
                << "Expected line of the form" << std::endl
                << " -blas -1 -N 10" << std::endl
	 	<< " -blas to indeicate using the cu libearies, N the number of primary points in Lee grid in one of the dimensions. The memroy limit is around 180 for block lanczos of 18 vectors " << std::endl;
      		std::abort();
	}
	
	bool use_block = true;
	unsigned int N = 10;
	unsigned int m = 5;
	double T_end = 1;
	unsigned int lc =  1+ (rand() % 100); // location of interest which is the a random index of 
					      // solution of vector lanczos or random row index of
					      // of the matrix solution in block lanczos
					      // in both case we are not computing the full solution
					      // but only the value of interest
	// for block lanczos lc detones a full row , meaning same index for all columns of the 
	// solution, the code need to be slightly changed if I am interested in different lc for every
	// column. 
						
	
	unsigned int fdtd_steps = 1000000;

	for (int l = 1; l < argc; l += 2)
	{
		std::string option = argv[l];
		
		if (option == "-N") N = static_cast<unsigned int>(std::stod(argv[l + 1]));
		if (option == "-m") m = static_cast<unsigned int>(std::stod(argv[l + 1]));
		
	}
	
	//print(N);

	if ( use_block )
	{
		test_BlockLanczos<double>( N, m, T_end, fdtd_steps,lc);
	}
	else
	{
		test_VectorLanczos<float>( N, m, T_end, fdtd_steps, lc);
	}
	//CudaDeviceInfo();

	
	return 0;

}
