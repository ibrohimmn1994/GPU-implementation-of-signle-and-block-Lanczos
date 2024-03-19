
#ifndef block_lanczos_hpp
#define block_lanczos_hpp

#include "copy_functions.hpp"
#include "../kernels/my_sqrtm_cusolver.hpp"
#include "../kernels/mm_tt2.hpp"
#include "../kernels/mm_ts.hpp"
#include "../kernels/mm_tt.hpp"
#include "../kernels/spmv_spmm.hpp"


void block_lanczos(	Ell_matrix<float> &A, 
			Dense_matrix<float> &B, 
			const unsigned int m,
			unsigned int lc,
			Vector<float> &q,
			Dense_matrix<float> *alpha,
			Dense_matrix<float> *beta,
			Dense_matrix<float> &Q0,
			Dense_matrix<float> &Q1,
			Dense_matrix<float> &W,
			const unsigned int n_blocks,
                        const unsigned int n_loads)
{			
		
	// beta = B'*B 
	mm_tt(n_blocks,n_loads, B, beta[0]);
		
	//beta = sqrtm(B'*B), beta[m] = inv(sqrtm(B'*B))
	my_sqrtm_cusolver<float>(beta[0], beta[m]);
							
	//Q0 = B * inv(sqrtm(B'*B))
	mm_ts(n_blocks,n_loads, B, beta[m], Q0);	

	// Q(0:length(Q0)) = Q0(lc,:)
	copy_row_to_vector<float>(lc, 0, Q0, q);
						
	// W = A*Q0
	spmm(A,Q0,W);				
					
	// alpha = 0.5*W'*Q0 + 0.5* Q0'*W
	mm_tt2(n_blocks, n_loads, W, Q0, alpha[0]);
	
	// // W = W - Q0*alpha
	mm_ts(n_blocks,n_loads,1, -1, Q0, alpha[0], W);		
		
	unsigned int j = 0;
	while( j <  m-1 )
	{
		++j;

		//beta = W'*W 
		mm_tt(n_blocks,n_loads, W, beta[j]);
	
		// beta = sqrtm(W'*W), beta[m] = inv(sqrtm(W'*W))
		my_sqrtm_cusolver<float>(beta[j], beta[m]);	

		// Q1 = W*beta_inv
		mm_ts(n_blocks,n_loads, W, beta[m], Q1);

		// W = A*Q1
		spmm(A,Q1,W);
		
		//  W = W - Q0*beta
		mm_ts(n_blocks,n_loads,1,-1, Q0, beta[j], W);	
						
		// alpha = 0.5*W'*Q1 + 0.5*Q1'*W
		mm_tt2( n_blocks, n_loads, W, Q1, alpha[j]);

		// W = W - Q1*alpha
		mm_ts(n_blocks,n_loads,1,-1, Q1, alpha[j], W);	

		Q0 = Q1;

		// Q(j*n:j*n+length(Q0)) = Q0(lc,:)
		copy_row_to_vector<float>(lc, j*N_COL, Q0, q);	
	}
			
}
/////////////////////////////////////////////////////////////////////////////////////

//  In the library based block lanczos I replace only mm_tt and m_tt2 kernels with my own as 
//  these are the only ones that perfromed considerably better



template< typename type_t>
void block_lanczos_blas(	Ell_matrix<type_t> &A, 
				Dense_matrix<type_t> &B, 
				const unsigned int m,
				unsigned int lc,
				Vector<type_t> &q,
				Dense_matrix<type_t> *alpha,
				Dense_matrix<type_t> *beta,
				Dense_matrix<type_t> &Q0,
				Dense_matrix<type_t> &Q1,
				Dense_matrix<type_t> &W,
				cusolver_args<type_t> &args,
				Vector<type_t> eigen_val,
				cublasHandle_t cublasH,
				const unsigned int n_blocks,
                        	const unsigned int n_loads)
{							// << kernel name in cublas >>
	//  beta[0] = full(B'*B)		
	mm_tt_cublas(B, beta[0], cublasH);				//option2: gemm OP_T	
	//mm_tt(n_blocks, n_loads, B, beta[0]);				// option3: own kernel
	
	//beta[0] = sqrtm(B'*B), beta[m] = inv(sqrtm(B'*B))
	//my_sqrtm_cusolver<type_t>(beta[0], beta[m]);
	sqrtm_cusolver(eigen_val, beta[0], beta[m], args);		//syevjBatched
	
	//Q0 = B * inv(sqrtm(B'*B))
	mm_cublas(0., 1., B, beta[m], Q0, cublasH);			//gemm OP_N
	//mm_ts(n_blocks,n_loads, B, beta[m], Q0);

	// Q(0:length(Q0)) = Q0(lc,:)
	copy_row_to_vector<type_t>(lc, 0, Q0, q);			// own kernel
					
	// W = A*Q0
	spmm(A,Q0,W);							//own kernel		

	//  alpha[0] = full(0.5*W'*Q0 + 0.5* Q0'*W)
	mm_tt2_cublas(W, Q0, alpha[0], cublasH);			//option2:  2 gemm OP_T
	//mm_tt2(n_blocks, n_loads,W,Q0, alpha[0]);			//option3: own kernel
	
	// // W = W - Q0*alpha		
	mm_cublas(1, -1, Q0, alpha[0], W, cublasH);			//gemm OP_N
	//mm_ts(n_blocks,n_loads,1, -1, Q0, alpha[0], W);

	unsigned int j = 0;
	while( j <  m-1 )
	{
		
		++j;
		// beta[j] = full(W*W') 
		mm_tt_cublas(W, beta[j], cublasH);			//option2:  gemm OP_T
		//mm_tt(n_blocks, n_loads, W, beta[j]);			//option3: own kernel

		// beta[j] = sqrtm(W'*W), beta[m] = inv(sqrtm(W'*W))
		//my_sqrtm_cusolver<type_t>(beta[0], beta[m]);
		sqrtm_cusolver(eigen_val, beta[j], beta[m], args);	//syevjBatched
	
		// Q1 = W*temp
		mm_cublas(0, 1, W, beta[m], Q1, cublasH);		//gemm OP_N
		//mm_ts(n_blocks,n_loads, W, beta[m], Q1);

		// W = A*Q1
		spmm(A,Q1,W);						//own kernel
		
		//  W = W - Q0*beta
		mm_cublas(1, -1, Q0, beta[j], W, cublasH);		//gemm OP_N
				
		// alpha[j] = full(0.5*W'*Q1 + 0.5*Q1'*W)
		mm_tt2_cublas(W, Q1, alpha[j], cublasH);		//option2: 2 gemm OP_T
		//mm_tt2(n_blocks, n_loads, W, Q1, alpha[j]);		// option3: own kernel

		// W = W - Q1*alpha	
		mm_cublas(1, -1, Q1, alpha[j], W, cublasH);		//gemm OP_N
		//mm_ts(n_blocks,n_loads,1,-1, Q1, alpha[j], W);

		Q0 = Q1;						//memcpy

		// Q(j*n:j*n+length(Q0)) = Q0(lc,:)
		copy_row_to_vector<type_t>(lc, j*N_COL, Q0, q);		// own kernel
	}	
}

#endif
