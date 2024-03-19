

/*
#include "../utils/cuda_utils.hpp"

#include "vector.hpp"
#include "dense_matrix.hpp"

#define N_COL 4
*/

namespace sqrtm{

// done by N_COL threads reduced in every iteration
template< typename type_t>
__device__ void Tridiagonalize(	const unsigned int tid,
				const unsigned int sgn,
				const unsigned int MASK,
				type_t (&A)[N_COL][N_COL+1],
				type_t *diag,
				type_t *upper_diag,
				type_t *u,
				type_t *v,
				type_t (&w)[N_COL][N_COL+1])
			
{
	unsigned int r,c;
	// flags determines if the thread does meaningful work
	// this is to avoid work inbalance and ifing out threads in every operation 
	int flag1 = 1;
	int flag2 = sgn * 1;//(tid/(1) - std::abs((int)(tid-(1)))/(1));
	for (unsigned int i = 0 ; i < N_COL - 2; ++i ) 
	{
		type_t dot_u1 = 0.; 
		int flag3 = sgn * (tid/(i+2) - std::abs((int)(tid-(i+2)))/(i+2)); // from i+2
		//////////////////////////////////////////////////////
		// u(j<i+1) = 0 v(j>=i+1) = A(j+N_COL*i)
		type_t ur = flag2 * A[i][tid];
		u[tid] = ur; 	
		dot_u1 = ur * ur;	
		// do shuflle reduction for dot_u1
#pragma unroll
		for ( r = N_COL/2; r > 0; r>>=1  )
			dot_u1 += __shfl_down_sync(MASK, dot_u1, r, N_COL);	
		// broadcast
		dot_u1 = __shfl_sync(MASK, dot_u1, 0);
		//////////////////////////////////////////////////////
		type_t u1 = A[i][i+1];
                // in Lanczos dot_u1 >=0
		ur *= flag3 * 1. / (u1 +  std::sqrt(dot_u1));
		type_t dot_u2 = flag3* ur * ur;
		//do shuffle reduction for dot_u2
#pragma unroll
		for ( r = N_COL/2; r > 0; r >>=1  )
                        dot_u2 += __shfl_down_sync(MASK, dot_u2, r, N_COL );
		// broadcast
		dot_u2++;
		dot_u2 = __shfl_sync(MASK, dot_u2, 0);
		// all u(j<i+2) = 0, v(i+1) = 1, v(j>=i+2) = A(tid+N_COL*i)
                u[tid] = flag3 * ur + (tid == (i+1));
		type_t sigma =  2./dot_u2;
		//////////////////////////////////////////////////////
		type_t alpha = 0;	  
		v[tid] = 0;
		// the root of this splitting is to reduce a bit computation down	
		for (c = i; c < tid; ++c) 
		{
			alpha += A[c][tid] * u[c];
		}
			
		for (c=tid ; c < N_COL; ++c) 
		{
			alpha += A[tid][c] * u[c];
		}
		alpha *= flag1 * sigma; // v(j<i) = 0;
		v[tid] = alpha;
		alpha *= u[tid];
#pragma unroll
		 for ( r = N_COL/2; r > 0; r >>=1  )
                        alpha += __shfl_down_sync(MASK, alpha, r, N_COL);
		alpha = __shfl_sync(MASK, alpha, 0);
		
		alpha /= dot_u2; 
		//////////////////////////////////////////////////////
		ur = u[tid]; 
		w[0][tid] = flag1 * (v[tid] - alpha * ur);
		if ( tid >= i )
		{
			// work inbalance
			for (c = tid ; c < N_COL; ++c) 
			{
				 A[tid][c] -= (ur*w[0][c] + w[0][tid]*u[c]);
			}
		}
		 __syncwarp(MASK);
		///////////////////////////////////////////
		if ( tid == i+1)
		{ 
			A[tid][i] = sigma; 
		}
		if ( tid > i + 1)
		{
			A[tid][i] = ur;
		}
		flag1 = flag2;
		flag2 = flag3;
		__syncwarp(MASK);			
	}
	diag[tid] = A[tid][tid];
        upper_diag[tid] = A[tid][tid+1];
}

template< typename type_t>
__device__ void Accumulate_HouseHolders(	const unsigned int tid,
						const unsigned int sgn,
						const unsigned int MASK,		
						type_t (&A)[N_COL][N_COL+1],
						type_t (&eigen_vec)[N_COL][N_COL+1],
						type_t *v)
{
	eigen_vec[tid][tid] = 1.;
	unsigned int c;
	for ( int i = N_COL - 3; i >= 0; --i)
	{
		int flag1 =  sgn * (tid/(i+2) - std::abs((int)(tid-(i+2)))/(i+2));
			//sgn * (tid/(i+2)) - (tid-(i+2)/(i+2));
	
		type_t sigma = A[i+1][i];//A[i + N_COL *( i + 1)];// all threads read this

		// all v(j<i+1) = 0, v(i+1) = 1, v(j>i+1) = A(i+N_COL*j)
		v[tid] =  flag1 * A[tid][i] + (tid == i+1);//A(i + N_COL*tid)

		type_t wr = 0;
		__syncwarp(MASK);

		for (c = i+1; c < N_COL; ++c)
                {
                	wr += v[c] * eigen_vec[c][tid];//eigen_vec[tid + N_COL*c];
                }		
		for (c = i+1; c < N_COL; ++c)
                {
                	eigen_vec[c][tid] -= v[c] * wr * sigma;//[tid + N_COL * c]
                }
		 __syncwarp(MASK);
	}	
}

template< typename type_t>
__device__ void Rotate_eigen_vec(	const unsigned int tid,
					const unsigned int MASK,
					type_t (&eigen_vec)[N_COL][N_COL+1],
					type_t *cos,
					type_t *sin,
					int *index,
					const unsigned int count  )
{	

	for ( unsigned int given = 0; given < count; given++ ) 
	{
		unsigned int j =  index[given];
		type_t cos_given = cos[given];
		type_t sin_given = sin[given];

		type_t val0 = eigen_vec[tid][j];
		type_t val1 = eigen_vec[tid][j + 1];
		eigen_vec[tid][j] = cos_given * val0 - sin_given * val1;//
		eigen_vec[tid][j + 1] = sin_given * val0 + cos_given * val1;//
		
		//__syncwarp(MASK); //no need to synchronize	
	}
}	


template< typename type_t, unsigned int global_iter>
__global__ void My_sqrtm_cusolver(	type_t *A,
					type_t *A_inv,
					type_t EPS )
{
	__shared__ type_t shmem_A[N_COL][N_COL+1];
  	__shared__ type_t diag[N_COL];
	__shared__ type_t upper_diag[N_COL];
	/*
	 u,v vector are used first in the Tridiagonalization and later
	  to store givens 
	 w is used first in the Tridigonalization and later to store eigenvectors
	*/
	/*
	 Note:: I have to much shared memory for this problem size therfore storing 
	 givens might seem waste of memory but it allows me to perfectly seperate
	 the parallelizable parts of the code from the serial
	*/

	__shared__ type_t u[global_iter * N_COL ];// cos
	__shared__ type_t v[global_iter * N_COL ];// sin
	__shared__ type_t w[N_COL][N_COL+1];

	__shared__ int index[global_iter * N_COL];

	const unsigned int tid = threadIdx.x;
	const unsigned int row = threadIdx.x%N_COL;
	const unsigned int col = threadIdx.x/N_COL;

	// count is first used to determin the flags parameters in Tridiagonalization
	// then later to count the givens
	unsigned int count = 1;
	if ( tid == 0)
	{
		count = 0;
	}
	shmem_A[ col ][ row ] = A[tid];	
	__syncthreads();

	unsigned MASK = __ballot_sync(0xffffffff, tid < N_COL);
	if(N_COL >= 4)	
	{
	if ( tid < N_COL )
	{	
	
		sqrtm::Tridiagonalize<type_t>(tid, count, MASK, shmem_A, diag, upper_diag, u, v, w);
	}
	__syncthreads();
	
	w[ col ][ row ] = 0;
	if ( tid < N_COL )
	{
		sqrtm::Accumulate_HouseHolders<type_t>(tid, count, MASK,  shmem_A, w, u);
	}
	}
	
	if ( tid == 0 )
	{

		type_t a00,a01,a11,s,x,y,a02,a12,a22,a23,tmp11,tmp12,tmp21,tmp22,cos,sin,tau;
		
		for ( unsigned int iter = 0; iter < global_iter; ++iter )
		{
			// << Bulge chasing >>
			/////////////////////////////////////////////////////
			int imin = -1, imax = -1;
			for ( int i = N_COL - 2; i >= 0; --i) 
			{
				a00 = diag[i];
				a01 = upper_diag[i];//check
				a11 = diag[i + 1];
				
				if(std::abs(a01) > EPS)
				{
					if (imax == -1) 
					{
						imax = i;
					}
					imin = i;
				}	
				else
				{
					if (imin >= 0) 
						break;
				}
			}
			if (imax == -1)
				break;
			////////////////////////////////////////////////////
			//<<< Start implicit shift >>>
			a00 = diag[imax];
        		a01 = upper_diag[imax];
        		a11 = diag[imax + 1];
        		s = a11 -  (a01*a01) / ((a00 - a11)*0.5  - 
					std::sqrt( (a00-a11)*(a00-a11)*0.25 + ( a01*a01)));
        		x = diag[imin] - s;
        		y = upper_diag[imin];
        		cos = 0; sin = 0; a02 = 0;
			for ( int i1 = imin; i1 <= imax; ++i1)	
			{	
				// << compute  Given >>
   			     	if (y != 0)
        			{
                			if (std::abs(y) > std::abs(x))
               				{
                        			tau = -x / y;
                        			sin = (1.) / std::sqrt((1) + tau * tau);
                       	 			cos = sin * tau;
                			}
                			else
                			{
                        			tau = -y / x;
                        			cos = (1.) / std::sqrt((1) + tau * tau);
                        			sin = cos * tau;
               	 			}
        			}
        			else
        			{
                			cos = 1; sin = 0;
        			}
        			u[count] = cos;
        			v[count] = sin;
                		index[count] = i1; // stored
                		if (i1 > imin)
                		{
                           		 upper_diag[i1-1] = cos * upper_diag[i1-1] - 
								sin * a02;
               			}
                		a11 = diag[i1];
                		a12 = upper_diag[i1];
                		a22 = diag[i1+1];
                		tmp11 = cos * a11 - sin * a12;
                		tmp12 = cos * a12 - sin * a22;
               		 	tmp21 = sin * a11 + cos * a12;
                		tmp22 = sin * a12 + cos * a22;

                		diag[i1] = cos * tmp11 - sin * tmp12;
                		upper_diag[i1] = sin * tmp11 + cos * tmp12;
                		diag[i1+1] = sin * tmp21 + cos * tmp22;
                		if (i1 < imax)
                		{
                        		a23 = upper_diag[i1+1];
                        		a02 = - sin * a23;
                        		upper_diag[i1+1] = cos * a23;
                        		x = upper_diag[i1];
                        		y = a02;
                		}
                		count++;
			}
		}
	}
	__syncthreads();	
	if (tid < N_COL)
	{
		count = __shfl_sync(MASK, count, 0);	
		sqrtm::Rotate_eigen_vec<type_t>( tid, MASK, w, u, v, index, count);

		/*
 * 			<<<     Important       >>>
 * 			The igen values should be positive as the input matrix
 * 			is positive definite in Lanczos otherwise no sqrtm can be obtained
 * 			however for single floating points a 0 value can be sligtly negative
 * 			therfore to insure positicity the abs() here is crucial in case of
 * 			single floatings. For double floatings there is no issue.
 		*/	
		
		shmem_A[0][tid] = std::sqrt(std::abs(diag[tid]));
                shmem_A[1][tid] = 1./std::sqrt(std::abs(diag[tid]));
	}
	__syncthreads();
	// do sqrtm
	type_t sum0 = 0, sum1 = 0;
#pragma unroll 
	for ( unsigned int i = 0; i < N_COL; ++i )
        {
	//if (tid == 0) printf("%f.",diag[i] );
                        sum0 +=   w[row][i] *  
                                  shmem_A[0][i] *
                                  w[col][i];
 
			sum1 +=   w[row][i] *   
                                  shmem_A[1][i] *
                                  w[col][i]; 
        }	

	A[row + N_COL*col] = sum0;
	A_inv[row + N_COL*col] = sum1;
}	
};



template< typename type_t>
void my_sqrtm_cusolver( Dense_matrix<type_t> &A,
			Dense_matrix<type_t> &A_inv) 
{
	type_t EPS = std::numeric_limits<type_t>::epsilon();
	sqrtm::My_sqrtm_cusolver<type_t, N_COL*3>
				<<<1, N_COL*N_COL>>>
                                        (A.data(),
                                        A_inv.data(),
					EPS);	
}









