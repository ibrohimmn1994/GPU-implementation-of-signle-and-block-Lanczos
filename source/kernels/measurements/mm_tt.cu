
#define N_COL 16

#include "../../utils/common.hpp"
#include "../../objects/vector.hpp"
#include "../../objects/dense_matrix.hpp"
#include "../../utils/lib_utils.hpp"


namespace tt{

template < 	typename type_t, typename copy4_t,  unsigned int WARPS_PER_BLOCK,
           	unsigned int LOADS_PER_WARP, unsigned int LOADS_PER_BLOCK >
__global__ void __launch_bounds__(WARPS_PER_BLOCK*WARP_SIZE)
		mm_tt(		const unsigned int number_of_loads,
				const unsigned int global_n_rows,
			 	type_t *T, // tall matrix
				type_t *R) // result matrix
{
	const unsigned int warpId = threadIdx.x / WARP_SIZE;
        const unsigned int laneId = threadIdx.x % WARP_SIZE;
       
	// shared memory is only for block level reduction
	__shared__ type_t shmem_warp_result[N_COL * WARPS_PER_BLOCK ]
					[ N_COL ];//<<--
	
	type_t *warp_ptr =  &T[blockIdx.x * LOADS_PER_BLOCK] +
				(warpId * LOADS_PER_WARP ); //<<--
	// registers are for thread level computation and reduction
	copy4_t fragment[N_COL] = {0};
	//////////////////////////////////////////////////////////
	if ( threadIdx.x < N_COL * N_COL/2)
        {
#pragma unroll
                for (unsigned int i = 0; i < WARPS_PER_BLOCK; ++i)
                {
                        shmem_warp_result[  threadIdx.x/N_COL + i*N_COL]
                                               [  threadIdx.x%N_COL] = 0;
                        shmem_warp_result[ threadIdx.x/N_COL + i*N_COL + (N_COL/2)]
                                               [  threadIdx.x%N_COL] = 0;
                }
        }

	__syncthreads();

	for ( unsigned int load = 0; load < number_of_loads; load++ )
	{

#pragma unroll
		for ( unsigned int i = 0; i < N_COL ; ++i )
		{
			fragment[i] = reinterpret_cast<copy4_t *>
                                        (warp_ptr +  i * global_n_rows
                                        + laneId  * 4)[0];
#pragma unroll
			for ( unsigned int j = 0; j < i+1; ++j )//N_COL
			{
		
				type_t result =	fragment[i].x * fragment[j].x + 
						fragment[i].y * fragment[j].y +
					 	fragment[i].z * fragment[j].x +
					 	fragment[i].w * fragment[j].w ;

				for ( unsigned int offset = WARP_SIZE/2; offset>0; offset >>= 1)
                                        result += __shfl_down_sync(0xffffffff, result, offset);
			
				if ( laneId == 0)
                                {
					
                                        shmem_warp_result[ ( warpId %  WARPS_PER_BLOCK ) * N_COL +
                                                 i ]
                                                [  j ]
                                                += result;		
                                }
			}
		}
		warp_ptr += gridDim.x * LOADS_PER_BLOCK;	
	}
	///////////////////////////////////////////////////////
		
	__syncthreads();
	// block level reduction
	// not enough threads to do everything here so let every thread do double the work
	if ( threadIdx.x < N_COL * N_COL/2)
	{
		type_t val_reduce1 = 0, val_reduce2 = 0;
			
			if ( threadIdx.x%N_COL < threadIdx.x/N_COL)
			{
			
			val_reduce1 += shmem_warp_result[  threadIdx.x/N_COL + 0*N_COL]
                                                        [  threadIdx.x%N_COL];
			val_reduce1 += shmem_warp_result[  threadIdx.x/N_COL + 1*N_COL]
                                                        [  threadIdx.x%N_COL];
			val_reduce1 += shmem_warp_result[  threadIdx.x/N_COL + 2*N_COL]
                                                        [  threadIdx.x%N_COL];
			val_reduce1 += shmem_warp_result[  threadIdx.x/N_COL + 3*N_COL]
                                                        [  threadIdx.x%N_COL];
			val_reduce1 += shmem_warp_result[  threadIdx.x/N_COL + 4*N_COL]
                                                        [  threadIdx.x%N_COL];
			val_reduce1 += shmem_warp_result[  threadIdx.x/N_COL + 5*N_COL]
                                                        [  threadIdx.x%N_COL];
			}
			else
			{
			val_reduce1 += shmem_warp_result[  threadIdx.x%N_COL + 0*N_COL]
                                                        [  threadIdx.x/N_COL];
			val_reduce1 += shmem_warp_result[  threadIdx.x%N_COL + 1*N_COL]
                                                        [  threadIdx.x/N_COL];
			val_reduce1 += shmem_warp_result[  threadIdx.x%N_COL + 2*N_COL]
                                                        [  threadIdx.x/N_COL];
			val_reduce1 += shmem_warp_result[  threadIdx.x%N_COL + 3*N_COL]
                                                        [  threadIdx.x/N_COL];
			val_reduce1 += shmem_warp_result[  threadIdx.x%N_COL + 4*N_COL]
                                                        [  threadIdx.x/N_COL];
			val_reduce1 += shmem_warp_result[  threadIdx.x%N_COL + 5*N_COL]
                                                        [  threadIdx.x/N_COL];
			}

			atomicAdd(  &R[ (threadIdx.x/N_COL) * N_COL + threadIdx.x%N_COL] ,
                                val_reduce1);

			if  ( threadIdx.x%N_COL < threadIdx.x/N_COL + N_COL/2)
			{
                        val_reduce2 += shmem_warp_result[ threadIdx.x/(N_COL) + 0*N_COL + (N_COL/2)]
                                                        [  threadIdx.x%N_COL];
			val_reduce2 += shmem_warp_result[ threadIdx.x/(N_COL) + 1*N_COL + (N_COL/2)]
                                                        [  threadIdx.x%N_COL];
			val_reduce2 += shmem_warp_result[ threadIdx.x/(N_COL) + 2*N_COL + (N_COL/2)]
                                                        [  threadIdx.x%N_COL];
			val_reduce2 += shmem_warp_result[ threadIdx.x/(N_COL) + 3*N_COL + (N_COL/2)]
                                                        [  threadIdx.x%N_COL];
			val_reduce2 += shmem_warp_result[ threadIdx.x/(N_COL) + 4*N_COL + (N_COL/2)]
                                                        [  threadIdx.x%N_COL];
			val_reduce2 += shmem_warp_result[ threadIdx.x/(N_COL) + 5*N_COL + (N_COL/2)]
                                                        [  threadIdx.x%N_COL];
			}
			else
			{
			val_reduce2 += shmem_warp_result[ threadIdx.x%(N_COL) + 0*N_COL ]
                                                        [  threadIdx.x/N_COL + (N_COL/2)];
			val_reduce2 += shmem_warp_result[ threadIdx.x%(N_COL) + 1*N_COL ]
                                                        [  threadIdx.x/N_COL + (N_COL/2)];
			val_reduce2 += shmem_warp_result[ threadIdx.x%(N_COL) + 2*N_COL ]
                                                        [  threadIdx.x/N_COL + (N_COL/2)];
			val_reduce2 += shmem_warp_result[ threadIdx.x%(N_COL) + 3*N_COL ]
                                                        [  threadIdx.x/N_COL + (N_COL/2)];
			val_reduce2 += shmem_warp_result[ threadIdx.x%(N_COL) + 4*N_COL ]
                                                        [  threadIdx.x/N_COL + (N_COL/2)];
			val_reduce2 += shmem_warp_result[ threadIdx.x%(N_COL) + 5*N_COL ]
                                                        [  threadIdx.x/N_COL + (N_COL/2)];
			}		

			atomicAdd(  &R[( threadIdx.x/N_COL + (N_COL/2) ) * N_COL + threadIdx.x%N_COL],
                                val_reduce2 );	
	}

}

};




void mm_tt(	const unsigned int n_blocks,
		const unsigned int n_loads,
		Dense_matrix<float> &T,
		Dense_matrix<float> &R)
{
	if ( T.memory_space() == MemorySpace::CUDA)
	{
#ifndef DISABLE_CUDA
		tt::mm_tt<float, float4, 6, 4*WARP_SIZE, 6*4*WARP_SIZE>
                                <<<n_blocks, 6*WARP_SIZE>>>
                                (n_loads,
				T.n_rows(),
                                T.data(),
                                R.data());

#endif
	}
	else
	{
		std::cout << "fill in later" << std::endl;
	}
}



#define Number float
#define Number4 float4

int main(int argc, char **argv)
{

        steady_clock time = steady_clock();

        const unsigned int n = 330;

        const unsigned int len = 40* 6*WARP_SIZE * n * 4;
        MemorySpace mem = MemorySpace::CUDA;
        Dense_matrix<Number> T( len, N_COL, mem);
        Dense_matrix<Number> R(N_COL, N_COL, mem);

        double best = 1e10;
        unsigned int rep , n_test = 10, n_rep = 100;
        T = 1;

	
        if(1)
        {
                for (unsigned int t = 0; t<n_test; ++t)
                {
                        cudaDeviceSynchronize();
                        time.start();
                        for (rep = 0; rep < n_rep ; ++ rep)
                        {
				mm_tt(40, n, T, R);
                        }

                        cudaDeviceSynchronize();
                        time.end();
                        best = std::min(best, time.duration()/n_rep);
                }
        }
        else
        {
                cublasHandle_t cublasH;
                CUBLAS_CHECK( cublasCreate(&cublasH) );
                Dense_matrix<float> temp(R); temp = 0;
                for (unsigned int t = 0; t<n_test; ++t)
                {
	
                        cudaDeviceSynchronize();
                        time.start();
                        for (rep = 0; rep < n_rep ; ++ rep)
                        {
			
                                  mm_tt_cublas(T, R, cublasH);
                        }

                        cudaDeviceSynchronize();
                        time.end();
			best = std::min(best, time.duration()/n_rep);
                }

                CUBLAS_CHECK( cublasDestroy ( cublasH ) );
        }
        std::cout << "rows " << len << std::endl;
        std::cout << "elapsed time: " << std::setw(11) << best << std::endl;

        std::cout << "Gb/s: " <<
        1e-9*( (T.memory_consumption()*2 + R.memory_consumption() ))/best << std::endl;

        std::cout << " Gflops/s: " <<
        1e-9*( ( 2*T.n_rows()*T.n_cols()*T.n_cols()  )  )/best<< std::endl;


        return 0;
}





