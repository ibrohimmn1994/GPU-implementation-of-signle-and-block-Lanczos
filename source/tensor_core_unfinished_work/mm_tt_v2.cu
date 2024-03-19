#include "../utils/cuda_utils.hpp"
#include <stdio.h>
#include <cuda.h>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <utility>
#include <memory>

#include <algorithm>

#include "vector.hpp"
#include "dense_matrix.hpp"


#include <cublas_v2.h>
///////////////////////////////////////



#define COPY4_t float4
#define COPY2_t float2
//#define NUMBER_OF_LOADS 2

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 6
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

// these will modified for double
#define LOADS_tt 4
#define LOADS_PER_WARP_tt LOADS_tt * WARP_SIZE
#define LOADS_PER_BLOCK_tt  LOADS_PER_WARP_tt * WARPS_PER_BLOCK


#define N_COL 4 
#define TILE_SIZE WARP_SIZE/N_COL

#define stride 0 // this is for memory alignment

// 12 * (16*16*4 + 16*16  )*4 for float
// 6 * (16*16*4 + 16*16  )*8 for double


namespace tt{
__device__ void loadFromGmem(	const unsigned int global_n_rows,
				const unsigned int warpId,
				const unsigned int laneId,
				float (&shmem)[][ LOADS_PER_WARP_tt + N_COL + stride],
				float *warp_ptr) // eithr 0 or shmem_off
{


#pragma unroll
	for ( unsigned int i = 0; i < N_COL; ++i )
	{	
		reinterpret_cast<COPY4_t *>
		(&(shmem[ warpId*N_COL  + i ]//[ warpId*N_COL +  laneId / (WARP_SIZE/2) + i ]
			[laneId * 4 ]))[0] //[laneId % ( WARP_SIZE/2 ) * 4 ])
		=
		reinterpret_cast<COPY4_t *>
		( warp_ptr  + i *global_n_rows
			+ laneId * 4)[0];	
	}
}

__device__ void processFromShmem(	const unsigned int global_n_rows,
                                	const unsigned int warpId,
                                	const unsigned int laneId,	
					float (&shmem)[][ LOADS_PER_WARP_tt + N_COL +stride],
					float2 *fragment,
                                	const unsigned int buffer)
{

// populate the registers
#pragma unroll
	for ( unsigned int i = 0; i < N_COL; ++ i )
	{
		fragment[i] =  reinterpret_cast<COPY2_t *>
					(&shmem[ warpId % ( WARPS_PER_BLOCK/2 ) * N_COL +        // warp location
                                         i +                                    	// columns of the tile
                                         buffer * ( WARPS_PER_BLOCK/2 ) * N_COL  ]      // buffer location

                                        [ laneId*2 +					// row of the tile
                                          warpId / ( WARPS_PER_BLOCK/2 ) * WARP_SIZE*2 ])[0];	// section of the buffer to be computed	
	}


#pragma unroll
	for ( unsigned int i = 0; i < N_COL ; ++i ) // N_COL covers 32 check this for double
	{
		 float result = 0; 	// 
#pragma unroll
		for ( unsigned int j = 0; j < N_COL; j++ )
		{	
			 result = fragment[i].x * fragment[j].x + fragment[i].y * fragment[j].y;
			// warp level reduction
			for ( unsigned int offset = WARP_SIZE/2; offset>0; offset /= 2)
				result += __shfl_down_sync(0xffffffff, result, offset);

			if ( laneId == 0 )
				shmem[warpId % ( WARPS_PER_BLOCK/2 ) * N_COL +
					 j + 
					buffer * ( WARPS_PER_BLOCK/2 ) * N_COL]
					[ LOADS_PER_WARP_tt + i] 
					+= result;
			__syncwarp();
				
		}		
	}	
}


template<const unsigned int NUMBER_OF_LOADS >
__global__ void //__launch_bounds__(NUM_THREADS)
		mm_tt(		const std::size_t global_n_rows,
			 	float *T,
				float *result)
{
	const unsigned int warpId = threadIdx.x / WARP_SIZE;
        const unsigned int laneId = threadIdx.x % WARP_SIZE;

 
	// shared memory is only for block level reduction
	extern __shared__ float shmem[][ LOADS_PER_WARP_tt + N_COL + stride];//<<--

	//const unsigned int shmem_off = N_COL * (WARPS_PER_BLOCK/2)

	//bool DOUBLE_BUFFERING == warpId/(WARPS_PER_BLOCK/2)
	
	float *warp_ptr =  &T[blockIdx.x * LOADS_PER_BLOCK_tt] +
			(warpId * LOADS_PER_WARP_tt ); //<<--

	// registers are for thread level computation and reduction
	float2 fragment[N_COL] = {0};	
	///////////////////////////////////////////////////////

	// prepare buffer 0
	if ( warpId/(WARPS_PER_BLOCK/2) == 0 )
	{
		tt::loadFromGmem(	global_n_rows, warpId, laneId, 
					shmem, warp_ptr);
	}
	//////////////////////////////////////////////////////////
	__syncthreads();

	for ( unsigned int load = 0; load < NUMBER_OF_LOADS - 1; load++ )
	{
		//warp_ptr += gridDim.x * LOADS_PER_BLOCK; // <<-- check this
		//lane_ptr = (float4 *)(warp_ptr ) + lane_ptr;	

		if (warpId/(WARPS_PER_BLOCK/2) == 0)

		{
			// process buffer 0
			tt::processFromShmem(	global_n_rows, warpId, laneId,
						shmem, fragment,
						0);
			__syncthreads();
			// process buffer 1
			tt::processFromShmem(   global_n_rows, warpId, laneId, 
						shmem, fragment,
                                                1);
			__syncthreads();
			// load into buffer 0
			tt::loadFromGmem( 	global_n_rows, warpId, laneId,
                                        	shmem, warp_ptr);
		}
		else
		{
			// load into buffer 1
			tt::loadFromGmem( 	global_n_rows, warpId, laneId,
                                                shmem, warp_ptr);
			__syncthreads();
			// process buffer 0
			tt::processFromShmem(	global_n_rows, warpId, laneId,
                                                shmem, fragment,
                                                0);
			__syncthreads();
			// process buffer 1
			tt::processFromShmem(	global_n_rows, warpId, laneId,
                                                shmem, fragment,
                                                1);
		}
		__syncthreads();
		warp_ptr += gridDim.x * LOADS_PER_BLOCK_tt;
	}

	// do the last section
	
	if (warpId/(WARPS_PER_BLOCK/2) == 0)
	{
		tt::processFromShmem(   global_n_rows, warpId, laneId,
                                        shmem, fragment,
                                        0);
		__syncthreads();
		tt::processFromShmem(   global_n_rows, warpId, laneId,
                                        shmem, fragment,
                                        1);
		
	}
	else
	{
		tt::loadFromGmem(       global_n_rows, warpId, laneId,
                                        shmem, warp_ptr);
		__syncthreads();

		tt::processFromShmem(   global_n_rows, warpId, laneId,
                                        shmem, fragment,
                                        0);

		tt::processFromShmem(   global_n_rows, warpId, laneId,
                                        shmem, fragment,
                                        1);
	}
	
	__syncthreads();
	///////////////////////////////////////////////////////

	// block level reduction
	
	if ( threadIdx.x < N_COL * N_COL/2)
	{
		float val_reduce1 = 0, val_reduce2 = 0;
#pragma unroll
		for (unsigned int i = 0; i < WARPS_PER_BLOCK; ++i)
                {
		
                        val_reduce1 += shmem[  threadIdx.x/N_COL + i*N_COL]
                                        [ LOADS_PER_WARP_tt + threadIdx.x%N_COL];

                        val_reduce2 += shmem[ threadIdx.x/N_COL + i*N_COL + (N_COL/2)]
                                        [ LOADS_PER_WARP_tt + threadIdx.x%N_COL];
                
		}
		// global reduction		
		atomicAdd( &result[(threadIdx.x/N_COL) * N_COL + threadIdx.x%N_COL],
				val_reduce1 );

                atomicAdd( &result[(threadIdx.x/N_COL + (N_COL/2)) * N_COL + threadIdx.x%N_COL],
				val_reduce2 );
	}
}

};




int main(int argc, char **argv)
{
	if (1)
	{
	const unsigned int len = 40* 6*32 * 100 * 4;	
	MemorySpace mem = MemorySpace::CUDA;
        Dense_matrix<float> T( len, N_COL, mem);
        Dense_matrix<float> R(N_COL, N_COL, mem);
      
       
        T = 1;
	double best = 1e10;
	unsigned int rep , n_test = 10, n_rep = 100;      
	for (unsigned int t = 0; t < n_test; ++t)
	{

		cudaDeviceSynchronize();
		auto t1 = std::chrono::steady_clock::now();
		for (rep = 0; rep < n_rep ; ++ rep)
		{
        	tt::mm_tt<100>
		<<<40,6*32, (N_COL*WARPS_PER_BLOCK) * (LOADS_PER_WARP_tt + N_COL + stride)*sizeof(float)>>>
				(T.n_rows(),
                              	T.data(),
                              	R.data());
		}

		cudaDeviceSynchronize();
		double time = std::chrono::duration_cast
					<std::chrono::duration<double>>(
                                        std::chrono::steady_clock::now() - t1)
                                        .count();
		best = std::min(best, time/n_rep);
	}
	std::cout << "elapsed time: " << std::setw(11) << best << std::endl;
	}
	else
	{

	 const unsigned int len = 1* 6*32 * 2 * 4;
	MemorySpace mem = MemorySpace::CUDA;
        Dense_matrix<float> T( len, N_COL, mem);
        Dense_matrix<float> R(N_COL, N_COL, mem);
	T = 1;
	 tt::mm_tt<2>
                <<<1,6*32, (N_COL*WARPS_PER_BLOCK) * (LOADS_PER_WARP_tt + N_COL + stride)*sizeof(float)>>>
                                (T.n_rows(),
                                T.data(),
                                R.data());


        R.print();
	}
 	//CudaDeviceInfo();	
	/*	
	MemorySpace mem = MemorySpace::CUDA;
	Dense_matrix<float> T(1024*2, N_COL, mem);
	Dense_matrix<float> S(N_COL, N_COL, mem);

	T = 1;
	S = 0;

	mm_ttd<<<1,8*32>>>(T.n_rows(), T, S);
	*/
                                
	return 0;
}














