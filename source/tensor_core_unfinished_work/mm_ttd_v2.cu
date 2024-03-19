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

#define NUMBER_OF_LOADS 2

#define WARP_SIZE 32
#define WARPS_PER_BLOCK 6
#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)

// these will modified for double
#define LOADS_ttd 4
#define LOADS_PER_WARP_ttd LOADS_ttd * WARP_SIZE/2
#define LOADS_PER_BLOCK_ttd  LOADS_PER_WARP_ttd * WARPS_PER_BLOCK


#define N_COL 4 
#define TILE_SIZE WARP_SIZE/N_COL


#define SHMEM_STRIDE_float N_COL // this will fully avoid bank conflict in block level reduction


//shared max
// 6 * ( (16*4*2 + 16 +1)*16  )*4 for float
// 4 * ( (14*4*2 + 14 +1)*14  )*8 for double

//register 
// (16+16)*4 for float 
// (16+16)*8 for double
namespace ttd{
__device__ void loadFromGmem(	const unsigned int global_n_rows,
				const unsigned int warpId,
				const unsigned int laneId,
				*float shmem,
				*float warp_ptr_a,
				*float warp_ptr_b) // eithr 0 or shmem_off
{
#pragma unroll
	for ( unsigned int i = 0; i < N_COL; i += 2 )
	{	
	//	*((int4 *)(shmem[ warpId*N_COL +  laneId / (WARP_SIZE/2) + i ][0]) + 
	//				laneId % ( WARP_SIZE/2 )) = 
	//	*((int4 *)(warp_ptr + (laneId / (WARP_SIZE/2) + i )*global_n_row ) + 
	//				laneId % (WARP_SIZE/2));

		reinterpret_cast<COPY4_t *>
		(& (shmem[ warpId*N_COL +  laneId / (WARP_SIZE/2) + i ]
			[laneId % ( WARP_SIZE/2 ) * 4 ]) )[0]
		=
		reinterpret_cast<COPY4_t *>
		(warp_ptr_a + (laneId / (WARP_SIZE/2) + i )*global_n_row
			+ laneId % (WARP_SIZE/2) * 4)[0];	

		reinterpret_cast<COPY4_t *>
                (& (shmem[ warpId*N_COL +  laneId / (WARP_SIZE/2) + i ]
                        [ (laneId % ( WARP_SIZE/2 ) * 4) + LOADS_PER_WARP_ttd ]) )[0]
                =
                reinterpret_cast<COPY4_t *>
                (warp_ptr_b + (laneId / (WARP_SIZE/2) + i )*global_n_row
                        + laneId % (WARP_SIZE/2) * 4)[0];
	}
}

__device__ void processFromShmem(	const unsigned int global_n_rows,
                                	const unsigned int warpId,
                                	const unsigned int laneId,	
					float* shmem,
					float* fragment_a,
					float* fragment_b,
                                	const unsigned int buffer)
{

// populate the registers

#pragma unroll
	for ( unsigned int i = 0; i < N_COL; ++ i )
	{
		fragment_a[i] = shmem[ warpId % ( WARPS_PER_BLOCK/2 ) * N_COL +        // warp location
                                         i +                                    	// columns of the tile
                                         buffer * ( WARPS_PER_BLOCK/2 ) * N_COL  ]      // buffer location

                                        [ laneId +					// row of the tile
                                          warpId / ( WARPS_PER_BLOCK/2 ) * WARP_SIZE ];	// section of the buffer to be computed	

		fragment_b[i] = shmem[ warpId % ( WARPS_PER_BLOCK/2 ) * N_COL +        
                                        i +                                            
                                        buffer * ( WARPS_PER_BLOCK/2 ) * N_COL  ]      

                                       [ laneId +                                     
					 LOADS_PER_WARP_ttd +
                                         warpId / ( WARPS_PER_BLOCK/2 ) * WARP_SIZE ];
	}


#pragma unroll
	for ( unsigned int i = 0; i < N_COL ; ++i ) // N_COL covers 32 check this for double
	{
		 float result = 0; 	//
		 
#pragma unroll
		for ( unsigned int j = 0; j < N_COL; j++ )
		{	
			 result = fragment_a[i]*fragment_b[j];
			// warp level reduction
			for ( unsigned int offset = WARP_SIZE/2; offset>0; offset /= 2)
				result += __shfl_down_sync(0xffffffff, result, offset);

			if ( laneId == 0 )
				shmem[warpId % ( WARPS_PER_BLOCK/2 ) * N_COL +
					 j + 
					buffer * ( WARPS_PER_BLOCK/2 ) * N_COL]
					[ 2*LOADS_PER_WARP_tt + i] 
					+= result;

				shmem[warpId % ( WARPS_PER_BLOCK/2 ) * N_COL +
                                         i +
                                        buffer * ( WARPS_PER_BLOCK/2 ) * N_COL]
                                        [ 2*LOADS_PER_WARP_tt + j]
                                        += result;
			__syncwarp();
				
		}		
	}	
}


template<const int NUM_THREADS >
__global__ void __launch_bounds__(NUM_THREADS)
		mm_ttd(		const unsigned int global_n_rows,
			 	const float *T1,
				const float *T2,
				float *result)
{
	const unsigned int warpId = threadIdx.x / WARP_SIZE;
        const unsigned int laneId = threadIdx.x % WARP_SIZE;

	const unsigned int lane_row = laneId%TILE_SIZE; 
	const unsigned int lane_col = laneId/TILE_SIZE
       
	// shared memory is only for block level reduction
	extern __shared__ float shmem[][ LOADS_PER_WARP_ttd*2 + N_COL + SHMEM_STRIDE_FLOAT ];//<<--

	//const unsigned int shmem_off = N_COL * (WARPS_PER_BLOCK/2)

	//bool DOUBLE_BUFFERING == warpId/(WARPS_PER_BLOCK/2)
	
	float *warp_ptr_a =  &T1[blockIdx.x * LOADS_PER_BLOCK_ttd] +
			(warpId * LOADS_PER_WARP_ttd ); //<<--
	float *warp_ptr_b =  &T2[blockIdx.x * LOADS_PER_BLOCK_ttd] +
                        (warpId * LOADS_PER_WARP_ttd );

	// registers are for thread level computation and reduction
	float fragment_a[N_COL] = {0}	
	float fragment_b[N_COL] = {0}
	///////////////////////////////////////////////////////

	// prepare buffer 0
	if ( warpId/(WARPS_PER_BLOCK/2) == 0 )
	{
		tt::loadFromGmem(	global_n_rows, laneId, warpId, 
					shmem, warp_ptr_a, warp_ptr_b);
	}
	//////////////////////////////////////////////////////////
	__syncthreads();

	for ( unsigned int load = 0; load < NUMBER_OF_LOADS - 1; load++ )
	{
		warp_ptr_a += gridDim.x * LOADS_PER_BLOCK_ttd; // <<-- check this
		warp_ptr_b += gridDim.x * LOADS_PER_BLOCK_ttd;
		//lane_ptr = (float4 *)(warp_ptr ) + lane_ptr;	

		if (warpId/(WARPS_PER_BLOCK/2) == 0)

		{
			// process buffer 0
			tt::processFromShmem(	global_n_rows, warpId, laneId,
						shmem, fragment_a, fragment_b,
						0, lane_row, lane_col);
			__syncthreads();
			// process buffer 1
			tt::processFromShmem(   global_n_rows, warpId, laneId, 
						shmem, fragment_a, fragment_b,
                                                1, lane_row, lane_col);
			__syncthreads();
			// load into buffer 0
			tt::loadFromGmem( 	global_n_rows, warpId, laneId,
                                        	shmem, warp_ptr_a, warp_ptr_b);
		}
		else
		{
			// load into buffer 1
			tt::loadFromGmem( 	global_n_rows, warpId, laneId,
                                                shmem, warp_ptr_a, warp_ptr_b);
			__syncthreads();
			// process buffer 0
			tt::processFromShmem(	global_n_rows, warpId, laneId,
                                                shmem, fragment_a, fragment_b,
                                                0, lane_row, lane_col);
			__syncthreads();
			// process buffer 1
			tt::processFromShmem(	global_n_rows, warpId, laneId,
                                                shmem, fragment_a, fragment_b
                                                1, lane_row, lane_col);
		}
		__syncthreads();
	}
	// do the last section
	if (warpId/(WARPS_PER_BLOCK/2) == 0)
	{
		tt::processFromShmem(   global_n_rows, warpId, laneId,
                                        shmem, fragment_a, fragment_b,
                                        0, lane_row, lane_col);
		__syncthreads();
		tt::processFromShmem(   global_n_rows, warpId, laneId,
                                        shmem, fragment_a, fragment_b
                                        1, lane_row, lane_col);
		
	}
	else
	{
		tt::loadFromGmem(       global_n_rows, warpId, laneId,
                                        shmem, warp_ptr_a, warp_ptr_b);
		__syncthreads();

		tt::processFromShmem(   global_n_rows, warpId, laneId,
                                        shmem, fragment_a, fragment_b
                                        0, lane_row, lane_col);

		tt::processFromShmem(   global_n_rows, warpId, laneId,
                                        shmem, fragment_a, fragment_b,
                                        1, lane_row, lane_col);
	}
	__syncthreads();
	///////////////////////////////////////////////////////

	// block level reduction
	// not enough threads to do everything here so let every thread do double the work
	if ( threadIdx <= N_COL * N_COL/2)
	{
		float val_reduce1 =  shmem[  threadIdx/(N_COL/2)]
					[ 2*LOADS_PER_WARP_ttd + threadIdx%N_COL];

		float val_reduce2 =  shmem[  threadIdx/(N_COL/2) +  (N_COL/2)]
                                        [ 2*LOADS_PER_WARP_ttd + threadIdx%N_COL];
		
#pragma unroll
		for (unsigned int i = 0; i < WARPS_PER_BLOCK; ++i)
                {
			
                        val_reduce1 += shmem[  threadIdx/(N_COL/2) + i*N_COL]
					[ 2*LOADS_PER_WARP_ttd + threadIdx%N_COL];

			val_reduce2 += shmem[ threadIdx/(N_COL/2) + i*N_COL + (N_COL/2)]
                                        [ 2*LOADS_PER_WARP_ttd + threadIdx%N_COL];
                }
		// global reduction		 
		
		atomicAdd( &val_reduce1, 
				result[threadIdx/(N_COL/2) * N_COL + threadIdx%N_COL] );

		atomicAdd( &val_reduce2, 
				result[(threadIdx/(N_COL/2) + (N_COL/2)) * N_COL + threadIdx%N_COL] );
	}
/*

	 	 val_reduce =  shmem[  threadIdx/(N_COL/2) + (N_COL/2)]
					[ 2*LOADS_PER_WARP_ttd + threadIdx%N_COL];

#pragma unroll
                for (unsigned int i = 0; i < WARPS_PER_BLOCK; ++i)
                {

                        val += shmem[ threadIdx/(N_COL/2) + i*N_COL + (N_COL/2)]
					[ 2*LOADS_PER_WARP_ttd + threadIdx%N_COL];
                }
                // global reduction

                atomicAdd( &val, result[threadIdx/N_COL * N_COL + threadIdx%N_COL] );
        }
*/
}

};




int main(int argc, char **argv)
{

 	CudaDeviceInfo();	
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














