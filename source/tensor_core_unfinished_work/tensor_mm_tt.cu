#include "../utils/cuda_utils.hpp"
#include <stdio.h>
#include <cuda.h>
#include <mma.h>
#include <vector>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

#include <iomanip>
#include <iostream>
#include <utility>
#include <memory>

#include <cooperative_groups.h>

#include "vector.hpp"
#include "dense_matrix.hpp"



#define WARP_SIZE 32
#define TILE_SIZE 16

/////////////////////////////////
#define M_TILES 1//256
#define N_TILES 1//256
#define K_TILES 64

#define M_GLOBAL (TILE_SIZE * M_TILES)
#define N_GLOBAL (TILE_SIZE * N_TILES)
#define K_GLOBAL (TILE_SIZE * K_TILES)
///////////////////////////////////////
using namespace nvcuda;


#define LOADS_PER_WARP 2
#define WARPS_PER_BLOCK 8
#define HALF_WARPS (WARPS_PER_BLOCK/2)

#define THREADS_PER_BLOCK (WARP_SIZE * WARPS_PER_BLOCK)         //
#define TILES_PER_LOAD 4 // <<----
#define TILES_PER_COMPUTE (TILES_PER_LOAD/2)

#define TILES_PER_BLOCK (TILES_PER_LOAD * WARPS_PER_BLOCK)              // 8*8

#define BYTES_PER_LINE (TILES_PER_LOAD * TILE_SIZE * sizeof(half))      //<<--
#define BYTES_PER_WARP (WARP_SIZE * sizeof(int4))               //
#define LINES_PER_WARP (BYTES_PER_WARP / BYTES_PER_LINE)
#define LANES_PER_LINE (WARP_SIZE / LINES_PER_WARP)             //

#define SHMEM 16 // this is for memory alignment

__global__ void h_mm_ttd(	const unsigned int global_n_rows,
			 	half *B,
				float*D)
{
	extern __shared__ half shmem[][TILES_PER_LOAD * TILE_SIZE + STRIDE];//<<--
	
	const unsigned int warpId = threadIdx.x / WARP_SIZE;
	const unsigned int laneId = threadIdx.x % WARP_SIZE;

	const unsigned int lane_col = laneId / LANES_PER_LINE;
	const unsigned int lane_row = laneId % LANES_PER_LINE;


	wmma::fragment<wmma::accumulator, TILE_SIZE, TILE_SIZE, TILE_SIZE, float> acc;
	wmma::fill_fragment(acc, 0.0f);

	wmma::fragment<wmma::matrix_a, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::row_major>a;
	wmma::fragment<wmma::matrix_b, TILE_SIZE, TILE_SIZE, TILE_SIZE, half, wmma::col_major>b;

	__syncthreads();

	half *warp_ptr =  &B[blockIdx.x * TILES_PER_BLOCK * TILE_SIZE] +
			(warpId * TILES_PER_LOAD * TILE_SIZE ); //<<--
		

	int4 *lane_ptr = (int4 *)(warp_ptr  + lane_col* global_n_rows) + lane_row;

	size_t shmem_idx =  warpId * TILE_SIZE + lane_col;

	///////////////////////////////////////////////////////
#pragma unroll
	for (unsigned int i = 0; i < ((WARP_SIZE/2 ) / LINES_PER_WARP) ; i++ )
	{
		
		*((int4 *)&shmem[shmem_idx][0] + lane_row) = *lane_ptr;
		lane_ptr = (int4 *)((half *)lane_ptr + n_rows * LINES_PER_WARP);
		shmem_idx += LINES_PER_WARP;
	}
	//////////////////////////////////////////////////////////
	__syncthreads();

	for ( unsigned int load = 1; load < LOADS_PER_WARP ; load++ )
	{
		
		warp_ptr += gridDim.x * TILES_PER_BLOCK * TILE_SIZE;
		lane_ptr = (int4 *)(warp_ptr  + lane_col* global_n_rows) + lane_row;
		shmem_idx =  warpId * TILE_SIZE + lane_col;
        	
#pragma unroll
		for (unsigned int i = 0; i < ((WARP_SIZE/2 ) / LINES_PER_WARP) ; i++)
		{
			*((int4 *)&shmem[shmem_idx][(load%2)*TILES_PER_LOAD*TILE_SIZE] 
					+ lane_row) = *lane_ptr;

                	lane_ptr = (int4 *)((half *)lane_ptr + n_global_rows * LINES_PER_WARP);
               	 	shmem_idx += LINES_PER_WARP;
		}

#pragma unroll
		for (unsigned int tile = 0; tile < TILES_PER_COMPUTE; tile++ )
		{
			const half *tile_ptr = &shmem[ warpId * TILE_SIZE]
                                [tile*TILE_SIZE + ((load-1)%2)*TILES_PER_LOAD*TILE_SIZE];

			wmma::load_matrix_sync(a, tile_ptr, TILE_SIZE * TILE_PER_LOAD + STRIDE);
			wmma::load_matrix_sync(b, tile_ptr, TILE_SIZE * TILE_PER_LOAD + STRIDE);
			wmma::mma_sync(acc, a, b, acc);
		}
		 __syncthreads();		
	}
	
	///////////////////////////////////////////////////////
#pragma unroll
	for (unsigned int tile = 0; tile < TILES_PER_COMPUTE; tile++ )
                {
                        const half *tile_ptr_a = &shmem[ warpId * TILE_SIZE]
                                [tile*TILE_SIZE + ((load-1)%2) * TILES_PER_LOAD * TILE_SIZE];

                        wmma::load_matrix_sync(a, tile_ptr_a, TILE_SIZE * TILE_PER_LOAD + STRIDE);
                        wmma::load_matrix_sync(b, tile_ptr_b, TILE_SIZE * TILE_PER_LOAD + STRIDE);

                        wmma::mma_sync(acc, a, b, acc);
                }
	__syncthreads(); // << dont need this
	//////////////////////////////////////////////////////////	
	
	float *tile_ptr = (float *)&shmem[warpId * TILE_SIZE * 2][0] ;
	wmma::store_matrix_sync(tile_ptr, acc, (TILES_PER_LOAD * TILE_SIZE + STRIDE)/2,  wmma::mem_col_major);

	
	//float *lane_reduction_ptr = (float *)&shmem[warpId*2][0] + laneId;
	tile_ptr = &( (float *)&shmem[warpId*2][0] + laneId );
	__syncthreads();
#pragma unroll
	for ( unsigned int n_sum = 0; n_sum < WARPS_PER_BLOCK ; n_sum++)
	{
		*tile_ptr += *(tile_ptr + 
			TILE_SIZE*(TILES_PER_LOAD * TILE_SIZE + STRIDE)/2 );//<-- moving in floats
	}

	__syncthreads();

	if ( laneId < WARP_SIZE/2)
	{
		atomicAdd( &( *tile_ptr + *(tile_ptr + TILE_SIZE) )
			D[laneID + TILE_SIZE*warpId]); //<<-- uncoalesced acess
	}
	// check reduction using fragments

	
}



















