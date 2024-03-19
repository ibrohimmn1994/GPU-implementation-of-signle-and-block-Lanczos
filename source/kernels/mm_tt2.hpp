


#if N_COL == 14 or N_COL == 18
#define STRIDE_MM_TT2 8
#else
#define STRIDE_MM_TT2 4
#endif



namespace tt2{

template < 	typename type_t, typename copy4_t, unsigned int WARPS_PER_BLOCK, 
		unsigned int LOADS_PER_WARP, unsigned int LOADS_PER_BLOCK, unsigned int PREF,
		unsigned int STRIDE >
__global__ void __launch_bounds__(WARPS_PER_BLOCK*WARP_SIZE)
		mm_tt2(		const unsigned int number_of_loads,
				const unsigned int global_n_rows,
			 	type_t *T1,
				type_t *T2,
				type_t *R)
{
	const unsigned int warpId = threadIdx.x / WARP_SIZE;
        const unsigned int laneId = threadIdx.x % WARP_SIZE;

	
	// shared memory is only for block level reduction
	__shared__ type_t shmem_warp_result[N_COL * WARPS_PER_BLOCK]
					[ N_COL  ];//<<--

	__shared__ type_t prefetch[WARPS_PER_BLOCK*WARP_SIZE][4*(PREF) + STRIDE];
		
	type_t *warp_ptr_a =  &T1[blockIdx.x * LOADS_PER_BLOCK] +
			(warpId * LOADS_PER_WARP ); //<<--

	type_t *warp_ptr_b =  &T2[blockIdx.x * LOADS_PER_BLOCK] +
                        (warpId * LOADS_PER_WARP );
	

	// registers are for thread level computation and reduction
	copy4_t fragment_b[N_COL] = {0};	
	copy4_t fragment_a ;
		
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
	//////////////////////////////////////////////////////////
	__syncthreads();

	for ( unsigned int load = 0; load < number_of_loads; load++ )
	{
#pragma unroll
		for ( unsigned int i = 0; i < N_COL; ++i )
		{	
			fragment_b[i] = reinterpret_cast<copy4_t *>
                			(warp_ptr_b +  i * global_n_rows
                        		+ laneId  * 4)[0];		
		}
			
#pragma unroll
		for ( unsigned int i = 0; i < N_COL ; ++i )
		{
			if (i%(PREF) == 0)
			{
#pragma unroll
				for( unsigned int j = 0; j < PREF; ++j )
				{
					reinterpret_cast<copy4_t *>
					(&prefetch[laneId+warpId*WARP_SIZE][j*4])[0] =  
					reinterpret_cast<copy4_t *> 
					(warp_ptr_a +  (j + (i/(PREF))*(PREF )) * global_n_rows
                                        	+ laneId  * 4)[0];
				}
			}
			
			fragment_a =  reinterpret_cast<copy4_t *>
                                 (&prefetch[laneId+warpId*WARP_SIZE][(i%(PREF))*4])[0];
			
#pragma unroll
			for ( unsigned int j = 0; j < N_COL; ++j )
			{
		
				type_t result =	0.5 *( 	fragment_a.x * fragment_b[j].x + 
							fragment_a.y * fragment_b[j].y +
						 	fragment_a.z * fragment_b[j].x +
						 	fragment_a.w * fragment_b[j].w) ;
			//	printf(" %f ", result);
	
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
		//////////////////////////////////////////////////////	
		warp_ptr_a += gridDim.x * LOADS_PER_BLOCK; // <<-- check this
                warp_ptr_b += gridDim.x * LOADS_PER_BLOCK;	
	}
	///////////////////////////////////////////////////////
	__syncthreads();
	// block level reduction
	// not enough threads to do everything here so let every thread do double the work
	if ( threadIdx.x < N_COL * N_COL/2)
	{
		type_t val_reduce1 = 0, val_reduce2 = 0;	
		
                        val_reduce1 += shmem_warp_result[  threadIdx.x/N_COL + 0*N_COL]
							[  threadIdx.x%N_COL]
					+ shmem_warp_result[  threadIdx.x%N_COL + 0*N_COL]
                                                        [  threadIdx.x/N_COL];
			val_reduce1 += shmem_warp_result[  threadIdx.x/N_COL + 1*N_COL]
                                                        [  threadIdx.x%N_COL]
                                        + shmem_warp_result[  threadIdx.x%N_COL + 1*N_COL]
                                                        [  threadIdx.x/N_COL];
			val_reduce1 += shmem_warp_result[  threadIdx.x/N_COL + 2*N_COL]
                                                        [  threadIdx.x%N_COL]
                                        + shmem_warp_result[  threadIdx.x%N_COL + 2*N_COL]
                                                        [  threadIdx.x/N_COL];
			val_reduce1 += shmem_warp_result[  threadIdx.x/N_COL + 3*N_COL]
                                                        [  threadIdx.x%N_COL]
                                        + shmem_warp_result[  threadIdx.x%N_COL + 3*N_COL]
                                                        [  threadIdx.x/N_COL];
			val_reduce1 += shmem_warp_result[  threadIdx.x/N_COL + 4*N_COL]
                                                        [  threadIdx.x%N_COL]
                                        + shmem_warp_result[  threadIdx.x%N_COL + 4*N_COL]
                                                        [  threadIdx.x/N_COL];
			val_reduce1 += shmem_warp_result[  threadIdx.x/N_COL + 5*N_COL]
                                                        [  threadIdx.x%N_COL]
                                        + shmem_warp_result[  threadIdx.x%N_COL + 5*N_COL]
                                                        [  threadIdx.x/N_COL];			
			
			 atomicAdd( &R[ (threadIdx.x/N_COL) * N_COL + threadIdx.x%N_COL],
                                val_reduce1 );


			val_reduce2 += shmem_warp_result[ threadIdx.x/N_COL + 0*N_COL + (N_COL/2)]
                                		        [  threadIdx.x%N_COL]
					+ shmem_warp_result[ threadIdx.x%N_COL + 0*N_COL ]
                                                        [  threadIdx.x/N_COL + (N_COL/2)];
			val_reduce2 += shmem_warp_result[ threadIdx.x/N_COL + 1*N_COL + (N_COL/2)]
                                                        [  threadIdx.x%N_COL]
                                        + shmem_warp_result[ threadIdx.x%N_COL + 1*N_COL ]
                                                        [  threadIdx.x/N_COL + (N_COL/2)];
			val_reduce2 += shmem_warp_result[ threadIdx.x/N_COL + 2*N_COL + (N_COL/2)]
                                                        [  threadIdx.x%N_COL]
                                        + shmem_warp_result[ threadIdx.x%N_COL + 2*N_COL ]
                                                        [  threadIdx.x/N_COL + (N_COL/2)];
			val_reduce2 += shmem_warp_result[ threadIdx.x/N_COL + 3*N_COL + (N_COL/2)]
                                                        [  threadIdx.x%N_COL]
                                        + shmem_warp_result[ threadIdx.x%N_COL + 3*N_COL ]
                                                        [  threadIdx.x/N_COL + (N_COL/2)];
			val_reduce2 += shmem_warp_result[ threadIdx.x/N_COL + 4*N_COL + (N_COL/2)]
                                                        [  threadIdx.x%N_COL]
                                        + shmem_warp_result[ threadIdx.x%N_COL + 4*N_COL ]
                                                        [  threadIdx.x/N_COL + (N_COL/2)];
			val_reduce2 += shmem_warp_result[ threadIdx.x/N_COL + 5*N_COL + (N_COL/2)]
                                                        [  threadIdx.x%N_COL]
                                        + shmem_warp_result[ threadIdx.x%N_COL + 5*N_COL ]
                                                        [  threadIdx.x/N_COL + (N_COL/2)];

			atomicAdd( &R[( (threadIdx.x/N_COL) + (N_COL/2) ) * N_COL + threadIdx.x%N_COL]
				, val_reduce2 );
	}
}

};



void mm_tt2(	const unsigned int n_blocks,
		const unsigned int n_loads,
		Dense_matrix<float> &T1,
		Dense_matrix<float> &T2,
		Dense_matrix<float> &R)
{
	//print(n_blocks); print(n_loads); print(T1.n_rows()); print(T2.n_rows());
	if( T1.memory_space() == MemorySpace::CUDA)
	{
#ifndef DISABLE_CUDA
		tt2::mm_tt2<float, float4, 6, 4*WARP_SIZE, 6*4*WARP_SIZE, N_COL/2, STRIDE_MM_TT2>
                                <<<n_blocks, 6*WARP_SIZE>>>
                                (
				n_loads,
				T1.n_rows(),
                                T1.data(),
                                T2.data(),
                                R.data());
#endif
	}
	else
	{
		std::cout << "fill in later" << std::endl;
	}

}

