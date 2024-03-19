

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
	__shared__ type_t shmem_warp_result[N_COL * WARPS_PER_BLOCK]
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
//#pragma unroll
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















