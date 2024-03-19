

namespace ts{

template < 	typename type_t, typename copy4_t, unsigned int WARPS_PER_BLOCK,
                unsigned int LOADS_PER_WARP, unsigned int LOADS_PER_BLOCK, unsigned int PREF >
__global__ void __launch_bounds__(WARPS_PER_BLOCK*WARP_SIZE)
		mm_ts1(		const unsigned int number_of_loads,
				const unsigned int global_n_rows,
			 	type_t *T, // tall matrix
				type_t *S, // small matrix
				type_t *R)	// result matrix
{
	const unsigned int warpId = threadIdx.x / WARP_SIZE;
        const unsigned int laneId = threadIdx.x % WARP_SIZE;
	
	__shared__ type_t shmem[N_COL * N_COL];
	
	__shared__ type_t prefetch[WARPS_PER_BLOCK*WARP_SIZE][4*(PREF)];
	
	// Number of threads might not be enough to store S in case
	// of size 16*16 so we do it in two stages
	if ( threadIdx.x < N_COL*(N_COL/2) )
	{
		shmem[threadIdx.x] =
                                S[threadIdx.x ];
                shmem[threadIdx.x + (N_COL/2)*N_COL] =
                                S[threadIdx.x + (N_COL/2)*N_COL ];	
	}
	
	type_t *warp_ptr_T =  &T[blockIdx.x * LOADS_PER_BLOCK] +
				(warpId * LOADS_PER_WARP); //<<--

	type_t *warp_ptr_R =  &R[blockIdx.x * LOADS_PER_BLOCK] +
                                (warpId * LOADS_PER_WARP );
	// registers are for thread level computation and reduction
	copy4_t fragment_R[N_COL] = {0};
	copy4_t fragment_T ;
	//copy4_t result; 

	//////////////////////////////////////////////////////////
	__syncthreads();
	
	for ( unsigned int load = 0; load < number_of_loads; load++ )
	{		
		unsigned int flag = 0;
#pragma unroll
		for ( unsigned int i = 0; i < N_COL ; ++i )
		{
			// prefetching
			if (i%(PREF) == 0)
			{				
#pragma unroll	
				for( unsigned int j = 0; j < PREF; ++j )
				{
				 reinterpret_cast<copy4_t *>
                                        (&prefetch[laneId+warpId*WARP_SIZE][j*4])[0] =
                                        reinterpret_cast<copy4_t *>
                                        (warp_ptr_T +  (j + flag*(PREF) ) * global_n_rows
                                                + laneId  * 4)[0];
				}	
				flag++;
			}

			//prefetched data into the register
			fragment_T =  reinterpret_cast<copy4_t *>
                                 (&prefetch[laneId+warpId*WARP_SIZE][(i%(PREF))*4])[0];

			if( i == 0 )
			{
#pragma unroll
				for ( unsigned int j = 0; j < N_COL; ++j )
				{
					type_t shmem_val = shmem[i*N_COL + j];
                                        fragment_R[j].x = fragment_T.x * shmem_val;
                                        fragment_R[j].y = fragment_T.y * shmem_val;
                                        fragment_R[j].z = fragment_T.z * shmem_val;
                                        fragment_R[j].w = fragment_T.w * shmem_val;
				}
			}
			else
			{
#pragma unroll
				for ( unsigned int j = 0; j < N_COL; ++j )
				{
					type_t shmem_val = shmem[i*N_COL + j];
					fragment_R[j].x += fragment_T.x * shmem_val;
                        		fragment_R[j].y += fragment_T.y * shmem_val;
                        		fragment_R[j].z += fragment_T.z * shmem_val;
                        		fragment_R[j].w += fragment_T.w * shmem_val;
				}
			}
		}
		for ( unsigned int j = 0; j < N_COL; ++j )
		{
			reinterpret_cast<copy4_t *>
                                (warp_ptr_R + j * global_n_rows + laneId * 4)[0]
                                        = fragment_R[j];
		}
		
		warp_ptr_T += gridDim.x * LOADS_PER_BLOCK; // <<-- check this
                warp_ptr_R += gridDim.x * LOADS_PER_BLOCK;
	}
	///////////////////////////////////////////////////////
}

template <      typename type_t, typename copy4_t, unsigned int WARPS_PER_BLOCK,
                unsigned int LOADS_PER_WARP, unsigned int LOADS_PER_BLOCK, unsigned int PREF >
__global__ void __launch_bounds__(WARPS_PER_BLOCK*WARP_SIZE)
		mm_ts2(		const unsigned int number_of_loads,
				const std::size_t global_n_rows,
				const type_t my_scalar,
				const type_t other_scalar,
			 	type_t *T, // tall matrix
				type_t *S, // small matrix
				type_t *R)	// result matrix
{
	const unsigned int warpId = threadIdx.x / WARP_SIZE;
        const unsigned int laneId = threadIdx.x % WARP_SIZE;
	
	__shared__ type_t shmem[N_COL * N_COL];
	
	__shared__ type_t prefetch[WARPS_PER_BLOCK*WARP_SIZE][4*(PREF) ];
	
	// Number of threads might not be enough to store S in case
	// of size 16*16 so we do it in two stages
	if ( threadIdx.x < N_COL*(N_COL/2) )
	{
		shmem[threadIdx.x] =
                                S[threadIdx.x ];
                shmem[threadIdx.x + (N_COL/2)*N_COL] =
                                S[threadIdx.x + (N_COL/2)*N_COL ];	
	}
	
	type_t *warp_ptr_T =  &T[blockIdx.x * LOADS_PER_BLOCK] +
				(warpId * LOADS_PER_WARP ); //<<--

	type_t *warp_ptr_R =  &R[blockIdx.x * LOADS_PER_BLOCK] +
                                (warpId * LOADS_PER_WARP);
	// registers are for thread level computation and reduction
	copy4_t fragment_R[N_COL] = {0};
	copy4_t fragment_T ;
	//copy4_t result; 

	//////////////////////////////////////////////////////////
	__syncthreads();
	
	for ( unsigned int load = 0; load < number_of_loads; load++ )
	{		

		unsigned int flag = 0;
#pragma unroll
		for ( unsigned int i = 0; i < N_COL ; ++i )
		{
			// prefetching
			if (i%(PREF) == 0)
			{				
#pragma unroll	
				for( unsigned int j = 0; j < PREF; ++j )
				{
				 reinterpret_cast<copy4_t *>
                                        (&prefetch[laneId+warpId*WARP_SIZE][j*4])[0] =
                                        reinterpret_cast<copy4_t *>
                                        (warp_ptr_T +  (j + flag*(PREF) ) * global_n_rows
                                                + laneId  * 4)[0];
				}
				
				flag++;
			}

			//prefetched data into the register
			fragment_T =  reinterpret_cast<copy4_t *>
                                 (&prefetch[laneId+warpId*WARP_SIZE][(i%(PREF))*4])[0];

			if( i == 0 )
			{
#pragma unroll
				for ( unsigned int j = 0; j < N_COL; ++j )
				{
					type_t shmem_val = shmem[i*N_COL + j];
                                        fragment_R[j].x = fragment_T.x * shmem_val;
                                        fragment_R[j].y = fragment_T.y * shmem_val;
                                        fragment_R[j].z = fragment_T.z * shmem_val;
                                        fragment_R[j].w = fragment_T.w * shmem_val;
				}
			}
			else
			{
#pragma unroll
				for ( unsigned int j = 0; j < N_COL; ++j )
				{
					type_t shmem_val = shmem[i*N_COL + j];
					fragment_R[j].x += fragment_T.x * shmem_val;
                        		fragment_R[j].y += fragment_T.y * shmem_val;
                        		fragment_R[j].z += fragment_T.z * shmem_val;
                        		fragment_R[j].w += fragment_T.w * shmem_val;
				}
			}
		}
#pragma unroll
		for ( unsigned int j = 0; j < N_COL; ++j )
		{
			copy4_t temp = reinterpret_cast<copy4_t *>
                                (warp_ptr_R + j * global_n_rows + laneId * 4)[0];
			temp.x =  temp.x - fragment_R[j].x;
			temp.y =  temp.y - fragment_R[j].y;
			temp.z =  temp.z - fragment_R[j].z;
			temp.w =  temp.w - fragment_R[j].w;
			
			reinterpret_cast<copy4_t *>
                                (warp_ptr_R + j * global_n_rows + laneId * 4)[0]
                                        = temp;
		}
		
		warp_ptr_T += gridDim.x * LOADS_PER_BLOCK; // <<-- check this
                warp_ptr_R += gridDim.x * LOADS_PER_BLOCK;
	}
	///////////////////////////////////////////////////////
}
};


void mm_ts(	const unsigned int n_blocks,
		const unsigned int n_loads,
		float my_scalar,
		float other_scalar,
		Dense_matrix<float> &T,
		Dense_matrix<float> &S,
		Dense_matrix<float> &R)
{
	if ( T.memory_space() == MemorySpace::CUDA)
	{
#ifndef DISABLE_CUDA
		ts::mm_ts2<float, float4, 6, 4*WARP_SIZE, 6*4*WARP_SIZE, N_COL/2>
                                <<<n_blocks, 6*WARP_SIZE>>>
                                (n_loads,
				T.n_rows(),
                                my_scalar,
                                other_scalar,
                                T.data(),
                                S.data(),
                                R.data());
#endif
	}
	else
	{
		std::cout << "fill later" << std::endl;
	}
}

void mm_ts(	const unsigned int n_blocks,
		const unsigned int n_loads,
		Dense_matrix<float> &T,
		Dense_matrix<float> &S,
		Dense_matrix<float> &R)
{
	if ( T.memory_space() == MemorySpace::CUDA)
	{
#ifndef DISABLE_CUDA
		ts::mm_ts1<float, float4, 6, 4*WARP_SIZE, 6*4*WARP_SIZE, N_COL/2>
                                <<<n_blocks, 6*WARP_SIZE>>>
                                (n_loads,
				T.n_rows(),
                                T.data(),
                                S.data(),
                                R.data());
#endif
	}
	else
	{
		std::cout << "fill later" << std::endl;
	}
}

/*
#define Number float
#define Number4 float4

int main(int argc, char **argv)
{

 	//CudaDeviceInfo();
	if (1)
	{
	const unsigned int len = 40* WARPS_PER_BLOCK*WARP_SIZE * 10 * 4;
        MemorySpace mem = MemorySpace::CUDA;
        Dense_matrix<Number> T( len, N_COL, mem);
	Dense_matrix<Number> S(N_COL, N_COL, mem);
        Dense_matrix<Number> R(T);

	Number my_scalar = 1.;
	Number other_scalar = 1.;

        double best = 1e10;
        unsigned int rep , n_test = 10, n_rep = 100;
        T = 1;
	S = 1;

        for (unsigned int t = 0; t<n_test; ++t)
        {
                cudaDeviceSynchronize();
                auto t1 = std::chrono::steady_clock::now();
                for (rep = 0; rep < n_rep ; ++ rep)
                {
                        ts::mm_ts1<Number, Number4, 10, WARPS_PER_BLOCK*WARP_SIZE>
                                <<<40,WARPS_PER_BLOCK*WARP_SIZE>>>
                                (T.n_rows(),
				my_scalar,
				other_scalar,
                                T.data(),
				S.data(),
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

        std::cout << "Gb/s: " <<
        1e-9*( (T.memory_consumption()*2 + S.memory_consumption() ))/best << std::endl;

        std::cout << " Gflops/s: " <<
        1e-9*( ( 2*T.n_rows()*T.n_cols()*T.n_cols()  )  )/best<< std::endl;

        std::cout << " Gflops: " <<
        1e-9*( ( 2*T.n_rows()*T.n_cols()*T.n_cols() + T.n_rows()*T.n_cols() )  )<< std::endl;	
	}
	else
	{
	const unsigned int len = 1* WARPS_PER_BLOCK*WARP_SIZE * 2 * 4;		
	MemorySpace mem = MemorySpace::CUDA;
	Dense_matrix<float> T( len, N_COL, mem);
	Dense_matrix<float> S(N_COL, N_COL, mem);
	Dense_matrix<float> R(T);
	float my_scalar = 1.;
	float other_scalar = 1.;

	T = 1;
	S = 1;

	ts::mm_ts2<float, float4, 2, WARPS_PER_BLOCK*WARP_SIZE>
				<<<1,WARPS_PER_BLOCK*WARP_SIZE>>>(T.n_rows(),
						my_scalar,
						other_scalar, 
						T.data(),
						S.data(),
						R.data());

	R.print();
	}
                                
	return 0;
}


*/











