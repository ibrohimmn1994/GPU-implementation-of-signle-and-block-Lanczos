
#ifndef tridiagonal_matrix_hpp
#define tridiagonal_matrix_hpp
 

#include <tuple>



#ifndef DISABLE_CUDA
namespace lt{

template <typename Number>
__global__ void insert_diag_blocks(      	const unsigned int T_n_rows,
                                 	      	const unsigned int block_n_rows,
                                       		const Number* block,
                                       		Number* T,
				 		const unsigned int block_num)
{
        const unsigned int idx = threadIdx.x;
        const unsigned int row = idx % block_n_rows;
        const unsigned int col = idx / block_n_rows;
	const unsigned int size = block_n_rows * block_n_rows;
        if (idx < size)
        {
                unsigned int index = ( row + block_num * block_n_rows ) +
                                ( col + block_num * block_n_rows )* T_n_rows;
                T[index] = block[idx];
        }
}
template<typename Number>
__global__ void insert_subdiag_blocks(       	const unsigned int T_n_rows,
                                      	   	const unsigned int block_n_rows,
                                           	const Number* block,
                                           	Number* T,
						const unsigned int block_num)
{
        const unsigned int idx = threadIdx.x;
        const unsigned int row = idx % block_n_rows;
        const unsigned int col = idx / block_n_rows;
	const unsigned int size = block_n_rows * block_n_rows;
        unsigned int index1, index2;
        if (idx < size)
        {
                Number scalar =  block[ idx ];
                index1 = ( row + (block_num - 1 ) * block_n_rows ) +
                        ( col + ( block_num  ) * block_n_rows ) * T_n_rows;
                index2 = ( row + ( block_num - 1 ) * block_n_rows ) * T_n_rows +
                        ( col + ( block_num  ) * block_n_rows);

                T[index1] = scalar;
                T[index2] = scalar;
        }
}


template <typename Number>
__global__ void insert_diag_entries(            const unsigned int n_entries,
                                                Number* T,
                                                Number* entries)
{
	const unsigned int tid = threadIdx.x;
	if ( tid < n_entries )
	{
		T[tid + tid * n_entries] = entries[tid];	
	}
}

template <typename Number>
__global__ void insert_subdiag_entries(         const unsigned int n_entries,
                                               	Number* T,
                                                Number* entries)
{
        const unsigned int tid = threadIdx.x;
        if ( tid < n_entries )
        {
                T[tid + 1 + tid * n_entries] = entries[tid];
		T[tid + (tid + 1) * n_entries] = entries[tid];
        }
}

}; // end namespace
#endif

////////////////////////////////////////////////////////////////////////////////////

//		<<<	Assembly of T	>>>


template<typename Number>
Dense_matrix<Number> Assemble_T(        const unsigned int n_blocks,
                                        Dense_matrix<Number> *diag_blocks,
                                        Dense_matrix<Number> *subdiag_blocks)
{
	const unsigned int block_dim = diag_blocks[0].n_rows();
	MemorySpace mem = diag_blocks[0].memory_space();
        Dense_matrix<Number> T(n_blocks*block_dim, n_blocks*block_dim, mem);

	if (mem == MemorySpace::CUDA)
	{
#ifndef DISABLE_CUDA
		for ( unsigned int  b = 1; b < n_blocks; ++b)
		{
			lt::insert_diag_blocks<Number>
                                <<<1,block_dim*block_dim>>>
                                        (T.n_rows(),
                                        block_dim,
                                        diag_blocks[b].data(),
                                        T.data(),
					b);
                	lt::insert_subdiag_blocks<Number>
                                <<<1,block_dim*block_dim>>>
                                        (T.n_rows(),
                                        block_dim,
                                        subdiag_blocks[b].data(),
                                        T.data(),
					b);
		}
		lt::insert_diag_blocks<Number>
                          <<<1,block_dim*block_dim>>>
                                 (T.n_rows(),
                                  block_dim,
                                  diag_blocks[0].data(),
                                  T.data(),
					0);	
#endif
	}
	else
	{
		unsigned int i, b, new_row_diag, new_col_diag,
                                new_row_subdiag, new_col_subdiag;
                const unsigned int size = block_dim*block_dim;
                for ( b = 1; b < n_blocks-1; ++b )
                {
                        for( i = 0; i < size; ++i )
                        {
                                new_row_diag   = (b*block_dim) + i%block_dim;
                                new_col_diag   = (b*block_dim) + i/block_dim;
                                new_row_subdiag    = ((b-1)*block_dim) + i%block_dim;
                                new_col_subdiag    = ((b)*block_dim) + i/block_dim;

                                T(new_row_diag + new_col_diag * T.n_rows()) =
                                               diag_blocks[b](i);
                                T(new_row_subdiag + new_col_subdiag * T.n_rows()) =
                                                subdiag_blocks[b](i);
                                T(new_col_diag + new_row_subdiag * T.n_rows()) =
                                                subdiag_blocks[b](i);
                        }
                }
                for (i = 0; i < size; ++i)
                {
                        T(i%block_dim+ i/block_dim*T.n_rows()) = diag_blocks[0](i);
                }
	}
	return T;
}


template<typename Number>
Dense_matrix<Number> Assemble_T(        const unsigned int n_entries,
                                        Number *diag_entries,
                                        Number *subdiag_entries,
					MemorySpace mem)
{       
        Dense_matrix<Number> T(n_entries, n_entries, mem);
        if (mem == MemorySpace::CUDA)
	{
#ifndef DISABLE_CUDA
		Number *d_diag_entries;
		Number *d_subdiag_entries;		
		AssertCuda( cudaMalloc(&d_diag_entries, n_entries * sizeof(Number)) );
		AssertCuda( cudaMalloc(&d_subdiag_entries, (n_entries-1) * sizeof(Number)) );

		AssertCuda( cudaMemcpy(d_diag_entries, diag_entries, 
				n_entries * sizeof(Number), cudaMemcpyHostToDevice) );	

		AssertCuda( cudaMemcpy(d_diag_entries, &(subdiag_entries[1]), 
				(n_entries - 1) * sizeof(Number), cudaMemcpyHostToDevice) );

		lt::insert_diag_entries<Number><<<1, n_entries>>>
				( n_entries,
                                  T.data(),
                                  d_diag_entries);		

		lt::insert_subdiag_entries<Number><<<1, n_entries-1>>>
                                ( n_entries-1,
                                  T.data(),
                                  d_subdiag_entries);

		AssertCuda( cudaFree(d_diag_entries) );
		AssertCuda( cudaFree(d_subdiag_entries) );
#endif
	}
	else
	{
		for ( unsigned int i = 1; i < n_entries ; ++i )
		{
			T( i + n_entries*i ) = diag_entries[i];
			T( i - 1 + n_entries*i ) = diag_entries[i];
			T( i + n_entries * (i - 1) ) = diag_entries[i];
		}
		T( 0 ) = diag_entries[0];
	}
	return T;
}


#endif
///////////////////////////////////////////////////////////////////////////////////
