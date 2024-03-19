
#ifndef ell_kernels_hpp
#define ell_kernels_hpp



#ifndef DISABLE_CUDA


namespace lm{
// basic ell-SpMV
template <typename Number>
__global__ void spmv_basic(	const unsigned int n_rows,
				const unsigned int size,
				const unsigned int width,
				Number* data,
				unsigned int* idx,
				const Number* array,
				Number* result)
{
	const unsigned int ind = threadIdx.x + blockIdx.x * blockDim.x;
	
	if ( ind < n_rows )
	{
		Number sum = 0;
		for (unsigned int i = 0; i < width; ++i )
		{
			sum += data[ind+i*n_rows] * array[idx[ind+i*n_rows]];
			
		}
		result[ind] = sum;
		
	}
}		
// basic spmm
template <typename Number>
__global__ void spmm_basic(	const unsigned int A_n_rows,
				const unsigned int A_width,
				const unsigned int B_size,
				const unsigned int B_n_rows,
				const unsigned int B_n_col,
				const Number* A_data,
				const unsigned int* A_idx,
				const Number* B,
				Number* result )
{
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if ( tid < A_n_rows )
        {
		for (unsigned int j = 0; j < B_n_col; ++j)
		{
                	Number sum = 0;
                	for (unsigned int i = 0; i < A_width; ++i )
                	{
                        	sum += A_data[tid+i*A_n_rows] * B[A_idx[tid+i*A_n_rows] + j*B_n_rows];

                	}
               	 	result[tid + j*B_n_rows] = sum;
		}
        }
}
template <typename Number>
__global__ void diag_inv(	const unsigned int size,
					Number* data)
{
	const unsigned int ind = threadIdx.x + blockIdx.x * blockDim.x;
	if (ind < size)
	{
		Number value =  1./data[ind];
		data[ind] = value;
	}
}
template <typename Number>
__global__ void diag_sqrt(	const unsigned int size,
					Number* data)
{
	const unsigned int ind = threadIdx.x + blockIdx.x * blockDim.x;
	if (ind < size)
	{
		Number value = std::sqrt(data[ind]);
		data[ind] = value;
	}
}
template <typename Number>
__global__ void mm_with_diag(	const std::size_t size,
				Number* data,
				const unsigned int* idx,
				const Number* diag )
{
	const unsigned int ind = threadIdx.x + blockIdx.x * blockDim.x;

	if (ind < size)
	{
		unsigned int col = idx[ind];
		data[ind] = data[ind] * diag[col];
	}
}
template <typename Number>
__global__ void change_major(	const unsigned int size,
				const unsigned int n_rows,
				const unsigned int width,
				const Number* data,
				const unsigned int* idx,
				Number* result_data,
				unsigned int* result_idx,
				const unsigned int stride)
{
	const unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int row, col, s, ind;
	if (i < size)
	{
		for ( s = 0; s < width/stride; ++s )
		{
			ind = i + s*stride*n_rows;
			row = i%n_rows;
			col = i/n_rows;
			result_data[row*stride + col + s*stride*n_rows]	 = data[ind];
			result_idx[row*stride + col + s*stride*n_rows] = idx[ind];
		}
	}
}
template<typename Number>
__global__ void padding(	const unsigned int n_rows,
				const unsigned int new_n_rows,
				const unsigned int stride,
				const unsigned int iter,
				const Number* data,
				const unsigned int* idx,
				Number * result_data,
				unsigned int* result_idx)
{
	const unsigned int ind = threadIdx.x + blockIdx.x * blockDim.x;

	if ( ind < n_rows*stride )
	{
		for ( unsigned int i = 0; i < iter; ++i )
		{
			result_data[ind + i*new_n_rows*stride] = data[ind +i*n_rows*stride];
			result_idx[ind + i*new_n_rows*stride] = idx[ind + i*n_rows*stride];
		}
	}
}

}//end namespace


#endif


#endif

//##################################################################################################################
