

#ifndef dense_kernels_hpp
#define dense_kernels_hpp

#ifndef DISABLE_CUDA

namespace dm{

template <typename Number>
__global__ void mv(	 const unsigned int n_rows,
			 const unsigned int n_col,
			 const Number* matrix,
			 const Number* array,
			 Number* result)
{
	const unsigned int ind = threadIdx.x + blockIdx.x * blockDim.x;
	Number sum = 0;
	if ( ind < n_rows )
	{
		
		for ( unsigned int col = 0; col < n_col; ++col )
		{
			sum += matrix[ind + col*n_rows] * array[col];			
			if ( ind == 0 )
			{
				printf( "%f ",  matrix[ind + col*n_rows]);
			}		
		}
		result[ind] = sum;
	}
}
// this one is basic the one below is replaed by matmat_TS_V2

template <typename Number>
__global__ void mm_add(	 const unsigned int size,
				 Number* data,
				 const Number* otherdata,
				 const Number my_scalar,
				 const Number other_scalar)
{
	const unsigned int ind = threadIdx.x + blockIdx.x * blockDim.x;
	
	if ( ind < size )	
	{
		data[ind] = my_scalar*data[ind] + other_scalar*otherdata[ind];
	}
}
// this is takes the output of cusolver to obtain for expm(T) 
// It is not part of the measurements so not optimized
//
template <typename Number>
__global__ void custom_mult(	const unsigned int n_rows,
				const Number* eigenval,
				const Number* eigenvec,
				Number* data)
{
	const unsigned int row = threadIdx.x + blockIdx.x * blockDim.x;
	const unsigned int col = threadIdx.y + blockIdx.y * blockDim.y;

	Number sum = 0;
	if ( row < n_rows && col < n_rows )
	{
		for ( unsigned int i = 0; i < n_rows; ++i ) 
		{
			sum += eigenvec[row+i*n_rows] *
				 std::exp(eigenval[i]) * 
				 eigenvec[col+i*n_rows];
		}
	}
	
	__syncthreads();
	if ( row < n_rows && col < n_rows )
	{
		data[row+col*n_rows] = sum;
	}
	
}
template<typename Number>
__global__ void diagonal_matrix_from_vector(	const unsigned int n_rows,
						const Number* array,
						Number* data)
{
	const unsigned int ind = threadIdx.x + blockIdx.x * blockDim.x;
	if (ind < n_rows)
	{
		data[ind+ind*n_rows] = array[ind];
	}
}
template <typename Number>
__global__ void matrix_padding(		const unsigned int n_rows,
					const unsigned int new_n_rows,
					const unsigned int n_col,
					const Number* data,
					Number* result_data)
{
	const unsigned int ind = threadIdx.x + blockIdx.x * blockDim.x;
	if( ind < n_rows )
	{
		for ( unsigned int j = 0; j < n_col; ++j )
		{
			result_data[ind + j*new_n_rows] = data[ind + j*n_rows];
		}
	} 
}
template <typename Number>
__global__ void add_scalar(	const unsigned int size,	
				Number scalar,
				Number* data)
{
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < size)
	{
		data[idx] = data[idx] + scalar;
	}
}

}// namespace
				
#endif

#endif
