
#ifndef vector_kernels_hpp
#define vector_kernels_hpp


#ifndef DISABLE_CUDA

namespace v{

template <typename Number>
__global__ void set_entries(	unsigned int N, 
				Number scalar, 
				Number* vector_result)
{
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < N)
	{
		vector_result[idx] = scalar;
	}
}
template <typename Number>
__global__ void vector_update(	unsigned int N, 
				Number scalar1, 
				Number scalar2, 
				const Number *array, 
				Number *result)
{
	const unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if ( idx < N )
	{
		result[idx] = scalar1*result[idx] + scalar2*array[idx];
	}
}
template <unsigned int block_size, typename Number>
__global__ void
do_dot(unsigned int n, Number *vector1, Number *vector2, Number *result)
{
  	__shared__ Number sdata[block_size];

  	unsigned int tid = threadIdx.x;
  	unsigned int i   = blockIdx.x * blockDim.x + tid;

  	if (i < n)
	{
    		sdata[tid] = vector1[i] * vector2[i];
	}
  	else
	{
    		sdata[tid] = 0;
	}
  	__syncthreads();

  	for (unsigned int s = block_size / 2; s > 0; s /= 2)
  	{
      		if (tid < s)
        	{
         		 sdata[tid] += sdata[tid + s];
          		__syncthreads();
        	}
    	}

  	if (tid == 0)
    		atomicAdd(result, sdata[0]);
}

template< typename Number>
__global__ void exp(	unsigned int size, Number* data)
{
	const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if ( tid < size )
	{
		data[tid] = std::exp( data[tid]);
	}
}
} // end namespace
#endif



#endif
//##################################################################################################################
