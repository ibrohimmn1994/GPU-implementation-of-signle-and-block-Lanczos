




/////////////////////////////////////////////////////////////////////////////////////////

//<<<	The texture part is not implemented		>>>
// the problem size in Lanczos does not fit in the texture and it expensive to bind repeatedly
#ifndef spmv_spmm_hpp
#define spmv_spmm_hpp
texture<float> texf;
texture<int2,1> texd;

texture<float, 2, cudaReadModeElementType> texf2D;
texture<int2, 2, cudaReadModeElementType> texd2D;

namespace text{

void bind_texture1D( float* x )
{
       AssertCuda( cudaBindTexture(NULL, texf, x) );
}
void bind_texture2D( float* x, const int rows,const int cols, size_t pitch )
{
        AssertCuda (cudaBindTexture2D(NULL, &texf2D, x, &texf2D.channelDesc,
                                       rows, cols, pitch));
}
void unbind_texture1D( float* x)
{
      AssertCuda(  cudaUnbindTexture(texf));
}
void unbind_texture2D( float* x)
{
      AssertCuda(  cudaUnbindTexture(&texf2D));
}
void bind_texture1D( double* x)
{
      AssertCuda(  cudaBindTexture(NULL, texd , x) );
}
void bind_texture2D( double* x, const int rows, const int cols, size_t pitch)
{
       AssertCuda (cudaBindTexture2D(NULL, &texd2D, x, &texd2D.channelDesc,
                                       rows, cols, pitch));
}
void unbind_texture1D( double* x)
{
        AssertCuda(  cudaUnbindTexture(texd) );
}
void unbind_texture2D( double* x)
{
        AssertCuda(  cudaUnbindTexture(&texd2D) );
}
template <bool cache>
__device__ float fetch_from_texture1D(const int i, float* x)
{
    if (cache)
        return tex1Dfetch(texf, i);
    else
        return x[i];
}
template <bool cache>
__device__ float fetch_from_texture2D(const int i, float* x,const int row, const int col)
{
    if (cache)
        return tex2D(texf2D, row, col);
    else
        return x[i];
}
template <bool cache>
__device__ double fetch_from_texture1D(const int i, double* x)
{

        if (cache)
        {
                int2 v = tex1Dfetch(texd, i);
                return __hiloint2double(v.y, v.x);
        }
        else
        {
                return x[i];
        }
}
template <bool cache>
__device__ double fetch_from_texture2D(const int i, double* x, const int row, const int col)
{

        if (cache)
        {
                int2 v = tex2D(texd2D, row, col);
                return __hiloint2double(v.y, v.x);
        }
        else
        {
                return x[i];
        }
}           
};


namespace ell{



template <typename type_t, typename copy4_t, bool cache>
__global__ void SpMV(			std::size_t ell_n_rows,
                                        type_t *ell_data,             // ell data
                                        unsigned int *ell_idx,        // ell indices
                                        type_t *vec_data,      // matrix with N_COL columns
                                        type_t *result_data )
{
        const unsigned int tid  = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < ell_n_rows)
	{
        	copy4_t fragment_data =  reinterpret_cast<copy4_t *>
                                        ( &ell_data[tid*4])[0];
        	int4 fragment_idx =  reinterpret_cast<int4 *>
                                        ( &ell_idx[tid*4])[0];

        	type_t fragment_result = 
			fragment_data.x *
               	 	text::fetch_from_texture1D<cache>(fragment_idx.x, vec_data) +
                	fragment_data.y *
                	text::fetch_from_texture1D<cache>(fragment_idx.y , vec_data) +
                	fragment_data.z *
                	text::fetch_from_texture1D<cache>(fragment_idx.z , vec_data) +
                	fragment_data.w *
                	text::fetch_from_texture1D<cache>(fragment_idx.w , vec_data);
        
        	result_data[tid] = fragment_result;
	}
 
       
}

template <typename type_t, typename copy4_t, bool cache, const unsigned int block_size>
__global__ void SpMM(			std::size_t ell_n_rows,
                                        type_t *ell_data,   	       	// ell data
                                        unsigned int *ell_idx,            	// ell indices
					unsigned int mat_n_rows,
                                        type_t *mat_data,      		// matrix with N_COL columns
                                        type_t *result_data )
{
        const unsigned int tid  = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ type_t shmem[block_size][4];
        
	if(tid < ell_n_rows)
	{
		copy4_t fragment_data =  reinterpret_cast<copy4_t *>
                                        ( &ell_data[tid*4])[0];
		int4 fragment_idx =  reinterpret_cast<int4 *>
                                        ( &ell_idx[tid*4])[0];	
		//type_t fragment_result[N_COL] = {0};
		
		copy4_t temp;
		temp.x = text::fetch_from_texture2D<cache>
                                (fragment_idx.x , mat_data,fragment_idx.x,0 );
                temp.y = text::fetch_from_texture2D<cache>
                                (fragment_idx.y , mat_data,fragment_idx.y,0);
                temp.z = text::fetch_from_texture2D<cache>
                                (fragment_idx.z , mat_data,fragment_idx.z,0);
                temp.w = text::fetch_from_texture2D<cache>
                                (fragment_idx.w , mat_data,fragment_idx.w,0);
#pragma unroll
		for ( unsigned int i = 1; i < N_COL; ++i )
		{
			shmem[threadIdx.x][0] = text::fetch_from_texture2D<cache>
			(fragment_idx.x + i*mat_n_rows, mat_data, fragment_idx.x, i);
			shmem[threadIdx.x][1] = text::fetch_from_texture2D<cache>
			(fragment_idx.y + i*mat_n_rows, mat_data, fragment_idx.y, i);
			shmem[threadIdx.x][2] = text::fetch_from_texture2D<cache>
			(fragment_idx.z + i*mat_n_rows, mat_data, fragment_idx.z, i);
			shmem[threadIdx.x][3] =  text::fetch_from_texture2D<cache>
			(fragment_idx.w + i*mat_n_rows, mat_data, fragment_idx.w, i);
				
		
			type_t fragment_result =
                		fragment_data.x * temp.x +
				fragment_data.y * temp.y +
				fragment_data.z * temp.z +
				fragment_data.w * temp.w ;

			 result_data[tid + (i-1)*mat_n_rows] = fragment_result;

			 temp = reinterpret_cast<copy4_t *>
                                 (&shmem[threadIdx.x][0])[0];

		}				
		 type_t fragment_result =
                                fragment_data.x * temp.x +
                                fragment_data.y * temp.y +
                                fragment_data.z * temp.z +
                                fragment_data.w * temp.w ;

                  result_data[tid + (N_COL-1)*mat_n_rows] = fragment_result;
	}
}

};







void spmv(	
		Ell_matrix<float> &A,
		Vector<float> &vec_in,
		Vector<float> &vec_out)
{
	
	if (A.memory_space() == MemorySpace::CUDA)
	{
#ifndef DISABLE_CUDA
		//text::bind_texture1D( vec_in.data() );
		
        	const unsigned int n_blocks =  CEIL_DIV(A.n_rows(), 256);
		ell::SpMV<float, float4, 0>
                        <<<n_blocks, 256>>>
                        (A.n_rows(), A.data(),
                        A.idx(), vec_in.data(), vec_out.data());

		//text::unbind_texture1D( vec_in.data());
#endif
	}
	else
	{
		std::cout << "implement later" << std::endl;
	}
}

void spmv( 	
		Ell_matrix<double> &A,
                Vector<double> &vec_in,
                Vector<double> &vec_out)
{

        if (A.memory_space() == MemorySpace::CUDA)
        {
#ifndef DISABLE_CUDA

		//text::bind_texture1D( vec_in.data() );
		
                const unsigned int n_blocks =  CEIL_DIV(A.n_rows(), 256);
                ell::SpMV<double, double4, 0>
                        <<<n_blocks, 256>>>
                        (A.n_rows(), A.data(),
          		A.idx(), vec_in.data(), vec_out.data());
		
		//text::unbind_texture1D( vec_in.data());
#endif
        }
        else
        {
                std::cout << "implement later" << std::endl;
        }
}

void spmm( 	
	        Ell_matrix<float> &A,
                Dense_matrix<float> &mat_in,
                Dense_matrix<float> &mat_out)
{
	if (A.memory_space() == MemorySpace::CUDA)
	{
#ifndef DISABLE_CUDA
		/*
		size_t pitch;
                float *x = 0;
                AssertCuda( cudaMallocPitch((void**)&x,&pitch,
                                mat_in.n_rows()*sizeof(float), mat_in.n_cols() ) );

                AssertCuda(cudaMemcpy2D(x, pitch, mat_in.data(), mat_in.n_rows()*sizeof(float),
                     mat_in.n_rows()*sizeof(float),mat_in.n_cols(),cudaMemcpyDeviceToDevice));

                text::bind_texture2D( x, mat_in.n_rows(), mat_in.n_cols(), pitch);		
		*/
		const unsigned int n_blocks =  CEIL_DIV(A.n_rows(), 256);
		ell::SpMM<float, float4, 0, 256>
                                <<<n_blocks, 256>>>
                        (A.n_rows(), A.data(), A.idx(), 
			mat_in.n_rows(), mat_in.data(), mat_out.data());
		
		/*
		text::unbind_texture2D(x);
                AssertCuda(  cudaFree(x) );
		*/
#endif
	}
	else
	{
		std::cout << " implement later" << std::endl;
	}
}

void spmm(     
		Ell_matrix<double> &A,
                Dense_matrix<double> &mat_in,
                Dense_matrix<double> &mat_out)
{
        if (A.memory_space() == MemorySpace::CUDA)
        {
#ifndef DISABLE_CUDA
		/*
		size_t pitch;
                double *x = 0;
                AssertCuda( cudaMallocPitch((void**)&x,&pitch,
                                mat_in.n_rows()*sizeof(double), mat_in.n_cols() ) );

                AssertCuda(cudaMemcpy2D(x, pitch, mat_in.data(), mat_in.n_rows()*sizeof(double),
                     mat_in.n_rows()*sizeof(double),mat_in.n_cols(),cudaMemcpyDeviceToDevice));

                text::bind_texture2D( x, mat_in.n_rows(), mat_in.n_cols(), pitch);		
		*/
                const unsigned int n_blocks =  CEIL_DIV(A.n_rows(), 256);
                ell::SpMM<double, double4, 0, 256>
                                <<<n_blocks, 256>>>
                        (A.n_rows(), A.data(), A.idx(), 
			mat_in.n_rows(), mat_in.data(), mat_out.data());
		/*
		text::unbind_texture2D(x);
                AssertCuda(  cudaFree(x) );
		*/
#endif
        }
        else
        {
                std::cout << " implement later" << std::endl;
        }
}


#endif



