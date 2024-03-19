
#ifndef cpy_funcs_hpp
#define cpy_funcs_hpp


#ifndef DISABLE_CUDA

namespace hf{

template<typename type_t>
__global__ void copy_row_to_vector(     const unsigned int n_rows,
					const unsigned int n_col,
					const unsigned int start_ind,
					unsigned int lc,
					const type_t* mat,
					type_t* vec)
{
	const unsigned int ind = threadIdx.x + blockIdx.x * blockDim.x;
	if ( ind < n_col )
	{
		
		vec[start_ind + ind] = mat[lc + ind*n_rows];
	}
}
} // end namespace
#endif
///////////////////////////////////////////////////////////////////////////////////



template<typename type_t>
void copy_row_to_vector(		unsigned int lc,
					const unsigned int vec_ind,
					Dense_matrix<type_t> &mat,
					Vector<type_t> &vec)
{
	if ( mat.memory_space() == MemorySpace::CUDA )
	{
#ifndef DISABLE_CUDA
		hf::copy_row_to_vector<type_t><<<1, 32>>>
						(       mat.n_rows(),
							mat.n_cols(),
							vec_ind,
							lc,
							mat.data(),
							vec.data());		
#endif
	}
	else
	{
		for ( unsigned int  i = 0; i < mat.n_cols(); ++i )
		{
			vec(vec_ind + i) = mat( lc + i * mat.n_rows());
		}
	}
	
}

template <typename type_t>
void copy_columns_to_matrix(    Dense_matrix<type_t> &mat1,
                                Dense_matrix<type_t> &mat2,
                                const unsigned int n_cols)
{
        if ( mat1.memory_space() == MemorySpace::CUDA )
        {
#ifndef DISABLE_CUDA

                AssertCuda( cudaMemcpy( mat2.data(), mat1.data(),
                           mat1.n_rows() * n_cols * sizeof(type_t), cudaMemcpyDeviceToDevice) );
#endif
        }
        else
        {
                for ( unsigned int i = 0; i < mat1.n_rows() * n_cols; ++i )
                        mat2(i) = mat1(i);
        }
}

template <typename type_t>
void copy_column_to_vector(	const Dense_matrix<type_t> &mat,
				Vector<type_t> &vec,
				const unsigned int col)
{
	if ( mat.memory_space() == MemorySpace::CUDA )
	{
#ifndef DISABLE_CUDA

		AssertCuda( cudaMemcpy( vec.data(), &( mat.data()[col * mat.n_rows() ]),
                                vec.size() * sizeof(type_t), cudaMemcpyDeviceToDevice) );
#endif
	}
	else
	{
		for ( unsigned int i = 0; i < vec.size(); ++i )
			vec(i) = mat(i + col * vec.size());
	}
}
template <typename type_t>
void copy_vector_to_column(      const Vector<type_t> &vec,
		 		Dense_matrix<type_t> &mat,
				const unsigned int col)
{
	if ( vec.memory_space() == MemorySpace::CUDA )
	{
#ifndef DISABLE_CUDA
		AssertCuda( cudaMemcpy(  &( mat.data()[col * mat.n_rows() ] ), vec.data(), 
				mat.n_rows() * sizeof(type_t), cudaMemcpyDeviceToDevice) );			  
#endif
	}
	else
	{
		for ( unsigned int i = 0; i < vec.size(); ++i )
			mat(i + col * vec.size()) = vec(i);
	}
}
template <typename type_t>
void copy_vector_element(       		const Vector<type_t> &vec1,
                                		const unsigned int src,
						Vector<type_t> &vec2, 
                                		const unsigned int dest)
{
        if ( vec1.memory_space() == MemorySpace::CUDA )
        {
#ifndef DISABLE_CUDA
             	AssertCuda( cudaMemcpy(&(vec2.data()[dest]), &(vec1.data()[src]), 
			sizeof(type_t), cudaMemcpyDeviceToDevice) );
#endif
    	}
      	else
     	{
  	        vec2(dest) = vec1(src);
        }
}
////////////////////////////////////////////////////////////////////////////////////////

#endif

