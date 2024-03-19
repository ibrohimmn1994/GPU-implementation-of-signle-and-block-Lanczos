
#ifndef ell_matrix_hpp
#define ell_matrix_hpp




#include "dense_matrix.hpp"
#include "../kernels/ell_kernels.hpp"
template <typename Number>
class Ell_matrix
{
private:
	std::size_t _n_rows;	
	std::size_t _n_cols;	// number of columns in the dense format
	std::size_t _size;	// number of nnz
	std::size_t _width;	// number of columns in the ell format

	Number* _data;		// nnz values
	unsigned int* _idx;	// nnz column indices
	MemorySpace _memory_space;

	void set_size(	const std::size_t n_rows, 
			const std::size_t size,
			const std::size_t n_cols)
	{	 
		if ( _memory_space == MemorySpace::CUDA )
		{
#ifndef DISABLE_CUDA
			AssertCuda(cudaFree(_data)); AssertCuda(cudaFree(_idx));

			AssertCuda( cudaMalloc(&_data, size * sizeof(Number)) );
			AssertCuda( cudaMalloc(&_idx, size * sizeof(unsigned int)));
#endif
		}
		else
		{
			delete[] _data; delete[] _idx;
			_data = new Number[size];
			_idx = new unsigned int[size];
		}
	
		_n_rows = n_rows;
		_n_cols = n_cols;
		_size = size;
		_width = size/n_rows;
		
	}
public:
	static const int block_size = Vector<Number>::block_size;
	Ell_matrix(	const std::size_t n_rows,
			const std::size_t size,
			const std::size_t n_cols,
			const MemorySpace memory_space)
	: _data(nullptr), _idx(nullptr), _memory_space(memory_space)
	{
		set_size(n_rows, size, n_cols);
		if ( _memory_space == MemorySpace::CUDA )
		{
#ifndef DISABLE_CUDA
			const unsigned int n_blocks =  CEIL_DIV(_size,block_size);
			v::set_entries<Number><<<block_size, n_blocks>>>
						( 	_size, 
							Number(0),
							_data) ;
			v::set_entries<unsigned int><<<block_size, n_blocks>>>
						(	_size,
							0U,
							_idx) ;
#endif
		}
		else
		{
			for ( std::size_t i = 0; i < _size; ++i )
			{
				_data[i] = 0; _idx[i] = 0;
			}
		}
	}
	Ell_matrix(const Ell_matrix &other)
	: _data(nullptr), _idx(nullptr), _memory_space(other._memory_space)
	{
		set_size(other._n_rows, other._size, other._n_cols);
		if ( _memory_space == MemorySpace::CUDA )
		{
#ifndef DISABLE_CUDA
			AssertCuda( cudaMemcpy(_data, other._data, _size*sizeof(Number), cudaMemcpyDeviceToDevice) );
			AssertCuda( cudaMemcpy(_idx, other._idx, _size*sizeof(unsigned int), cudaMemcpyDeviceToDevice) );
#endif
		}
		else
		{
			for ( std::size_t i = 0; i < _size; ++i )
			{
				_data[i] = other._data[i];
				_idx[i] = other._idx[i];
			}
		}
	}
	~Ell_matrix()
	{
		if ( _memory_space == MemorySpace::CUDA )
		{
#ifndef DISABLE_CUDA
			AssertCuda(cudaFree(_data)); 
			AssertCuda(cudaFree(_idx));
#endif
		}
		else
		{
			delete[] _data; delete[] _idx;
		}
	}
	// () this for data , [] this for idx
	const Number &operator()(const std::size_t index) const
	{
		 return _data[index];
	}
	Number &operator()(const std::size_t index)
	{
		 return _data[index];
	}
	const std::size_t &operator[](const std::size_t index) const
	{
		return _idx[index];
	}
	unsigned int &operator[](const std::size_t index)
	{
		return _idx[index];	
	}

	std::size_t size() const
	{
		return _size;
	}
	std::size_t n_rows() const 
	{
		return _n_rows;
	}
	std::size_t n_cols() const
	{
		return _n_cols;
	}
	std::size_t width() const
	{
		return _width;
	}
	Number* data()
	{
		return _data;
	}
	const Number* data() const
	{
		return _data;
	}
	unsigned int* idx()
	{
		return _idx;
	}
	const unsigned int* idx() const
	{
		return _idx;
	}
	MemorySpace memory_space() const
	{
		return _memory_space;
	}
	Ell_matrix &operator=(const Ell_matrix &other)
	{
		set_size(other._n_rows, other._size, other._n_cols);
		if ( _memory_space == MemorySpace::CUDA )
		{
#ifndef DISABLE_CUDA
			AssertCuda( cudaMemcpy(_data, other._data, _size*sizeof(Number), cudaMemcpyDeviceToDevice) );
			AssertCuda( cudaMemcpy(_idx, other._idx, _size*sizeof(unsigned int), cudaMemcpyDeviceToDevice) );
#endif
		}
		else
		{
			for (std::size_t i = 0; i < _size; ++i )
			{
				_data[i] = other._data[i];
				_idx[i] = other._idx[i];
			}
		}
		return *this;
	}
	const Ell_matrix copy_to_device() const
	{
		if ( _memory_space == MemorySpace::CUDA )
		{
			std::cout << "You are already in the device" << std::endl;
			return *this;
		}
		else
		{
#ifndef DISABLE_CUDA
			Ell_matrix<Number> other(_n_rows, _size, _n_cols , MemorySpace::CUDA);
			AssertCuda( cudaMemcpy(other._data, _data, _size * sizeof(Number), cudaMemcpyHostToDevice) );
			AssertCuda( cudaMemcpy(other._idx, _idx, _size * sizeof(unsigned int), cudaMemcpyHostToDevice) );
			return other;
#endif
		}
	}

	const Ell_matrix copy_to_host() const
	{
		if ( _memory_space == MemorySpace::Host )
		{
			std::cout << "You are already in the host" << std::endl;
			return *this;
		}
		else
		{
#ifndef DISABLE_CUDA
			Ell_matrix<Number> other(_n_rows, _size, _n_cols,  MemorySpace::Host);
			AssertCuda( cudaMemcpy(other._data, _data, _size * sizeof(Number), cudaMemcpyDeviceToHost) );
			AssertCuda( cudaMemcpy(other._idx, _idx, _size * sizeof(unsigned int), cudaMemcpyDeviceToHost) );
			return other;
#endif
		}
	
	}
	std::size_t memory_consumption() const
	{
		return _size * (sizeof(Number) + sizeof(unsigned int));
	}
	void spmv(	Vector<Number> &vec, 
			Vector<Number> &result) const
	{
		if ( _memory_space == MemorySpace::CUDA )
		{
#ifndef DISABLE_CUDA		
			const unsigned int n_blocks = CEIL_DIV(_n_rows,block_size);
			lm::spmv_basic<Number>
					<<<n_blocks, block_size>>>
					(	_n_rows, 
						_size, 
						_width,
						_data, 
						_idx, 
						vec.data(),
						result.data() ) ;			
#endif
		}
		else
		{
			result = 0.;
			for ( std::size_t i = 0; i < _size; ++i )			
				result( i%_n_rows ) += _data[i] * vec( _idx[i]);
		}
	}
	void mult_scalar(	Number scalar)
	{
		if ( _memory_space == MemorySpace::CUDA )
		{
			std::cout << " not implemented " << std::endl;
		}
		else
		{
			for ( std::size_t i = 0; i < _size; ++i )
			{
				_data[i] = _data[i] * scalar;
			}
		}
	}
	void spmm(	const Dense_matrix<Number> &mat, 
			Dense_matrix<Number> &result) const
	{	
		if ( _memory_space == MemorySpace::CUDA )
		{
#ifndef DISABLE_CUDA			
			const unsigned int n_blocks = CEIL_DIV(_n_rows,block_size); 
			lm::spmm_basic<Number>
					<<<n_blocks,block_size>>>
					(	_n_rows,
						_width, 
						mat.size(),
						mat.n_rows(),
						mat.n_cols(),
						_data, 
						_idx, 
						mat.data(),
						result.data()) ;
#endif
		}
		else
		{
			result = 0;
			std::size_t mat_col, i;
			for (  mat_col = 0; mat_col < mat.n_cols(); ++mat_col )
			{
				for (  i = 0; i < _size; ++i )
				{
					result( i%(_n_rows) + mat_col*_n_rows ) += 
						_data[i] * mat( _idx[i] + mat_col*_n_rows );
					
				}
			}
		}		
	}
	void diag_inv()
	{
		if ( _memory_space == MemorySpace::CUDA )
		{
#ifndef DISABLE_CUDA
			const unsigned int n_blocks = CEIL_DIV(_size,block_size);
			lm::diag_inv<Number><<<n_blocks, block_size>>>
							(	_size,
								_data);
#endif
		}
		else
		{
			for ( std::size_t  i = 0; i < _size; ++i )
			{
				_data[i] = 1./_data[i];
			}
		} 
	}
	void diag_sqrt()
	{
		if ( _memory_space == MemorySpace::CUDA )
		{
#ifndef DISABLE_CUDA
			const unsigned int n_blocks = CEIL_DIV(_size,block_size);
			lm::diag_sqrt<Number><<<n_blocks, block_size>>>
							(	_size,
								_data);		
#endif
		}
		else
		{
			for ( std::size_t i = 0; i < _size; ++i )
			{
				_data[i] = std::sqrt(_data[i]);
			}
		}	
	}
	void mult_diagonal(	const Ell_matrix &diag	) 
	{
		if ( _memory_space == MemorySpace::CUDA)
		{
#ifndef DISABLE_CUDA
			const unsigned int n_blocks = CEIL_DIV(_size,block_size);
			lm::mm_with_diag<Number>
					<<<n_blocks, block_size>>>
						(	_size,
							_data,
							_idx,
							diag._data);
#endif
		}
		else
		{
			for (std::size_t i = 0; i < _size; ++i )
			{
				_data[i] = _data[i] * diag._data[_idx[i]];
			}
		}
	}
	void change_order(	const unsigned int stride)
	{
		Ell_matrix<Number> result(_n_rows, _size, _n_cols, _memory_space);

		if ( _memory_space == MemorySpace::CUDA )
		{
#ifndef DISABLE_CUDA
			const unsigned int new_size = _size*stride/_width;
			const unsigned int n_blocks = CEIL_DIV(new_size,block_size);
			lm::change_major<Number>
						<<<n_blocks, block_size>>>
						(	new_size,
							_n_rows,
							_width,
							_data,
							_idx,
							result.data(),
							result.idx(),
							stride);
#endif
		}
		else
		{		
			std::size_t row, col ,i , s, ind;
			for ( s = 0; s < _width/stride; ++s )
			{
				 
				for ( i = 0; i < _size/stride ; ++i)
				{
					
					ind = i + s*stride*_n_rows;
					row = i%_n_rows;
					col = i/_n_rows;
					result._data[ row*stride + col +s*stride*_n_rows ]
								 = _data[ind];
					result._idx[  row*stride + col +s*stride*_n_rows]
								 = _idx[ind];
				}
			}
		}
		*this = result;
	}
	void padding( const unsigned int pads)
	{
		const unsigned int _stride = 4;
		const unsigned int new_n_rows = ( (_n_rows*_stride + pads - 1)/pads) *
								 (pads/_stride) ;
		const unsigned int new_size = new_n_rows * _width;	
		Ell_matrix<Number> result(new_n_rows, new_size, _n_cols, _memory_space);
		if ( _memory_space == MemorySpace::CUDA)
		{
#ifndef DISABLE_CUDA
			const unsigned int n_blocks = CEIL_DIV(_n_rows*_stride,block_size);
			const unsigned int iter = _width/_stride;
			
			lm::padding<Number>
					<<<n_blocks, block_size>>>
						(	_n_rows,
							new_n_rows,
							_stride,
							iter,
							_data,
							_idx,
							result._data,
							result._idx	);
#endif
		}
		else
		{
			std::size_t j,i;
			for ( j = 0 ; j < _width/_stride; ++j )
			{
				for ( i = 0; i < new_n_rows*_stride; ++i )
				{
					result._data[i + j*new_n_rows*_stride] = 
						(i < _n_rows*_stride) ?
						 _data[i + j*_n_rows*_stride] : 0.0;

					result._idx[i + j*new_n_rows*_stride] = 
						(i< _n_rows*_stride) ?	
						 _idx[i + j*_n_rows*_stride] : 0.0;				
				}
			}
		
		}
		*this = result;
	}
	void print() const
	{
		if ( _memory_space == MemorySpace::CUDA )
		{
			Ell_matrix<Number> host_copy = this->copy_to_host();
			host_copy.print();
		}
		else
		{
			std::cout << std::endl;
			std::size_t i;
			for ( i = 0; i < _size; ++i )
			{
				std::cout << _data[i] << " " ;
			}
			std::cout << std::endl << std::endl;

			for ( i = 0; i < _size; ++i )
			{
				std::cout << _idx[i] << " ";
			}
			std::cout << std::endl << std::endl;
		}
	}
	void print_as_ell() const
	{
		if ( _memory_space == MemorySpace::CUDA )
		{
			Ell_matrix<Number> host_copy = this->copy_to_host();
			host_copy.print_as_ell();
		}
		else
		{	
			std::size_t row, col;
			std::cout << std::endl;
			for ( row = 0; row < _n_rows; ++row )
			{
				for ( col = 0; col < _width; ++col )
				{
					std::cout << _data[row + col*_n_rows] << " " ;
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
			for (row = 0; row < _n_rows; ++row)
			{
				for (col = 0; col < _width; ++col)
				{
					std::cout << _idx[row + col*_n_rows] << " " ;
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;			
		}
	}
	void print_as_dense() const
	{
		if ( _memory_space == MemorySpace::CUDA )
		{
			Ell_matrix host_copy = this->copy_to_host();
			host_copy.print_as_dense();
		}
		else
		{		
			Dense_matrix<Number> dense(_n_rows,_n_cols, _memory_space);
			std::cout << std::endl;
			for ( std::size_t i = 0; i < _size; ++i)
			{
				if (_data[i] != 0)
				{
				
					dense(i%_n_rows + _idx[i]*_n_rows) = _data[i];
				
				} 
			}
			dense.print();
			std::cout << std::endl;	
		}
	}
	void info() const
	{	
		std::cout << std::endl;
		std::cout << " n_rows: " << _n_rows << std::endl;
		std::cout << ", n_cols: " << _n_cols << std::endl;
		std::cout << ", width: " << _width << std::endl;
		std::cout << ", size: " << _size << std::endl;
		if (_memory_space == MemorySpace::CUDA)
                {
                        std::cout<< " memory space in device" << std::endl;
                }
                else
                {
                        std::cout<< " memory space in host " << std::endl;
                }
	}
};

#endif
