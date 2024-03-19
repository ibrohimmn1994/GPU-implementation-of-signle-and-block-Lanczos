
#ifndef dense_matrix_hpp
#define dense_matrix_hpp


#include "vector.hpp"
#include "../kernels/dense_kernels.hpp"

//column major
template <typename Number>
class Dense_matrix
{
private:
	std::size_t _n_rows;
	std::size_t _n_cols;
	std::size_t _size;
	Number *_data;
	MemorySpace _memory_space;

	void set_size(	const std::size_t n_rows, 
		 	const std::size_t n_cols)
	{
		_size = n_rows*n_cols;
		if ( _memory_space == MemorySpace::CUDA )
		{
#ifndef DISABLE_CUDA
			AssertCuda( cudaFree(_data) );
			AssertCuda( cudaMalloc(&_data, _size * sizeof(Number)) );
#endif
		}
		else
		{
			delete[] _data;
			_data = new Number[_size];
		}
		_n_rows = n_rows;
		_n_cols = n_cols;
	}
public:
	static const int block_size = Vector<Number>::block_size;
	Dense_matrix()
	:_data(nullptr)
	{}
	Dense_matrix(	const std::size_t n_rows, 
			const std::size_t n_cols, 
			const MemorySpace memory_space)
	: _data(nullptr), _memory_space(memory_space)
	{
		set_size(n_rows, n_cols);
		if ( _memory_space == MemorySpace::CUDA)
		{
#ifndef DISABLE_CUDA
			const unsigned int n_blocks = CEIL_DIV(_size,block_size);
			v::set_entries<Number><<<n_blocks, block_size>>>
					(	_size, 
						Number(0), 
						_data) ;
#endif
		}
		else
		{
			for ( std::size_t i = 0; i < _size; ++i )
				_data[i] = 0.0;
		}	
	}
	Dense_matrix(const Dense_matrix &other)
	: _data(nullptr), _memory_space(other._memory_space)
	{
		set_size(other._n_rows, other._n_cols);
		if ( _memory_space == MemorySpace::CUDA )
		{
#ifndef DISABLE_CUDA
			AssertCuda( cudaMemcpy(_data, other._data, _size * sizeof(Number), cudaMemcpyDeviceToDevice) );
#endif
		}
		else
	 	{
			for ( std::size_t i = 0; i < _size; ++i )
				_data[i] = other._data[i];
		}
		
	}
	Dense_matrix(	const std::size_t n_rows,
			const std::size_t n_cols, 
			std::vector<Number> &array, 
			const MemorySpace memory_space )
	: _data(nullptr), _memory_space(memory_space)
	{
			set_size(n_rows, n_cols);
			if ( _memory_space == MemorySpace::CUDA )
			{
#ifndef DISABLE_CUDA
				Number *array_new = new Number[_size];
				for ( std::size_t i = 0; i < _size; ++i )
					array_new[i] = array[i];
				AssertCuda( cudaMemcpy(_data, array_new, _size * sizeof(Number), cudaMemcpyHostToDevice) );
				delete[] array_new;
#endif
			}
			else
			{
				for( std::size_t i = 0; i < _size ; ++i )
					_data[i] = array[i];
			}	
	}
	Dense_matrix( const Vector<Number> &vec )
	: _data(nullptr), _memory_space(vec.memory_space())
	{
		set_size(vec.size(), vec.size());
		if ( _memory_space == MemorySpace::CUDA )
		{
#ifndef DISABLE_CUDA
                        v::set_entries<Number><<< CEIL_DIV(_size,block_size), block_size>>>
                                        (       _size,
                                                Number(0),
                                                _data) ;

			dm::diagonal_matrix_from_vector
					<Number><<< CEIL_DIV(_n_rows,block_size),block_size>>>
							(       _n_rows,
								vec.data(),
								_data);				
#endif
		}
		else
		{	
			std::size_t i;
			for ( i = 0; i < _size; ++i )
				_data[i] = 0;
			for ( i = 0; i < _n_rows; ++i )
				_data[i + i *_n_rows] = vec(i);
		}
	}
	~Dense_matrix()
	{
		if ( _memory_space == MemorySpace::CUDA )
		{
#ifndef DISABLE_CUDA
			cudaFree(_data);
#endif
		}
		else
		{
			delete[] _data;
		}
	}
	Dense_matrix &operator=(const Dense_matrix &other)
	{
		if ( other._memory_space == MemorySpace::CUDA )
		{
#ifndef DISABLE_CUDA
			_memory_space = MemorySpace::CUDA;
			set_size(other._n_rows, other._n_cols);
			AssertCuda( cudaMemcpy(_data, other._data, _size * sizeof(Number), cudaMemcpyDeviceToDevice) );
#endif
		}
		else
		{
			_memory_space = MemorySpace::Host;
                        set_size(other._n_rows, other._n_cols);
			for ( std::size_t i = 0; i < _size; ++i )
				_data[i] = other._data[i];
		}
		return *this;
	}
	 Dense_matrix &operator=(const Number scalar)
	{
		if ( _memory_space == MemorySpace::CUDA)
		{
#ifndef DISABLE_CUDA
		
			const unsigned int n_blocks = CEIL_DIV(_size,block_size);
			v::set_entries<Number><<<n_blocks, block_size>>>
						(       _size,
							scalar,
							_data);
#endif
		}
		else
		{
			 for ( std::size_t i = 0; i < _size; ++i )
				_data[i] = scalar;
		}
		return *this;
	}
	const Dense_matrix copy_to_device() const
	{
		if ( _memory_space == MemorySpace::CUDA )
		{
			std::cout << " You are already in the device " << std::endl;
			return *this;
		}
		else
		{
#ifndef DISABLE_CUDA
			Dense_matrix<Number> other(_n_rows, _n_cols, MemorySpace::CUDA);
			AssertCuda( cudaMemcpy(other._data, _data, _size * sizeof(Number), cudaMemcpyHostToDevice) );
			return other;
#endif
		}
	}
	const Dense_matrix copy_to_host() const
	{
		if ( _memory_space == MemorySpace::Host )
		{
			std::cout << " You are already in the host " << std::endl;
			return *this;
		}
		else
		{
#ifndef DISABLE_CUDA
			Dense_matrix<Number> other(_n_rows, _n_cols, MemorySpace::Host);
			AssertCuda( cudaMemcpy(other._data, _data, _size * sizeof(Number), cudaMemcpyDeviceToHost) );
			return other;
#endif
		}
	}
	const Number &operator()( const std::size_t row,
				  const std::size_t col) const
	{
		return _data[row + _n_rows*col];
	}
	Number &operator()(	const std::size_t row,
				const std::size_t col)
	{
		return _data[row + _n_rows*col];
	}
	const Number &operator()(const std::size_t index) const
	{
		return _data[index];
	}
	Number &operator()(const std::size_t index)
	{
		return _data[index];
	}

	Number *data() 
	{
		return _data;
	}
	
	const Number *data() const
	{
		return _data;
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
	MemorySpace memory_space() const
	{
		return _memory_space;
	}

	std::size_t memory_consumption() const
	{
		return _size * sizeof(Number);
	}

	void mv(	const Vector<Number> &vec, 
		 	Vector<Number> &result) const
	{		
		if ( _memory_space == MemorySpace::CUDA )
		{
#ifndef DISABLE_CUDA
			const unsigned int n_blocks =  CEIL_DIV(_n_rows,block_size);
			dm::mv<Number><<<n_blocks, block_size>>>
						(	_n_rows, 
							_n_cols, 
							_data, 
							vec.data(), 
							result.data() ) ;
#endif
		}
		else
		{
			Number sum;
			std::size_t col, row;
			for ( col = 0; col < _n_cols; ++col )
			{
				sum = 0;
				for ( row = 0; row < _n_rows; ++row )
				{
					sum += _data[row + col*_n_rows] * vec(col);
				}
				result(row) = sum;			
			}
		}	
	}
	void mm (	const Number my_scalar,
                      	const Number other_scalar,
                       	Dense_matrix &other1,
                       	Dense_matrix &other2)
	{
		Number sum;
                std::size_t col, row, el;
                Dense_matrix temp(*this);
                for ( col = 0; col < _n_cols; ++col )
                {
                        for ( row = 0; row < _n_rows; ++row )
                        {
                                sum = 0;
                                for ( el = 0; el < other1._n_cols; ++el)
                                {
                                        sum += other1._data[row + _n_rows*el] *
                                                other2._data[el + other1._n_cols*col];
                                }
                                temp._data[row + _n_rows*col] =
                                        my_scalar*_data[row + _n_rows*col] + other_scalar*sum;
                        }
                }
                for ( el = 0; el < _size; ++el )
                     _data[el] = temp._data[el];
	}
	void tra() 
	{	
		Dense_matrix T(_n_cols, _n_rows, _memory_space);
		std::size_t col, row;
		for ( col = 0; col < _n_cols; ++col )
		{
			for (  row = 0; row < _n_rows; ++row )
				T._data[col + _n_cols*row] = _data[row + _n_rows*col];
		}
		*this = T;
	}
	// this = my_scalar*this + other_scalar*other
	void sadd(	const Number my_scalar,
			const Number other_scalar, 
			const Dense_matrix &other)
	{	
		if ( _memory_space == MemorySpace::CUDA )
		{
#ifndef DISABLE_CUDA
			const unsigned int n_blocks =  CEIL_DIV(_size,block_size);
			dm::mm_add<Number>
					<<<n_blocks, block_size>>>
						(	_size,  
							_data, 
							other._data, 
							my_scalar, 
							other_scalar) ;
#endif
		}
		else
		{
			std::size_t ind, col, row;
			for ( col = 0; col < _n_cols; ++col )
			{	
				for ( row = 0; row < _n_rows; ++row )
				{
					ind = row + col*_n_rows;	
					_data[ind] = my_scalar*_data[ind] + 
							other_scalar*other._data[ind];
				}
			}
		}
	}
	// this  = scalar * this	
	void mult_scalar(const Number scalar) 
	{
		sadd(scalar, 0, *this);
	}
	void add(	const Number other_scalar,
			const Dense_matrix &other) 
	{
		sadd(1., other_scalar, other);
	}

	void add_scalar(const Number scalar)
	{
		if ( _memory_space == MemorySpace::CUDA)
		{
#ifndef DISABLE_CUDA
			const unsigned int n_blocks =  CEIL_DIV(_size,block_size);
			dm::add_scalar<Number>
					<<<n_blocks,block_size>>>
						(	_size,
							scalar, 
							_data);
#endif
		}
		else
		{
			for ( std::size_t  i = 0; i < _size; ++i )
			{
				_data[i] += scalar;
			}
		}
	}
	// this = U * T * U' and T is diagonal
	void custom_mult(	const Vector<Number> &eigval,
				const Dense_matrix &eigvec)
	{
		if ( _memory_space == MemorySpace::CUDA )
		{
#ifndef DISABLE_CUDA
			dim3 dim_grid(  CEIL_DIV(_n_rows,32),  CEIL_DIV(_n_rows,32) );
			dim3 dim_block(32, 32);
			dm::custom_mult<Number>
					<<<dim_grid, dim_block>>>
						(	_n_rows, 
							eigval.data(), 
							eigvec._data, 
							_data);
			
#endif
		}
		else
		{
			std::size_t col, row, el;
			Number sum;
			for ( col = 0; col < _n_rows; ++col )
			{
				for ( row = 0; row < _n_rows; ++row)
				{
					sum = 0;
					for ( el = 0; el < _n_rows; ++el )
					{
						sum += eigvec._data[row+el*_n_rows] * 
							std::exp(eigval(el)) *
							eigvec._data[col+el*_n_rows];
					}
					_data[row+_n_rows*col] = sum;
				}
			}
		}
	}
	void padding(const unsigned int pads)
	{
		const unsigned int new_n_rows = ((_n_rows + pads - 1 )/pads) * pads;
		Dense_matrix<Number> result(new_n_rows, _n_cols, _memory_space);
		if( _memory_space == MemorySpace::CUDA)
		{
#ifndef DISABLE_CUDA
			const unsigned int n_blocks = CEIL_DIV(_n_rows,block_size);
			dm::matrix_padding<Number>
						<<<n_blocks, block_size>>>
							(	_n_rows,
								new_n_rows,
								_n_cols,
								_data,
								result.data());
#endif
		}
		else
		{
			unsigned int i,j;
			for ( j = 0; j < _n_cols; ++j)
			{
				for ( i = 0; i < new_n_rows; ++i )
				{
					result._data[i+j*new_n_rows] =
						 (i<_n_rows) ? _data[i+j*_n_rows] : 0.0;
				}
			}
		}
		*this = result;
	}
	void print() const
	{
		if ( _memory_space == MemorySpace::CUDA )
		{
			Dense_matrix<Number> host_copy = this->copy_to_host();
			host_copy.print();
				
		}
		else
		{
			std::cout << std::endl;
			std::size_t row, col;
			for ( row = 0; row < _n_rows; ++row )
			{
				for ( col = 0; col < _n_cols; ++col )
				{
					std::cout << _data[row + _n_rows*col] << " ";	
				}
				std::cout << std::endl;
			}
			std::cout << std::endl << std::endl;
		}
	}
	void info() const
	{
		std::cout << std::endl;
		std::cout<< " n_rows: " << _n_rows << std::endl;
		std::cout<< " , n_cols: " << _n_cols << std::endl;
		std::cout<< " , size: " << _size << std::endl;
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
