
#ifndef vector_hpp
#define vector_hpp



#include "../kernels/vector_kernels.hpp"



enum class MemorySpace
{
	Host,
	CUDA
};

template <typename Number>
class Vector
{
private:
	Number * _data;
	std::size_t _size;
	MemorySpace _memory_space;	
	void set_size( std::size_t size )
	{
		if ( _memory_space == MemorySpace::CUDA )
		{
#ifndef DISABLE_CUDA
			AssertCuda(cudaFree(_data)) ;
			AssertCuda( cudaMalloc(&_data, size * sizeof(Number)) );
#endif
		}
		else
		{
			delete[] _data;
			_data = new Number[size];
		}
		_size = size;
	}
	void assert_size( const Vector &other ) const
	{
		if ( _size != other._size )
		{
			std::cout << "The size of the two vectors does not match" << std::endl;
			std::abort();
		}
	}
public :
	static const int block_size = 32;// 256
	
	Vector( std::vector<Number> &array, 
		const MemorySpace memory_space )
	: _data(nullptr), _memory_space(memory_space)
	{
		set_size(array.size());
		if ( _memory_space == MemorySpace::CUDA)
		{
#ifndef DISABLE_CUDA
			Number* array_new = new Number[_size];
			for ( std::size_t i = 0; i < _size; ++i )
				array_new[i] = array[i];

			AssertCuda( cudaMemcpy(_data, array_new, _size * sizeof(Number), cudaMemcpyHostToDevice) );
			delete[] array_new;
#endif
		}
		else
		{
			for ( std::size_t i = 0; i < _size; ++i )
				_data[i] = array[i];
			
		}
	}
	Vector(	const std::size_t size, 
		const MemorySpace memory_space)
	: _data(nullptr), _memory_space(memory_space)
	{
		set_size(size);
		if ( _memory_space == MemorySpace::CUDA )
		{
#ifndef DISABLE_CUDA
			const unsigned int n_blocks =  CEIL_DIV(_size,block_size);
			v::set_entries<Number><<<n_blocks, block_size>>>
							(	_size, 
								Number(0), 
								_data) ;
#endif
		}
		else
		{
			for ( std::size_t i = 0; i < _size; ++i )
				_data[i] = 0;
		}
	}
	Vector(std::size_t size, const Number scalar, const MemorySpace memory_space)
	: _data(nullptr), _memory_space(memory_space)
	{
		set_size(size);
		if ( _memory_space == MemorySpace::CUDA )
		{
#ifndef DISABLE_CUDA
			const unsigned int n_blocks = CEIL_DIV(_size,block_size); 
			v::set_entries<Number><<<n_blocks, block_size>>>
							(	_size, 
								scalar, 
								_data) ;
#endif
		}
		else
		{
			for ( unsigned int i = 0; i < _size; ++i )
				_data[i] = scalar;
		}
	}	
	Vector(const Vector &other)
	: _data(nullptr), _memory_space(other._memory_space)
	{
		set_size(other._size);
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
	~Vector()
	{
		if (_memory_space == MemorySpace::CUDA)
		{
#ifndef DISABLE_CUDA
			AssertCuda( cudaFree(_data) );
#endif
		}
		else
		{
			delete[] _data;
		}
	}
	Vector &operator=( const Vector &other)
	{
		set_size(other._size);
		if ( _memory_space == MemorySpace::CUDA )
		{
#ifndef DISABLE_CUDA
		
			AssertCuda( cudaMemcpy(_data, other._data, _size*sizeof(Number), cudaMemcpyDeviceToDevice) );
#endif
		}
		else
		{
			for ( std::size_t i = 0; i < _size; ++i )
				_data[i] = other._data[i];
		}
		return *this;
	}
	const Number &operator()(const std::size_t index) const
	{
		return _data[index];
	}
	Number &operator()(const std::size_t index)
	{
		return _data[index];
	}
	MemorySpace memory_space() const
	{
		return _memory_space;
	}
	Vector &operator=(const Number scalar)
	{
		if ( _memory_space == MemorySpace::CUDA )
		{
#ifndef DISABLE_CUDA
			const unsigned int n_blocks = CEIL_DIV(_size,block_size);
			v::set_entries<Number><<<n_blocks, block_size>>>
								(	_size, 
									scalar, 
									_data) ;
#endif
		}
		else
		{
			for ( std::size_t i = 0; i < _size; ++i )
				_data[i] = scalar;
		}
		return *this;
	}
	void add_scalar(const Number scalar) 
	{
		Vector<Number> vec(_size, _memory_space); 	
		vec = scalar;
		sadd(1., 1., vec );
	}
	void mult_scalar(const Number scalar)
	{
		sadd(0., scalar, *this);
	}
	void add(	const Number vec_scalar, 
			const Vector &vec) 
	{
		sadd(1., vec_scalar, vec);
		
	}
	void sadd(	const Number my_scalar, 
			const Number vec_scalar, 
			const Vector &vec)
	{
		if ( _memory_space == MemorySpace::CUDA )
		{
#ifndef DISABLE_CUDA
			assert_size(vec);
		
			const unsigned int n_blocks = CEIL_DIV(_size,block_size);
			v::vector_update<Number><<<n_blocks, block_size>>>
							(	_size, 
								my_scalar, 
								vec_scalar, 
								vec._data, 
								_data) ;
#endif
		}
		else
		{
			for ( std::size_t i = 0; i < _size; ++i )
				_data[i] = my_scalar*_data[i] + vec_scalar*vec._data[i];
		}
	}
	Number l2_norm() const
	{
		const Number norm_squared = norm_square();
		if ( std::isfinite(norm_squared) )
			return std::sqrt(norm_squared);
		else
		{
			std::cout << "The norm is not finite" << std::endl;
			std::abort();
			return 0;
		}
	}
	Number norm_square() const
	{
		return dot(*this);
	}
	Number dot(	const Vector &other) const
	{
		assert_size(other);
		Number sum = 0;
		if ( _memory_space == MemorySpace::CUDA )
		{
#ifndef DISABLE_CUDA
			Number *d_sum;
			AssertCuda( cudaMalloc(&d_sum, sizeof(Number)) );
			AssertCuda( cudaMemset(d_sum, 0, sizeof(Number)));

			const unsigned int n_blocks = CEIL_DIV(_size,block_size);
			v::do_dot<block_size, Number><<<n_blocks, block_size>>>
						(	_size,
							_data,
							other._data,
							d_sum) ;
			 cudaMemcpy(&sum, d_sum, sizeof(Number), cudaMemcpyDeviceToHost);	
#endif
		}
		else
		{
			for ( std::size_t i = 0; i < _size; ++i )
				sum += _data[i] * other._data[i];	
		}
		return sum;
	}
	const Vector copy_to_device() const
	{
		if ( _memory_space == MemorySpace::Host )
		{
#ifndef DISABLE_CUDA
			Vector<Number> other(_size, MemorySpace::CUDA);

			AssertCuda( cudaMemcpy(other._data, _data, _size * sizeof(Number), cudaMemcpyHostToDevice) );
			return other;
#endif
		}
		else
		{
			std::cout << "You are already in the device" << std::endl;
			return *this;
		}
	}
	const Vector copy_to_host() const
	{
		if ( _memory_space == MemorySpace::CUDA )
		{
#ifndef DISABLE_CUDA
			Vector<Number> other(_size, MemorySpace::Host);
			AssertCuda( cudaMemcpy(other._data, _data, _size * sizeof(Number), cudaMemcpyDeviceToHost) );
			return other;
#endif
		}
		else
		{
			std::cout << "You are laready in the host " << std::endl;
			return *this;
		}	
	}
	Number* data()
	{
		return _data;
	}
	const Number* data() const
	{
		return _data;
	}
	
	std::size_t size() const
	{
		return _size;
	}
	std::size_t memory_consumption() const
        {
                return _size * sizeof(Number);
        }

	void exp()
        {
                if( _memory_space == MemorySpace::CUDA)
                {
#ifndef DISABLE_CUDA
			v::exp<Number>
				<<< CEIL_DIV(_size,block_size), block_size>>>
					(_size, _data);

#endif
                }
                else
                {
			for ( unsigned int  i = 0; i < _size; ++ i )
			{
				_data[i] = std::exp(_data[i]);
			}
                }
        }
	void print() const
	{
		if ( _memory_space == MemorySpace::CUDA )
		{
			Vector<Number> host_copy =  this->copy_to_host();
			host_copy.print();
		}
		else
		{
			std::cout << std::endl;
			for ( std::size_t i =  0; i < _size; ++i )
			{
				std::cout << _data[i] << " ";
			}
			std::cout << std::endl;		
		}
	}
	void info() const
        {
                std::cout << std::endl;
		std::cout << "size : " << _size << std::endl;
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
