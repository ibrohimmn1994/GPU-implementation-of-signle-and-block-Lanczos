




#ifndef cuda_utils_hpp
#define cuda_utils_hpp

#include <memory>
#include <utility>
#include <iostream>//
#include <vector>
#include <fstream>
#include <iomanip>//
#include <chrono>
//#include <stdio.h>
#include <stdbool.h>
#include <random>





void print( const float scalar)
{
	std::cout << scalar << std::endl;
}
void print( const double scalar)
{
        std::cout << scalar << std::endl;
}
void print( const int scalar)
{
        std::cout << scalar << std::endl;
}
void print( const unsigned scalar)
{
        std::cout << scalar << std::endl;
}
void print( const std::size_t scalar)
{
        std::cout << scalar << std::endl;
}


class steady_clock{
	private:
		std::chrono::time_point<std::chrono::steady_clock> t1;
		std::chrono::time_point<std::chrono::steady_clock> t2;
	public:
		steady_clock(){}
		~steady_clock(){}
		void start()
		{
			t1 = std::chrono::steady_clock::now();
		}
		void end()
		{
			t2 = std::chrono::steady_clock::now();
		}
		double duration()
		{
			return std::chrono::duration_cast<std::chrono::duration<double>>(
                                        t2 - t1).count();
		}
};

#ifdef DISABLE_CUDA
#define AssertCuda(err)
#define CUDA_CHECK(err)
#define CUSOLVER_CHECK(err)

#else
#include <cuda.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cusolverDn.h>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

#define WARP_SIZE 32

#define CUSOLVER_CHECK(err)									\
	do											\
	{											\
		cusolverStatus_t err_ = (err);							\
		if (err_ != CUSOLVER_STATUS_SUCCESS) 						\
		{										\
			 printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);	\
			 throw std::runtime_error("cusolver error");				\
		}										\
	} while (0)			

#define CUBLAS_CHECK(err)									\
	do											\
	{											\
		cublasStatus_t err_ = (err);							\
		if (err_ != CUBLAS_STATUS_SUCCESS) 						\
		{										\
			printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);		\
			throw std::runtime_error("cublas error");				\
		}										\
	} while(0)	       
       
#define AssertCuda(err)										\
	if (err != cudaSuccess)									\
	{											\
		std::cout << "The cuda call in " << __FILE__ << " on line "			\
		<< __LINE__ << " resulted in the error ' "					\
		<< cudaGetErrorString(err) << " ' " << std::endl;				\
		std::abort();									\
	}											
								

void CudaDeviceInfo() 
{

	int deviceId;
	cudaGetDevice(&deviceId);
	cudaDeviceProp props{};
	cudaGetDeviceProperties(&props, deviceId);

	std::cout << " Device ID: " <<deviceId << std::endl;
	std::cout << "Name: " <<  props.name<< std::endl;
	std::cout << " Compute Capability: " << props.major <<", " <<  props.major<< std::endl;
	std::cout << "memoryBusWidth: " <<  props.memoryBusWidth<< std::endl;
	std::cout << "maxThreadsPerBlock: " << props.maxThreadsPerBlock << std::endl;
	std::cout << "maxThreadsPerMultiProcessor "<< props.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "maxRegsPerBlock" <<  props.regsPerBlock<< std::endl;
	std::cout << "maxRegsPerMultiProcessor: " << props.regsPerMultiprocessor << std::endl;
	std::cout << "sharedMemPerBlock kB : " << props.sharedMemPerBlock / 1024 << std::endl;
	std::cout << "sharedMemPerMultiprocessor kB : " << props.sharedMemPerMultiprocessor / 1024<< std::endl;
	std::cout << "multiProcessorCount: " << props.multiProcessorCount << std::endl;
	std::cout << "texturePitch: " << props.maxTexture2DLinear[0] << std::endl;
}

#endif

#endif
