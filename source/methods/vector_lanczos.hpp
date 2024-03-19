#ifndef vector_lanczos_hpp
#define vector_lanczos_hpp


#include "copy_functions.hpp"
#include "../kernels/spmv_spmm.hpp"

template <typename type_t>
void vector_lanczos(	Ell_matrix<type_t> &A, 
			Vector<type_t> &b, 
			const unsigned int m,
			const unsigned int lc,  
			Vector<type_t> &q, 	
			type_t *alpha,
			type_t *beta,
			Vector<type_t> &q0,
			Vector<type_t> &q1,
			Vector<type_t> &w)
{
	// beta = morm(b)
	beta[0] = b.l2_norm();		
	 
	//q0 = b/norm(b)
	q0.mult_scalar(1./beta[0]); 		
 	
	//q(0)  = q0(lc)
	copy_vector_element<type_t>(q0, lc, q, 0);

	// w = A*q0
	spmv(A, q0 ,w);

	// alpha = w.q0
	alpha[0] = w.dot(q0);		

	// w = w - alpha*q0
	w.add(-alpha[0], q0);			

	unsigned int j = 0;	
	while ( j < m-1 )
	{
		++j;		

		// beta = morm(b)
		beta[j] = w.l2_norm();		

		// q1 = w/norm(w)
		q1 = w;			
		q1.mult_scalar(1./beta[j]);	

		// w = A*q1
		spmv(A, q1, w);

		//w = w - beta*q0
		w.add(-beta[j], q0);		

		// alpha = w.q1
		alpha[j] = w.dot(q1);		

		// w = w - alpha*q1
		w.add(-alpha[j], q1);		
		
		q0 = q1;

		//q(j) = q0(lc)
		copy_vector_element<type_t>(q0, lc, q, j);
	}
}


template <typename type_t>
void vector_lanczos_blas(	Ell_matrix<type_t> &A, 
				Vector<type_t> &b, 
				const unsigned int m,
				const unsigned int lc, 		
				Vector<type_t> &q, 	
				type_t *alpha,
				type_t *beta,
				Vector<type_t> &q0,
				Vector<type_t> &q1,
				Vector<type_t> &w,
				cublasHandle_t cublasH)
{						// << kernel name in cublas >>
	// beta = morm(b)
	l2_norm_cublas(b, &(beta[0]), cublasH);		// nrm2
	
	// Q0 = b/norm(b)			
	mult_scalar_cublas(q0, 1./beta[0], cublasH);	//scal
 	
	//q_(0)  = q0(lc)
	copy_vector_element<type_t>(q0, lc, q, 0);	//memcpy

	// w = A*q0
	spmv(A, q0 ,w);					//own kernel
	
	// alpha = w.q0
	dot_cublas(w, q0, &(alpha[0]), cublasH);	//dot
	
	// w = w - alpha*q0
	vec_add_cublas( -alpha[0], q0, w, cublasH);	//axpy

	unsigned int j = 0;	
	while ( j < m-1 )
	{
		++j;		

		// beta = morm(b)
		l2_norm_cublas(w, &(beta[j]), cublasH);		//nrm2
			
		// q1 = w/norm(w)
		q1 = w;						// memcpy
		mult_scalar_cublas(q1, 1./beta[j], cublasH);	//scal
		// w = A*q1
		spmv(A, q1, w);					//own kernel

		// w = w - beta*q0
		vec_add_cublas( -beta[j], w, q0, cublasH);	//axpy

		// alpha = w.q1
		dot_cublas(w, q1, &(alpha[j]), cublasH);	//dot

		// w = w - alpha*q1
		vec_add_cublas( -alpha[j], q1, w, cublasH);

		q0 = q1;					//memcpy

		//q(j) = q0(lc)
		copy_vector_element<type_t>(q0, lc, q, j);	//memcpy
	}
}
#endif
