


#define DISABLE_CUDA
#include "../utils/cuda_utils.hpp"


#include <vector>
#include <cmath>


#include <iomanip>
#include <iostream>
#include <utility>
#include <memory>

#include <algorithm>


#include "vector.hpp"
#include "dense_matrix.hpp"



#define N_COL 4

namespace cpu{

// done by N_COL threads reduced in every iteration
void Tridiagonalize(	Dense_matrix<double> &A,
			Vector<double> &diag,
			Vector<double> &upper_diag,
			Vector<double> &sigma_vec)
			
{
	unsigned int r,c,k;
	//unsigned int N_COL = A.n_rows();

	Vector<double> v(N_COL, A.memory_space());
	Vector<double> w(v);
	Vector<double> u(v);


	for (unsigned int i = 0 ; i < N_COL - 2; ++i ) 
	{
		k = i + 1;
		double dot_u1 = 0.; // shmem
		u = 0; 
		for ( r = k; r < N_COL; ++r ) // signle threads 
		{
                    	u(r) = A(r + N_COL * i);
		 	dot_u1 += u(r) * u(r); // shufle reduction
                }
	 	 //--
	 	 
		double dot_u2 = 1.; //shmem
                // in Lanczos dot_u1 >=0
                double u1 = A(k + N_COL * i);// shared
                for (r = k + 1; r < N_COL; ++r) // single threads
		{
                	        u(r) *= 1. / (u1 +  std::sqrt(dot_u1));;
                        	dot_u2 += u(r) * u(r);//shufle reduction
                }
		 //--
		u(k) = 1;
		double sigma =  2./dot_u2; // shmem
		double alpha = 0;	   //shmem
		v = 0;
		for ( r = i; r < N_COL; ++r ) 
		{	
			//check this
			// ideally we need only the second one but because we store
			// the in upper diagonal so it is so
			
			for (c = i; c < r; ++c) 
			{
				v(r) += A(r + N_COL * c) * u(c);
			}
			
			for (c=r ; c < N_COL; ++c) 
			{
				v(r) += A(c + N_COL * r) * u(c);
			}
			v(r) *= sigma;
			alpha += v(r) * u(r); // shufle reduction
		}
		alpha /= dot_u2;
		 //--
		 
		for ( r = i; r < N_COL; ++r ) //single thread
		{
			w(r) = v(r) - alpha * u(r);
		}
		 
		for ( r = i; r < N_COL; ++r ) 
		{
			//w(r) = v(r) - alpha * u(r);
			//A(r + N_COL * r) -=  u(r)*w(r) * 2;
			for (c = r ; c < N_COL; ++c) 
			{
				A(c + N_COL * r) -= u(r)*w(c) + w(r)*u(c);
			}
		}
		 //--
		//A(i + N_COL * k) = sigma;
		sigma_vec(i) = sigma;
		for ( r = k + 1; r < N_COL; ++r ) 
		{
			A(i + N_COL * r) = u(r);
		}
	}
		//---
	for ( unsigned int j = 0; j < N_COL - 1; ++j ) 
	{
		diag(j) = A( j* (N_COL + 1) );
		upper_diag(j) = A( j * ( N_COL + 1 ) + 1);
	}
	diag(N_COL - 1) = A( (N_COL - 1) *(N_COL + 1));		
		//--
	
}


//serial
void Given(	const double x, 
		const double y, 
		Vector<double> &cos, 
		Vector<double> &sin,
		const int count)
{
	double tau;
	double cos_val;
	double sin_val;
	if (y != 0) 
	{
		if (std::abs(y) > std::abs(x)) 
		{
			tau = -x / y;
			sin_val = (1.) / std::sqrt((1) + tau * tau);
			cos_val = sin_val * tau;
		}
		else
		{
			tau = -y / x;
			cos_val = (1.) / std::sqrt((1) + tau * tau);
			sin_val = cos_val * tau;
		}
	}
	else
	{
		cos_val = 1; sin_val = 0;
	}
	cos(count) = cos_val;
	sin(count) = sin_val;
		
}
//serial
void Implicit_Wilkinson(	const int imin,
				const int imax,
				Vector<double> &diag,
				Vector<double> &upper_diag,
				Vector<double> &cos,
				Vector<double> &sin,
				Vector<int> &index,
				unsigned int *count )
{
	double a00 = diag(imax);
	double a01 = upper_diag(imax);
	double a11 = diag(imax + 1);

	double dif = (a00 - a11) * 0.5;
	double sgn = (dif >= 0 ? -1 : -1);

	double a01_sqr = a01 * a01;
	double u = a11 - a01_sqr / (dif + sgn * std::sqrt(dif * dif + a01_sqr));

	double x = diag(imin) - u;
	double y = upper_diag(imin);
	double a12, a22, a23, tmp11, tmp12, tmp21, tmp22, cs = 0, sn = 0;
	double a02 = 0;

//	int i0 = imin - 1, i2 = imin + 1;
	
//	for ( int i1 = imin; i1 <= imax; ++i0, ++i1, ++i2) 
	for ( int i1 = imin; i1 <= imax; ++i1)
	{
		int i0 = i1 - 1;
		int i2 = i1 + 1;		
		cpu::Given(x, y, cos, sin, *count);
		index(*count) = i1;
		

	
		if (i1 > imin) 
		{
                	    upper_diag(i0) = cos(*count) * upper_diag(i0) - sin(*count) * a02;
       	 	}

		a11 = diag(i1);
		a12 = upper_diag(i1);
		a22 = diag(i2);
		tmp11 = cos(*count) * a11 - sin(*count) * a12;
		tmp12 = cos(*count) * a12 - sin(*count) * a22;
		tmp21 = sin(*count) * a11 + cos(*count) * a12;
		tmp22 = sin(*count) * a12 + cos(*count) * a22;

		diag(i1) = cos(*count) * tmp11 - sin(*count) * tmp12;
		upper_diag(i1) = sin(*count) * tmp11 + cos(*count) * tmp12;
		diag(i2) = sin(*count) * tmp21 + cos(*count) * tmp22;
		if (i1 < imax) 
		{
			a23 = upper_diag(i2);
			a02 = -sin(*count) * a23;
			upper_diag(i2) = cos(*count) * a23;
			x = upper_diag(i1);
			y = a02;
		}	
		(*count)++;
	}

	
}

// should have stored the householder from tridigonlization
// done byN_COL threads
void Accumulate_HouseHolders(	Dense_matrix<double> &A,
				Dense_matrix<double> &eigen_vec,
				Vector<double> &sigma_vec)
{
	eigen_vec.set_to_identity();
	
	
	Vector<double> v( N_COL, eigen_vec.memory_space());
	Vector<double> w( N_COL, eigen_vec.memory_space());
	unsigned int r, c;
	//unsigned int rmin;
	// this is done once
	////////////////////////////////////
	
	//for ( int i = N_COL - 3, rmin = i + 1; i >= 0; --i, --rmin)
	for ( int i = N_COL - 3; i >= 0; --i)
	{
	
		//double sigma = A(i + N_COL *( i + 1));//(i*N_COL + i + 1);
		double sigma = sigma_vec(i);
		v = 0.;
		//
		v(i+1) = 1.;

		for (r = i+2; r < N_COL; ++r)
		{
			v(r) = A(i + N_COL*r);
		}
		
		w = 0;
		for (r = 0; r < N_COL; ++r) 
		{
			
			for (c = i+1; c < N_COL; ++c) 
			{
				w(r) += v(c) * eigen_vec(r + N_COL*c);	
			}
			//w(r) *= sigma;

			for (c = i+1; c < N_COL; ++c)
                        {
                                eigen_vec(r + N_COL * c) -= v(c) * w(r)*sigma;
                        }
		}
	}
}

// can be done by N_COL threads
void Rotate_eigen_vec(	Dense_matrix<double> &eigen_vec,
			const Vector<double> &cos,
			const Vector<double> &sin,
			const Vector<int> &index,
			const unsigned int count  )
{		
	for ( unsigned int given = 0; given < count; given++ ) 
	{
		for ( unsigned int r = 0; r < N_COL; ++r) 
		{
			int j =  index(given) + r*N_COL ;//
			double val0 = eigen_vec(j);
			double val1 = eigen_vec(j + 1);
			double temp0 = cos(given) * val0 - sin(given) * val1;//
			double temp1 = sin(given) * val0 + cos(given) * val1;//
			eigen_vec(j) = temp0;
			eigen_vec(j + 1) = temp1;
		}
	}
	
	//eigen_vec.print();
	
		
	
}

void my_eigen_solver(	Dense_matrix<double> &A,
			const unsigned int global_iter)
{

	
	
	Vector<double> diag(N_COL, MemorySpace::Host); // this will be the eigen values
	Vector<double> upper_diag(N_COL-1, MemorySpace::Host);
	Dense_matrix<double> eigen_vec(A); // <<--

	Vector<double> sigma_vec(N_COL, MemorySpace::Host);
	
	const unsigned int iteration = N_COL*3;
	Vector<double> cos(iteration*N_COL, MemorySpace::Host);
	Vector<double> sin(iteration*N_COL, MemorySpace::Host);
	Vector<int> index(iteration*N_COL, MemorySpace::Host);

	unsigned int count = 0;
		
	int imin = -1, imax = -1;	
	cpu::Tridiagonalize(A, diag, upper_diag, sigma_vec);

	diag.print();
	upper_diag.print();

	cpu::Accumulate_HouseHolders(A, eigen_vec, sigma_vec);
	eigen_vec.print();

	for ( unsigned int iter = 0; iter < iteration; ++iter )
	{
		
		
		int imin = -1, imax = -1;
		// bulge chasing
		for ( int i = N_COL - 2; i >= 0; --i) 
		{
			
			double a00 = diag(i);
			double a01 = upper_diag(i);
			double a11 = diag(i + 1);
			double sum = std::abs(a00) + std::abs(a11);
			if (sum + std::abs(a01) != sum) 
			{
				if (imax == -1) 
				{
					imax = i;
				}
				imin = i;
			}	
			else
			{
				if (imin >= 0) 
					break;
			}
		}

		if (imax == -1)
			break;
		
		//std::cout << imin << ", " << imax << std::endl;
		cpu::Implicit_Wilkinson(imin, imax, diag, upper_diag, cos,sin, index,&count);
		//std::cout << count << std::endl;

	}

	//diag.print();
	

	cpu::Rotate_eigen_vec( eigen_vec,cos, sin,index, count  );	
	eigen_vec.print();

	A = 0;
	eigen_vec.tra();
        A.custom_mult2(diag, eigen_vec);
        A.print();

		
	
}
	
};


int main(int argc, char **argv)
{
	std::vector<double> array = {4,1,-2,2, 1,2,0,1, -2,0,3,-2, 2,1,-2,-1};
	Dense_matrix<double> A(N_COL,N_COL, array, MemorySpace::Host);

	cpu::my_eigen_solver(A, 2);

	/*
	Vector<double> diag(N_COL , MemorySpace::Host); // this will be the eigen values
        Vector<double> upper_diag(N_COL -1, MemorySpace::Host);

	A.print();
	cpu::Tridiagonalize(A, diag, upper_diag);

	diag.print();
	upper_diag.print();
	*/
	
	return 0 ;
}

