
#ifndef build_ell_utils_hpp
#define build_ell_utils_hpp
 

/////////////////////////////////////////////////////////////////////////////////////////////
//	<<<	Assembly of A	>>>
template <typename Number>
Ell_matrix<Number> Ikron(Ell_matrix<Number> &I, Ell_matrix<Number> &X)
{

	const unsigned int I_rows = I.n_rows();
	const unsigned int X_width = X.width();
	const unsigned int X_rows = X.n_rows();
	const unsigned int X_size = X.size();
	const unsigned int X_cols = X.n_cols();
	unsigned int i,j;

	Ell_matrix<Number> result(I_rows*X_rows, I_rows*X_size, I_rows*X_cols, X.memory_space());

	for ( j = 0; j < X_width; ++j )
	{
		for ( i = 0; i < X_rows*I_rows; ++i)
		{
			result(i + j*(X_rows*I_rows)) = X(i%X_rows+j*X_rows);
			result[i+j*(X_rows*I_rows)] = X[i%X_rows+j*X_rows] + X_cols*(i/X_rows);

		}
	}

	return result;
	
}



template <typename Number>
Ell_matrix<Number> kronI(Ell_matrix<Number> &X, Ell_matrix<Number> &I)
{
	const unsigned int I_rows = I.n_rows();
//	const unsigned int x_width = X.width();
	const unsigned int X_rows = X.n_rows();
	const unsigned int X_size = X.size();
	const unsigned int X_cols = X.n_cols();
	unsigned int i,j;
	Ell_matrix<Number> result(I_rows*X_rows, I_rows*X_size, I_rows*X_cols, X.memory_space());

	for ( j = 0; j < I_rows; ++j )
	{
		for ( i = 0; i < X_size; ++i )
		{
			result(i*I_rows + j) = X(i) * I(j);
			result[i*I_rows + j] = X[i]*I_rows + j ;
		}
	}
	
	return result;
}


template<typename Number>
void insert(	Ell_matrix<Number> &D,
		Ell_matrix<Number> &d,
		const unsigned int r_loc,
		const unsigned int c_loc,
		const unsigned int c_shift )
{
	const unsigned int d_cols = d.width();
	const unsigned int d_rows = d.n_rows();
	const unsigned int D_rows = D.n_rows();
	unsigned int i,j;
	const unsigned int start = r_loc + c_loc*D_rows;
	for ( j = 0; j < d_cols; ++j )
	{
		for ( i = 0; i < d_rows; ++i)
		{
			D(start + i + j*D_rows) = d(i + j*d_rows);
			D[start + i + j*D_rows] = d[i + j*d_rows] + c_shift ;
		}
	}
}

template<typename Number>
Vector<Number> Diff(	const Vector<Number> &other)
{
	Vector<Number> diff(other.size()-1, MemorySpace::Host);
	for ( unsigned int i = 0; i < diff.size(); ++i )
	{
		diff(i) = other(i+1) - other(i);
	}
	return diff;
}

template<typename Number>
Vector<Number> Linspace(	const Number x_l,
				const Number x_r,
				const unsigned int N)
{
	Number h = (x_r - x_l)/(N-1);
	Vector<Number> grid(N, MemorySpace::Host);
	for ( unsigned int i = 0; i < N; ++i )
	{
		grid(i) = x_l + i * h;
	}
	return grid;
}

template<typename Number>
Ell_matrix<Number> diag(	const unsigned int N, 
				const Number scalar)
{
	Ell_matrix<Number> I(N, N, N, MemorySpace::Host);
	for ( unsigned int i = 0; i < N; ++i )
	{
		I(i) = scalar;
		I[i] = i;
	}
	return I;
}


template<typename Number>
Dense_matrix<Number> bidiagonal(	const unsigned int N,
					Number diagonal,
					Number upper_diagonal)
{
	Dense_matrix<Number> result(N,N+1,MemorySpace::Host);

	unsigned int i;
	for ( i = 0; i < N; ++i )
	{
		result(i + N*i) = diagonal;
		result(i+ N*(i+1)) = upper_diagonal;
	}
	
	
	return result;
}


template<typename Number>
Ell_matrix<Number> dense_to_ell(	const Dense_matrix<Number> &dense,
					unsigned int width)
{
	const unsigned int n_rows = dense.n_rows();
	const unsigned int n_cols = dense.n_cols();
	Ell_matrix<Number> ell(n_rows, width*n_rows, 
			       n_cols, dense.memory_space());

	unsigned int i,j,ind;
	for ( i = 0; i < n_rows; ++i )
	{
		ind = 0;
		for ( j = 0; j < n_cols; ++j )
		{
			if (dense(i + j*n_rows) != 0)
			{
				ell(i +ind*n_rows) = dense(i + j*n_rows);
				ell[i + ind*n_rows] = j;
				ind++;
			}	
		}
	}
	return ell;
	
}

template<typename Number>
Dense_matrix<Number> diag_inv(	Dense_matrix<Number> &diag_mat)
{
	Dense_matrix<Number> result(diag_mat);
	
	const unsigned int n_rows = diag_mat.n_rows();
	for ( unsigned int i = 0; i < n_rows; ++i )
	{
		result(i + n_rows*i) = 1./diag_mat(i + n_rows*i);
	}
	return result;
}


template<typename Number>
std::tuple< Vector<Number>,Vector<Number>, Vector<Number> >
grid_3D(                Vector<Number>&x,
                        Vector<Number>&y,
                        Vector<Number>&z)
{
        unsigned int xsize = x.size(), ysize = y.size(), zsize = z.size();
        unsigned int size = xsize * ysize * zsize;
        Vector<Number> X(size, x.memory_space());
        Vector<Number> Y(size, x.memory_space());
        Vector<Number> Z(size, x.memory_space());

        for ( unsigned int i = 0; i < size; ++i )
        {
                X(i) = x(i%xsize);
                Y(i) = y( (i/xsize) % ysize );
                Z(i) = z( (i/(xsize*ysize)) % zsize );
        }
        return std::make_tuple(X,Y,Z);
}

template<typename Number>
Vector<Number> gaussian_3D(                     const unsigned int N,
                                                Vector<Number>&x,
                                                Vector<Number>&y,
                                                Vector<Number>&z,
                                                Number shift)
{
        auto grid_3d = grid_3D<Number>(x,y,z);
        Vector<Number> X = std::get<0>(grid_3d);
        Vector<Number> Y = std::get<1>(grid_3d);
        Vector<Number> Z = std::get<2>(grid_3d);


        Vector<Number> result(N, x.memory_space());
        for ( unsigned int i = 0; i < X.size(); ++i )
        {
                result(i) = std::exp(   -std::pow(X(i)-shift,2)
                                        -std::pow(Y(i)-shift,2)
                                        -std::pow(Z(i)-shift,2)  );
        }
        return result;

}

template<typename Number>
Vector<Number> gaussian_vector_b(        const unsigned int N, const unsigned int n_rows)
{
        const Number x_l = 0., x_r = 1., y_l = 0., y_r = 1., z_l = 0., z_r = 1.;
        const Number hx = (x_r - x_l)/(N + 1), hy =  (y_r - y_l)/(N + 1),  hz = (z_r - z_l)/(N + 1);
        Vector<Number> x_p = Linspace<Number>(x_l+hx, x_r-hx, N);
        Vector<Number> y_p = Linspace<Number>(y_l+hy, y_r-hy, N);
        Vector<Number> z_d = Linspace<Number>(z_l+hz/2, z_r-hz/2, N+1);
        Vector<Number> b_host(n_rows, MemorySpace::Host) ;
        b_host = gaussian_3D<Number>(n_rows,x_p,y_p,z_d,0.5);
        return b_host;
}

template<typename Number>
Vector<Number> random_vector_b(	const unsigned int n_rows)
{
	Vector<Number> b_host(n_rows, MemorySpace::Host);
	for ( unsigned int i = 0; i < n_rows; ++i )
	{
		b_host(i) = ((Number) rand() / (RAND_MAX)) + 1;
	}
	return b_host;
}

template<typename Number>
Dense_matrix<Number> gaussian_matrix_B(  const unsigned int N, const unsigned int n_rows)
{
        const Number x_l = 0., x_r = 1., y_l = 0., y_r = 1., z_l = 0., z_r = 1.;
        const Number hx = (x_r - x_l)/(N + 1), hy =  (y_r - y_l)/(N + 1),  hz = (z_r - z_l)/(N + 1);
        Vector<Number> x_p = Linspace<Number>(x_l+hx, x_r-hx, N);
        Vector<Number> y_p = Linspace<Number>(y_l+hy, y_r-hy, N);
        Vector<Number> z_d = Linspace<Number>(z_l+hz/2, z_r-hz/2, N+1);

        Vector<Number> b_host(n_rows, MemorySpace::Host) ;
        Dense_matrix<Number> B_host(n_rows, N_COL,  MemorySpace::Host);
        for ( unsigned int  i = 0; i < N_COL; ++i )
        {
                b_host = gaussian_3D<Number>(n_rows,x_p,y_p,z_d,0.1*(i+1));
                copy_vector_to_column(b_host, B_host, i);
        }

        return B_host;
}

template<typename Number>
Dense_matrix<Number> random_matrix_B( const unsigned int n_rows)
{
        Dense_matrix<Number> B_host(n_rows, N_COL, MemorySpace::Host);
        for ( unsigned int i = 0; i < n_rows * N_COL ; ++i )
        {
                B_host(i) = ((Number) rand() / (RAND_MAX)) + 1;
        }
        return B_host;
}

#endif
