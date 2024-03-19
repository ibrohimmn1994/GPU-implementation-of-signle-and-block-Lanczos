
#ifndef fdtd_hpp
#define fdtd_hpp


template<typename type_t>
type_t fdtd_vector(	Ell_matrix<type_t> &A,
			Vector<type_t> &u0, 
			const unsigned int Nsteps,
			const double T_end,
			const unsigned int lc)
{

	type_t dt = T_end/Nsteps;
	Vector<type_t> dudt(u0);
	Vector<type_t> u(u0);

	for ( unsigned int  i = 0; i < Nsteps; ++i )
	{

		//A.ell_SpMV(u,dudt);
		spmv(A,u,dudt);
		u.add(dt,dudt);
		
	}

	Vector<type_t> u_host = u.copy_to_host();
	type_t result = u_host(lc);

	return result;
}

template<typename type_t>
Vector<type_t> ftdt_block(	Ell_matrix<type_t> &A,
				Dense_matrix<type_t> &U0,
				const unsigned int Nsteps,
				const double T_end,
				unsigned int lc)
{

	type_t dt = T_end/Nsteps;
	
	Dense_matrix<type_t> dUdT(U0);
	Dense_matrix<type_t> U(U0);

	for ( unsigned int i = 0; i < Nsteps; ++i )
	{
		spmm(A,U,dUdT);
		U.sadd(1,dt, dUdT);
	}
	Vector<type_t> result(U0.n_cols(), U0.memory_space());
	copy_row_to_vector(lc, 0,U, result);

	return result;
	
}


#endif
