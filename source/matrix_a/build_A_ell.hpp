
#ifndef build_A_ell_hpp
#define build_A_ell_hpp
 

#include "build_ell_utils.hpp"

template <typename Number> 
std::pair< Ell_matrix<Number>, Ell_matrix<Number> >
 Matrix_A(	const unsigned int Nx, 
		const unsigned int Ny,
		const unsigned int Nz)
{
	MemorySpace mem = MemorySpace::Host;

	const Number x_l = 0., x_r = 1., y_l = 0., y_r = 1., z_l = 0., z_r = 1.;
	const unsigned int Nx_plus = Nx + 2,  Ny_plus = Ny + 2,  Nz_plus = Nz + 2;

	//const unsigned int Nx = N, Ny = N, Nz = N;


	const Number hx = (x_r - x_l)/(Nx_plus - 1);
	const Number hy = (y_r - y_l)/(Ny_plus - 1);
	const Number hz = (z_r - z_l)/(Nz_plus - 1);

	// x_p, x_d, y_p, y_d, z_p, z_d;

	Vector<Number> x_p = Linspace<Number>(x_l, x_r, Nx_plus);
	Vector<Number> x_d = Linspace<Number>(x_l, x_r-hx, Nx_plus-1);
	x_d.add_scalar(hx/2);

	Vector<Number> y_p = Linspace<Number>(y_l, y_r, Ny_plus);
	Vector<Number> y_d = Linspace<Number>(y_l, y_r-hy, Ny_plus-1);
	y_d.add_scalar(hy/2);

	Vector<Number> z_p = Linspace<Number>(z_l, z_r, Nz_plus);
	Vector<Number> z_d = Linspace<Number>(z_l, z_r-hz, Nz_plus-1);
	z_d.add_scalar(hz/2);
	

	// delta_x_p, delta_x_d, delta_y_p, delta_y_d, delta_z_p, delta_z_d;
	
	Vector<Number> delta_x_p = Diff<Number>(x_p); Vector<Number> delta_x_d = Diff<Number>(x_d);
	Vector<Number> delta_y_p = Diff<Number>(y_p); Vector<Number> delta_y_d = Diff<Number>(y_d);
	Vector<Number> delta_z_p = Diff<Number>(z_p); Vector<Number> delta_z_d = Diff<Number>(z_d);

	
	
 
	Dense_matrix<Number> W_x_(delta_x_p); Dense_matrix<Number> W_x_hat_(delta_x_d);
	Dense_matrix<Number> W_y_(delta_y_p); Dense_matrix<Number> W_y_hat_(delta_y_d);
	Dense_matrix<Number> W_z_(delta_z_p); Dense_matrix<Number> W_z_hat_(delta_z_d);


	
	// W_x_inv, W_y_inv, W_z_inv, W_x_hat_inv, W_y_hat_inv, W_z_hat_inv;

	Dense_matrix<Number> W_x_inv     = diag_inv<Number>( W_x_)	;  
	Dense_matrix<Number> W_x_hat_inv = diag_inv<Number>(W_x_hat_)	;
	Dense_matrix<Number> W_y_inv	 = diag_inv<Number>( W_y_)	;
	Dense_matrix<Number> W_y_hat_inv = diag_inv<Number>(W_y_hat_)	;
	Dense_matrix<Number> W_z_inv 	 = diag_inv<Number>(W_z_)	;
	Dense_matrix<Number> W_z_hat_inv = diag_inv<Number>(W_z_hat_)	;

	
	// bidiag_x, bidiag_x_T, bidiag_y, bidiag_y_T, bidiag_z, bidiag_z_T;
	
	
	Dense_matrix<Number> bidiag_x = bidiagonal<Number>(Nx,1.,-1.);
	Dense_matrix<Number> bidiag_x_T = bidiag_x;	bidiag_x_T.tra();
	

	
	Dense_matrix<Number> bidiag_y = bidiagonal<Number>(Ny,1.,-1.);
	Dense_matrix<Number> bidiag_y_T = bidiag_y;	bidiag_y_T.tra();
	

	Dense_matrix<Number> bidiag_z = bidiagonal<Number>(Nz,1.,-1.);
	Dense_matrix<Number> bidiag_z_T = bidiag_z;	bidiag_z_T.tra();

	
	
	// X, X_hat, Y, Y_hat, Z, Z_hat;

	Dense_matrix<Number> X_(Nx+1,Nx,mem);	   	X_.mm(0.,1.,W_x_inv,bidiag_x_T);
	Dense_matrix<Number> Y_(Ny+1,Ny,mem);     	Y_.mm(0.,1.,W_y_inv,bidiag_y_T);
	Dense_matrix<Number> Z_(Nz+1,Nz,mem);   	Z_.mm(0.,1.,W_z_inv,bidiag_z_T);
	

	
	
	Dense_matrix<Number> X_hat_(Nx,Nx+1,mem);	X_hat_.mm(0.,1.,W_x_hat_inv,bidiag_x);
	X_hat_.mult_scalar(-1.);
	Dense_matrix<Number> Y_hat_(Ny,Ny+1,mem);	Y_hat_.mm(0.,1.,W_y_hat_inv,bidiag_y);
	Y_hat_.mult_scalar(-1.);
	Dense_matrix<Number> Z_hat_(Nz,Nz+1,mem);	Z_hat_.mm(0.,1.,W_z_hat_inv,bidiag_z);
	Z_hat_.mult_scalar(-1.);

	////////////////////////////////////////////////////////
	//		conver to Ellpack on CPU
	// X Y Z X_hat, Y_hat Z_hat W_x W_y W_z W_x_hat W_y_hat W_z_hat
	unsigned int nnz_per_row = 2;
	Ell_matrix<Number> X = dense_to_ell(X_,nnz_per_row);
	Ell_matrix<Number> Y = dense_to_ell(Y_,nnz_per_row);
	Ell_matrix<Number> Z = dense_to_ell(Z_,nnz_per_row);

	nnz_per_row = 2;
	Ell_matrix<Number> X_hat = dense_to_ell(X_hat_,nnz_per_row);
 	Ell_matrix<Number> Y_hat = dense_to_ell(Y_hat_,nnz_per_row);
 	Ell_matrix<Number> Z_hat = dense_to_ell(Z_hat_,nnz_per_row);

	nnz_per_row = 1;
	Ell_matrix<Number> W_x = dense_to_ell(W_x_,nnz_per_row);
	Ell_matrix<Number> W_y = dense_to_ell(W_y_,nnz_per_row);
	Ell_matrix<Number> W_z = dense_to_ell(W_z_,nnz_per_row);
	Ell_matrix<Number> W_x_hat = dense_to_ell(W_x_hat_,nnz_per_row);
	Ell_matrix<Number> W_y_hat = dense_to_ell(W_y_hat_,nnz_per_row);
	Ell_matrix<Number> W_z_hat = dense_to_ell(W_z_hat_,nnz_per_row);
 	
	

	//Ix, Ix_plus, Iy, Iy_plus, Iz, Iz_plus;
	
	Ell_matrix<Number> Ix = diag<Number>(Nx,1.);
	Ell_matrix<Number> Iy = diag<Number>(Ny,1.); 
	Ell_matrix<Number> Iz = diag<Number>(Nz,1.); 

	Ell_matrix<Number> Ix_plus = diag<Number>(Nx+1,1.);
        Ell_matrix<Number> Iy_plus = diag<Number>(Ny+1,1.);
        Ell_matrix<Number> Iz_plus = diag<Number>(Nz+1,1.);
	
	// De_12, D2_13, De_21, De_23, De_31, De_32;
	
	Ell_matrix<Number> temp = kronI(Iy_plus,Ix);
	Ell_matrix<Number> De_12 = kronI(Z,temp);
		
	temp = kronI(Y,Ix);
	Ell_matrix<Number> De_13 = Ikron(Iz_plus,temp);
	temp = Ikron(Iy,Ix_plus);
	Ell_matrix<Number> De_21 = kronI(Z,temp);
	temp = Ikron(Iy,X); 
	Ell_matrix<Number> De_23 = Ikron(Iz_plus,temp);
	temp = kronI(Y,Ix_plus);
	Ell_matrix<Number> De_31 = Ikron(Iz,temp);
	temp = Ikron(Iy_plus, X);
	Ell_matrix<Number> De_32 = Ikron(Iz,temp);
	

	De_12.mult_scalar(-1.); De_23.mult_scalar(-1.); De_31.mult_scalar(-1.);
	
	
	
	temp = Ikron(Iy,Ix_plus);
	Ell_matrix<Number> Dh_12 = kronI(Z_hat,temp);
	temp = kronI(Y_hat,Ix_plus);
   	Ell_matrix<Number> Dh_13 = Ikron(Iz,temp);
	temp = Ikron(Iy_plus,Ix);
	Ell_matrix<Number> Dh_21 = kronI(Z_hat,temp);
	temp = Ikron(Iy_plus, X_hat);
	Ell_matrix<Number> Dh_23 = Ikron(Iz, temp);
	temp = kronI(Y_hat,Ix);
	Ell_matrix<Number> Dh_31 = Ikron(Iz_plus,temp);
	temp = Ikron(Iy,X_hat);
	Ell_matrix<Number> Dh_32 = Ikron(Iz_plus,temp);

	
		
	Dh_13.mult_scalar(-1.); Dh_21.mult_scalar(-1.); Dh_32.mult_scalar(-1.);
	
	unsigned int De_rows =  De_12.n_rows() + De_21.n_rows() + De_31.n_rows();
	unsigned int De_size =  De_12.size() + De_21.size() + De_31.size() + De_13.size() + De_23.size() + De_32.size();

	unsigned int Dh_rows =  Dh_12.n_rows() + Dh_21.n_rows() + Dh_31.n_rows();
	unsigned int Dh_size =  Dh_12.size() + Dh_21.size() + Dh_31.size() + Dh_13.size() + Dh_23.size() + Dh_32.size();

	Ell_matrix<Number> De(De_rows, De_size, Dh_rows, mem);
	Ell_matrix<Number> Dh(Dh_rows, Dh_size, De_rows, mem);



	unsigned int D_rows = De_rows + Dh_rows;
	unsigned int D_size = De_size + Dh_size;
	Ell_matrix<Number> D(D_rows, D_size, D_rows, mem);

	
	unsigned int shift1, shift2;
	shift1 = Dh_12.n_rows(); shift2 = Dh_21.n_rows();

		
	insert(De, De_12, 0, 0, shift1);
	insert(De, De_13, 0, De_12.width(), shift1+shift2);
	insert(De, De_21, De_12.n_rows(),0, 0);
	insert(De, De_23, De_12.n_rows(), De_12.width(), shift1+shift2);
	insert(De, De_31, De_12.n_rows()+De_21.n_rows(), 0, 0);	
	insert(De, De_32, De_12.n_rows()+De_21.n_rows(), De_12.width(),shift1);

	
	
	
	shift1 = De_12.n_rows(); shift2 = De_21.n_rows();
	
	insert(Dh, Dh_12, 0, 0, shift1);
	insert(Dh, Dh_13, 0, Dh_12.width(), shift1+shift2);
	insert(Dh, Dh_21, Dh_12.n_rows(),0, 0);
	insert(Dh, Dh_23, Dh_12.n_rows(), Dh_12.width(), shift1+shift2);
	insert(Dh, Dh_31, Dh_12.n_rows()+Dh_21.n_rows(), 0, 0);
	insert(Dh, Dh_32, Dh_12.n_rows()+Dh_21.n_rows(), Dh_12.width(),shift1);
	
	
	
	insert(D, Dh, 0, 0, Dh_12.n_rows()+Dh_21.n_rows()+Dh_31.n_rows());
	insert(D, De, Dh.n_rows(), 0, 0 );
	
	temp = kronI(W_y_hat,W_x);
	Ell_matrix<Number> We_11 = kronI(W_z_hat,temp);
	temp = kronI(W_y,W_x_hat);
	Ell_matrix<Number> We_22 = kronI(W_z_hat,temp);
	temp = kronI(W_y_hat,W_x_hat);
	Ell_matrix<Number> We_33 = kronI(W_z,temp);

	temp = kronI(W_y,W_x_hat);
	Ell_matrix<Number> Wh_11 = kronI(W_z,temp);
	temp = kronI(W_y_hat,W_x);
	Ell_matrix<Number> Wh_22 = kronI(W_z,temp);
	temp = kronI(W_y,W_x);
	Ell_matrix<Number> Wh_33 = kronI(W_z_hat,temp);
	
	
	unsigned int We_rows = We_11.n_rows() + We_22.n_rows() + We_33.n_rows(); 
	unsigned int We_size = We_11.size() + We_22.size() + We_33.size();
	unsigned int Wh_rows = Wh_11.n_rows() + Wh_22.n_rows() + Wh_33.n_rows();
	unsigned int Wh_size= Wh_11.size() + Wh_22.size() + Wh_33.size();

	Ell_matrix<Number> We(We_rows, We_size, We_rows, mem);
	Ell_matrix<Number> Wh(Wh_rows, Wh_size, Wh_rows, mem);

	insert(We, We_11, 0, 0, 0);
	insert(We, We_22, We_11.n_rows(), 0, We_11.n_rows());
	insert(We, We_33, We_11.n_rows()+We_22.n_rows(), 0,We_11.n_rows()+We_22.n_rows());

	insert(Wh, Wh_11, 0, 0, 0);
	insert(Wh, Wh_22, Wh_11.n_rows(), 0, Wh_11.n_rows());
	insert(Wh, Wh_33, Wh_11.n_rows()+Wh_22.n_rows(), 0,Wh_11.n_rows()+Wh_22.n_rows());

	Wh.mult_scalar(-1.);

	Ell_matrix<Number> W(We.n_rows()+Wh.n_rows(), We.size()+Wh.size(),We.n_rows()+Wh.n_rows(), mem);

	insert(W, We, 0, 0, 0);
	insert(W, Wh, We.n_rows(), 0, We.n_rows());
		
	return std::make_pair(D, W);

	
}
#endif


