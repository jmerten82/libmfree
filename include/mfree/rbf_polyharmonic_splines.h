/*** rbf_polyharmonic_splines.h
The explcit implementation of polharmonic splines up to tenth order. 
 
Julian Merten
Universiy of Oxford
Dec 2016
julian.merten@physics.ox.ac.uk
http://www.julianmerten.net
***/

#ifndef    RBF_PHS_H
#define    RBF_PHS_H

#include <cmath>
#include <limits>
#include <mfree/radial_basis_function.h>

using namespace std;

/**
   A first order polyhoarmonic spline.
   phi(r) = r
**/

class phs_first_order : public radial_basis_function
{

 public:
 
 phs_first_order(double x = 0., double y = 0., double z = 0.) :  radial_basis_function(x,y,z) {};

 phs_first_order(coordinate input) :  radial_basis_function(input) {};
  
 phs_first_order(phs_first_order &input) : radial_basis_function(input) {};

  double operator() (double radius);
  using radial_basis_function::operator();
  double Dx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxyz(double x_in = 0., double y_in = 0., double z_in = 0.);
};


/**
   A second order polyhoarmonic spline:
   phi(r) = r^2*log(r)
**/

class phs_second_order : public radial_basis_function
{
  
 public:

 phs_second_order(double x = 0., double y = 0., double z = 0.) :  radial_basis_function(x,y,z) {};
  
 phs_second_order(coordinate input) :  radial_basis_function(input) {};
  
 phs_second_order(phs_second_order &input) : radial_basis_function(input) {};

  double operator() (double radius);
  using radial_basis_function::operator();
  double Dx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxyz(double x_in = 0., double y_in = 0., double z_in = 0.);
};

/**
   A third order polyhoarmonic spline:
   phi(r) = r^3
**/

class phs_third_order : public radial_basis_function
{

 public:

 phs_third_order(double x = 0., double y = 0., double z = 0.) :  radial_basis_function(x,y,z) {};

 phs_third_order(coordinate input) :  radial_basis_function(input) {};

 phs_third_order(phs_third_order &input) : radial_basis_function(input) {};

  double operator() (double radius);
  using radial_basis_function::operator();
  double Dx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxyz(double x_in = 0., double y_in = 0., double z_in = 0.);

};


/**
   A fourth order polyhoarmonic spline:
   phi(r) = r^4*log(r)
**/

class phs_fourth_order : public radial_basis_function
{

 public:
  
 phs_fourth_order(double x = 0., double y = 0., double z = 0.) :  radial_basis_function(x,y,z) {};

 phs_fourth_order(coordinate input) :  radial_basis_function(input) {};

 phs_fourth_order(phs_fourth_order &input) : radial_basis_function(input) {};

  double operator() (double radius);
  using radial_basis_function::operator();
  double Dx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxyz(double x_in = 0., double y_in = 0., double z_in = 0.);
};

/**
   A fifth order polyhoarmonic spline:
   phi(r) = r^5
**/

class phs_fifth_order : public radial_basis_function
{

 public:

 phs_fifth_order(double x = 0., double y = 0., double z = 0.) :  radial_basis_function(x,y,z) {};

 phs_fifth_order(coordinate input) :  radial_basis_function(input) {};

 phs_fifth_order(phs_fifth_order &input) : radial_basis_function(input) {};

  double operator() (double radius);
  using radial_basis_function::operator();
  double Dx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxyz(double x_in = 0., double y_in = 0., double z_in = 0.);

};

/**
   A sixth order polyhoarmonic spline:
   phi(r) = r^6*log(r)
**/

class phs_sixth_order : public radial_basis_function
{

 public:

 phs_sixth_order(double x = 0., double y = 0., double z = 0.) :  radial_basis_function(x,y,z) {};

 phs_sixth_order(coordinate input) :  radial_basis_function(input) {};

 phs_sixth_order(phs_sixth_order &input) : radial_basis_function(input) {};

  double operator() (double radius);
  using radial_basis_function::operator();
  double Dx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxyz(double x_in = 0., double y_in = 0., double z_in = 0.);
};

/**
   A seventh order polyhoarmonic spline:
   phi(r) = r^7
**/

class phs_seventh_order : public radial_basis_function
{

 public:

 phs_seventh_order(double x = 0., double y = 0., double z = 0.) :  radial_basis_function(x,y,z) {};

 phs_seventh_order(coordinate input) :  radial_basis_function(input) {};

 phs_seventh_order(phs_seventh_order &input) : radial_basis_function(input) {};

  double operator() (double radius);
  using radial_basis_function::operator();
  double Dx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxyz(double x_in = 0., double y_in = 0., double z_in = 0.);
};

/**
   A eighth order polyhoarmonic spline:
   phi(r) = r^8*log(r)
**/

class phs_eighth_order : public radial_basis_function
{

 public:

 phs_eighth_order(double x = 0., double y = 0., double z = 0.) :  radial_basis_function(x,y,z) {};

 phs_eighth_order(coordinate input) :  radial_basis_function(input) {};

 phs_eighth_order(phs_eighth_order &input) : radial_basis_function(input) {};

  double operator() (double radius);
  using radial_basis_function::operator();
  double Dx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxyz(double x_in = 0., double y_in = 0., double z_in = 0.);
};

/**
   A nineth order polyhoarmonic spline:
   phi(r) = r^9
**/

class phs_nineth_order : public radial_basis_function
{

 public:

 phs_nineth_order(double x = 0., double y = 0., double z = 0.) :  radial_basis_function(x,y,z) {};

 phs_nineth_order(coordinate input) :  radial_basis_function(input) {};

 phs_nineth_order(phs_nineth_order &input) : radial_basis_function(input) {};

  double operator() (double radius);
  using radial_basis_function::operator();
  double Dx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxyz(double x_in = 0., double y_in = 0., double z_in = 0.);
};

/**
   A tenth order polyhoarmonic spline:
   phi(r) = r^10*log(r)
**/

class phs_tenth_order : public radial_basis_function
{

 public:

 phs_tenth_order(double x = 0., double y = 0., double z = 0.) :  radial_basis_function(x,y,z) {};

 phs_tenth_order(coordinate input) :  radial_basis_function(input) {};

 phs_tenth_order(phs_tenth_order &input) : radial_basis_function(input) {};

  double operator() (double radius);
  using radial_basis_function::operator();
  double Dx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxyz(double x_in = 0., double y_in = 0., double z_in = 0.);
};

#endif    /*RBF_PHS_H*/
