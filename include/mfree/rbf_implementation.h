/*** rbf_implementation.h
Explicit implementations of radial basis functions. 
Depending on their number of parameters, they derive from
the base classes in radial basis function
 
Julian Merten
Universiy of Oxford
Jul 2015
julian.merten@physics.ox.ac.uk
http://www.julianmerten.net
***/

#ifndef    RBF_IMPLEMENTATION_H
#define    RBF_IMPLEMENTATION_H

#include <cmath>
#include <mfree/radial_basis_function.h>

/**
   A Gaussian radial basis function implementing
   \phi(r) = Exp(-(\epsilon*r)^2)
**/

class gaussian_rbf : public radial_basis_function_shape
{

 protected:

  /*
    Constants often used in the differentiation.
  */

  double Etwo, Efour, Esix, twoEtwo, fourEfour,threeEfour, eightEfour, nineEfour, twentyfourEfour, eightEsix, fifteenEsix,fortyeightEsix;

  /*
    See base class radial_basis function for explanation. 
  */
 public:

  gaussian_rbf(double x = 0., double y = 0., double z = 0., double shape = 1.);

  gaussian_rbf(coordinate input, double shape = 1.);

  gaussian_rbf(gaussian_rbf &input);

  void operator += (double input);
  void operator -= (double input);
  void operator *= (double input);
  void operator /= (double input);
  void set_epsilon(double epsilon_in);

  double operator() (double radius);
  double operator() (double x_in, double y_in, double z_in = 0.);
  double operator() (coordinate x_in);
  double Dx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dx(coordinate x_in);
  double Dy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dy(coordinate x_in);
  double Dz(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dz(coordinate x_in);
  double Dxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxx(coordinate x_in);
  double Dyy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dyy(coordinate x_in);
  double Dzz(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dzz(coordinate x_in);
  double Dxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxy(coordinate x_in);
  double Dxz(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxz(coordinate x_in);
  double Dyz(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dyz(coordinate x_in);
  double Dxxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxx(coordinate x_in);
  double Dyyy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dyyy(coordinate x_in);
  double Dzzz(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dzzz(coordinate x_in);
  double Dxxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxy(coordinate x_in);
  double Dxxz(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxz(coordinate x_in);
  double Dyyz(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dyyz(coordinate x_in);
  double Dxyy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxyy(coordinate x_in);
  double Dxzz(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxzz(coordinate x_in);
  double Dyzz(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dyzz(coordinate x_in);
  double Dxyz(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxyz(coordinate x_in);
};

/**
   A cubic spline radial basis function implementing
   \phi(r) = r^3

**/

class cubic_spline_rbf : public radial_basis_function
{
  /*
    See base class for function explanations.
  */

 public:

 cubic_spline_rbf(double x = 0., double y = 0., double z = 0.) :  radial_basis_function(x,y,z) {};

 cubic_spline_rbf(coordinate input) :  radial_basis_function(input) {};

 cubic_spline_rbf(cubic_spline_rbf &input) : radial_basis_function(input) {};

  double operator() (double radius);
  double operator() (double x_in, double y_in, double z_in = 0.);
  double operator() (coordinate x_in);
  double Dx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dx(coordinate x_in);
  double Dy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dy(coordinate x_in);
  double Dz(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dz(coordinate x_in);
  double Dxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxx(coordinate x_in);
  double Dyy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dyy(coordinate x_in);
  double Dzz(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dzz(coordinate x_in);
  double Dxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxy(coordinate x_in);
  double Dxz(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxz(coordinate x_in);
  double Dyz(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dyz(coordinate x_in);
  double Dxxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxx(coordinate x_in);
  double Dyyy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dyyy(coordinate x_in);
  double Dzzz(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dzzz(coordinate x_in);
  double Dxxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxy(coordinate x_in);
  double Dxxz(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxz(coordinate x_in);
  double Dyyz(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dyyz(coordinate x_in);
  double Dxyy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxyy(coordinate x_in);
  double Dxzz(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxzz(coordinate x_in);
  double Dyzz(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dyzz(coordinate x_in);
  double Dxyz(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxyz(coordinate x_in);
};



#endif    /*RBF_IMPLEMENTATION_H*/
