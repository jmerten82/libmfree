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
  using radial_basis_function::operator();
  double Dx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxyz(double x_in = 0., double y_in = 0., double z_in = 0.);
};

/**
   The multiquadric radial basis function:
   phi(r) = sqrt(1+(eps*r)^2)
**/

class multiquadric_rbf : public radial_basis_function_shape
{

 protected:

  /*
    Constants often used in the differentiation.
  */

  double Etwo, Efour, Esix;

  /*
    See base class radial_basis function for explanation. 
  */
 public:

  multiquadric_rbf(double x = 0., double y = 0., double z = 0., double shape = 1.);

  multiquadric_rbf(coordinate input, double shape = 1.);

  multiquadric_rbf(multiquadric_rbf &input);

  void set_epsilon(double epsilon_in);

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
   Implements inverse quadric RBF:
   phi(r) = (1.+(eps*r)^2)^-1/2
**/

class inverse_multiquadric_rbf : public radial_basis_function_shape
{

 protected:

  /*
    Constants often used in the differentiation.
  */

  double Etwo, Efour, Esix;

  /*
    See base class radial_basis function for explanation. 
  */
 public:

  inverse_multiquadric_rbf(double x = 0., double y = 0., double z = 0., double shape = 1.);

  inverse_multiquadric_rbf(coordinate input, double shape = 1.);

  inverse_multiquadric_rbf(inverse_multiquadric_rbf &input);

  void set_epsilon(double epsilon_in);

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
   Implements inverse quadric RBF:
   phi(r) = (1.+(eps*r)^2)^-1
**/

class inverse_quadratic_rbf : public radial_basis_function_shape
{

 protected:

  /*
    Constants often used in the differentiation.
  */

  double Etwo, Efour, Esix, TwoEtwo;

  /*
    See base class radial_basis function for explanation. 
  */
 public:

  inverse_quadratic_rbf(double x = 0., double y = 0., double z = 0., double shape = 1.);

  inverse_quadratic_rbf(coordinate input, double shape = 1.);

  inverse_quadratic_rbf(inverse_quadratic_rbf &input);

  void set_epsilon(double epsilon_in);

  double operator() (double radius);
  using radial_basis_function::operator();
  double Dx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxyz(double x_in = 0., double y_in = 0., double z_in = 0.);
};





#endif    /*RBF_IMPLEMENTATION_H*/
