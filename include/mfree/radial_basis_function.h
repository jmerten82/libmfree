/*** radial_basis_functions.h
This abstract class provides a radial basis functions and its
derivatives up to third order. The functions are hard-wired
due to the fact that we analytically implement the derivatives.
 
Julian Merten
Universiy of Oxford
Feb 2015
julian.merten@physics.ox.ac.uk
http://www.julianmerten.net
***/

#ifndef    RADIAL_BASIS_FUNCTIONS_H
#define    RADIAL_BASIS_FUNCTIONS_H

#include <cmath>
#include <stdexcept>

using namespace std;

struct coordinate 
{

  double x; //x coordinate 
  double y; //y coordinate
  double z; //z coordinate

  //Using a constructor to initialise entries to 0
  coordinate() : x(0.), y(0.), z(0.)
  {
  }
};
/**

    Abstract base class for a radial basis function. Delivers the 
    full interface and parameter handling of a general function.

**/


class radial_basis_function
{

 protected:

  /*
    The shape parameter of the radial basis function.
  */

  double epsilon;

  /*
    The original of the radial basis funtions's coordinate system. Can
    be seen also as an offset of the coordinate system. 
  */

  double x_0, y_0, z_0;

  /*
    Constants often used in the differentiation.
  */

  double Etwo, Efour, Esix, twoEtwo, fourEfour,threeEfour, eightEfour, nineEfour, twentyfourEfour, eightEsix, fifteenEsix,fortyeightEsix;

 public:

  /*
    Standard constructor. Needs the type and the epsilon value. If not 
    given they are set to default values.
  */ 

  radial_basis_function(double epsilon = 1.);

  /*
    Standard destructor. In fact, does nothing.
  */

  ~radial_basis_function();

  /*
    Here you can set the epsilon or also also called shape parameter
    of the radial basis function.
  */

  void set_epsilon(double epsilon_in);

  /*
    Returns the shape parameter the radial basis function is currently set to. 
  */

  double show_epsilon();

  /*
    Here you can set the coordinate origin of the radial basis 
    function.
  */

  void set_coordinate_offset(double x_in = 0., double y_in = 0., double z_in = 0.);
  
  /*
    Same function as above but using a coordinate structure.
  */

  void set_coordinate_offset(coordinate x_in);

  /*
    Returns the current 2D coordinate offset of the radial basis function.
  */

  coordinate show_coordinate_offset();

  /*
    The standard bracket operator evaluates the radial basis function
    for a given radius r. No information about coordinates are used
    here. It is implemented as a pure virtual function.
  */

  virtual double operator() (double radius) = 0;

  /*
    This bracket operator evaluates the radial basis function
    at a specific coordinate from which the origin
    coordinate of the radial basis function is subtracted to
    evaluate the radius. This function is general and can be used for
    the 1D, 2D and 3D case. Also this is implemented as pure virtual function
  */

  virtual double operator() (double x_in, double y_in, double z_in = 0.) = 0;

  /*
    The same function as above but using a coordinate structure.
  */

  virtual double operator() (coordinate x_in) = 0;

  /*
    This function returns the x derivative of the 
    radial basis function at a certain coordinate (x_in, y_in);
  */

  virtual double Dx(double x_in = 0., double y_in = 0., double z_in = 0.) = 0;

  /*
    The same function as above but using a 2D coordinate structure.
  */

  virtual double Dx(coordinate x_in) = 0;

  /*
    This function returns the y derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  virtual double Dy(double x_in = 0., double y_in = 0., double z_in = 0.) = 0;

  /*
    The same function as above but using a coordinate structure.
  */

  virtual double Dy(coordinate x_in) = 0;

  /*
    This function returns the z derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  virtual double Dz(double x_in = 0., double y_in = 0., double z_in = 0.) = 0;

  /*
    The same function as above but using a coordinate structure.
  */

  virtual double Dz(coordinate x_in) = 0;

  /*
    This function returns the second x derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  virtual double Dxx(double x_in = 0., double y_in = 0., double z_in = 0.) = 0;

  /*
    The same function as above but using a coordinate structure.
  */

  virtual double Dxx(coordinate x_in) = 0;

  /*
    This function returns the second y derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  virtual double Dyy(double x_in = 0., double y_in = 0., double z_in = 0.) = 0;

  /*
    The same function as above but using a coordinate structure.
  */

  virtual double Dyy(coordinate x_in) = 0;

  /*
    This function returns the second z derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  virtual double Dzz(double x_in = 0., double y_in = 0., double z_in = 0.) = 0;

  /*
    The same function as above but using a coordinate structure.
  */

  virtual double Dzz(coordinate x_in) = 0;

  /*
    This function returns the second mixed x and y derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  virtual double Dxy(double x_in = 0., double y_in = 0., double z_in = 0.) = 0;

  /*
    The same function as above but using a coordinate structure.
  */

  virtual double Dxy(coordinate x_in) = 0;

  /*
    This function returns the second mixed x and z derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  virtual double Dxz(double x_in = 0., double y_in = 0., double z_in = 0.) = 0;

  /*
    The same function as above but using a coordinate structure.
  */

  virtual double Dxz(coordinate x_in) = 0;

  /*
    This function returns the second mixed y and z derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  virtual double Dyz(double x_in = 0., double y_in = 0., double z_in = 0.) = 0;

  /*
    The same function as above but using a coordinate structure.
  */

  virtual double Dyz(coordinate x_in) = 0;

  /*
    This function returns the third x derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  virtual double Dxxx(double x_in = 0., double y_in = 0., double z_in = 0.) = 0;

  /*
    The same function as above but using a coordinate structure.
  */

  virtual double Dxxx(coordinate x_in) = 0;

  /*
    This function returns the third y derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  virtual double Dyyy(double x_in = 0., double y_in = 0., double z_in = 0.) = 0;

  /*
    The same function as above but using a coordinate structure.
  */

  virtual double Dyyy(coordinate x_in) = 0;

  /*
    This function returns the third  z derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in, z_in);
  */

  virtual double Dzzz(double x_in = 0., double y_in = 0., double z_in = 0.) = 0;

  /*
    The same function as above but using a coordinate structure.
  */

  virtual double Dzzz(coordinate x_in) = 0;

  /*
    This function returns the third mixed xx and y derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in, z_in);
  */

  virtual double Dxxy(double x_in = 0., double y_in = 0., double z_in = 0.) = 0;

  /*
    The same function as above but using a coordinate structure.
  */

  virtual double Dxxy(coordinate x_in) = 0;

  /*
    This function returns the third mixed xx and z derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in, z_in);
  */

  virtual double Dxxz(double x_in = 0., double y_in = 0., double z_in = 0.) = 0;

  /*
    The same function as above but using a coordinate structure.
  */

  virtual double Dxxz(coordinate x_in) = 0;
  /*
    This function returns the third mixed yy and z derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in, z_in);
  */

  virtual double Dyyz(double x_in = 0., double y_in = 0., double z_in = 0.) = 0;

  /*
    The same function as above but using a coordinate structure.
  */

  virtual double Dyyz(coordinate x_in) = 0;

  /*
    This function returns the third mixed x and yy derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  virtual double Dxyy(double x_in = 0., double y_in = 0., double z_in = 0.) = 0;

  /*
    The same function as above but using a coordinate structure.
  */

  virtual double Dxyy(coordinate x_in) = 0;

  /*
    This function returns the third mixed x and zz derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  virtual double Dxzz(double x_in = 0., double y_in = 0., double z_in = 0.) = 0;

  /*
    The same function as above but using a coordinate structure.
  */

  virtual double Dxzz(coordinate x_in) = 0;

  /*
    This function returns the third mixed y and zz derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  virtual double Dyzz(double x_in = 0., double y_in = 0., double z_in = 0.) = 0;

  /*
    The same function as above but using a coordinate structure.
  */

  virtual double Dyzz(coordinate x_in) = 0;

  /*
    This function returns the third mixed x, y and z derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  virtual double Dxyz(double x_in = 0., double y_in = 0., double z_in = 0.) = 0;

  /*
    The same function as above but using a coordinate structure.
  */

  virtual double Dxyz(coordinate x_in) = 0;

};

/**
   A Gaussian radial basis function implementing
   \phi(r) = Exp(-(\epsilon*r)^2)
**/

class gaussian_rbf : public radial_basis_function
{

  /*
    See base class radial_basis function for explanation. 
  */
 public:

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

#endif    /*RADIAL_BASIS_FUNCTIONS_H*/
