/*** radial_basis_functions.h
This abstract class provides a radial basis functions and its
derivatives up to third order. The functions are hard-wired
due to the fact that we analytically implement the derivatives.

The classes here are base classes. Explicit RBFs are defined elsewhere
but inherit the functionality of the base classes. 
 
Julian Merten
Universiy of Oxford
Feb 2015
julian.merten@physics.ox.ac.uk
http://www.julianmerten.net
***/

#ifndef    RADIAL_BASIS_FUNCTIONS_H
#define    RADIAL_BASIS_FUNCTIONS_H

#include <stdexcept>
#include <cmath>

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
    The original of the radial basis funtions's coordinate system. Can
    be seen also as an offset of the coordinate system. 
  */

  double x_0, y_0, z_0;

 public:

  /*
    Standard constructor. Does nothing but setting the origin to the input
    or 0.
  */ 

 radial_basis_function(double x = 0., double y = 0., double z = 0.) : x_0(x), y_0(y), z_0(z){};

  /*
    Another constructor, using a coordinate structure.
  */

 radial_basis_function(coordinate input) : x_0(input.x), y_0(input.y), z_0(input.z){};

  /*
    Copy constructor.
  */

 radial_basis_function(radial_basis_function &input) : x_0(input.x_0), y_0(input.y_0), z_0(input.z_0){};

  /*
    Standard destructor. In fact, does nothing.
  */

  ~radial_basis_function();


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

  double operator() (double x_in, double y_in, double z_in = 0.);

  /*
    The same function as above but using a coordinate structure.
  */

  double operator() (coordinate x_in);

  /*
    This function returns the x derivative of the 
    radial basis function at a certain coordinate (x_in, y_in);
  */

  virtual double Dx(double x_in = 0., double y_in = 0., double z_in = 0.) = 0;

  /*
    The same function as above but using a 2D coordinate structure.
  */

  double Dx(coordinate x_in);

  /*
    This function returns the y derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  double Dy(double x_in = 0., double y_in = 0., double z_in = 0.);

  /*
    The same function as above but using a coordinate structure.
  */

  double Dy(coordinate x_in);

  /*
    This function returns the z derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  double Dz(double x_in = 0., double y_in = 0., double z_in = 0.);

  /*
    The same function as above but using a coordinate structure.
  */

  double Dz(coordinate x_in);

  /*
    This function returns the second x derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  virtual double Dxx(double x_in = 0., double y_in = 0., double z_in = 0.) = 0;

  /*
    The same function as above but using a coordinate structure.
  */

  double Dxx(coordinate x_in);

  /*
    This function returns the second y derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  double Dyy(double x_in = 0., double y_in = 0., double z_in = 0.);

  /*
    The same function as above but using a coordinate structure.
  */

  double Dyy(coordinate x_in);

  /*
    This function returns the second z derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  double Dzz(double x_in = 0., double y_in = 0., double z_in = 0.);

  /*
    The same function as above but using a coordinate structure.
  */

  double Dzz(coordinate x_in);

  /*
    This function returns the second mixed x and y derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  virtual double Dxy(double x_in = 0., double y_in = 0., double z_in = 0.) = 0;

  /*
    The same function as above but using a coordinate structure.
  */

  double Dxy(coordinate x_in);

  /*
    This function returns the second mixed x and z derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  double Dxz(double x_in = 0., double y_in = 0., double z_in = 0.);

  /*
    The same function as above but using a coordinate structure.
  */

  double Dxz(coordinate x_in);

  /*
    This function returns the second mixed y and z derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  double Dyz(double x_in = 0., double y_in = 0., double z_in = 0.);

  /*
    The same function as above but using a coordinate structure.
  */

  double Dyz(coordinate x_in);

  /*
    This function returns the third x derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  virtual double Dxxx(double x_in = 0., double y_in = 0., double z_in = 0.) = 0;

  /*
    The same function as above but using a coordinate structure.
  */

  double Dxxx(coordinate x_in);

  /*
    This function returns the third y derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  double Dyyy(double x_in = 0., double y_in = 0., double z_in = 0.);

  /*
    The same function as above but using a coordinate structure.
  */

  double Dyyy(coordinate x_in);

  /*
    This function returns the third  z derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in, z_in);
  */

  double Dzzz(double x_in = 0., double y_in = 0., double z_in = 0.);

  /*
    The same function as above but using a coordinate structure.
  */

  double Dzzz(coordinate x_in);

  /*
    This function returns the third mixed xx and y derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in, z_in);
  */

  virtual double Dxxy(double x_in = 0., double y_in = 0., double z_in = 0.) = 0;

  /*
    The same function as above but using a coordinate structure.
  */

  double Dxxy(coordinate x_in);

  /*
    This function returns the third mixed xx and z derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in, z_in);
  */

  double Dxxz(double x_in = 0., double y_in = 0., double z_in = 0.);

  /*
    The same function as above but using a coordinate structure.
  */

  double Dxxz(coordinate x_in);
  /*
    This function returns the third mixed yy and z derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in, z_in);
  */

  double Dyyz(double x_in = 0., double y_in = 0., double z_in = 0.);

  /*
    The same function as above but using a coordinate structure.
  */

  double Dyyz(coordinate x_in);

  /*
    This function returns the third mixed x and yy derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  double Dxyy(double x_in = 0., double y_in = 0., double z_in = 0.);

  /*
    The same function as above but using a coordinate structure.
  */

  double Dxyy(coordinate x_in);

  /*
    This function returns the third mixed x and zz derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  double Dxzz(double x_in = 0., double y_in = 0., double z_in = 0.);

  /*
    The same function as above but using a coordinate structure.
  */

  double Dxzz(coordinate x_in);

  /*
    This function returns the third mixed y and zz derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  double Dyzz(double x_in = 0., double y_in = 0., double z_in = 0.);

  /*
    The same function as above but using a coordinate structure.
  */

  double Dyzz(coordinate x_in);

  /*
    This function returns the third mixed x, y and z derivative of the 
    radial basis function at a certain coordinate (x_in, y_in, z_in);
  */

  virtual double Dxyz(double x_in = 0., double y_in = 0., double z_in = 0.) = 0;

  /*
    The same function as above but using a coordinate structure.
  */

  double Dxyz(coordinate x_in);

};


/**
   The second RBF base class, which additionally allows for 
   a shape parameter in the RBF.
**/

class radial_basis_function_shape : public radial_basis_function
{

 protected:

  /*
    The shape parameter of the radial basis function.
  */

  double epsilon;


 public:

  /*
    Standard constructor, setting origin and shape to 0.
  */

 radial_basis_function_shape(double x = 0., double y = 0., double z = 0., double shape = 0.) : radial_basis_function(x,y,z), epsilon(shape) {};

  /*
    Standard constructor using a coordinate structure instead. 
  */

  radial_basis_function_shape(coordinate input, double shape = 0.) : radial_basis_function(input),  epsilon(shape) {};

  /*
    Copy constructor.
  */

 radial_basis_function_shape(radial_basis_function_shape &input) : radial_basis_function(input.show_coordinate_offset()), epsilon(input.epsilon) {};

  /*
    Destructor doing nothing.
  */

  ~radial_basis_function_shape() {};


  /*
    Here you can set the epsilon or also also called shape parameter
    of the radial basis function.
  */

  virtual void set_epsilon(double epsilon_in);

  /*
    Returns the shape parameter the radial basis function is currently set to. 
  */

  double show_epsilon();

  /*
    Increment operator which increases the shape parameter of the 
    RBF by a double value. 
  */

  virtual void operator += (double input);

  /*
    Decrement operator which decreases the shape parameter of the 
    RBF by a double value. 
  */

  virtual void operator -= (double input);

  /*
    Multiplication operator which multiplies the shape parameter of the 
    RBF with a double value. 
  */

  virtual void operator *= (double input);

  /*
    Division operator which divides the shape parameter of the 
    RBF with a double value. 
  */

  virtual void operator /= (double input);

};

#endif    /*RADIAL_BASIS_FUNCTIONS_H*/
