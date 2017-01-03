/*** rbf_wendland.h
Explicit implementations of Wendland kernels
of different order and for different dimensions.
Wendland (1995)
http://link.springer.com/article/10.1007/BF02123482
 
Julian Merten
Universiy of Oxford
Jan 2017
julian.merten@physics.ox.ac.uk
http://www.julianmerten.net
***/

#ifndef    RBF_WENDLAND_H
#define    RBF_WENDLAND_H

#include <cmath>
#include <mfree/radial_basis_function.h>

using namespace std;

/**
   One dimensional C0 Wendland kernel:
   phi(r) = max(1-epsilon*r,0)
**/

class wendland_C0_1D : public radial_basis_function_shape
{

 protected:
  
  double E1,E2,E3;

 public:
  
 wendland_C0_1D(double x = 0., double y = 0., double z = 0., double shape = 1.) :  radial_basis_function_shape(x,y,z,shape), E1(3.*shape), E2(-3.*shape), E3(-shape) {};

 wendland_C0_1D(coordinate input, double shape = 1.) :  radial_basis_function_shape(input,shape), E1(3.*shape), E2(-3.*shape), E3(-shape) {};

 wendland_C0_1D(wendland_C0_1D &input) :  radial_basis_function_shape(input), E1(input.E1), E2(input.E2), E3(input.E3) {};


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
   One dimensional C2 Wendland kernel:
   phi(r) = max(1-epsilon*r,0)^3 * (3r+1)
**/

class wendland_C2_1D : public radial_basis_function_shape
{

 protected:
  
  double E1,E2,E3;

 public:
  
 wendland_C2_1D(double x = 0., double y = 0., double z = 0., double shape = 1.) :  radial_basis_function_shape(x,y,z,shape), E1(3.*shape), E2(12.*epsilon*epsilon), E3(24.*epsilon*epsilon*epsilon) {};

 wendland_C2_1D(coordinate input, double shape = 1.) :  radial_basis_function_shape(input,shape), E1(3.*shape), E2(12.*epsilon*epsilon), E3(24.*epsilon*epsilon*epsilon) {};

 wendland_C2_1D(wendland_C2_1D &input) :  radial_basis_function_shape(input), E1(input.E1), E2(input.E2), E3(input.E3) {};


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
   One dimensional C4 Wendland kernel:
   phi(r) = max(1-epsilon*r,0)^5 * (8r^2+5r+1)
**/

class wendland_C4_1D : public radial_basis_function_shape
{

 protected:
  
  double E1,E2,E3,E4;

 public:
  
 wendland_C4_1D(double x = 0., double y = 0., double z = 0., double shape = 1.) :  radial_basis_function_shape(x,y,z,shape), E1(3.*shape), E2(14.*shape*shape), E3(4.*shape*shape), E4(280.*shape*shape*shape*shape) {};

 wendland_C4_1D(coordinate input, double shape = 1.) :  radial_basis_function_shape(input,shape), E1(3.*shape), E2(14.*shape*shape), E3(4.*shape*shape), E4(280.*shape*shape*shape*shape) {};

 wendland_C4_1D(wendland_C4_1D &input) :  radial_basis_function_shape(input), E1(input.E1), E2(input.E2), E3(input.E3), E4(input.E4) {};


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
   Two or three dimensional C0 Wendland kernel:
   phi(r) = max(1-epsilon*r,0)^2
**/

class wendland_C0 : public radial_basis_function_shape
{

 public:
  
 wendland_C0(double x = 0., double y = 0., double z = 0., double shape = 1.) :  radial_basis_function_shape(x,y,z,shape) {};

 wendland_C0(coordinate input, double shape = 1.) :  radial_basis_function_shape(input,shape) {};

 wendland_C0(wendland_C0 &input) :  radial_basis_function_shape(input) {};
  
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
   Two or three dimensional C2 Wendland kernel:
   phi(r) = max(1-epsilon*r,0)^4 * (4r+1)
**/

class wendland_C2 : public radial_basis_function_shape
{

 protected:
  
  double E1,E2,E3;

 public:
  
 wendland_C2(double x = 0., double y = 0., double z = 0., double shape = 1.) :  radial_basis_function_shape(x,y,z,shape), E1(4.*shape), E2(20.*shape*shape), E3(60.*shape*shape*shape) {};

 wendland_C2(coordinate input, double shape = 1.) :  radial_basis_function_shape(input), E1(4.*shape), E2(20.*shape*shape), E3(60.*shape*shape*shape) {};

 wendland_C2(wendland_C2 &input) :  radial_basis_function_shape(input), E1(input.E1), E2(input.E2), E3(input.E3) {};
  
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
   Two or three dimensional C4 Wendland kernel:
   phi(r) = max(1-epsilon*r,0)^6 * (35r^2+18r+3)
**/

class wendland_C4 : public radial_basis_function_shape
{

 protected:
  
  double E1,E2,E3;

 public:
  
 wendland_C4(double x = 0., double y = 0., double z = 0., double shape = 1.) :  radial_basis_function_shape(x,y,z,shape) {};

 wendland_C4(coordinate input) :  radial_basis_function_shape(input) {};

 wendland_C4(wendland_C4 &input) :  radial_basis_function_shape(input) {};
  
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
   Four or five dimensional C0 Wendland kernel:
   phi(r) = max(1-epsilon*r,0)^3
**/

class wendland_C0_5D : public radial_basis_function_shape
{

 public:
  
 wendland_C0_5D(double x = 0., double y = 0., double z = 0., double shape = 1.) :  radial_basis_function_shape(x,y,z,shape) {};

 wendland_C0_5D(coordinate input) :  radial_basis_function_shape(input) {};

 wendland_C0_5D(wendland_C0_5D &input) :  radial_basis_function_shape(input) {};
  
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
   Four or five dimensional C2 Wendland kernel:
   phi(r) = max(1-epsilon*r,0)^5 * (5r+1)
**/

class wendland_C2_5D : public radial_basis_function_shape
{

 public:
  
 wendland_C2_5D(double x = 0., double y = 0., double z = 0., double shape = 1.) :  radial_basis_function_shape(x,y,z,shape) {};

 wendland_C2_5D(coordinate input) :  radial_basis_function_shape(input) {};

 wendland_C2_5D(wendland_C2_5D &input) :  radial_basis_function_shape(input) {};
  
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
   Four or five dimensional C4 Wendland kernel:
   phi(r) = max(1-epsilon*r,0)^7 * (16r^2+7r+3)
**/

class wendland_C4_5D : public radial_basis_function_shape
{

 public:
  
 wendland_C4_5D(double x = 0., double y = 0., double z = 0., double shape = 1.) :  radial_basis_function_shape(x,y,z,shape) {};

 wendland_C4_5D(coordinate input) :  radial_basis_function_shape(input) {};

 wendland_C4_5D(wendland_C4_5D &input) :  radial_basis_function_shape(input) {};
  
  double operator() (double radius);
  using radial_basis_function::operator();
  double Dx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxx(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxxy(double x_in = 0., double y_in = 0., double z_in = 0.);
  double Dxyz(double x_in = 0., double y_in = 0., double z_in = 0.);

};


#endif    /*RBF_WENDLAND_H*/

