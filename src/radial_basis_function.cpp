/*** radial_basis_functions.cpp
This class provides a set of radial basis functions and their
derivatives up to third order. The functions are hard-wired
due to the fact that we analytically implement the derivatives.
 
Julian Merten
Universiy of Oxford
Feb 2015
julian.merten@physics.ox.ac.uk
http://www.julianmerten.net
***/

#include <mfree/radial_basis_function.h>



radial_basis_function::~radial_basis_function()
{
}

double radial_basis_function::show_epsilon()
{
  return 0.;
}


void radial_basis_function::set_coordinate_offset(double x_in, double y_in, double z_in)
{
  x_0 = x_in;
  y_0 = y_in;
  z_0 = z_in;
}

void radial_basis_function::set_coordinate_offset(coordinate x_in)
{

  x_0 = x_in.x; 
  y_0 = x_in.y; 
  z_0 = x_in.z;
}

coordinate radial_basis_function::show_coordinate_offset()
{

  coordinate x_out;
  x_out.x = x_0;
  x_out.y = y_0;
  x_out.z = z_0;
  return x_out;
}

double radial_basis_function::operator() (double x_in, double y_in, double z_in)
{

  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  return (*this)(radius);
}

double radial_basis_function::operator() (coordinate x_in)
{
  return (*this)(x_in.x,x_in.y,x_in.z);
}

double radial_basis_function::Dx(coordinate x_in)
{
  return this->Dx(x_in.x,x_in.y,x_in.z);
}
double radial_basis_function::Dy(coordinate x_in)
{
  return this->Dy(x_in.x,x_in.y,x_in.z);
}
double radial_basis_function::Dz(coordinate x_in)
{
  return this->Dz(x_in.x,x_in.y,x_in.z);
}
double radial_basis_function::Dxx(coordinate x_in)
{
  return this->Dxx(x_in.x,x_in.y,x_in.z);
}
double radial_basis_function::Dyy(coordinate x_in)
{
  return this->Dyy(x_in.x,x_in.y,x_in.z);
}
double radial_basis_function::Dzz(coordinate x_in)
{
  return this->Dzz(x_in.x,x_in.y,x_in.z);
}
double radial_basis_function::Dxy(coordinate x_in)
{
  return this->Dxy(x_in.x,x_in.y,x_in.z);
}
double radial_basis_function::Dxz(coordinate x_in)
{
  return this->Dxz(x_in.x,x_in.y,x_in.z);
}
double radial_basis_function::Dyz(coordinate x_in)
{
  return this->Dyz(x_in.x,x_in.y,x_in.z);
}
double radial_basis_function::Dxxx(coordinate x_in)
{
  return this->Dxxx(x_in.x,x_in.y,x_in.z);
}
double radial_basis_function::Dyyy(coordinate x_in)
{
  return this->Dyyy(x_in.x,x_in.y,x_in.z);
}
double radial_basis_function::Dzzz(coordinate x_in)
{
  return this->Dzzz(x_in.x,x_in.y,x_in.z);
}
double radial_basis_function::Dxxy(coordinate x_in)
{
  return this->Dxxy(x_in.x,x_in.y,x_in.z);
}
double radial_basis_function::Dxxz(coordinate x_in)
{
  return this->Dxxz(x_in.x,x_in.y,x_in.z);
}
double radial_basis_function::Dyyz(coordinate x_in)
{
  return this->Dyyz(x_in.x,x_in.y,x_in.z);
}
double radial_basis_function::Dxyy(coordinate x_in)
{
  return this->Dxyy(x_in.x,x_in.y,x_in.z);
}
double radial_basis_function::Dxzz(coordinate x_in)
{
  return this->Dxzz(x_in.x,x_in.y,x_in.z);
}
double radial_basis_function::Dyzz(coordinate x_in)
{
  return this->Dyzz(x_in.x,x_in.y,x_in.z);
}
double radial_basis_function::Dxyz(coordinate x_in)
{
  return this->Dxyz(x_in.x,x_in.y,x_in.z);
}

double radial_basis_function::Dy(double x_in, double y_in, double z_in)
{
  coordinate shuffle = this->show_coordinate_offset();
  this->set_coordinate_offset(shuffle.y,shuffle.x,shuffle.z);
  double x = this->Dx(y_in,x_in,z_in);
  this->set_coordinate_offset(shuffle);
  return x;
}

double radial_basis_function::Dz(double x_in, double y_in, double z_in)
{
  coordinate shuffle = this->show_coordinate_offset();
  this->set_coordinate_offset(shuffle.z,shuffle.x,shuffle.y);
  double x = this->Dx(z_in,x_in,y_in);
  this->set_coordinate_offset(shuffle);
  return x;
}

double radial_basis_function::Dyy(double x_in , double y_in, double z_in)
{
  coordinate shuffle = this->show_coordinate_offset();
  this->set_coordinate_offset(shuffle.y,shuffle.x,shuffle.z);
  double x = this->Dxx(y_in,x_in,z_in);
  this->set_coordinate_offset(shuffle);
  return x;
}

double radial_basis_function::Dzz(double x_in , double y_in, double z_in)
{
  coordinate shuffle = this->show_coordinate_offset();
  this->set_coordinate_offset(shuffle.z,shuffle.x,shuffle.y);
  double x = this->Dxx(z_in,x_in,y_in);
  this->set_coordinate_offset(shuffle);
  return x;
}

double radial_basis_function::Dxz(double x_in , double y_in, double z_in)
{
  coordinate shuffle = this->show_coordinate_offset();
  this->set_coordinate_offset(shuffle.x,shuffle.z,shuffle.y);
  double x = this->Dxy(x_in,z_in,y_in);
  this->set_coordinate_offset(shuffle);
  return x;
}
  
double radial_basis_function::Dyz(double x_in , double y_in, double z_in)
{
  coordinate shuffle = this->show_coordinate_offset();
  this->set_coordinate_offset(shuffle.y,shuffle.z,shuffle.x);
  double x = this->Dxy(y_in,z_in,x_in);
  this->set_coordinate_offset(shuffle);
  return x;
}

double radial_basis_function::Dyyy(double x_in , double y_in, double z_in)
{
  coordinate shuffle = this->show_coordinate_offset();
  this->set_coordinate_offset(shuffle.y,shuffle.x,shuffle.z);
  double x = this->Dxxx(y_in,x_in,z_in);
  this->set_coordinate_offset(shuffle);
  return x;
}

double radial_basis_function::Dzzz(double x_in , double y_in, double z_in)
{
  coordinate shuffle = this->show_coordinate_offset();
  this->set_coordinate_offset(shuffle.z,shuffle.x,shuffle.y);
  double x = this->Dxxx(z_in,x_in,y_in);
  this->set_coordinate_offset(shuffle);
  return x;
}

double radial_basis_function::Dxxz(double x_in , double y_in, double z_in)
{
  coordinate shuffle = this->show_coordinate_offset();
  this->set_coordinate_offset(shuffle.x,shuffle.z,shuffle.y);
  double x = this->Dxxy(x_in,z_in,y_in);
  this->set_coordinate_offset(shuffle);
  return x;
}

double radial_basis_function::Dyyz(double x_in , double y_in, double z_in)
{
  coordinate shuffle = this->show_coordinate_offset();
  this->set_coordinate_offset(shuffle.y,shuffle.z,shuffle.x);
  double x = this->Dxxy(y_in,z_in,x_in);
  this->set_coordinate_offset(shuffle);
  return x;  
}

double radial_basis_function::Dxyy(double x_in , double y_in, double z_in)
{
  coordinate shuffle = this->show_coordinate_offset();
  this->set_coordinate_offset(shuffle.y,shuffle.x,shuffle.z);
  double x = this->Dxxy(y_in,x_in,z_in);
  this->set_coordinate_offset(shuffle);
  return x;  
}

double radial_basis_function::Dxzz(double x_in , double y_in, double z_in)
{
  coordinate shuffle = this->show_coordinate_offset();
  this->set_coordinate_offset(shuffle.z,shuffle.x,shuffle.y);
  double x = this->Dxxy(z_in,x_in,y_in);
  this->set_coordinate_offset(shuffle);
  return x;  
}

double radial_basis_function::Dyzz(double x_in , double y_in, double z_in)
{
  coordinate shuffle = this->show_coordinate_offset();
  this->set_coordinate_offset(shuffle.z,shuffle.y,shuffle.x);
  double x = this->Dxxy(z_in,y_in,x_in);
  this->set_coordinate_offset(shuffle);
  return x;  
}

void radial_basis_function_shape::set_epsilon(double epsilon_in)
{

  //Wondering if I should introduce a check that epsilon is 
  //positive, decided against since it makes things more flexible.

  epsilon = epsilon_in;
}

double radial_basis_function_shape::show_epsilon()
{

  return epsilon;
}


void radial_basis_function_shape::operator+=(double input)
{
  epsilon += input;
}

void radial_basis_function_shape::operator-=(double input)
{
  epsilon -= input;
}

void radial_basis_function_shape::operator*=(double input)
{
  epsilon *= input;
}

void radial_basis_function_shape::operator/=(double input)
{
  epsilon /= input;
}

