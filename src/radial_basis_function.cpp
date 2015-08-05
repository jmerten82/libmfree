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

