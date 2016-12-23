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

#include <mfree/rbf_implementation.h>

using namespace std;

gaussian_rbf::gaussian_rbf(double x, double y, double z, double shape) : radial_basis_function_shape(x,y,z,shape)
{
  Etwo = epsilon*epsilon;
  Efour = Etwo*Etwo;
  Esix = Etwo*Efour;
  twoEtwo = 2.*Etwo;
  fourEfour = 4.*Efour;
  threeEfour = 3.*Efour;
  eightEfour = 8.*Efour;
  nineEfour = 9.*Efour;
  twentyfourEfour = 24.*Efour;
  eightEsix = 8.*Esix;
  fifteenEsix = 15.*Esix;
  fortyeightEsix = 48.*Esix;
}

gaussian_rbf::gaussian_rbf(coordinate input, double shape) : radial_basis_function_shape(input.x,input.y,input.z,shape)
{
  Etwo = epsilon*epsilon;
  Efour = Etwo*Etwo;
  Esix = Etwo*Efour;
  twoEtwo = 2.*Etwo;
  fourEfour = 4.*Efour;
  threeEfour = 3.*Efour;
  eightEfour = 8.*Efour;
  nineEfour = 9.*Efour;
  twentyfourEfour = 24.*Efour;
  eightEsix = 8.*Esix;
  fifteenEsix = 15.*Esix;
  fortyeightEsix = 48.*Esix;
}

gaussian_rbf::gaussian_rbf(gaussian_rbf &input) : radial_basis_function_shape(input.x_0,input.y_0,input.z_0,input.epsilon)
{
  Etwo = epsilon*epsilon;
  Efour = Etwo*Etwo;
  Esix = Etwo*Efour;
  twoEtwo = 2.*Etwo;
  fourEfour = 4.*Efour;
  threeEfour = 3.*Efour;
  eightEfour = 8.*Efour;
  nineEfour = 9.*Efour;
  twentyfourEfour = 24.*Efour;
  eightEsix = 8.*Esix;
  fifteenEsix = 15.*Esix;
  fortyeightEsix = 48.*Esix;
}

void gaussian_rbf::operator+=(double input)
{
  epsilon += input;

  Etwo = epsilon*epsilon;
  Efour = Etwo*Etwo;
  Esix = Etwo*Efour;
  twoEtwo = 2.*Etwo;
  fourEfour = 4.*Efour;
  threeEfour = 3.*Efour;
  eightEfour = 8.*Efour;
  nineEfour = 9.*Efour;
  twentyfourEfour = 24.*Efour;
  eightEsix = 8.*Esix;
  fifteenEsix = 15.*Esix;
  fortyeightEsix = 48.*Esix;
}

void gaussian_rbf::operator-=(double input)
{
  epsilon -= input;

  Etwo = epsilon*epsilon;
  Efour = Etwo*Etwo;
  Esix = Etwo*Efour;
  twoEtwo = 2.*Etwo;
  fourEfour = 4.*Efour;
  threeEfour = 3.*Efour;
  eightEfour = 8.*Efour;
  nineEfour = 9.*Efour;
  twentyfourEfour = 24.*Efour;
  eightEsix = 8.*Esix;
  fifteenEsix = 15.*Esix;
  fortyeightEsix = 48.*Esix;
}

void gaussian_rbf::operator*=(double input)
{
  epsilon *= input;

  Etwo = epsilon*epsilon;
  Efour = Etwo*Etwo;
  Esix = Etwo*Efour;
  twoEtwo = 2.*Etwo;
  fourEfour = 4.*Efour;
  threeEfour = 3.*Efour;
  eightEfour = 8.*Efour;
  nineEfour = 9.*Efour;
  twentyfourEfour = 24.*Efour;
  eightEsix = 8.*Esix;
  fifteenEsix = 15.*Esix;
  fortyeightEsix = 48.*Esix;
}

void gaussian_rbf::operator/=(double input)
{
  epsilon /= input;

  Etwo = epsilon*epsilon;
  Efour = Etwo*Etwo;
  Esix = Etwo*Efour;
  twoEtwo = 2.*Etwo;
  fourEfour = 4.*Efour;
  threeEfour = 3.*Efour;
  eightEfour = 8.*Efour;
  nineEfour = 9.*Efour;
  twentyfourEfour = 24.*Efour;
  eightEsix = 8.*Esix;
  fifteenEsix = 15.*Esix;
  fortyeightEsix = 48.*Esix;
}


void gaussian_rbf::set_epsilon(double input)
{
  epsilon = input;

  Etwo = epsilon*epsilon;
  Efour = Etwo*Etwo;
  Esix = Etwo*Efour;
  twoEtwo = 2.*Etwo;
  fourEfour = 4.*Efour;
  threeEfour = 3.*Efour;
  eightEfour = 8.*Efour;
  nineEfour = 9.*Efour;
  twentyfourEfour = 24.*Efour;
  eightEsix = 8.*Esix;
  fifteenEsix = 15.*Esix;
  fortyeightEsix = 48.*Esix;
}

double gaussian_rbf::operator() (double radius)
{

  return exp(-Etwo*radius*radius);      
}

double gaussian_rbf::Dx(double x_in, double y_in, double z_in)
{

  double radius = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  return -twoEtwo*(x_in-x_0)*(* this)(radius);
}

double gaussian_rbf::Dxx(double x_in, double y_in, double z_in)
{

  double radius = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  double x = x_in-x_0;
  return (fourEfour*x*x-twoEtwo)*(* this)(radius);
}


double gaussian_rbf::Dxy(double x_in, double y_in, double z_in)
{

  double radius = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  return fourEfour*(x_in-x_0)*(y_in-y_0)*(* this)(radius);
}

double gaussian_rbf::Dxxx(double x_in, double y_in, double z_in)
{

  double radius = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  double x = x_in-x_0;
  return -fourEfour*x*(twoEtwo*x*x-3.)*(* this)(radius);
}


double gaussian_rbf::Dxxy(double x_in, double y_in, double z_in)
{

  double radius = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  double x = x_in - x_0;
  return -fourEfour*(y_in-y_0)*(twoEtwo*x*x-1.)*(* this)(radius);
}

double gaussian_rbf::Dxyz(double x_in, double y_in, double z_in)
{

  double radius = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  return -eightEsix*(y_in-y_0)*(y_in-y_0)*(z_in-z_0)*(* this)(radius);
}
