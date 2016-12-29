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

multiquadric_rbf::multiquadric_rbf(double x, double y, double z, double shape) : radial_basis_function_shape(x,y,z,shape)
{
  Etwo = epsilon*epsilon;
  Efour = Etwo*Etwo;
  Esix = Etwo*Efour;
}

multiquadric_rbf::multiquadric_rbf(coordinate input, double shape) : radial_basis_function_shape(input.x,input.y,input.z,shape)
{
  Etwo = epsilon*epsilon;
  Efour = Etwo*Etwo;
  Esix = Etwo*Efour;
}

multiquadric_rbf::multiquadric_rbf(multiquadric_rbf &input) : radial_basis_function_shape(input.x_0,input.y_0,input.z_0,input.epsilon)
{
  Etwo = epsilon*epsilon;
  Efour = Etwo*Etwo;
  Esix = Etwo*Efour;
}
void multiquadric_rbf::set_epsilon(double input)
{
  epsilon = input;

  Etwo = epsilon*epsilon;
  Efour = Etwo*Etwo;
  Esix = Etwo*Efour;
}

double multiquadric_rbf::operator() (double radius)
{
  return sqrt(1.+Etwo*radius*radius);
}

double multiquadric_rbf::Dx(double x_in, double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  return Etwo*x*pow(1.+Etwo*(x*x+y*y+z*z),-0.5);
}

double multiquadric_rbf::Dxx(double x_in, double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  return (Etwo+Efour*aux)*pow(1.+Etwo*(x*x+aux),-1.5);
}

double multiquadric_rbf::Dxy(double x_in, double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  return Efour*x*y*pow(1.+Etwo*(x*x+y*y+z*z),-1.5);
}

double multiquadric_rbf::Dxxx(double x_in, double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  return 3.*Efour*x*(1.+Etwo*aux)*pow(1.+Etwo*(x*x+aux),-2.5);
}

double multiquadric_rbf::Dxxy(double x_in, double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  return -y*(Efour+Esix*(-2.*x*x+aux))*pow(1.+Etwo*(x*x+aux),-2.5);
}

double multiquadric_rbf::Dxyz(double x_in, double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  return 3.*Esix*x*y*z*pow(1.+Etwo*(x*x+y*y+z*z),-2.5);
}

inverse_multiquadric_rbf::inverse_multiquadric_rbf(double x, double y, double z, double shape) : radial_basis_function_shape(x,y,z,shape)
{
  Etwo = epsilon*epsilon;
  Efour = Etwo*Etwo;
  Esix = Etwo*Efour;
}

inverse_multiquadric_rbf::inverse_multiquadric_rbf(coordinate input, double shape) : radial_basis_function_shape(input.x,input.y,input.z,shape)
{
  Etwo = epsilon*epsilon;
  Efour = Etwo*Etwo;
  Esix = Etwo*Efour;
}

inverse_multiquadric_rbf::inverse_multiquadric_rbf(inverse_multiquadric_rbf &input) : radial_basis_function_shape(input.x_0,input.y_0,input.z_0,input.epsilon)
{
  Etwo = epsilon*epsilon;
  Efour = Etwo*Etwo;
  Esix = Etwo*Efour;
}
void inverse_multiquadric_rbf::set_epsilon(double input)
{
  epsilon = input;

  Etwo = epsilon*epsilon;
  Efour = Etwo*Etwo;
  Esix = Etwo*Efour;
}

double inverse_multiquadric_rbf::operator() (double radius)
{
  return 1./sqrt(1.+Etwo*radius*radius);
}

double inverse_multiquadric_rbf::Dx(double x_in, double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  return Etwo*x*pow(1.+Etwo*(x*x+y*y+z*z),-1.5);
}

double inverse_multiquadric_rbf::Dxx(double x_in, double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  return Etwo*(-1.+Etwo*(2.*x*x-aux))*pow(1.+Etwo*(aux+x*x),-2.5);
}

double inverse_multiquadric_rbf::Dxy(double x_in, double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  return Efour*x*y*pow(1.+Etwo*(x*x+y*y+z*z),-1.5);
}

double inverse_multiquadric_rbf::Dxxx(double x_in, double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  return 3.*Efour*x*(3.+Etwo*(-2.*x*x+3.*aux))*pow(1.+Etwo*(x*x+aux),-3.5);
}

double inverse_multiquadric_rbf::Dxxy(double x_in, double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  return 3.*y*(Efour+Esix*(-4.*x*x+aux))*pow(1.+Etwo*(x*x+aux),-3.5);
}

double inverse_multiquadric_rbf::Dxyz(double x_in, double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  return 15.*Esix*x*y*z*pow(1.+Etwo*(x*x+y*y+z*z),-3.5);
}

inverse_quadratic_rbf::inverse_quadratic_rbf(double x, double y, double z, double shape) : radial_basis_function_shape(x,y,z,shape)
{
  Etwo = epsilon*epsilon;
  Efour = Etwo*Etwo;
  Esix = Etwo*Efour;
  TwoEtwo = 2.*Etwo;
}

inverse_quadratic_rbf::inverse_quadratic_rbf(coordinate input, double shape) : radial_basis_function_shape(input.x,input.y,input.z,shape)
{
  Etwo = epsilon*epsilon;
  Efour = Etwo*Etwo;
  Esix = Etwo*Efour;
  TwoEtwo = 2.*Etwo;
}

inverse_quadratic_rbf::inverse_quadratic_rbf(inverse_quadratic_rbf &input) : radial_basis_function_shape(input.x_0,input.y_0,input.z_0,input.epsilon)
{
  Etwo = epsilon*epsilon;
  Efour = Etwo*Etwo;
  Esix = Etwo*Efour;
  TwoEtwo = 2.*Etwo;
}
void inverse_quadratic_rbf::set_epsilon(double input)
{
  epsilon = input;

  Etwo = epsilon*epsilon;
  Efour = Etwo*Etwo;
  Esix = Etwo*Efour;
  TwoEtwo = 2.*Etwo;
}

double inverse_quadratic_rbf::operator() (double radius)
{
  return 1./(1.+Etwo*radius*radius);
}

double inverse_quadratic_rbf::Dx(double x_in, double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = 1.+Etwo*(x*x+y*y+z*z);
  return TwoEtwo*x/(aux*aux);
}

double inverse_quadratic_rbf::Dxx(double x_in, double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  double aux2 = 1.+Etwo*(x*x+y*y+z*z);
  return (-TwoEtwo+Efour*(6.*x*x-2.*(aux)))/(aux2*aux2*aux2);
}

double inverse_quadratic_rbf::Dxy(double x_in, double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux2 = 1.+Etwo*(x*x+y*y+z*z);
  return 8.*Efour*x*y/(aux2*aux2*aux2);
}

double inverse_quadratic_rbf::Dxxx(double x_in, double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  double aux2 = 1.+Etwo*(x*x+y*y+z*z);
  double aux3 = aux2*aux2;
  return 24.*Efour*x*(1.+Etwo*(-x*x+aux))/(aux3*aux3);
}

double inverse_quadratic_rbf::Dxxy(double x_in, double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  double aux2 = 1.+Etwo*(x*x+y*y+z*z);
  double aux3 = aux2*aux2;
  return 8.*y*(Efour+Esix*(-5.*x*x+aux))/(aux3*aux3);
}

double inverse_quadratic_rbf::Dxyz(double x_in, double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux2 = 1.+Etwo*(x*x+y*y+z*z);
  double aux3 = aux2*aux2;
  return -48.*Esix*x*y*z/(aux3*aux3);
}
