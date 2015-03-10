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

radial_basis_function::radial_basis_function(double epsilon_in) :  epsilon(epsilon_in)
{

  x_0 = 0.;
  y_0 = 0.;
  z_0 = 0.;

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

radial_basis_function::~radial_basis_function()
{
}

void radial_basis_function::set_epsilon(double epsilon_in)
{

  //Wondering if I should introduce a check that epsilon is 
  //positive, decided against since it makes things more flexible.

  epsilon = epsilon_in;
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

double  radial_basis_function::show_epsilon()
{

  return epsilon;
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

double gaussian_rbf::operator() (double radius)
{

  return exp(-Etwo*radius*radius);      
}

double gaussian_rbf::operator() (double x_in, double y_in, double z_in)
{

  double radius = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  return (* this)(radius);
};

double gaussian_rbf::operator() (coordinate x_in)
{

  double radius = sqrt(pow(x_in.x-x_0,2.)+pow(x_in.y-y_0,2.)+pow(x_in.z-z_0,2.));
  return (* this)(radius);
};

double gaussian_rbf::Dx(double x_in, double y_in, double z_in)
{

  double radius = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  return -twoEtwo*(x_in-x_0)*(* this)(radius);
}

double gaussian_rbf::Dy(double x_in, double y_in, double z_in)
{

  double radius = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  return -twoEtwo*(y_in-y_0)*(* this)(radius);
}

double gaussian_rbf::Dz(double x_in, double y_in, double z_in)
{

  double radius = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  return -twoEtwo*(z_in-z_0)*(* this)(radius);
}

double gaussian_rbf::Dxx(double x_in, double y_in, double z_in)
{

  double radius = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  double x = x_in-x_0;
  return (fourEfour*x*x-twoEtwo)*(* this)(radius);
}

double gaussian_rbf::Dyy(double x_in, double y_in, double z_in)
{

  double radius = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  double y = y_in - y_0;
  return twoEtwo*(twoEtwo*y*y-1.)*(* this)(radius);
}

double gaussian_rbf::Dzz(double x_in, double y_in, double z_in)
{

  double radius = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  double z = z_in - z_0;
  return twoEtwo*(twoEtwo*z*z-1.)*(* this)(radius);
}

double gaussian_rbf::Dxy(double x_in, double y_in, double z_in)
{

  double radius = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  return fourEfour*(x_in-x_0)*(y_in-y_0)*(* this)(radius);
}

double gaussian_rbf::Dxz(double x_in, double y_in, double z_in)
{

  double radius = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  return fourEfour*(x_in-x_0)*(z_in-z_0)*(* this)(radius);
}

double gaussian_rbf::Dyz(double x_in, double y_in, double z_in)
{

  double radius = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  return fourEfour*(y_in-y_0)*(z_in-z_0)*(* this)(radius);
}

double gaussian_rbf::Dxxx(double x_in, double y_in, double z_in)
{

  double radius = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  double x = x_in-x_0;
  return -fourEfour*x*(twoEtwo*x*x-3.)*(* this)(radius);
}

double gaussian_rbf::Dyyy(double x_in, double y_in, double z_in)
{

  double radius = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  double y = y_in-y_0;
  return -fourEfour*y*(twoEtwo*y*y-3.)*(* this)(radius);
}

double gaussian_rbf::Dzzz(double x_in, double y_in, double z_in)
{

  double radius = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  double z = z_in-z_0;
  return -fourEfour*z*(twoEtwo*z*z-3.)*(* this)(radius);
}

double gaussian_rbf::Dxxy(double x_in, double y_in, double z_in)
{

  double radius = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  double x = x_in - x_0;
  return -fourEfour*(y_in-y_0)*(twoEtwo*x*x-1.)*(* this)(radius);
}

double gaussian_rbf::Dxxz(double x_in, double y_in, double z_in)
{

  double radius = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  double x = x_in - x_0;
  return -fourEfour*(z_in-z_0)*(twoEtwo*x*x-1.)*(* this)(radius);
}

double gaussian_rbf::Dyyz(double x_in, double y_in, double z_in)
{

  double radius = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  double y = y_in - y_0;
  return -fourEfour*(z_in-z_0)*(twoEtwo*y*y-1.)*(* this)(radius);
}

double gaussian_rbf::Dxyy(double x_in, double y_in, double z_in)
{

  double radius = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  double y = y_in - y_0;
  return -fourEfour*(x_in-x_0)*(twoEtwo*y*y-1.)*(* this)(radius);
}

double gaussian_rbf::Dxzz(double x_in, double y_in, double z_in)
{

  double radius = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  double z = z_in - z_0;
  return -fourEfour*(x_in-x_0)*(twoEtwo*z*z-1.)*(* this)(radius);
}

double gaussian_rbf::Dyzz(double x_in, double y_in, double z_in)
{

  double radius = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  double z = z_in - z_0;
  return -fourEfour*(y_in-y_0)*(twoEtwo*z*z-1.)*(* this)(radius);
}

double gaussian_rbf::Dxyz(double x_in, double y_in, double z_in)
{

  double radius = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  return -eightEsix*(y_in-y_0)*(y_in-y_0)*(z_in-z_0)*(* this)(radius);
}

double gaussian_rbf::Dx(coordinate x_in)
{

  return this->Dx(x_in.x,x_in.y,x_in.z);
}

double gaussian_rbf::Dy(coordinate x_in)
{

  return this->Dy(x_in.x,x_in.y,x_in.z); 
}

double gaussian_rbf::Dz(coordinate x_in)
{

  return this->Dz(x_in.x,x_in.y,x_in.z); 
}

double gaussian_rbf::Dxx(coordinate x_in)
{

  return this->Dxx(x_in.x,x_in.y,x_in.z);
}

double gaussian_rbf::Dyy(coordinate x_in)
{

  return this->Dyy(x_in.x,x_in.y,x_in.z);
}
double gaussian_rbf::Dzz(coordinate x_in)
{

  return this->Dzz(x_in.x,x_in.y,x_in.z);
}

double gaussian_rbf::Dxy(coordinate x_in)
{

  return this->Dxy(x_in.x,x_in.y,x_in.z);
}
double gaussian_rbf::Dxz(coordinate x_in)
{

  return this->Dxz(x_in.x,x_in.y,x_in.z);
}
double gaussian_rbf::Dyz(coordinate x_in)
{

  return this->Dyz(x_in.x,x_in.y,x_in.z);
}
double gaussian_rbf::Dxxx(coordinate x_in)
{

  return this->Dxxx(x_in.x,x_in.y,x_in.z);
}

double gaussian_rbf::Dyyy(coordinate x_in)
{

  return this->Dyyy(x_in.x,x_in.y,x_in.z);
}
double gaussian_rbf::Dzzz(coordinate x_in)
{

  return this->Dzzz(x_in.x,x_in.y,x_in.z);
}

double gaussian_rbf::Dxxy(coordinate x_in)
{

  return this->Dxxy(x_in.x,x_in.y,x_in.z);
}
double gaussian_rbf::Dxxz(coordinate x_in)
{

  return this->Dxxz(x_in.x,x_in.y,x_in.z);
}
double gaussian_rbf::Dyyz(coordinate x_in)
{

  return this->Dyyz(x_in.x,x_in.y,x_in.z);
}

double gaussian_rbf::Dxyy(coordinate x_in)
{

  return this->Dxyy(x_in.x,x_in.y,x_in.z);
}
double gaussian_rbf::Dxzz(coordinate x_in)
{

  return this->Dxzz(x_in.x,x_in.y,x_in.z);
}
double gaussian_rbf::Dyzz(coordinate x_in)
{

  return this->Dyzz(x_in.x,x_in.y,x_in.z);
}
double gaussian_rbf::Dxyz(coordinate x_in)
{

  return this->Dxyz(x_in.x,x_in.y,x_in.z);
}

double cubic_spline_rbf::operator() (double radius)
{
  return radius*radius*radius;
}

double cubic_spline_rbf::operator() (double x_in , double y_in, double z_in)
{
  double value = pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.);
  return pow(value,1.5);
}

double cubic_spline_rbf::operator() (coordinate x_in)
{
  double value = pow(x_in.x-x_0,2.)+pow(x_in.y-y_0,2.)+pow(x_in.z-z_0,2.);
  return pow(value,1.5);
}

double cubic_spline_rbf::Dx(double x_in , double y_in, double z_in)
{
  double x = x_in - x_0;
  return 3.*x*sqrt(pow(x,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
}

double cubic_spline_rbf::Dy(double x_in , double y_in, double z_in)
{
  double y = y_in - y_0;
  return 3.*y*sqrt(pow(x_in-x_0,2.)+pow(y,2.)+pow(z_in-z_0,2.));
}

double cubic_spline_rbf::Dz(double x_in , double y_in, double z_in)
{
  double z = z_in - z_0;
  return 3.*z*sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z,2.));
}

double cubic_spline_rbf::Dxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double r = sqrt(pow(x,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  return 3. * (x*x/r + r);
}

double cubic_spline_rbf::Dyy(double x_in , double y_in, double z_in)
{
  double y = y_in-y_0;
  double r = sqrt(pow(x_in-x_0,2.)+pow(y,2.)+pow(z_in-z_0,2.));
  return 3. * (y*y/r + r);
}

double cubic_spline_rbf::Dzz(double x_in , double y_in, double z_in)
{
  double z = z_in-z_0;
  double r = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z,2.));
  return 3. * (z*z/r + r);
}

double cubic_spline_rbf::Dxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double r = sqrt(pow(x,2.)+pow(y,2.)+pow(z_in-z_0,2.));
  return 3.*x*y/r;
}

double cubic_spline_rbf::Dxz(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double z = z_in-z_0;
  double r = sqrt(pow(x,2.)+pow(y_in-y_0,2.)+pow(z,2.));
  return 3.*x*z/r;
}
  
double cubic_spline_rbf::Dyz(double x_in , double y_in, double z_in)
{
  double y = y_in-y_0;
  double z = z_in-z_0;
  double r = sqrt(pow(x_in-x_0,2.)+pow(y,2.)+pow(z,2.));
  return 3.*y*z/r;
}

double cubic_spline_rbf::Dxxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double r = sqrt(pow(x,2.)+pow(y_in-y_0,2.)+pow(z_in-z_0,2.));
  return 3.*x*(3./r - x*x*pow(r,-3));
}

double cubic_spline_rbf::Dyyy(double x_in , double y_in, double z_in)
{
  double y = y_in-y_0;
  double r = sqrt(pow(x_in-x_0,2.)+pow(y,2.)+pow(z_in-z_0,2.));
  return 3.*y*(3./r - y*y*pow(r,-3));
}

double cubic_spline_rbf::Dzzz(double x_in , double y_in, double z_in)
{
  double z = z_in-z_0;
  double r = sqrt(pow(x_in-x_0,2.)+pow(y_in-y_0,2.)+pow(z,2.));
  return 3.*z*(3./r - z*z*pow(r,-3));
}

double cubic_spline_rbf::Dxxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double r = sqrt(pow(x,2.)+pow(y,2.)+pow(z_in-z_0,2.));
  return 3.*y*(1./r - x*x*pow(r,-3));
}

double cubic_spline_rbf::Dxxz(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double z = z_in-z_0;
  double r = sqrt(pow(x,2.)+pow(y_in-y_0,2.)+pow(z,2.));
  return 3.*z*(1./r - x*x*pow(r,-3));
}

double cubic_spline_rbf::Dyyz(double x_in , double y_in, double z_in)
{
  double y = y_in-y_0;
  double z = z_in-z_0;
  double r = sqrt(pow(x_in-x_0,2.)+pow(y,2.)+pow(z,2.));
  return 3.*z*(1./r - y*y*pow(r,-3));
}

double cubic_spline_rbf::Dxyy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double r = sqrt(pow(x,2.)+pow(y,2.)+pow(z_in-z_0,2.));
  return 3.*x*(1./r - y*y*pow(r,-3));
}

double cubic_spline_rbf::Dxzz(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double z = z_in-z_0;
  double r = sqrt(pow(x,2.)+pow(y_in-y_0,2.)+pow(z,2.));
  return 3.*x*(1./r - z*z*pow(r,-3));
}

double cubic_spline_rbf::Dyzz(double x_in , double y_in, double z_in)
{
  double y = y_in-y_0;
  double z = z_in-z_0;
  double r = sqrt(pow(x_in-x_0,2.)+pow(y,2.)+pow(z,2.));
  return 3.*y*(1./r - z*z*pow(r,-3));
}

double cubic_spline_rbf::Dxyz(double x_in , double y_in, double z_in)
{

  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double pseudo_r = x*x+y*y+z*z;
  return -3.*x*y*z*pow(pseudo_r,-1.5);
}

double cubic_spline_rbf::Dx(coordinate x_in)
{

  return this->Dx(x_in.x,x_in.y,x_in.z);
}

double cubic_spline_rbf::Dy(coordinate x_in)
{

  return this->Dy(x_in.x,x_in.y,x_in.z); 
}

double cubic_spline_rbf::Dz(coordinate x_in)
{

  return this->Dz(x_in.x,x_in.y,x_in.z); 
}

double cubic_spline_rbf::Dxx(coordinate x_in)
{

  return this->Dxx(x_in.x,x_in.y,x_in.z);
}

double cubic_spline_rbf::Dyy(coordinate x_in)
{

  return this->Dyy(x_in.x,x_in.y,x_in.z);
}
double cubic_spline_rbf::Dzz(coordinate x_in)
{

  return this->Dzz(x_in.x,x_in.y,x_in.z);
}

double cubic_spline_rbf::Dxy(coordinate x_in)
{

  return this->Dxy(x_in.x,x_in.y,x_in.z);
}
double cubic_spline_rbf::Dxz(coordinate x_in)
{

  return this->Dxz(x_in.x,x_in.y,x_in.z);
}
double cubic_spline_rbf::Dyz(coordinate x_in)
{

  return this->Dyz(x_in.x,x_in.y,x_in.z);
}
double cubic_spline_rbf::Dxxx(coordinate x_in)
{

  return this->Dxxx(x_in.x,x_in.y,x_in.z);
}

double cubic_spline_rbf::Dyyy(coordinate x_in)
{

  return this->Dyyy(x_in.x,x_in.y,x_in.z);
}
double cubic_spline_rbf::Dzzz(coordinate x_in)
{

  return this->Dzzz(x_in.x,x_in.y,x_in.z);
}

double cubic_spline_rbf::Dxxy(coordinate x_in)
{

  return this->Dxxy(x_in.x,x_in.y,x_in.z);
}
double cubic_spline_rbf::Dxxz(coordinate x_in)
{

  return this->Dxxz(x_in.x,x_in.y,x_in.z);
}
double cubic_spline_rbf::Dyyz(coordinate x_in)
{

  return this->Dyyz(x_in.x,x_in.y,x_in.z);
}

double cubic_spline_rbf::Dxyy(coordinate x_in)
{

  return this->Dxyy(x_in.x,x_in.y,x_in.z);
}
double cubic_spline_rbf::Dxzz(coordinate x_in)
{

  return this->Dxzz(x_in.x,x_in.y,x_in.z);
}
double cubic_spline_rbf::Dyzz(coordinate x_in)
{

  return this->Dyzz(x_in.x,x_in.y,x_in.z);
}
double cubic_spline_rbf::Dxyz(coordinate x_in)
{

  return this->Dxyz(x_in.x,x_in.y,x_in.z);
}

