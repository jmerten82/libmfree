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
