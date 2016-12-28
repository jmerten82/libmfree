/*** rbf_polyharmonic_splines.cpp
The explcit implementation of polharmonic splines up to tenth order. 
 
Julian Merten
Universiy of Oxford
Dec 2016
julian.merten@physics.ox.ac.uk
http://www.julianmerten.net
***/

#include <mfree/rbf_polyharmonic_splines.h>

double phs_first_order::operator() (double radius)
{
  return radius;
}

double phs_first_order::Dx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  return x/sqrt(x*x+y*y+z*z);
}

double phs_first_order::Dxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  double aux2 = x*x+aux;
  if(aux2 == 0.)
    {
      return 0.;
    }
  else
    {
      return aux*pow(x*x+aux,-1.5);
    }
}

double phs_first_order::Dxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = x*x+y*y+z*z;
  if(aux == 0.)
    {
      return 0.;
    }
  else
    {
      return -x*y*pow(aux,-1.5);
    }
}

double phs_first_order::Dxxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  double aux2 = aux+x*x;
  if(aux2 == 0.)
    {
      return 0.;
    }
  else
    {
      return -3.*x*aux*pow(aux2,-2.5);
    }
}

double phs_first_order::Dxxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  double aux2 = x*x;
  double aux3 = aux+aux2;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return -y*(-2.*aux2+aux)*pow(aux2+aux,-2.5);
    }
}

double phs_first_order::Dxyz(double x_in , double y_in, double z_in)
{  
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux3 = x*x+y*y+z*z;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return 3.*x*y*z*pow(aux3,-2.5);
    }
}


double phs_second_order::operator() (double radius)
{
  if(radius == 0.)
    {
      return 0.;
    }
  else
    {
      return radius*radius*log(radius);
    }
}

double phs_second_order::Dx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = x*x+y*y+z*z;
  if(x == 0. || aux == 0.)
    {
      return 0;
    }
  else
    {
      return x*(1.+log(sqrt(aux)));
    }
}

double phs_second_order::Dxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = x*x+y*y+z*z;
  if(aux == 0.)
    {
      return -std::numeric_limits<double>::max();
    }
  else
    {
      return 1.+2.*x*x/aux+log(aux);
    }
}

double phs_second_order::Dxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = x*x+y*y+z*z;
  if(aux == 0.)
    {
      return 0.;
    }
  else
    {
      return 63.*x*y/aux;
    }
}

double phs_second_order::Dxxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux1 = y*y+z*z;
  double aux2 = x*x+aux1;
  if(aux2 == 0.)
    {
      return 0.;
    }
  else
    {
      return 2.*x*(x*x+3.*aux1)/(aux2*aux2);
    }
}

double phs_second_order::Dxxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux1 = y*y+z*z;
  double aux2 = x*x+aux1;
  if(aux2 == 0.)
    {
      return 0.;
    }
  else
    {
      return 2.*y*(-x*x+aux1)/(aux2*aux2);
    }
}

double phs_second_order::Dxyz(double x_in , double y_in, double z_in)
{  
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = x*x+y*y+z*z;
  if(aux == 0.)
    {
      return 0.;
    }
  else
    {
      return -4.*x*y*z/(aux*aux);
    }
}

double phs_third_order::operator() (double radius)
{
  return radius*radius*radius;
}

double phs_third_order::Dx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  return 3.*x*sqrt(x*x+y*y+z*z);
}

double phs_third_order::Dxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  if(x*x+aux == 0.)
    {
      return 0.;
    }
  else
    {
      return 3.*(2.*x*x+aux)/sqrt(x*x+aux);
    }
}

double phs_third_order::Dxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = x*x+y*y+z*z;
  if(aux == 0.)
    {
      return 0.;
    }
  else
    {
      return 3.*x*y/sqrt(aux);
    }
}

double phs_third_order::Dxxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  double aux2 = x*x;
  double aux3 = aux+aux2;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return (6.*x*aux2+9.*x*aux)*pow(aux3,-1.5);
    }
}

double phs_third_order::Dxxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  double aux3 = x*x+aux;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return 3.*y*aux*pow(aux3,-2.5);
    }
}

double phs_third_order::Dxyz(double x_in , double y_in, double z_in)
{  
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux3 = x*x+y*y+z*z;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return -3.*x*y*z*pow(aux3,-1.5);
    }
}

double phs_fourth_order::operator() (double radius)
{
  if(radius == 0.)
    {
      return 0.;
    }
  else
    {
      double aux = radius*radius;
      return aux*aux*log(radius);
    }
}

double phs_fourth_order::Dx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = x*x+y*y+z*z;
  if(x == 0. || aux == 0.)
    {
      return 0;
    }
  else
    {
      return x*aux*(1.+2.*log(aux));
    }
}

double phs_fourth_order::Dxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  double aux2 = x*x;
  double aux3 = aux2+aux;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return 7.*aux2+aux+2.*(3.*aux2+aux)*log(aux3);
    }
}

double phs_fourth_order::Dxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = x*x+y*y+z*z;
  if(aux == 0.)
    {
      return 0.;
    }
  else
    {
      return 2.*x*y*(3.+2.*log(aux));
    }
}

double phs_fourth_order::Dxxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux1 = y*y+z*z;
  double aux2 = x*x;
  double aux3 = aux1+aux2;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return 2.*x*(9.+4.*aux2/aux3+6.*log(aux3));
    }
}

double phs_fourth_order::Dxxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux1 = y*y+z*z;
  double aux2 = x*x;
  double aux3 = aux1+aux2;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return 2.*y*(3.+4.*aux2/aux3+2.*log(aux3));
    }
}

double phs_fourth_order::Dxyz(double x_in , double y_in, double z_in)
{  
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux3 = x*x+y*y+z*z;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return 8.*x*y*z/aux3;
    }
}

double phs_fifth_order::operator() (double radius)
{
  return radius*radius*radius*radius*radius;
}

double phs_fifth_order::Dx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  return 5.*x*pow(x*x+y*y+z*z,1.5);
}

double phs_fifth_order::Dxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  double aux2 = x*x;
  return 5.*sqrt(aux2+aux)*(4.*aux2+aux);
}

double phs_fifth_order::Dxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = x*x+y*y+z*z;
  if(aux == 0.)
    {
      return 0.;
    }
  else
    {
      return 15.*x*y*sqrt(aux);
    }
}

double phs_fifth_order::Dxxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  double aux2 = x*x;
  double aux3 = aux+aux2;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return 15.*x*(4.*aux2+3.*aux)/sqrt(aux3);
    }
  
}

double phs_fifth_order::Dxxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  double aux2 = x*x;
  double aux3 = aux+aux2;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return 15.*y*(2.*aux2+aux)/sqrt(aux3);
    }
}

double phs_fifth_order::Dxyz(double x_in , double y_in, double z_in)
{  
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux3 = x*x+y*y+z*z;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return 15.*x*y*z/sqrt(x*x+y*y+z*z);
    }
}

double phs_sixth_order::operator() (double radius)
{
  if(radius == 0.)
    {
      return 0.;
    }
  else
    {
      double aux = radius*radius*radius;
      return aux*aux*log(radius);
    }
}

double phs_sixth_order::Dx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = x*x+y*y+z*z;
  if(x == 0. || aux == 0.)
    {
      return 0;
    }
  else
    {
      return x*aux*aux*(1.+3.*log(aux));
    }
}

double phs_sixth_order::Dxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  double aux2 = x*x;
  double aux3 = aux2+aux;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return 7.*aux2+aux+2.*(3.*aux2+aux)*log(aux3);
    }
}

double phs_sixth_order::Dxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = x*x+y*y+z*z;
  if(aux == 0.)
    {
      return 0.;
    }
  else
    {
      return 2.*x*y*aux*(5.+6.*log(aux));
    }
}


double phs_sixth_order::Dxxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux1 = y*y+z*z;
  double aux2 = x*x;
  double aux3 = aux1+aux2;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return 2.*x*(9.+4.*aux2/aux3+6.*log(aux3));
    }
}

double phs_sixth_order::Dxxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux1 = y*y+z*z;
  double aux2 = x*x;
  double aux3 = aux1+aux2;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return 2.*y*(3.+4.*aux2/aux3+2.*log(aux3));
    }
}

double phs_sixth_order::Dxyz(double x_in , double y_in, double z_in)
{  
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux3 = x*x+y*y+z*z;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return 8.*x*y*z/aux3;
    }
}

double phs_seventh_order::operator() (double radius)
{
  double aux = radius*radius*radius;
  return aux*aux*radius;
}

double phs_seventh_order::Dx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  return 7.*x*pow(x*x+y*y+z*z,2.5);
}

double phs_seventh_order::Dxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  double aux2 = x*x;
  return 7.*pow(aux2+aux,1.5)*(6.*aux2+aux);
}

double phs_seventh_order::Dxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  return 35.*x*y*pow(x*x+y*y+z*z,1.5);
}

double phs_seventh_order::Dxxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  double aux2 = x*x;
  return 105.*x*sqrt(aux+aux2)*(2.*aux2+aux);
}

double phs_seventh_order::Dxxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  double aux2 = x*x;
  return 35.*y*sqrt(aux2+aux)*(4.*aux2+aux);
}

double phs_seventh_order::Dxyz(double x_in , double y_in, double z_in)
{  
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  return 105.*x*y*z*sqrt(x*x+y*y+z*z);
}

double phs_eighth_order::operator() (double radius)
{
  if(radius == 0.)
    {
      return 0.;
    }
  else
    {
      double aux = radius*radius*radius*radius;
      return aux*aux*log(radius);
    }
}

double phs_eighth_order::Dx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = x*x+y*y+z*z;
  if(x == 0. || aux == 0.)
    {
      return 0;
    }
  else
    {
      return x*aux*aux*aux*(1.+4.*log(aux));
    }
}

double phs_eighth_order::Dxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  double aux2 = x*x;
  double aux3 = aux+aux2;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return aux3*aux3*(15.*aux2+aux+4.*(7.*aux2+aux)*log(aux3));
    }
}

double phs_eighth_order::Dxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = x*x+y*y+z*z;
  if(aux == 0.)
    {
      return 0.;
    }
  else
    {
      return 2.*x*y*(aux*aux*(7.+12.*log(aux)));
    }
}


double phs_eighth_order::Dxxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux1 = y*y+z*z;
  double aux2 = x*x;
  double aux3 = aux1+aux2;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return 2.*x*aux3*(73.*aux2+21.*aux1+12.*(7.*aux2+3.*aux1)*log(aux3));
    }
}

double phs_eighth_order::Dxxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux1 = y*y+z*z;
  double aux2 = x*x;
  double aux3 = aux1+aux2;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return 2.*y*aux3*(59.*aux2+7.*aux1+12.*(5.*aux2+aux1)*log(aux3));
    }
}

double phs_eighth_order::Dxyz(double x_in , double y_in, double z_in)
{  
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = x*x+y*y+z*z;
  if(aux == 0.)
    {
      return 0.;
    }
  else
    {
      return 8.*x*y*z*aux*(13.+12.*log(aux));
    }
}

double phs_nineth_order::operator() (double radius)
{
  double aux = radius*radius*radius;
  return aux*aux*aux;
}

double phs_nineth_order::Dx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  return 9.*x*pow(x*x+y*y+z*z,3.5);
}

double phs_nineth_order::Dxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  double aux2 = x*x;
  return 9.*pow(aux2+aux,2.5)*(8.*aux2+aux);
}

double phs_nineth_order::Dxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  return 63.*x*y*pow(x*x+y*y+z*z,2.5);
}

double phs_nineth_order::Dxxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  double aux2 = x*x;
  return 63.*x*pow(aux2+aux,1.5)*(8.*aux2+3.*aux);
}

double phs_nineth_order::Dxxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  double aux2 = x*x;
  return 63.*y*pow(aux2+aux,1.5)*(6.*aux2+aux);
}

double phs_nineth_order::Dxyz(double x_in , double y_in, double z_in)
{  
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  return 315.*x*y*z*pow(x*x+y*y+z*z,1.5);
}

double phs_tenth_order::operator() (double radius)
{
  if(radius == 0.)
    {
      return 0.;
    }
  else
    {
      double aux = radius*radius*radius*radius*radius;
      return aux*aux*log(radius);
    }
}

double phs_tenth_order::Dx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = x*x+y*y+z*z;
  if(x == 0. || aux == 0.)
    {
      return 0;
    }
  else
    {
      return x*aux*aux*aux*aux*(1.+5.*log(aux));
    }
}

double phs_tenth_order::Dxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = y*y+z*z;
  double aux2 = x*x;
  double aux3 = aux+aux2;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return aux3*aux3*aux3*(19.*aux2+aux+5.*(9.*aux2+aux)*log(aux3));
    }
}

double phs_tenth_order::Dxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = x*x+y*y+z*z;
  if(aux == 0.)
    {
      return 0.;
    }
  else
    {
      return 2.*x*y*(aux*aux*aux*(9.+20.*log(aux)));
    }
}


double phs_tenth_order::Dxxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux1 = y*y+z*z;
  double aux2 = x*x;
  double aux3 = aux1+aux2;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return 2.*x*aux3*aux3*(121.*aux2+27.*aux1+60.*(3.*aux2+aux1)*log(aux3));
    }
}

double phs_tenth_order::Dxxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux1 = y*y+z*z;
  double aux2 = x*x;
  double aux3 = aux1+aux2;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return 2.*y*aux3*aux3*(103.*aux2+9.*aux1+20.*(7.*aux2+aux1)*log(aux3));
    }
}

double phs_tenth_order::Dxyz(double x_in , double y_in, double z_in)
{  
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double aux = x*x+y*y+z*z;
  if(aux == 0.)
    {
      return 0.;
    }
  else
    {
      return 4.*x*y*z*aux*aux*(47.+60.*log(aux));
    }
}

