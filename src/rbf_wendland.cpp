/*** rbf_wendland.cpp
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

#include <mfree/rbf_wendland.h>

void wendland_C0_1D::set_epsilon(double epsilon_in)
{
  epsilon = epsilon_in;
  E1 = 3.*epsilon;
  E2 = -E1;
  E3 = -epsilon;
}

double wendland_C0_1D::operator() (double radius)
{
  double aux = 1.+E3*radius;
  if(aux > 0.)
    {
      return aux;
    }
  else
    {
      return 0.;
    }
}

double wendland_C0_1D::Dx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1.+E3*radius;

  if(aux > 0.)
    {
      return E3*x/radius;
    }
  else
    {
      return 0.;
    }
}

double wendland_C0_1D::Dxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1.+E3*radius;

  if(aux > 0.)
    {
      return E3*(y*y+z*z)/(radius*radius*radius);
    }
  else
    {
      return 0.;
    }
}

double wendland_C0_1D::Dxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1.+E3*radius;

  if(aux > 0.)
    {
      return epsilon*x*y/(radius*radius*radius);
    }
  else
    {
      return 0.;
    }
}

double wendland_C0_1D::Dxxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1.+E3*radius;

  if(aux > 0.)
    {
      double aux2 = radius*radius;
      return E1*x*(y*y+z*z)/(aux2*aux2*radius);
    }
  else
    {
      return 0.;
    }
}

double wendland_C0_1D::Dxxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1.+E3*radius;

  if(aux > 0.)
    {
      double aux2 = radius*radius;
      return epsilon*y*(-2.*x*x+y*y+z*z)/(aux2*aux2*radius);
    }
  else
    {
      return 0.;
    }
}

double wendland_C0_1D::Dxyz(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1.+E3*radius;

  if(aux > 0.)
    {
      double aux2 = radius*radius;
      return E2*x*y*z/(aux2*aux2*radius);
    }
  else
    {
      return 0.;
    }
}

void wendland_C2_1D::set_epsilon(double epsilon_in)
{
  epsilon = epsilon_in;
  E1 = 3.*epsilon;
  E2 = 12.*epsilon*epsilon;
  E3 = 24.*epsilon*epsilon*epsilon;

}

double wendland_C2_1D::operator() (double radius)
{
  double native = epsilon*radius;
  double aux = 1. -native;
  if(aux > 0.)
    {
      return aux*aux*aux*(3.*native+1.);
    }
  else
    {
      return 0.;
    }
}

double wendland_C2_1D::Dx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = epsilon*radius - 1.;
      return -E2*x*aux2*aux2;
    }
  else
    {
      return 0.;
    }
}

double wendland_C2_1D::Dxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux1 = epsilon*radius;
      double aux2 = aux1 - 1.;
      return -E2/(radius*radius)*aux2*((y*y+z*z)*aux2+x*x*(3.*aux1-1.));
    }
  else
    {
      return 0.;
    }
}

double wendland_C2_1D::Dxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      return E3*x*y*(1./radius-epsilon);
    }
  else
    {
      return 0.;
    }
}

double wendland_C2_1D::Dxxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = radius*radius;
      aux2 *= aux2;
      return E3*x*(-E1*aux2+radius*(2.*x*x+3.*(y*y+z*z)))/aux2;
    }
  else
    {
      return 0.;
    }
}

double wendland_C2_1D::Dxxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = radius*radius*radius;
      return E3*y*(y*y+z*z-epsilon*aux2)/aux2;
    }
  else
    {
      return 0.;
    }
}

double wendland_C2_1D::Dxyz(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = aux*aux*aux;
      return -E3*x*y*z/aux2;
    }
  else
    {
      return 0.;
    }
}

void wendland_C4_1D::set_epsilon(double epsilon_in)
{
  epsilon = epsilon_in;
  E1 = 3.*epsilon;
  E2 = 14.*epsilon*epsilon;
  E3 = 4.*epsilon*epsilon;
  E4 = 280.*epsilon*epsilon*epsilon*epsilon;

}

double wendland_C4_1D::operator() (double radius)
{
  double native = epsilon*radius;
  double aux = 1. -native;
  double aux2 = aux*aux;
  if(aux > 0.)
    {
      return aux2*aux2*aux*(8.*native*native+5.*native+1.);
    }
  else
    {
      return 0.;
    }
}

double wendland_C4_1D::Dx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = epsilon*radius - 1.;
      return -E2*x*aux2*aux2*aux2*aux2*(radius+4.*epsilon*radius*radius)/radius;
    }
  else
    {
      return 0.;
    }
}

double wendland_C4_1D::Dxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux1 = epsilon*radius;
      double aux2 = aux1 - 1.;
      return -E2/radius*aux2*aux2*aux2*(-radius-E1*radius*radius+4.*epsilon*epsilon*radius*(6.*x*x+y*y+z*z));
    }
  else
    {
      return 0.;
    }
}

double wendland_C4_1D::Dxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = epsilon*radius-1.;
      return -E4*x*y*aux2*aux2*aux2;
    }
  else
    {
      return 0.;
    }
}

double wendland_C4_1D::Dxxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = epsilon*radius-1.;
      return 3.*E4*x*aux2*aux2*(epsilon*(2.*x*x+y*y+z*z)-radius)/radius;
    }
  else
    {
      return 0.;
    }
}

double wendland_C4_1D::Dxxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = epsilon*radius-1.;
      return E4*y*aux2*aux2*(epsilon*(4.*x*x+y*y+z*z)-radius)/radius;
    }
  else
    {
      return 0.;
    }
}

double wendland_C4_1D::Dxyz(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = epsilon*radius-1.;
      return 3.*E4*epsilon*x*y*z*aux2*aux2/radius;
    }
  else
    {
      return 0.;
    }
}

double wendland_C0::operator() (double radius)
{
  double aux = 1.-epsilon*radius;
  if(aux > 0.)
    {
      return aux*aux;
    }
  else
    {
      return 0.;
    }
}

double wendland_C0::Dx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1.-epsilon*radius;

  if(aux > 0.)
    {
      return 2.*epsilon*x*(epsilon-1./radius);
    }
  else
    {
      return 0.;
    }
}

double wendland_C0::Dxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1.-epsilon*radius;

  if(aux > 0.)
    {
      return 2.*epsilon*(epsilon+ (-y*y-z*z)/(radius*radius*radius));
    }
  else
    {
      return 0.;
    }
}

double wendland_C0::Dxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1.-epsilon*radius;

  if(aux > 0.)
    {
      return 2.*epsilon*x*y/(radius*radius*radius);
    }
  else
    {
      return 0.;
    }
}

double wendland_C0::Dxxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1.-epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = radius*radius;
      return 6.*epsilon*x*(y*y+z*z)/(aux2*aux2*radius);
    }
  else
    {
      return 0.;
    }
}

double wendland_C0::Dxxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1.-epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = radius*radius;
      return 2.*epsilon*y*(-2.*x*x+y*y+z*z)/(aux2*aux2*radius);
    }
  else
    {
      return 0.;
    }
}

double wendland_C0::Dxyz(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1.-epsilon*radius;

  if(aux > 0.)
    {      
      double aux2 = radius*radius;
      return 6.*epsilon*x*y*z/(aux2*aux2*radius);
    }
  else
    {
      return 0.;
    }
}

void wendland_C2::set_epsilon(double epsilon_in)
{
  epsilon = epsilon_in;
  E1 = 4.*epsilon;
  E2 = 5.*E1*epsilon;
  E3 = 3.*E2*epsilon;
}

double wendland_C2::operator() (double radius)
{
  double native = epsilon*radius;
  double aux = 1. -native;
  double aux2 = aux*aux;
  if(aux > 0.)
    {
      return aux2*aux2*(4.*native+1.);
    }
  else
    {
      return 0.;
    }
}

double wendland_C2::Dx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = epsilon*radius - 1.;
      return E2*x*aux2*aux2*aux2;
    }
  else
    {
      return 0.;
    }
}

double wendland_C2::Dxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = epsilon*radius -1.;
      return E2/(radius*radius)*aux2*aux2*((y*y+z*z)*aux2+x*x*(-1.+E1*radius));
    }
  else
    {
      return 0.;
    }
}

double wendland_C2::Dxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = epsilon*radius -1.;
      return E3*x*y*aux2*aux2/(radius*radius);
    }
  else
    {
      return 0.;
    }
}

double wendland_C2::Dxxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = radius*radius;
      double aux3 = epsilon*radius -1.;
      double aux4 = y*y+z*z;
      double aux5 = x*x;
      return E3*x/(aux2*aux2)*aux3*(-radius*(2.*aux5+3.*aux4)+epsilon*aux2*(4.*aux5+3.*aux4));
      
    }
  else
    {
      return 0.;
    }
}

double wendland_C2::Dxxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = radius*radius;
      double aux3 = epsilon*radius -1.;
      double aux4 = y*y+z*z;
      double aux5 = x*x;
      return E3*y/(aux2*aux2)*aux3*(-aux4*radius+epsilon*aux2*(2.*aux5+aux4));
    }
  else
    {
      return 0.;
    }
}

double wendland_C2::Dxyz(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = radius*radius;
      double aux3 = epsilon*radius -1.;
      return E3*x*y*z*aux3/(aux2*radius);
    }
  else
    {
      return 0.;
    }
}

void wendland_C4::set_epsilon(double epsilon_in)
{
  epsilon = epsilon_in;
  E1 = 5.*epsilon;
  E2 = 56.*epsilon*epsilon;
  E3= 30.*E2*epsilon*epsilon;
}

double wendland_C4::operator() (double radius)
{
  double native = epsilon*radius;
  double aux = 1. -native;
  double aux2 = aux*aux*aux;
  aux2 *= aux2;
  if(aux > 0.)
    {
      return aux2*(35.*native*native+18.*native+3.);
    }
  else
    {
      return 0.;
    }
}

double wendland_C4::Dx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = epsilon*radius - 1.;
      double aux3 = aux2*aux2;
      return E2*x*aux3*aux3*aux2*(1.+E1*radius);
    }
  else
    {
      return 0.;
    }
}

double wendland_C4::Dxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = epsilon*radius -1.;
      aux2 *= aux2;
      return E2*aux2*(-1.-4*epsilon*radius+E1*epsilon*(7.*x*x+y*y+z*z));
    }
  else
    {
      return 0.;
    }
}

double wendland_C4::Dxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = epsilon*radius -1.;
      aux2 *= aux2;
      aux2 *= aux2;
      return E3*x*y*aux2;
    }
  else
    {
      return 0.;
    }
}

double wendland_C4::Dxxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = radius*radius;
      double aux3 = epsilon*radius -1.;
      double aux4 = y*y+z*z;
      double aux5 = x*x;
      return 4.*epsilon*E3*x*y*aux3*aux3*(3.*aux4*aux3+aux5*(-2.+5.*epsilon*radius))/(radius*radius*radius);  
    }
  else
    {
      return 0.;
    }
}

double wendland_C4::Dxxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = radius*radius;
      double aux3 = epsilon*radius -1.;
      return E3*y*aux3*aux3*aux3*(-radius+epsilon*(5.*x*x+y*y+z*z))/radius;
    }
  else
    {
      return 0.;
    }
}

double wendland_C4::Dxyz(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = radius*radius;
      double aux3 = epsilon*radius -1.;
      return 4.*epsilon*E3*x*y*z*aux3*aux3*aux3/radius;
    }
  else
    {
      return 0.;
    }
}

void wendland_C0_5D::set_epsilon(double epsilon_in)
{
  epsilon = epsilon_in;
  E1 = 3.*epsilon;
  E2 = -E1;
  E3 = epsilon*epsilon;
}

double wendland_C0_5D::operator() (double radius)
{
  double aux = 1.-epsilon*radius;
  if(aux > 0.)
    {
      return aux*aux*aux;
    }
  else
    {
      return 0.;
    }
}

double wendland_C0_5D::Dx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1.-epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = epsilon*radius-1.;
      return E2*x*aux2*aux2/radius;
    }
  else
    {
      return 0.;
    }
}

double wendland_C0_5D::Dxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1.-epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = epsilon*radius-1.;
      return E2*aux2/(radius*radius*radius)*(2.*epsilon*x*x*radius+y*y*aux2+z*z*aux2);
    }
  else
    {
      return 0.;
    }
}

double wendland_C0_5D::Dxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1.-epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = E3*radius*radius-1.;
      return E2*x*y*aux2/(radius*radius*radius);
    }
  else
    {
      return 0.;
    }
}

double wendland_C0_5D::Dxxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1.-epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = E3*radius*radius-1.;
      double aux3 = x*x;
      double aux4 = y*y+z*z;
      double aux5 = radius*radius;
      return E1*x/(aux5*aux5*radius)*(-3.*aux4+E3*aux5*(2.*aux3+3.*aux4));
    }
  else
    {
      return 0.;
    }
}

double wendland_C0_5D::Dxxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1.-epsilon*radius;

  if(aux > 0.)
    {
      double aux4 = y*y+z*z;
      double aux5 = radius*radius;
      return E2*y/(aux5*aux5*radius)*(aux4*(-1.+E3*aux4)+x*x*(2.*E3*aux4));
    }
  else
    {
      return 0.;
    }
}

double wendland_C0_5D::Dxyz(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1.-epsilon*radius;

  if(aux > 0.)
    {
      double aux5 = radius*radius;
      return E1*x*y*z/(aux5*aux5*radius)*(-3.+E3*aux5);
    }
  else
    {
      return 0.;
    }
}

void wendland_C2_5D::set_epsilon(double epsilon_in)
{
  epsilon = epsilon_in;
  E1 = 5.*epsilon;
  E2 = 30.*epsilon*epsilon;
  E3 = 120.*epsilon*epsilon*epsilon;

}

double wendland_C2_5D::operator() (double radius)
{
  double native = epsilon*radius;
  double aux = 1. -native;
  double aux2 = aux*aux;
  if(aux > 0.)
    {
      return aux2*aux2*aux*(5.*native+1.);
    }
  else
    {
      return 0.;
    }
}

double wendland_C2_5D::Dx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = epsilon*radius - 1.;
      double aux3 = aux2*aux2;
      return -E2*x*aux3*aux3;
    }
  else
    {
      return 0.;
    }
}

double wendland_C2_5D::Dxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux1 = epsilon*radius;
      double aux2 = aux1 - 1.;
      return -E2/(radius*radius)*aux2*aux2*aux2*((y*y+z*z)*aux2+x*x*(5.*aux1-1.));
    }
  else
    {
      return 0.;
    }
}

double wendland_C2_5D::Dxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = epsilon*radius-1.;
      return E3*x*y*aux2*aux2*aux2/radius;
    }
  else
    {
      return 0.;
    }
}

double wendland_C2_5D::Dxxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = radius*radius;
      aux2 *= aux2;
      double aux3 = epsilon*radius-1.;
      double aux4 = x*x;
      double aux5 = y*y+z*z;
      return -E3*x*aux3*aux3/aux2*(-radius*(2.*aux4+3.*aux5)+epsilon*aux2*(5.*aux4+3.*aux5));
    }
  else
    {
      return 0.;
    }
}

double wendland_C2_5D::Dxxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = radius*radius;
      aux2 *= aux2;
      double aux3 = epsilon*radius-1.;
      double aux4 = x*x;
      double aux5 = y*y+z*z;
      return -E3*y*aux3*aux3/aux2*(-aux5*radius+epsilon*aux2*(3.*aux4+aux5));
    }
  else
    {
      return 0.;
    }
}

double wendland_C2_5D::Dxyz(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = aux*aux*aux;
      double aux3 = epsilon*epsilon;
      return E3*x*y*z*(-E3/60.-1./(radius*radius*radius)+E2/(10.*radius));
    }
  else
    {
      return 0.;
    }
}

void wendland_C4_5D::set_epsilon(double epsilon_in)
{
  epsilon = epsilon_in;
  E1 = 5.*epsilon;
  E2 = 6.*epsilon*epsilon;
  E3 = 24.*epsilon*epsilon;
  E4 = 1008.*epsilon*epsilon*epsilon*epsilon;

}

double wendland_C4_5D::operator() (double radius)
{
  double native = epsilon*radius;
  double aux = 1. -native;
  double aux2 = aux*aux*aux;
  if(aux > 0.)
    {
      return aux2*aux2*aux*(16.*native*native+7.*native+1.);
    }
  else
    {
      return 0.;
    }
}

double wendland_C4_5D::Dx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = epsilon*radius - 1.;
      double aux3 = aux2*aux2*aux2;
      return -E3*x*aux3*aux3*(radius+6.*epsilon*radius*radius)/radius;
    }
  else
    {
      return 0.;
    }
}

double wendland_C4_5D::Dxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux1 = epsilon*radius;
      double aux2 = aux1 - 1.;
      double aux3 = aux2*aux2;
      return -E2/radius*aux3*aux3*aux2*(-radius-E1*radius*radius+E2*radius*(8.*x*x+y*y+z*z));
    }
  else
    {
      return 0.;
    }
}

double wendland_C4_5D::Dxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = epsilon*radius-1.;
      double aux3 = aux2*aux2;
      return -E4*x*y*aux3*aux3*aux2;
    }
  else
    {
      return 0.;
    }
}

double wendland_C4_5D::Dxxx(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = epsilon*radius-1.;
      aux2 *= aux2;
      aux2 *= aux2;
      return -E4*x*aux2*(8.*epsilon*x*x+3.*epsilon*(y*y+z*z)-3.*radius)/radius;
    }
  else
    {
      return 0.;
    }
}

double wendland_C4_5D::Dxxy(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = epsilon*radius-1.;
      aux2 *= aux2;
      aux2 *= aux2;
      return -E4*y*aux2*(epsilon*(6.*x*x+y*y+z*z)-radius)/radius;
    }
  else
    {
      return 0.;
    }
}

double wendland_C4_5D::Dxyz(double x_in , double y_in, double z_in)
{
  double x = x_in-x_0;
  double y = y_in-y_0;
  double z = z_in-z_0;
  double radius = sqrt(x*x+y*y+z*z);
  double aux = 1. -epsilon*radius;

  if(aux > 0.)
    {
      double aux2 = epsilon*radius-1.;
      aux2 *= aux2;
      aux2 *= aux2;
      return -E1*E4*x*y*z*aux2/radius;
    }
  else
    {
      return 0.;
    }
}



