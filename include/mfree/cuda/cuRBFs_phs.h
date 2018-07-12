/*** /cuda/cuRBFs_phs.h
     This implements polyharmonic spline radial basis functions.
     See cuRBFs.h for the general template.
     

Julian Merten
INAF OA Bologna
Jul 2018
julian.merten@oabo.inaf.it
http://www.julianmerten.net
***/

#ifndef    CUDA_RBFPHS_H
#define    CUDA_RBFPHS_H

#include <mfree/cuda/cuRBFs.h>

/**
   Polyharmonic spline implementations. From order 1 to 10.
**/

__device__ inline double phs1(double x, double y, double shape_squared)
{
  return sqrt(x*x+y*y);
};

__device__ inline double phs1_dx(double x, double y, double shape_squared)
{
  double r2 = (x*x+y*y);
  if(r2 == 0.)
    {
      return 0.;
    }
  else
    {
      return x/sqrt(r2);
    }
};

__device__ inline double phs1_dxx(double x, double y, double shape_squared)
{
  double r2 = x*x+y*y;
  if(r2 == 0.)
    {
      return 0.;
    }
  else
    {
      return y*y*pow(r2,-1.5);
    }
};

__device__ inline double phs1_dxy(double x, double y, double shape_squared)
{
  double r2 = x*x+y*y;
  if(r2 == 0.)
    {
      return 0.;
    }
  else
    {
      return -x*y*pow(r2,-1.5);
    }
};

__device__ inline double phs1_laplace(double x, double y, double shape_squared)
{
  return 1./sqrt(x*x+y*y);
};

__device__ inline double phs1_neg_laplace(double x, double y, double shape_squared)
{
  double r2 = x*x+y*y;
  return (y*y-x*x)*pow(r2,-1.5);
};

__device__ inline double phs2(double x, double y, double shape_squared)
{
  double r2 = x*x+y*y; 
  if(r2 == 0.)
    {
      return 0.;
    }
  else
    {
      return r2*log(sqrt(r2));
    }
};

__device__ inline double phs2_dx(double x, double y, double shape_squared)
{
  if(x == 0.)
    {
      return 0;
    }
  else
    {
      double r = sqrt(x*x+y*y);
      if(r == 0.)
	{
	  return 0.;
	}
      else
	{
	  return x*(1.+log(r));
	}
    }

};

__device__ inline double phs2_dxx(double x, double y, double shape_squared)
{
  double r2 = x*x+y*y;
  if(r2 == 0.)
    {
      return 1.e308;
    }
  else
    {
      return 1.+2.*x*x/r2+log(r2);
    }
};

__device__ inline double phs2_dxy(double x, double y, double shape_squared)
{
  double r2 = x*x+y*y;
  if(r2 == 0.)
    {
      return 0.;
    }
  else
    {
      return 63.*x*y/r2;
    }
};

__device__ inline double phs2_laplace(double x, double y, double shape_squared)
{
  double r2 = x*x+y*y;
  if(r2 == 0.)
    {
      return 1.e308;
    }
  else
    {
      return 2.*(1.+x*x/r2+y*y/r2+log(r2));
    }

};

__device__ inline double phs2_neg_laplace(double x, double y, double shape_squared)
{
  double r2 = x*x+y*y;
  if(r2 == 0.)
    {
      return 1.e308;
    }
  else
    {
      return 2.*(1.+x*x/r2-y*y/r2+log(r2));
    }
};

  

__device__ inline double phs3(double x, double y, double shape_squared)
{
  return pow(x*x+y*y,1.5);

};

__device__ inline double phs3_dx(double x, double y, double shape_squared)
{
  double r = sqrt(x*x+y*y);
  return 3.*x*r;

};

__device__ inline double phs3_dxx(double x, double y, double shape_squared)
{
  double r = sqrt(x*x+y*y);
  if(r == 0.)
    {
      return 0;
    }
  else
    {
      return 3.*(2.*x*x+y*y)/r;
    }
};

__device__ inline double phs3_dxy(double x, double y, double shape_squared)
{
  double r = sqrt(x*x+y*y);

  if(r == 0.)
    {
      return 0.;
    }
  else
    {
      return 3.*x*y/r;
    }
};

__device__ inline double phs3_laplace(double x, double y, double shape_squared)
{
  double r = sqrt(x*x+y*y);
  if(r == 0.)
    {
      return 0;
    }
  else
    {
      return 6.*r;
    }
};

__device__ inline double phs3_neg_laplace(double x, double y, double shape_squared)
{
  double r = sqrt(x*x+y*y);
  if(r == 0.)
    {
      return 0;
    }
  else
    {
      return 3.*(x*x-y*y)/r;
    }
};

__device__ inline double phs4(double x, double y, double shape_squared)
{
  double r = sqrt(x*x+y*y);
  if(r == 0.)
    {
      return 0.;
    }
  else
    {
      double r2 = r*r;
      return r2*r2*log(r);
    }
};

__device__ inline double phs4_dx(double x, double y, double shape_squared)
{
  if(x == 0)
    {
      return 0;
    }
  else
    {
      double r2 = x*x+y*y;
      if(r2 == 0.)
	{
	  return 0.;
	}
      else
	{
	  return x*r2*(1.+2.*log(r2));
	}
    }
};

__device__ inline double phs4_dxx(double x, double y, double shape_squared)
{
  double r2 = x*x+y*y;
  if(r2 == 0.)
    {
      return 0.;
    }
  else
    {
      return 7.*x*x+y*y+2.*(3.*x*x+y*y)*log(r2);
    }
};

__device__ inline double phs4_dxy(double x, double y, double shape_squared)
{
  double r2 = x*x+y*y;
  if(r2 == 0.)
    {
      return 0.;
    }
  else
    {
      return 2.*x*y*(3.+2.*log(r2));
    }
};

__device__ inline double phs4_laplace(double x, double y, double shape_squared)
{
  double r2 = x*x+y*y;
  if(r2 == 0.)
    {
      return 0.;
    }
  else
    {
      return 8.*r2+8.*r2*log(r2);
    }
};

__device__ inline double phs4_neg_laplace(double x, double y, double shape_squared)
{
  double r2 = x*x+y*y;
  if(r2 == 0.)
    {
      return 0.;
    }
  else
    {
      return 6.*(x*x-y*y)+4.*(x*x-y*y)*log(r2);
    }
};

__device__ inline double phs5(double x, double y, double shape_squared)
{
  return pow(x*x+y*y,2.5);
};

__device__ inline double phs5_dx(double x, double y, double shape_squared)
{
  return 5.*x*pow(x*x+y*y,1.5);
};

__device__ inline double phs5_dxx(double x, double y, double shape_squared)
{
  double aux = y*y;
  double aux2 = x*x;
  return 5.*sqrt(aux2+aux)*(4.*aux2+aux);
};

__device__ inline double phs5_dxy(double x, double y, double shape_squared)
{
  return 15.*x*y*sqrt(x*x+y*y);
};

__device__ inline double phs5_laplace(double x, double y, double shape_squared)
{
  return 25.*pow(x*x+y*y,1.5);
};

__device__ inline double phs5_neg_laplace(double x, double y, double shape_squared)
{
  double aux = y*y;
  double aux2 = x*x;
  return 15.*sqrt(aux2+aux)*(aux2-aux);
};

__device__ inline double phs6(double x, double y, double shape_squared)
{
  double r = sqrt(x*x+y*y);
  if(r == 0.)
    {
      return 0.;
    }
  else
    {
      double aux = r*r*r;
      return aux*aux*log(r);
    }
};

__device__ inline double phs6_dx(double x, double y, double shape_squared)
{
  if(x == 0.)
    {
      return 0.;
    }
  else
    {
      double r2 = x*x+y*y;
      return x*r2*r2*log(1.+3.*r2);
    }
};

__device__ inline double phs6_dxx(double x, double y, double shape_squared)
{
  double aux = y*y;
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
};

__device__ inline double phs6_dxy(double x, double y, double shape_squared)
{
  double r2 = x*x+y*y;
  if(r2 == 0.)
    {
      return 0.;
    }
  else
    {
      return 2.*x*y*r2*(5.+6.*log(r2));
    }
};

__device__ inline double phs6_laplace(double x, double y, double shape_squared)
{
  double aux = y*y;
  double aux2 = x*x;
  double aux3 = aux2+aux;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return 8.*aux3+2.*8.*aux3*log(aux3);
    }
};

__device__ inline double phs6_neg_laplace(double x, double y, double shape_squared)
{
  double aux = y*y;
  double aux2 = x*x;
  double aux3 = aux2+aux;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return 6.*(aux2-aux)+4.*(aux2-aux)*log(aux3);
    }
};

__device__ inline double phs7(double x, double y, double shape_squared)
{
  return pow(x*x+y*y,3.5);
};

__device__ inline double phs7_dx(double x, double y, double shape_squared)
{
  return 7.*x*pow(x*x+y*y,2.5);
};

__device__ inline double phs7_dxx(double x, double y, double shape_squared)
{
  double aux = y*y;
  double aux2 = x*x;
  return 7.*pow(aux2+aux,1.5)*(6.*aux2+aux);
};

__device__ inline double phs7_dxy(double x, double y, double shape_squared)
{
  return 35.*x*y*pow(x*x+y*y,1.5);
};

__device__ inline double phs7_laplace(double x, double y, double shape_squared)
{
  return 49.*pow(x*x+y*y,3.5);
};

__device__ inline double phs7_neg_laplace(double x, double y, double shape_squared)
{
  double aux = y*y;
  double aux2 = x*x;
  return 35.*pow(aux2+aux,1.5)*(aux2-aux);
};

__device__ inline double phs8(double x, double y, double shape_squared)
{
  double r2 = x*x+y*y;
  return r2*r2*r2*r2*log(sqrt(r2));
};

__device__ inline double phs8_dx(double x, double y, double shape_squared)
{
  if(x == 0.)
    {
      return 0.;
    }
  else
    {
      double r2 = x*x+y*y;
      if(r2 == 0.)
	{
	  return 0.;
	}
      else
	{
	  return x*r2*r2*r2*(1.+4.*log(r2));
	}
    }

};

__device__ inline double phs8_dxx(double x, double y, double shape_squared)
{
  double aux = y*y;
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
};

__device__ inline double phs8_dxy(double x, double y, double shape_squared)
{
  double r2 = x*x+y*y;
  if(r2 == 0.)
    {
      return 0.;
    }
  else
    {
      return 2.*x*y*(r2*r2*(7.+12.*log(r2)));
    }
};

__device__ inline double phs8_laplace(double x, double y, double shape_squared)
{
  double aux = y*y;
  double aux2 = x*x;
  double aux3 = aux+aux2;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return aux3*aux3*aux3*(16.+32.*log(aux3));
    }
};

__device__ inline double phs8_neg_laplace(double x, double y, double shape_squared)
{
  double aux = y*y;
  double aux2 = x*x;
  double aux3 = aux+aux2;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return aux3*aux3*(14.*aux2-aux)+24.*(aux2-aux)*log(aux3);
    }
};

__device__ inline double phs9(double x, double y, double shape_squared)
{
  return pow(x*x+y*y,4.5);
};

__device__ inline double phs9_dx(double x, double y, double shape_squared)
{
  return 9.*x*pow(x*x+y*y,3.5);
};

__device__ inline double phs9_dxx(double x, double y, double shape_squared)
{
  double aux = y*y;
  double aux2 = x*x;
  return 9.*pow(aux2+aux,2.5)*(8.*aux2+aux);
};

__device__ inline double phs9_dxy(double x, double y, double shape_squared)
{
  return 63.*x*y*pow(x*x+y*y,2.5);
};

__device__ inline double phs9_laplace(double x, double y, double shape_squared)
{
  return 81.*pow(x*x+y*y,3.5);
};

__device__ inline double phs9_neg_laplace(double x, double y, double shape_squared)
{
  double aux = y*y;
  double aux2 = x*x;
  return 63.*pow(aux2+aux,2.5)*(aux2-aux);
};

__device__ inline double phs10(double x, double y, double shape_squared)
{
  double r2 = x*x+y*y;
  if(r2 == 0.)
    {
      return 0.;
    }
  else
    {
      double r = sqrt(r2);
      return r2*r2*r2*r2*r2*log(r);
    }
};

__device__ inline double phs10_dx(double x, double y, double shape_squared)
{
  if(x == 0.)
    {
      return 0;
    }
  else
    {
      double r2 = x*x+y*y;
      if(r2 == 0.)
	{
	  return 0.;
	}
      else
	{
	  return x*r2*r2*r2*r2*(1.+5.*log(r2));
	}
    }
};

__device__ inline double phs10_dxx(double x, double y, double shape_squared)
{
  double aux = y*y;
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
};

__device__ inline double phs10_dxy(double x, double y, double shape_squared)
{
  double r2 = x*x+y*y;
  if(r2 == 0.)
    {
      return 0.;
    }
  else
    {
      return 2.*x*y*(r2*r2*r2*(9.+20.*log(r2)));
    }
};

__device__ inline double phs10_laplace(double x, double y, double shape_squared)
{
  double aux = y*y;
  double aux2 = x*x;
  double aux3 = aux+aux2;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return aux3*aux3*aux3*aux3*(20.+50*log(aux3));
    }
};

__device__ inline double phs10_neg_laplace(double x, double y, double shape_squared)
{
  double aux = y*y;
  double aux2 = x*x;
  double aux3 = aux+aux2;
  if(aux3 == 0.)
    {
      return 0.;
    }
  else
    {
      return aux3*aux3*aux3*(18.*(aux2-aux)+40.*(aux2-aux)*log(aux3));
    }
};
  

#endif /* CUDA_RBFPHS_H */
