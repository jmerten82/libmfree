/*** /cuda/cuRBFs_wendland.h
     This implements Wendland radial basis functions.
     See cuRBFs.h for the general template.
     

Julian Merten
INAF OA Bologna
Jul 2018
julian.merten@oabo.inaf.it
http://www.julianmerten.net
***/

#ifndef    CUDA_RBFW_H
#define    CUDA_RBFW_H

#include <mfree/cuda/cuRBFs.h>

/**
   Wendland C0 implementation.
**/

__device__ inline double wc0(double x, double y, double shape_squared)
{
  double aux = 1. - sqrt((x*x+y*y)*shape_squared);
  if(aux > 0.)
    {
      return aux*aux;
    }
  else
    {
      return 0.;
    }
};

__device__ inline double wc0_dx(double x, double y, double shape_squared)
{
  double r = sqrt(x*x+y*y);
  double shape = sqrt(shape_squared);
  double aux = 1.-sqrt(shape*r);
  if(aux > 0.)
    {
      return 2.*shape*x*(shape-1./r);
    }
  else
    {
      return 0.;
    }
};

__device__ inline double wc0_dxx(double x, double y, double shape_squared)
{
  double shape = sqrt(shape_squared);
  double r = sqrt(x*x+y*y);
  double aux = 1.-shape*r;
  if(aux > 0.)
    {
      return 2.*shape*(shape-y*y)/(r*r*r);
    }
  else
    {
      return 0.;
    }
};

__device__ inline double wc0_dxy(double x, double y, double shape_squared)
{
  double shape = sqrt(shape_squared);
  double r = sqrt(x*x+y*y);
  double aux = 1. - shape*r;
  if(aux > 0.)
    {
      return 2.*x*y*shape/(r*r*r);
    } 
  else
    {
      return 0.;
    }
};

__device__ inline double wc0_laplace(double x, double y, double shape_squared)
{
  double shape = sqrt(shape_squared);
  double r = sqrt(x*x+y*y);
  double aux = 1.-shape*r;
  if(aux > 0.)
    {
      return 2.*shape/(r*r*r)*(2.*shape-r*r);
    }
  else
    {
      return 0.;
    }
};

__device__ inline double wc0_neg_laplace(double x, double y, double shape_squared)
{
  double shape = sqrt(shape_squared);
  double r = sqrt(x*x+y*y);
  double aux = 1.-shape*r;
  if(aux > 0.)
    {
      return 2.*shape/(r*r*r)*(x*x-y*y);
    }
  else
    {
      return 0.;
    }
};

/**
   Wendland C2 implementation.
**/

__device__ inline double wc2(double x, double y, double shape_squared)
{
  double aux0 = sqrt((x*x+y*y)*shape_squared);
  double aux = 1. - aux0;
  if(aux > 0.)
    {
      double aux2 = aux*aux;      
      return aux2*aux2*(4.*aux0+1);
    }
  else
    {
      return 0.;
    }
};

__device__ inline double wc2_dx(double x, double y, double shape_squared)
{
  double aux0 = sqrt((x*x+y*y)*shape_squared);
  double aux = 1. - aux0;
  if(aux > 0.)
    {
      double aux2 = aux0 - 1.;
      return 20.*shape_squared*aux2*aux2*aux2;
    }
  else
    {
      return 0.;
    }
};

__device__ inline double wc2_dxx(double x, double y, double shape_squared)
{
  double r = sqrt(x*x+y*y);
  double shape = sqrt(shape_squared);
  double aux0 = r*shape;
  double aux = 1. - aux0;
  if(aux > 0.)
    {
      double aux2 = aux0 - 1.;
      return 20.*shape_squared/(r*r)*aux2*aux2*(y*y*aux2+x*x*(-1.+4.*shape*r));
    }
  else
    {
      return 0.;
    }
};

__device__ inline double wc2_dxy(double x, double y, double shape_squared)
{
  double shape = sqrt(shape_squared);
  double r = sqrt(x*x+y*y);
  double aux0 = shape*r;
  double aux = 1. - aux0;
  if(aux > 0.)
    {
      double aux2 = aux0 -1.;
      return 60.*shape_squared*shape*x*y*aux2*aux2/(r*r);
    } 
  else
    {
      return 0.;
    }
};

__device__ inline double wc2_laplace(double x, double y, double shape_squared)
{
  double r = sqrt(x*x+y*y);
  double shape = sqrt(shape_squared);
  double aux0 = r*shape;
  double aux = 1. - aux0;
  if(aux > 0.)
    {
      double aux2 = aux0 - 1.;
      return 20.*shape_squared/(r*r)*aux2*aux2*((y*y*aux2+x*x*(-1.+4.*shape*r))+(x*x*aux2+y*y*(-1.+4.*shape*r)));
    }
  else
    {
      return 0.;
    }
};

__device__ inline double wc2_neg_laplace(double x, double y, double shape_squared)
{
  double r = sqrt(x*x+y*y);
  double shape = sqrt(shape_squared);
  double aux0 = r*shape;
  double aux = 1. - aux0;
  if(aux > 0.)
    {
      double aux2 = aux0 - 1.;
      return 20.*shape_squared/(r*r)*aux2*aux2*((y*y*aux2+x*x*(-1.+4.*shape*r))-(x*x*aux2+y*y*(-1.+4.*shape*r)));
    }
  else
    {
      return 0.;
    }
};

/**
   Wendland C4 implementation.
**/

__device__ inline double wc4(double x, double y, double shape_squared)
{
  double aux0 = sqrt((x*x+y*y)*shape_squared);
  double aux = 1. - aux0;
  if(aux > 0.)
    {
      double aux2 = aux*aux*aux;      
      return aux2*aux2*(35.*aux0*aux0+18.*aux0+3.);
    }
  else
    {
      return 0.;
    }
};

__device__ inline double wc4_dx(double x, double y, double shape_squared)
{
  double sr = sqrt((x*x+y*y)*shape_squared);
  double aux = 1.-sr;
  if(aux > 0.)
    {
      double aux2 = sr - 1.;
      double aux3 = aux2*aux2;
      return 56.*shape_squared*aux3*aux3*aux2*(1.+5.*sr);
    }
  else
    {
      return 0.;
    }
};

__device__ inline double wc4_dxx(double x, double y, double shape_squared)
{
  double sr = sqrt((x*x+y*y)*shape_squared);
  double aux = 1.-sr;
  if(aux > 0.)
    {
      double aux2 = sr - 1.;
      aux2 *= aux2;
      return 20.*shape_squared*aux2*(-1.-4.*sr+4.*shape_squared*(7.*x*x+y*y));
    }
  else
    {
      return 0.;
    }
};

__device__ inline double wc4_dxy(double x, double y, double shape_squared)
{
  double sr = sqrt((x*x+y*y)*shape_squared);
  double aux = 1. - sr;
  if(aux > 0.)
    {
      double aux2 = sr -1.;
      return 1680.*shape_squared*shape_squared*x*y*aux2*aux2*aux2;
    } 
  else
    {
      return 0.;
    }
};

__device__ inline double wc4_laplace(double x, double y, double shape_squared)
{
  double sr = sqrt((x*x+y*y)*shape_squared);
  double aux = 1.-sr;
  if(aux > 0.)
    {
      double aux2 = sr - 1.;
      aux2 *= aux2;
      return 20.*shape_squared*aux2*(-2.-4.*sr+24.*shape_squared*(x*x+y*y));
    }
  else
    {
      return 0.;
    }
};

__device__ inline double wc4_neg_laplace(double x, double y, double shape_squared)
{
  double sr = sqrt((x*x+y*y)*shape_squared);
  double aux = 1.-sr;
  if(aux > 0.)
    {
      double aux2 = sr - 1.;
      aux2 *= aux2;
      return 20.*shape_squared*aux2*(-4.*sr+32.*shape_squared*(x*x-y*y));
    }
  else
    {
      return 0.;
    }
};

/**
   The following is a class-like implementaion of RBFs. 
   In this header, Wendland RBFs. We will not use inheritance, 
   since we cannot provide an object to a kernel which contains virtual
   functions.
**/


class wc0_rbf
{
 public:
  __device__ double operator() (double x, double y, double shape_squared)
  {
    return wc0(x,y,shape_squared);
  };
  __device__ double D(double x, double y, double shape_squared, int selection)
  {
    double res;
    switch(selection)
      {
      case 1: res = wc0_dx(x,y,shape_squared);
      case 2: res = wc0_dx(y,x,shape_squared);
      case 3: res = wc0_dxx(x,y,shape_squared);
      case 4: res = wc0_dxx(y,x,shape_squared);
      case 5: res = wc0_dxy(x,y,shape_squared);
      case 6: res = wc0_laplace(x,y,shape_squared);
      case 7: res = wc0_neg_laplace(x,y,shape_squared);
      }
    return res;
  };
  __device__ double dx(double x, double y, double shape_squared)
  {
    return wc0_dx(x,y,shape_squared);
  };
  __device__ double dxx(double x, double y, double shape_squared)
  {
    return wc0_dxx(x,y,shape_squared);
  };
  __device__ double dxy(double x, double y, double shape_squared)
  {
    return wc0_dxy(x,y,shape_squared);
  };
  __device__ double laplace(double x, double y, double shape_squared)
  {
    return wc0_laplace(x,y,shape_squared);
  }
  __device__ double neg_laplace(double x, double y, double shape_squared)
  {
    return wc0_neg_laplace(x,y,shape_squared);
  }
};

class wc2_rbf
{
 public:
  __device__ double operator() (double x, double y, double shape_squared)
  {
    return wc2(x,y,shape_squared);
  };
  __device__ double D(double x, double y, double shape_squared, int selection)
  {
    double res;
    switch(selection)
      {
      case 1: res = wc2_dx(x,y,shape_squared);
      case 2: res = wc2_dx(y,x,shape_squared);
      case 3: res = wc2_dxx(x,y,shape_squared);
      case 4: res = wc2_dxx(y,x,shape_squared);
      case 5: res = wc2_dxy(x,y,shape_squared);
      case 6: res = wc2_laplace(x,y,shape_squared);
      case 7: res = wc2_neg_laplace(x,y,shape_squared);
      }
    return res;
  };
  __device__ double dx(double x, double y, double shape_squared)
  {
    return wc2_dx(x,y,shape_squared);
  };
  __device__ double dxx(double x, double y, double shape_squared)
  {
    return wc2_dxx(x,y,shape_squared);
  };
  __device__ double dxy(double x, double y, double shape_squared)
  {
    return wc2_dxy(x,y,shape_squared);
  };
  __device__ double laplace(double x, double y, double shape_squared)
  {
    return wc2_laplace(x,y,shape_squared);
  }
  __device__ double neg_laplace(double x, double y, double shape_squared)
  {
    return wc2_neg_laplace(x,y,shape_squared);
  }
};

class wc4_rbf
{
 public:
  __device__ double operator() (double x, double y, double shape_squared)
  {
    return wc4(x,y,shape_squared);
  };
  __device__ double D(double x, double y, double shape_squared, int selection)
  {
    double res;
    switch(selection)
      {
      case 1: res = wc4_dx(x,y,shape_squared);
      case 2: res = wc4_dx(y,x,shape_squared);
      case 3: res = wc4_dxx(x,y,shape_squared);
      case 4: res = wc4_dxx(y,x,shape_squared);
      case 5: res = wc4_dxy(x,y,shape_squared);
      case 6: res = wc4_laplace(x,y,shape_squared);
      case 7: res = wc4_neg_laplace(x,y,shape_squared);
      }
    return res;
  };
  __device__ double dx(double x, double y, double shape_squared)
  {
    return wc4_dx(x,y,shape_squared);
  };
  __device__ double dxx(double x, double y, double shape_squared)
  {
    return wc4_dxx(x,y,shape_squared);
  };
  __device__ double dxy(double x, double y, double shape_squared)
  {
    return wc4_dxy(x,y,shape_squared);
  };
  __device__ double laplace(double x, double y, double shape_squared)
  {
    return wc4_laplace(x,y,shape_squared);
  }
  __device__ double neg_laplace(double x, double y, double shape_squared)
  {
    return wc4_neg_laplace(x,y,shape_squared);
  }
};



#endif /* CUDA_RBFW_H */
