/*** /cuda/cuRBFs.h
     This is the template for the acutal RBFs
     which are implemented as device functions. 
     This header implements GA as default the default
     template for device RBFs.

Julian Merten
INAF OA Bologna
Jul 2018
julian.merten@oabo.inaf.it
http://www.julianmerten.net
***/

#ifndef    CUDA_RBF_H
#define    CUDA_RBF_H

#include <mfree/cuda/cuRBFs_quadric.h>

/**
   Default template function for a device RBF. The first template
   parameter is the RBF type. This header implements
   
   template parameter t
   'g': Gaussian
   
   The second template parameter is a more specific order parameter 
   for previously selected RBFs. Implemented here are
   
   Gaussian:
   template parameter o
   '0.': Standard Gaussian (GA)
     
   The third template parameter queries the requested derivative.
   Implemented here are:
   
   template parameter d
   '0': Function itself
   '1': d/dx
   '2': d/dy
   '3': d^2/dxx
   '4': d^2/dyy
   '5': d^2/dxy
   '6': d^2/dxx+d^2/dyy
   '7': d^2/dxx-d^2/dyy
**/

/*
  The default template, implements a GA.
*/

template<char t, double o, int d> static __device__  double rbf(double x, double y, double shape_squared)
{
  return exp(-r_squared*shape_squared);
}

//x derivative as specialisation.

template<> __device__ double rbf<'g',0.,1>(double x, double y, double shape_squared)
{
  return -2.*shape_squared*x*exp(-shape_squared*r_squared);
}

//y derivative as specialisation.

template<> __device__ double rbf<'g',0.,2>(double x, double y, double shape_squared)
{
  return -2.*shape_squared*y*exp(-shape_squared*r_squared);
}

//xx derivative as specialisation.

template<> __device__ double rbf<'g',0.,3>(double x, double y, double shape_squared)
{
  return 2.*shape_squared*(2.*shape_squared*x*x-1.)*exp(-shape_squared*r_squared);
}

//yy derivative as specialisation.

template<> __device__ double rbf<'g',0.,4>(double x, double y, double shape_squared)
{
  return 2.*shape_squared*(2.*shape_squared*y*y-1.)*exp(-shape_squared*r_squared);
}

//xy derivative as specialisation.

template<> __device__ double rbf<'g',0.,5>(double x, double y, double shape_squared)
{
  return 4.*shape_squared*x*y*exp(-shape_squared*r_squared);
}

//Laplacian as specialisation

template<> __device__ double rbf<'g',0.,6>(double x, double y, double shape_squared)
{
  return 2.*shape_squared*exp(-shape_squared*r_squared)*((2.*shape_squared*x*x-1.)+(2.*shape_squared*y*y-1.));
}

//Negative Laplacian as specialisation

template<> __device__ double rbf<'g',0.,7>(double x, double y, double shape_squared)
{
  return 2.*shape_squared*exp(-shape_squared*r_squared)*((2.*shape_squared*x*x-1.)-(2.*shape_squared*y*y-1.));
}


#endif /* CUDA_RBF_H */
