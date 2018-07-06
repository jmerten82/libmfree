/*** /cuda/cuRBFs_quadric.h
     This implements quadric radial basis functions.
     See cuRBFs.h for the general template.
     

Julian Merten
INAF OA Bologna
Jul 2018
julian.merten@oabo.inaf.it
http://www.julianmerten.net
***/

#ifndef    CUDA_RBFQ_H
#define    CUDA_RBFQ_H

#include <mfree/cuda/cuRBFs.h>

/**
   Quadric RBFs implemented as templates.
   
   template parameter t
   'q': Quadric
   
   The second template parameter is a more specific order parameter 
   for previously selected RBFs. Implemented here are
   
   Quadric:
   template parameter o
   '12': Multi-Quadric (MQ)
   '-12': Inverse-Multi-Quadric (IMQ)
   '-1': Inverse-Quadric (IQ)
     
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

template<char t, int o, int d> static __device__  double rbf(double x, double y, double shape_squared);

/*
  The MQ implementations.
*/

template<> __device__ double rbf<'q',12,0>(double x, double y, double shape_squared)
{
  return sqrt(1.+(x*x+y*y)*shape_squared);
}

template<> __device__ double rbf<'q',12,1>(double x, double y, double shape_squared)
{
  return shape_squared*x*pow(1.+(x*x+y*y)*shape_squared,-.5);
}

template<> __device__ double rbf<'q',12,2>(double x, double y, double shape_squared)
{
  return shape_squared*y*pow(1.+(x*x+y*y)*shape_squared,-.5);
}

template<> __device__ double rbf<'q',12,3>(double x, double y, double shape_squared)
{
  double aux = y*y;
  return shape_squared*(1+shape_squared*aux)*pow(1.+shape_squared*(x*x+aux),-1.5);
}

template<> __device__ double rbf<'q',12,4>(double x, double y, double shape_squared)
{
  double aux = x*x;
  return shape_squared*(1+shape_squared*aux)*pow(1.+shape_squared*(x*x+aux),-1.5);
}

template<> __device__ double rbf<'q',12,5>(double x, double y, double shape_squared)
{
  return shape_squared*shape_squared*pow(1.+shape_squared*(x*x+y*y),-1.5);
}

template<> __device__ double rbf<'q',12,6>(double x, double y, double shape_squared)
{
  double aux1 = x*x;
  double aux2 = y*y;
  return shape_squared*pow(1.+shape_squared*(aux1+aux2),-1.5)*(2.+shape_squared*aux1 +shape_squared*aux2);
}

template<> __device__ double rbf<'q',12,7>(double x, double y, double shape_squared)
{
  double aux1 = x*x;
  double aux2 = y*y;
  return shape_squared*pow(1.+shape_squared*(aux1+aux2),-1.5)*(shape_squared*aux1 - shape_squared*aux2);
}

/*
  IMQ implementations
*/

template<> __device__ double rbf<'q',-12,0>(double x, double y, double shape_squared)
{
  return pow(1.+shape_squared*(x*x+y*y),-0.5);
}

template<> __device__ double rbf<'q',-12,1>(double x, double y, double shape_squared)
{
  return shape_squared*x*pow(1.+shape_squared*(x*x+y*y),-1.5);
}

template<> __device__ double rbf<'q',-12,2>(double x, double y, double shape_squared)
{
  return shape_squared*y*pow(1.+shape_squared*(x*x+y*y),-1.5);
}

template<> __device__ double rbf<'q',-12,3>(double x, double y, double shape_squared)
{
  double aux1 = x*x;
  double aux2 = y*y;
  return shape_squared*(-1.+shape_squared*(2.*aux1-aux2))*pow(1.+shape_squared*(aux1+aux2),-2.5);
}

template<> __device__ double rbf<'q',-12,4>(double x, double y, double shape_squared)
{
  double aux2 = x*x;
  double aux1 = y*y;
  return shape_squared*(-1.+shape_squared*(2.*aux1-aux2))*pow(1.+shape_squared*(aux1+aux2),-2.5);
}

template<> __device__ double rbf<'q',-12,5>(double x, double y, double shape_squared)
{
  return shape_squared*shape_squared*x*y*pow(1.+shape_squared*(x*x+y*y),-1.5);
}

template<> __device__ double rbf<'q',-12,6>(double x, double y, double shape_squared)
{
  double aux1 = x*x;
  double aux2 = y*y;
  return shape_squared*pow(1.+shape_squared*(aux1+aux2),-2.5)*(-2.+shape_squared*((2.*aux1-aux2)+(2.*aux2-aux1)));
}

template<> __device__ double rbf<'q',-12,7>(double x, double y, double shape_squared)
{
  double aux1 = x*x;
  double aux2 = y*y;
  return shape_squared*pow(1.+shape_squared*(aux1+aux2),-2.5)*(-2.+shape_squared*((2.*aux1-aux2)-(2.*aux2-aux1)));
}

/*
  IQ implementations
*/

template<> __device__ double rbf<'q',-1,0>(double x, double y, double shape_squared)
{
  return 1./(1.+shape_squared*(x*x+y*y));
}

template<> __device__ double rbf<'q',-1,1>(double x, double y, double shape_squared)
{
  double aux = 1.+shape_squared*(x*x+y*y);
  return 2.*shape_squared*x/(aux*aux);
}

template<> __device__ double rbf<'q',-1,2>(double x, double y, double shape_squared)
{
  double aux = 1.+shape_squared*(x*x+y*y);
  return 2.*shape_squared*y/(aux*aux);
}

template<> __device__ double rbf<'q',-1,3>(double x, double y, double shape_squared)
{
  double aux = 1.+shape_squared*(x*x+y*y);
  return shape_squared*(-2.+shape_squared*(6.*x*x-2.*y*y))/(aux*aux*aux);
}

template<> __device__ double rbf<'q',-1,4>(double x, double y, double shape_squared)
{
  double aux = 1.+shape_squared*(x*x+y*y);
  return shape_squared*(-2.+shape_squared*(6.*y*y-2.*x*x))/(aux*aux*aux);
}

template<> __device__ double rbf<'q',-1,5>(double x, double y, double shape_squared)
{
  double aux = 1.+shape_squared*(x*x+y*y);
  return 8.*shape_squared*shape_squared*x*y/(aux*aux*aux);
}

template<> __device__ double rbf<'q',-1,6>(double x, double y, double shape_squared)
{
  double aux = 1.+shape_squared*(x*x+y*y);
  return shape_squared*(-4.+shape_squared*((6.*x*x-2.*y*y)+(6.*y*y-2.*x*x)))/(aux*aux*aux);
}

template<> __device__ double rbf<'q',-1,7>(double x, double y, double shape_squared)
{
  double aux = 1.+shape_squared*(x*x+y*y);
  return shape_squared*(shape_squared*((6.*x*x-2.*y*y)-(6.*y*y-2.*x*x)))/(aux*aux*aux);
}




#endif /* CUDA_RBFQ_H */
