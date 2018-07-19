/*** /cuda/cuRBFs.h
     This implements radial basis functions for device execution.
     Firstly, we define the actual function and its derivative as
     inline function. Then we implement the RBF as a class and
     as templates. Currently it looks like the class implementation
     will be prefered. 

     This header implements a Gaussian RBF (GA).

Julian Merten
INAF OA Bologna
Jul 2018
julian.merten@oabo.inaf.it
http://www.julianmerten.net
***/

#ifndef    CUDA_RBF_H
#define    CUDA_RBF_H

#include <mfree/cuda/cuda_manager.h>
#include <mfree/cuda/cuRBFs_quadric.h>
#include <mfree/cuda/cuRBFs_wendland.h>
#include <mfree/cuda/cuRBFs_phs.h>

/**
   Here the very basic implementations of the functions we need, we
   will use this mainly for performance checks later.
**/

__device__ inline double ga(double x, double y, double shape_squared)
{
  return exp(-(x*x+y*y)*shape_squared);
};

__device__ inline double ga_dx(double x, double y, double shape_squared)
{
  return -2.*shape_squared*x*exp(-shape_squared*(x*x+y*y));
};

__device__ inline double ga_dxx(double x, double y, double shape_squared)
{
  double aux = x*x;
  return 2.*shape_squared*(2.*shape_squared*aux-1.)*exp(-shape_squared*(aux+y*y));
};

__device__ inline double ga_dxy(double x, double y, double shape_squared)
{
  return 4.*shape_squared*x*y*exp(-shape_squared*(x*x+y*y));
};

__device__ inline double ga_laplace(double x, double y, double shape_squared)
{
  double aux1 = x*x;
  double aux2 = y*y;
  return 2.*shape_squared*exp(-shape_squared*(aux1+aux2))*((2.*shape_squared*aux1-1.)+(2.*shape_squared*aux2-1.));
};

__device__ inline double ga_neg_laplace(double x, double y, double shape_squared)
{
  double aux1 = x*x;
  double aux2 = y*y;
  return 2.*shape_squared*exp(-shape_squared*(aux1+aux2))*((2.*shape_squared*aux1-1.)-(2.*shape_squared*aux2-1.));
};


/**
   The following is a class-like implementaion of RBFs. 
   In this header, a Gaussian RBF. We will not use inheritance, 
   since we cannot provide an object to a kernel which contains virtual
   functions.
**/

class ga_rbf
{
 public:
  __device__ double operator() (double x, double y, double shape_squared)
  {
    return ga(x,y,shape_squared);
  };
  __device__ double D(double x, double y, double shape_squared, int selection)
  {
    double res;
    switch(selection)
      {
      case 1: res = ga_dx(x,y,shape_squared);
	break;
      case 2: res = ga_dx(y,x,shape_squared);
	break;
      case 3: res = ga_dxx(x,y,shape_squared);
	break;
      case 4: res = ga_dxx(y,x,shape_squared);
	break;
      case 5: res = ga_dxy(x,y,shape_squared);
	break;
      case 6: res = ga_laplace(x,y,shape_squared);
	break;
      case 7: res = ga_neg_laplace(x,y,shape_squared);
	break;
      }
    return res;
  };
  __device__ double dx(double x, double y, double shape_squared)
  {
    return ga_dx(x,y,shape_squared);
  };
  __device__ double dxx(double x, double y, double shape_squared)
  {
    return ga_dxx(x,y,shape_squared);
  };
  __device__ double dxy(double x, double y, double shape_squared)
  {
    return ga_dxy(x,y,shape_squared);
  };
  __device__ double laplace(double x, double y, double shape_squared)
  {
    return ga_laplace(x,y,shape_squared);
  }
  __device__ double neg_laplace(double x, double y, double shape_squared)
  {
    return ga_neg_laplace(x,y,shape_squared);
  }
};

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

template<char t, int o, int d> static __device__  double rbf(double x, double y, double shape_squared)
{
  return exp(-(x*x+y*y)*shape_squared);
}

//x derivative as specialisation.

template<> __device__ double rbf<'g',0,1>(double x, double y, double shape_squared)
{
  return -2.*shape_squared*x*exp(-shape_squared*(x*x+y*y));
}

//y derivative as specialisation.

template<> __device__ double rbf<'g',0,2>(double x, double y, double shape_squared)
{
  return -2.*shape_squared*y*exp(-shape_squared*(x*x+y*y));
}

//xx derivative as specialisation.

template<> __device__ double rbf<'g',0,3>(double x, double y, double shape_squared)
{
  double aux = x*x;
  return 2.*shape_squared*(2.*shape_squared*aux-1.)*exp(-shape_squared*(aux+y*y));
}

//yy derivative as specialisation.

template<> __device__ double rbf<'g',0,4>(double x, double y, double shape_squared)
{
  double aux =y*y;
  return 2.*shape_squared*(2.*shape_squared*aux-1.)*exp(-shape_squared*(x*x+aux));
}

//xy derivative as specialisation.

template<> __device__ double rbf<'g',0,5>(double x, double y, double shape_squared)
{
  return 4.*shape_squared*x*y*exp(-shape_squared*(x*x+y*y));
}

//Laplacian as specialisation

template<> __device__ double rbf<'g',0,6>(double x, double y, double shape_squared)
{
  double aux1 = x*x;
  double aux2 = y*y;
  return 2.*shape_squared*exp(-shape_squared*(aux1+aux2))*((2.*shape_squared*aux1-1.)+(2.*shape_squared*aux2-1.));
}

//Negative Laplacian as specialisation

template<> __device__ double rbf<'g',0,7>(double x, double y, double shape_squared)
{
  double aux1 = x*x;
  double aux2 = y*y;
  return 2.*shape_squared*exp(-shape_squared*(aux1+aux2))*((2.*shape_squared*aux1-1.)-(2.*shape_squared*aux2-1.));
}


/**
   Some helper functions to fill the polynomial part of the 
   RBF linear system of equations.
**/

/*
  Given a x and y component this writes the polynomial portion of the 
  RBF findif linear system into a pre-defined array.
*/

__device__ inline void row_vector_from_polynomial(double x, double y, int col_stride, int pdeg, double *row_ptr, double *col_ptr)
{
  double x_values[MAX_PDEG];
  double y_values[MAX_PDEG];
  x_values[0] = 1.;
  y_values[0] = 1.;

  for(int i = 1; i <= pdeg; i++)
    {
      x_values[i] = x_values[i-1]*x;
      y_values[i] = y_values[i-1]*y;
    }
  row_ptr[0] = 1.;
  col_ptr[0] = 1.;
  int counter = 1;
  for(int i = 0; i < pdeg; i++)
    {
      int max_grade = i+1;
      for(int j = 0; j <= max_grade; j++)
	{
	  double value = x_values[max_grade-j]*y_values[j];
	  row_ptr[counter] = value;
	  col_ptr[counter*col_stride] = value;
	  counter++;
	}
    }
}

/*
  This is a simpler version of the above which only writes into one pointer.
*/

__device__ inline void row_vector_from_polynomial_simpler(double x, double y, int pdeg, double *row_ptr)
{
  double x_values[MAX_PDEG];
  double y_values[MAX_PDEG];
  x_values[0] = 1.;
  y_values[0] = 1.;

  for(int i = 1; i <= pdeg; i++)
    {
      x_values[i] = x_values[i-1]*x;
      y_values[i] = y_values[i-1]*y;
    }
  row_ptr[0] = 1.;
  int counter = 1;
  for(int i = 0; i < pdeg; i++)
    {
      int max_grade = i+1;
      for(int j = 0; j <= max_grade; j++)
	{
	  double value = x_values[max_grade-j]*y_values[j];
	  row_ptr[counter] = value;
	  counter++;
	}
    }
}


#endif /* CUDA_RBF_H */
