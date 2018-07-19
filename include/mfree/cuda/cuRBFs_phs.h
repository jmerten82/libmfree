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


class phs1_rbf
{
 public:
  __device__ double operator() (double x, double y, double shape_squared)
  {
    return phs1(x,y,shape_squared);
  };
  __device__ double D(double x, double y, double shape_squared, int selection)
  {
    double res;
    switch(selection)
      {
      case 1: res = phs1_dx(x,y,shape_squared);
	break;
      case 2: res = phs1_dx(y,x,shape_squared);
	break;
      case 3: res = phs1_dxx(x,y,shape_squared);
	break;
      case 4: res = phs1_dxx(y,x,shape_squared);
	break;
      case 5: res = phs1_dxy(x,y,shape_squared);
	break;
      case 6: res = phs1_laplace(x,y,shape_squared);
	break;
      case 7: res = phs1_neg_laplace(x,y,shape_squared);
	break;
      }
    return res;
  };
  __device__ double dx(double x, double y, double shape_squared)
  {
    return phs1_dx(x,y,shape_squared);
  };
  __device__ double dxx(double x, double y, double shape_squared)
  {
    return phs1_dxx(x,y,shape_squared);
  };
  __device__ double dxy(double x, double y, double shape_squared)
  {
    return phs1_dxy(x,y,shape_squared);
  };
  __device__ double laplace(double x, double y, double shape_squared)
  {
    return phs1_laplace(x,y,shape_squared);
  }
  __device__ double neg_laplace(double x, double y, double shape_squared)
  {
    return phs1_neg_laplace(x,y,shape_squared);
  }
};

class phs2_rbf
{
 public:
  __device__ double operator() (double x, double y, double shape_squared)
  {
    return phs2(x,y,shape_squared);
  };
  __device__ double D(double x, double y, double shape_squared, int selection)
  {
    double res;
    switch(selection)
      {
      case 1: res = phs2_dx(x,y,shape_squared);
	break;
      case 2: res = phs2_dx(y,x,shape_squared);
	break;
      case 3: res = phs2_dxx(x,y,shape_squared);
	break;
      case 4: res = phs2_dxx(y,x,shape_squared);
	break;
      case 5: res = phs2_dxy(x,y,shape_squared);
	break;
      case 6: res = phs2_laplace(x,y,shape_squared);
	break;
      case 7: res = phs2_neg_laplace(x,y,shape_squared);
	break;
      }
    return res;
  };
  __device__ double dx(double x, double y, double shape_squared)
  {
    return phs2_dx(x,y,shape_squared);
  };
  __device__ double dxx(double x, double y, double shape_squared)
  {
    return phs2_dxx(x,y,shape_squared);
  };
  __device__ double dxy(double x, double y, double shape_squared)
  {
    return phs2_dxy(x,y,shape_squared);
  };
  __device__ double laplace(double x, double y, double shape_squared)
  {
    return phs2_laplace(x,y,shape_squared);
  }
  __device__ double neg_laplace(double x, double y, double shape_squared)
  {
    return phs2_neg_laplace(x,y,shape_squared);
  }
};

class phs3_rbf
{
 public:
  __device__ double operator() (double x, double y, double shape_squared)
  {
    return phs3(x,y,shape_squared);
  };
  __device__ double D(double x, double y, double shape_squared, int selection)
  {
    double res;
    switch(selection)
      {
      case 1: res = phs3_dx(x,y,shape_squared);
	break;
      case 2: res = phs3_dx(y,x,shape_squared);
	break;
      case 3: res = phs3_dxx(x,y,shape_squared);
	break;
      case 4: res = phs3_dxx(y,x,shape_squared);
	break;
      case 5: res = phs3_dxy(x,y,shape_squared);
	break;
      case 6: res = phs3_laplace(x,y,shape_squared);
	break;
      case 7: res = phs3_neg_laplace(x,y,shape_squared);
	break;
      }
    return res;
  };
  __device__ double dx(double x, double y, double shape_squared)
  {
    return phs3_dx(x,y,shape_squared);
  };
  __device__ double dxx(double x, double y, double shape_squared)
  {
    return phs3_dxx(x,y,shape_squared);
  };
  __device__ double dxy(double x, double y, double shape_squared)
  {
    return phs3_dxy(x,y,shape_squared);
  };
  __device__ double laplace(double x, double y, double shape_squared)
  {
    return phs3_laplace(x,y,shape_squared);
  }
  __device__ double neg_laplace(double x, double y, double shape_squared)
  {
    return phs3_neg_laplace(x,y,shape_squared);
  }
};

class phs4_rbf
{
 public:
  __device__ double operator() (double x, double y, double shape_squared)
  {
    return phs4(x,y,shape_squared);
  };
  __device__ double D(double x, double y, double shape_squared, int selection)
  {
    double res;
    switch(selection)
      {
      case 1: res = phs4_dx(x,y,shape_squared);
	break;
      case 2: res = phs4_dx(y,x,shape_squared);
	break;
      case 3: res = phs4_dxx(x,y,shape_squared);
	break;
      case 4: res = phs4_dxx(y,x,shape_squared);
	break;
      case 5: res = phs4_dxy(x,y,shape_squared);
	break;
      case 6: res = phs4_laplace(x,y,shape_squared);
	break;
      case 7: res = phs4_neg_laplace(x,y,shape_squared);
	break;
      }
    return res;
  };
  __device__ double dx(double x, double y, double shape_squared)
  {
    return phs4_dx(x,y,shape_squared);
  };
  __device__ double dxx(double x, double y, double shape_squared)
  {
    return phs4_dxx(x,y,shape_squared);
  };
  __device__ double dxy(double x, double y, double shape_squared)
  {
    return phs4_dxy(x,y,shape_squared);
  };
  __device__ double laplace(double x, double y, double shape_squared)
  {
    return phs4_laplace(x,y,shape_squared);
  }
  __device__ double neg_laplace(double x, double y, double shape_squared)
  {
    return phs4_neg_laplace(x,y,shape_squared);
  }
};
  
class phs5_rbf
{
 public:
  __device__ double operator() (double x, double y, double shape_squared)
  {
    return phs5(x,y,shape_squared);
  };
  __device__ double D(double x, double y, double shape_squared, int selection)
  {
    double res;
    switch(selection)
      {
      case 1: res = phs5_dx(x,y,shape_squared);
	break;
      case 2: res = phs5_dx(y,x,shape_squared);
	break;
      case 3: res = phs5_dxx(x,y,shape_squared);
	break;
      case 4: res = phs5_dxx(y,x,shape_squared);
	break;
      case 5: res = phs5_dxy(x,y,shape_squared);
	break;
      case 6: res = phs5_laplace(x,y,shape_squared);
	break;
      case 7: res = phs5_neg_laplace(x,y,shape_squared);
	break;
      }
    return res;
  };
  __device__ double dx(double x, double y, double shape_squared)
  {
    return phs5_dx(x,y,shape_squared);
  };
  __device__ double dxx(double x, double y, double shape_squared)
  {
    return phs5_dxx(x,y,shape_squared);
  };
  __device__ double dxy(double x, double y, double shape_squared)
  {
    return phs5_dxy(x,y,shape_squared);
  };
  __device__ double laplace(double x, double y, double shape_squared)
  {
    return phs5_laplace(x,y,shape_squared);
  }
  __device__ double neg_laplace(double x, double y, double shape_squared)
  {
    return phs5_neg_laplace(x,y,shape_squared);
  }
};

class phs6_rbf
{
 public:
  __device__ double operator() (double x, double y, double shape_squared)
  {
    return phs6(x,y,shape_squared);
  };
  __device__ double D(double x, double y, double shape_squared, int selection)
  {
    double res;
    switch(selection)
      {
      case 1: res = phs6_dx(x,y,shape_squared);
	break;
      case 2: res = phs6_dx(y,x,shape_squared);
	break;
      case 3: res = phs6_dxx(x,y,shape_squared);
	break;
      case 4: res = phs6_dxx(y,x,shape_squared);
	break;
      case 5: res = phs6_dxy(x,y,shape_squared);
	break;
      case 6: res = phs6_laplace(x,y,shape_squared);
	break;
      case 7: res = phs6_neg_laplace(x,y,shape_squared);
	break;
      }
    return res;
  };
  __device__ double dx(double x, double y, double shape_squared)
  {
    return phs6_dx(x,y,shape_squared);
  };
  __device__ double dxx(double x, double y, double shape_squared)
  {
    return phs6_dxx(x,y,shape_squared);
  };
  __device__ double dxy(double x, double y, double shape_squared)
  {
    return phs6_dxy(x,y,shape_squared);
  };
  __device__ double laplace(double x, double y, double shape_squared)
  {
    return phs6_laplace(x,y,shape_squared);
  }
  __device__ double neg_laplace(double x, double y, double shape_squared)
  {
    return phs6_neg_laplace(x,y,shape_squared);
  }
};

class phs7_rbf
{
 public:
  __device__ double operator() (double x, double y, double shape_squared)
  {
    return phs7(x,y,shape_squared);
  };
  __device__ double D(double x, double y, double shape_squared, int selection)
  {
    double res;
    switch(selection)
      {
      case 1: res = phs7_dx(x,y,shape_squared);
	break;
      case 2: res = phs7_dx(y,x,shape_squared);
	break;
      case 3: res = phs7_dxx(x,y,shape_squared);
	break;
      case 4: res = phs7_dxx(y,x,shape_squared);
	break;
      case 5: res = phs7_dxy(x,y,shape_squared);
	break;
      case 6: res = phs7_laplace(x,y,shape_squared);
	break;
      case 7: res = phs7_neg_laplace(x,y,shape_squared);
	break;
      }
    return res;
  };
  __device__ double dx(double x, double y, double shape_squared)
  {
    return phs7_dx(x,y,shape_squared);
  };
  __device__ double dxx(double x, double y, double shape_squared)
  {
    return phs7_dxx(x,y,shape_squared);
  };
  __device__ double dxy(double x, double y, double shape_squared)
  {
    return phs7_dxy(x,y,shape_squared);
  };
  __device__ double laplace(double x, double y, double shape_squared)
  {
    return phs7_laplace(x,y,shape_squared);
  }
  __device__ double neg_laplace(double x, double y, double shape_squared)
  {
    return phs7_neg_laplace(x,y,shape_squared);
  }
};

class phs8_rbf
{
 public:
  __device__ double operator() (double x, double y, double shape_squared)
  {
    return phs8(x,y,shape_squared);
  };
  __device__ double D(double x, double y, double shape_squared, int selection)
  {
    double res;
    switch(selection)
      {
      case 1: res = phs8_dx(x,y,shape_squared);
	break;
      case 2: res = phs8_dx(y,x,shape_squared);
	break;
      case 3: res = phs8_dxx(x,y,shape_squared);
	break;
      case 4: res = phs8_dxx(y,x,shape_squared);
	break;
      case 5: res = phs8_dxy(x,y,shape_squared);
	break;
      case 6: res = phs8_laplace(x,y,shape_squared);
	break;
      case 7: res = phs8_neg_laplace(x,y,shape_squared);
	break;
      }
    return res;
  };
  __device__ double dx(double x, double y, double shape_squared)
  {
    return phs8_dx(x,y,shape_squared);
  };
  __device__ double dxx(double x, double y, double shape_squared)
  {
    return phs8_dxx(x,y,shape_squared);
  };
  __device__ double dxy(double x, double y, double shape_squared)
  {
    return phs8_dxy(x,y,shape_squared);
  };
  __device__ double laplace(double x, double y, double shape_squared)
  {
    return phs8_laplace(x,y,shape_squared);
  }
  __device__ double neg_laplace(double x, double y, double shape_squared)
  {
    return phs8_neg_laplace(x,y,shape_squared);
  }
};

class phs9_rbf
{
 public:
  __device__ double operator() (double x, double y, double shape_squared)
  {
    return phs9(x,y,shape_squared);
  };
  __device__ double D(double x, double y, double shape_squared, int selection)
  {
    double res;
    switch(selection)
      {
      case 1: res = phs9_dx(x,y,shape_squared);
	break;
      case 2: res = phs9_dx(y,x,shape_squared);
	break;
      case 3: res = phs9_dxx(x,y,shape_squared);
	break;
      case 4: res = phs9_dxx(y,x,shape_squared);
	break;
      case 5: res = phs9_dxy(x,y,shape_squared);
	break;
      case 6: res = phs9_laplace(x,y,shape_squared);
	break;
      case 7: res = phs9_neg_laplace(x,y,shape_squared);
	break;
      }
    return res;
  };
  __device__ double dx(double x, double y, double shape_squared)
  {
    return phs9_dx(x,y,shape_squared);
  };
  __device__ double dxx(double x, double y, double shape_squared)
  {
    return phs9_dxx(x,y,shape_squared);
  };
  __device__ double dxy(double x, double y, double shape_squared)
  {
    return phs9_dxy(x,y,shape_squared);
  };
  __device__ double laplace(double x, double y, double shape_squared)
  {
    return phs9_laplace(x,y,shape_squared);
  }
  __device__ double neg_laplace(double x, double y, double shape_squared)
  {
    return phs9_neg_laplace(x,y,shape_squared);
  }
};

class phs10_rbf
{
 public:
  __device__ double operator() (double x, double y, double shape_squared)
  {
    return phs10(x,y,shape_squared);
  };
  __device__ double D(double x, double y, double shape_squared, int selection)
  {
    double res;
    switch(selection)
      {
      case 1: res = phs10_dx(x,y,shape_squared);
	break;
      case 2: res = phs10_dx(y,x,shape_squared);
	break;
      case 3: res = phs10_dxx(x,y,shape_squared);
	break;
      case 4: res = phs10_dxx(y,x,shape_squared);
	break;
      case 5: res = phs10_dxy(x,y,shape_squared);
	break;
      case 6: res = phs10_laplace(x,y,shape_squared);
	break;
      case 7: res = phs10_neg_laplace(x,y,shape_squared);
	break;
      }
    return res;
  };
  __device__ double dx(double x, double y, double shape_squared)
  {
    return phs10_dx(x,y,shape_squared);
  };
  __device__ double dxx(double x, double y, double shape_squared)
  {
    return phs10_dxx(x,y,shape_squared);
  };
  __device__ double dxy(double x, double y, double shape_squared)
  {
    return phs10_dxy(x,y,shape_squared);
  };
  __device__ double laplace(double x, double y, double shape_squared)
  {
    return phs10_laplace(x,y,shape_squared);
  }
  __device__ double neg_laplace(double x, double y, double shape_squared)
  {
    return phs10_neg_laplace(x,y,shape_squared);
  }
};

#endif /* CUDA_RBFPHS_H */
