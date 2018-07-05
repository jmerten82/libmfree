/*** /cuda/cuRBFs.cu
     This is the template for the acutal RBFs
     which are implemented as device functions. 
     This header implements GA as default the default
     template for device RBFs.
     This header also contains some declarations for 
     helper functions which are then defined in the respective source file. 

Julian Merten
INAF OA Bologna
Jul 2018
julian.merten@oabo.inaf.it
http://www.julianmerten.net
***/

#include <mfree/cuda/cuRBFs.h>

__device__ void row_vector_from_polynomial(double x, double y, int col_stride, int pdeg, double *row_ptr, double *col_ptr)
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

__device__ void row_vector_from_polynomial_simpler(double x, double y, int pdeg, double *row_ptr)
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

