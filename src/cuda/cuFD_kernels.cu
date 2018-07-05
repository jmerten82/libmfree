/*** /cuda/cuda_cuFD_kernels.cu
     These are the CUDA kernels which are associated with
     the libmfree cuFD host drivers.

Julian Merten
INAF OA Bologna
Jul 2018
julian.merten@oabo.inaf.it
http://www.julianmerten.net
***/

#include <mfree/cuda/cuFD_kernels.h>

__global__ void cuFD_weights_set_zeros(int matrix_stride, int polynomial, int nn,  double *A)
{
  
  int offsetA = blockIdx.x*matrix_stride*matrix_stride + (nn+threadIdx.x)*matrix_stride + nn;
  
  for(int i = 0; i < polynomial; i++)
    {
      A[offsetA+i] = 0.;
    }
}





