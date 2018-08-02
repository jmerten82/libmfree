/*** /cuda/cuda_cuIP_kernels.cu
     These are the CUDA kernels which are associated
     with the device interpolation routines of libmfree.

Julian Merten
INAF OAS Bologna
Jul 2018
julian.merten@inaf.it
http://www.julianmerten.net
***/

#include <mfree/cuda/cuIP_kernels.h>

__global__ void cuIP_vector_part(int* tree, double *interpolant,int matrix_stride, double* b)
{
  int index = tree[blockIdx.x*blockDim.x+threadIdx.x];
  int offsetb = blockIdx.x*matrix_stride;
  b[offsetb+threadIdx.x] = interpolant[index];

  //ugly, pot probably not too harmfull, actually I checked and their is virtually no runtime difference
  if(threadIdx.x == 0)
    {
      for(int i = blockDim.x; i < matrix_stride; i++)
	{
	  b[offsetb+i] = 0.;
	}
    }
}

__global__ void cuIP_optimise_const_part(int* tree, double *interpolation_coordinates, double *interpolant, double *interpolant_coordinates, int matrix_stride, int pdeg, double* A, double *b)
{
  int offsetb = blockIdx.x*matrix_stride;
  int offsetA = offsetb*matrix_stride;

  //Getting all nearest neighbours in the shared memory
  __shared__ double coordinates[MAX_NN][2];
  int index = tree[blockIdx.x*blockDim.x+threadIdx.x];
  coordinates[threadIdx.x][0] = interpolant_coordinates[index*2] - interpolation_coordinates[blockIdx.x*2];
  coordinates[threadIdx.x][1] = interpolant_coordinates[index*2+1] - interpolation_coordinates[blockIdx.x*2+1];
  __syncthreads();

  row_vector_from_polynomial(coordinates[threadIdx.x][0],coordinates[threadIdx.x][1],matrix_stride, pdeg,&A[offsetA+threadIdx.x*matrix_stride+blockDim.x],&A[offsetA+blockDim.x*matrix_stride+threadIdx.x]);

  b[offsetb+threadIdx.x] = interpolant[index];

  //ugly, pot probably not too harmfull, actually I checked and their is virtually no runtime difference
  if(threadIdx.x == 0)
    {
      for(int i = blockDim.x; i < matrix_stride; i++)
	{
	  b[offsetb+i] = 0.;
	}
    }
}
