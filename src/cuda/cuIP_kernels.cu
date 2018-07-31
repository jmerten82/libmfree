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

__global__ cuIP_vector_part(int* index_map, double *interpolant,int matrix_stride, double* b)
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

__global__ void cuIP_product_pow2(int *tree, double *interpolation_coordinates, double *interpolant_coordinates, double *shapes, double *w, double *i, int matrix_stride)
{
  int index_c = tree[blockIdx.x*blockDim.x+threadIdx.x];
  double x = interpolant_coordinates[index_c*2] - interpolation_coordinates[blockIdx.x*2];
  double y = interpolant_coordinates[index_c*2+1] - interpolation_coordinates[blockIdx.x*2+1];
  __shared__ double product[SHAPE_OPT_MAX_NN];

  product[threadIdx.x] = ga(x*x+y*y,shapes[blockIdx.x])*w[blockIdx.x*matrix_stride+threadIdx.x];
  __syncthreads();

  index_c = blockDim.x /2;
  while(index_c != 0)
    {
      if(threadIdx.x < index_c)
	{
	  product[threadIdx.x] += product[threadIdx.x +index_c];
	}
      __syncthreads();
      index_c /= 2;
    }

  if(threadIdx.x == 0)
    {
      i[blockIdx.x] = product[0]+w[blockIdx.x*matrix_stride+blockDim.x];
    }
}

__global__ void cuIP_product(int *tree, double *interpolation_coordinates, double *interpolant_coordinates, double *shapes, double *w, double *i, int matrix_stride)
{
  int index_c = tree[blockIdx.x*blockDim.x+threadIdx.x];
  double x = interpolant_coordinates[index_c*2] - interpolation_coordinates[blockIdx.x*2];
  double y = interpolant_coordinates[index_c*2+1] - interpolation_coordinates[blockIdx.x*2+1];
  __shared__ double product[SHAPE_OPT_MAX_NN];

  product[threadIdx.x] = ga(x*x+y*y,shapes[blockIdx.x])*w[blockIdx.x*matrix_stride+threadIdx.x];
  __syncthreads();


  if(threadIdx.x == 0)
    {
      double sum = w[blockIdx.x*matrix_stride+blockDim.x]; //This is the const.
      for(index_c = 0; index_c < blockDim.x; index_c++)
	{
	  sum += product[index_c];
	}  
      i[blockIdx.x] = sum; 
    }
}

__global__ void cuIP_optimise_const_part(int* tree, double *interpolation_coordinates, double *interpolant, double *interpolant_coordinates, int matrix_stride, int pdeg, double* A, double *b)
{
  int offsetb = blockIdx.x*matrix_stride;
  int offsetA = offsetb*matrix_stride;

  //Getting all nearest neighbours in the shared memory
  __shared__ double coordinates[SHAPE_OPT_MAX_NN][2];
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

__global__ void cuIP_optimise_calculate_IP_and_compare_pow2(int *tree, double *interpolation_coordinates, double *interpolant_coordinates, double *shapes, double *w, double *truth, double *err, int matrix_stride)
{
  int index_c = tree[blockIdx.x*blockDim.x+threadIdx.x];
  double x = interpolant_coordinates[index_c*2] - interpolation_coordinates[blockIdx.x*2];
  double y = interpolant_coordinates[index_c*2+1] - interpolation_coordinates[blockIdx.x*2+1];
  __shared__ double product[SHAPE_OPT_MAX_NN];

  product[threadIdx.x] = ga(x*x+y*y,shapes[blockIdx.x])*w[blockIdx.x*matrix_stride+threadIdx.x];
  __syncthreads();

  index_c = blockDim.x /2;
  while(index_c != 0)
    {
      if(threadIdx.x < index_c)
	{
	  product[threadIdx.x] += product[threadIdx.x +index_c];
	}
      __syncthreads();
      index_c /= 2;
    }

  if(threadIdx.x == 0)
    {
      double correct = truth[blockIdx.x]; 
      err[blockIdx.x] = abs((product[0]+w[blockIdx.x*matrix_stride+blockDim.x]-correct)/correct);
    }
}

__global__ void cuIP_optimise_calculate_IP_and_compare(int *tree, double *interpolation_coordinates, double *interpolant_coordinates, double *shapes, double *w, double *truth, double *err, int matrix_stride)
{
  int index_c = tree[blockIdx.x*blockDim.x+threadIdx.x];
  double x = interpolant_coordinates[index_c*2] - interpolation_coordinates[blockIdx.x*2];
  double y = interpolant_coordinates[index_c*2+1] - interpolation_coordinates[blockIdx.x*2+1];
  __shared__ double product[SHAPE_OPT_MAX_NN];

  product[threadIdx.x] = ga(x*x+y*y,shapes[blockIdx.x])*w[blockIdx.x*matrix_stride+threadIdx.x];
  __syncthreads();


  if(threadIdx.x == 0)
    {
      double sum = w[blockIdx.x*matrix_stride+blockDim.x]; //This is the const.
      for(int index = 0; index < blockDim.x; index++)
	{
	  sum += product[index];
	} 
      double correct = truth[blockIdx.x];
      err[blockIdx.x] = abs((sum - correct)/correct); 
    }
}
