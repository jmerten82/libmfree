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

__global__ void cuFD_differentiate_product_pow2(double *f, double *w, double *d)
{
  int index = blockIdx.x*blockDim.x+threadIdx.x;
  __shared__ double product[MAX_NN];

  product[threadIdx.x] = f[index]*w[index];
  __syncthreads();

  index = blockDim.x /2;
  while(index != 0)
    {
      if(threadIdx.x < index)
	{
	  product[threadIdx.x] += product[threadIdx.x +index];
	}
      __syncthreads();
      index /= 2;
    }
  if(threadIdx.x == 0)
    {
      d[blockIdx.x] = product[0];
    }
}

__global__ void cuFD_differentiate_product(double *f, double *w, double *d)
{
  int index = blockIdx.x*blockDim.x+threadIdx.x;
  __shared__ double product[MAX_NN];

  product[threadIdx.x] = f[index]*w[index];
  __syncthreads();

  if(threadIdx.x == 0)
    {
      double sum = 0.;
      for(index = 0; index < blockDim.x; index++)
	{
	  sum += product[index];
	}      
      d[blockIdx.x] = sum; 
    }
}



__global__ void cuFD_optimise_const_part(int* tree, double *all_coordinates, int matrix_stride, int pdeg, double* A, double *b, int derivative_order)
{

  int offsetb = blockIdx.x*matrix_stride;
  int offsetA = offsetb*matrix_stride;

  //Getting all nearest neighbours in the shared memory
  __shared__ double coordinates[MAX_NN][2];
  int index = tree[blockIdx.x*blockDim.x+threadIdx.x];
  coordinates[threadIdx.x][0] = all_coordinates[index*2] - all_coordinates[blockIdx.x*2];
  coordinates[threadIdx.x][1] = all_coordinates[index*2+1] - all_coordinates[blockIdx.x*2+1];
  __syncthreads();

  row_vector_from_polynomial(coordinates[threadIdx.x][0],coordinates[threadIdx.x][1],matrix_stride, pdeg,&A[offsetA+threadIdx.x*matrix_stride+blockDim.x],&A[offsetA+blockDim.x*matrix_stride+threadIdx.x]);
  b[offsetb+threadIdx.x] = 0;
 
  //ugly, pot probably not too harmfull, actually I checked and their is virtually no runtime difference
  if(threadIdx.x == 0)
    {
      switch(derivative_order)
	{
	case 1:
	  {
	    for(int i = blockDim.x+2; i < matrix_stride; i++)
	      {
		b[offsetb+i] = 0;
	      }
	    b[offsetb+blockDim.x] = 0.;
	    if((matrix_stride - blockDim.x) > 1)
	      {
		b[offsetb+blockDim.x+1] = 1.;
	      }
	  }
	case 2:
	  {
	    for(int i = blockDim.x+3; i < matrix_stride; i++)
	      {
		b[offsetb+i] = 0;
	      }
	    b[offsetb+blockDim.x] = 0.;
	    if((matrix_stride - blockDim.x) > 2)
	      {
		b[offsetb+blockDim.x+1] = 0.;
		b[offsetb+blockDim.x+2] = 1.;
	      }
	  }
	case 3:
	  {
	    for(int i = blockDim.x+4; i < matrix_stride; i++)
	      {
		b[offsetb+i] = 0;
	      }
	    b[offsetb+blockDim.x] = 0.;
	    if((matrix_stride - blockDim.x) > 2)
	      {
		b[offsetb+blockDim.x+1] = 0.;
		b[offsetb+blockDim.x+2] = 0.;
		if((matrix_stride - blockDim.x) > 3)
		  {
		    b[offsetb+blockDim.x+3] = 2.;
		  }
	      }

	  }
	case 4:
	  {
	    for(int i = blockDim.x+6; i < matrix_stride; i++)
	      {
		b[offsetb+i] = 0;
	      }
	    b[offsetb+blockDim.x] = 0.;
	    if((matrix_stride - blockDim.x) > 2)
	      {
		b[offsetb+blockDim.x+1] = 0.;
		b[offsetb+blockDim.x+2] = 0.;
		if((matrix_stride - blockDim.x) > 5)
		  {
		    b[offsetb+blockDim.x+3] = 0.;
		    b[offsetb+blockDim.x+4] = 0.;
		    b[offsetb+blockDim.x+5] = 2.;
		  }
	      }
	  }
	case 5:
	  {
	    for(int i = blockDim.x+5; i < matrix_stride; i++)
	      {
		b[offsetb+i] = 0;
	      }
	    b[offsetb+blockDim.x] = 0.;
	    if((matrix_stride - blockDim.x) > 2)
	      {
		b[offsetb+blockDim.x+1] = 0.;
		b[offsetb+blockDim.x+2] = 0.;
		if((matrix_stride - blockDim.x) > 4)
		  {
		    b[offsetb+blockDim.x+3] = 0.;
		    b[offsetb+blockDim.x+4] = 1.;
		  }
	      }
	  }
	case 6:
	  {
	    for(int i = blockDim.x+6; i < matrix_stride; i++)
	      {
		b[offsetb+i] = 0;
	      }
	    b[offsetb+blockDim.x] = 0.;
	    if((matrix_stride - blockDim.x) > 2)
	      {
		b[offsetb+blockDim.x+1] = 0.;
		b[offsetb+blockDim.x+2] = 0.;
		if((matrix_stride - blockDim.x) > 5)
		  {
		    b[offsetb+blockDim.x+3] = 2.;
		    b[offsetb+blockDim.x+4] = 0.;
		    b[offsetb+blockDim.x+5] = 2.;
		  }
	      }
	  }
	case 7:
	  {
	    for(int i = blockDim.x+6; i < matrix_stride; i++)
	      {
		b[offsetb+i] = 0;
	      }
	    b[offsetb+blockDim.x] = 0.;
	    if((matrix_stride - blockDim.x) > 2)
	      {
		b[offsetb+blockDim.x+1] = 0.;
		b[offsetb+blockDim.x+2] = 0.;
		if((matrix_stride - blockDim.x) > 5)
		  {
		    b[offsetb+blockDim.x+3] = 2.;
		    b[offsetb+blockDim.x+4] = 0.;
		    b[offsetb+blockDim.x+5] = -2.;
		  }
	      }
	  }
	}
    }
}

__global__ void cuFD_optimise_const_part(int* tree, double *all_coordinates, int matrix_stride, int pdeg, double* A, double *b, int derivative_order, double factor)
{

  int offsetb = blockIdx.x*matrix_stride;
  int offsetA = offsetb*matrix_stride;

  //Getting all nearest neighbours in the shared memory
  __shared__ double coordinates[MAX_NN][2];
  int index = tree[blockIdx.x*blockDim.x+threadIdx.x];
  coordinates[threadIdx.x][0] = all_coordinates[index*2] - all_coordinates[blockIdx.x*2];
  coordinates[threadIdx.x][1] = all_coordinates[index*2+1] - all_coordinates[blockIdx.x*2+1];
  __syncthreads();

  row_vector_from_polynomial(coordinates[threadIdx.x][0],coordinates[threadIdx.x][1],matrix_stride, pdeg,&A[offsetA+threadIdx.x*matrix_stride+blockDim.x],&A[offsetA+blockDim.x*matrix_stride+threadIdx.x]);
  b[offsetb+threadIdx.x] = 0;
 
  //ugly, pot probably not too harmfull, actually I checked and their is virtually no runtime difference
  if(threadIdx.x == 0)
    {
      switch(derivative_order)
	{
	case 1:
	  {
	    for(int i = blockDim.x+2; i < matrix_stride; i++)
	      {
		b[offsetb+i] = 0;
	      }
	    b[offsetb+blockDim.x] = 0.;
	    if((matrix_stride - blockDim.x) > 1)
	      {
		b[offsetb+blockDim.x+1] = factor;
	      }
	  }
	case 2:
	  {
	    for(int i = blockDim.x+3; i < matrix_stride; i++)
	      {
		b[offsetb+i] = 0;
	      }
	    b[offsetb+blockDim.x] = 0.;
	    if((matrix_stride - blockDim.x) > 2)
	      {
		b[offsetb+blockDim.x+1] = 0.;
		b[offsetb+blockDim.x+2] = factor;
	      }
	  }
	case 3:
	  {
	    for(int i = blockDim.x+4; i < matrix_stride; i++)
	      {
		b[offsetb+i] = 0;
	      }
	    b[offsetb+blockDim.x] = 0.;
	    if((matrix_stride - blockDim.x) > 2)
	      {
		b[offsetb+blockDim.x+1] = 0.;
		b[offsetb+blockDim.x+2] = 0.;
		if((matrix_stride - blockDim.x) > 3)
		  {
		    b[offsetb+blockDim.x+3] = 2.*factor;
		  }
	      }

	  }
	case 4:
	  {
	    for(int i = blockDim.x+6; i < matrix_stride; i++)
	      {
		b[offsetb+i] = 0;
	      }
	    b[offsetb+blockDim.x] = 0.;
	    if((matrix_stride - blockDim.x) > 2)
	      {
		b[offsetb+blockDim.x+1] = 0.;
		b[offsetb+blockDim.x+2] = 0.;
		if((matrix_stride - blockDim.x) > 5)
		  {
		    b[offsetb+blockDim.x+3] = 0.;
		    b[offsetb+blockDim.x+4] = 0.;
		    b[offsetb+blockDim.x+5] = 2.*factor;
		  }
	      }
	  }
	case 5:
	  {
	    for(int i = blockDim.x+5; i < matrix_stride; i++)
	      {
		b[offsetb+i] = 0;
	      }
	    b[offsetb+blockDim.x] = 0.;
	    if((matrix_stride - blockDim.x) > 2)
	      {
		b[offsetb+blockDim.x+1] = 0.;
		b[offsetb+blockDim.x+2] = 0.;
		if((matrix_stride - blockDim.x) > 4)
		  {
		    b[offsetb+blockDim.x+3] = 0.;
		    b[offsetb+blockDim.x+4] = factor;
		  }
	      }
	  }
	case 6:
	  {
	    for(int i = blockDim.x+6; i < matrix_stride; i++)
	      {
		b[offsetb+i] = 0;
	      }
	    b[offsetb+blockDim.x] = 0.;
	    if((matrix_stride - blockDim.x) > 2)
	      {
		b[offsetb+blockDim.x+1] = 0.;
		b[offsetb+blockDim.x+2] = 0.;
		if((matrix_stride - blockDim.x) > 5)
		  {
		    b[offsetb+blockDim.x+3] = 2.*factor;
		    b[offsetb+blockDim.x+4] = 0.;
		    b[offsetb+blockDim.x+5] = 2.*factor;
		  }
	      }
	  }
	case 7:
	  {
	    for(int i = blockDim.x+6; i < matrix_stride; i++)
	      {
		b[offsetb+i] = 0;
	      }
	    b[offsetb+blockDim.x] = 0.;
	    if((matrix_stride - blockDim.x) > 2)
	      {
		b[offsetb+blockDim.x+1] = 0.;
		b[offsetb+blockDim.x+2] = 0.;
		if((matrix_stride - blockDim.x) > 5)
		  {
		    b[offsetb+blockDim.x+3] = 2.*factor;
		    b[offsetb+blockDim.x+4] = 0.;
		    b[offsetb+blockDim.x+5] = -2.*factor;
		  }
	      }
	  }
	}
    }
}




