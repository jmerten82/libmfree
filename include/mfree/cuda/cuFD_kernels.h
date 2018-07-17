/*** /cuda/cuda_cuFD_kernels.h
     These are the CUDA kernels which are associated with
     the libmfree cuFD host drivers.

Julian Merten
INAF OA Bologna
Jul 2018
julian.merten@oabo.inaf.it
http://www.julianmerten.net
***/

#ifndef    CUDA_FD_KERNELS_H
#define    CUDA_FD_KERNELS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <mfree/cuda/cuda_manager.h>
#include <mfree/cuda/cuRBFs.h>


/*** 
    The following kernels are needed for FD operations on the device. 
    They are split into different categories. 
    
    weights: Calculate finite differencing weights for given RBFs and shapes
    differentiate: Calculate derivatives from weights
    optimise: routines tuned for shape parameter optimisation

    For all these routines the number of threads is the number nearest
    neighbours and the number of blocks is the number of mesh-free nodes.

***/

/**
   weights
**/

/*                                                                              
  This sets the lower right block of the coefficient matrix to 0.               
*/

__global__ void cuFD_weights_set_zeros(int matrix_stride, int polynomial, int n\
n, double* A);


/*
  This CUDA kernel fills the coefficient matrix for weight calculation.
  This is implemented as a template with the underlying RBF as template
  parameter. RBF must be implemented as a class and an object passed
  to __global__ function. 
  Since this is a template, we write the full definition in here.
*/

template<class T> __global__ void cuFD_weights_matrix_part(int* tree, double *all_coordinates, double *shapes,int matrix_stride, int pdeg, double* A, T *rbf)
{
  int offsetA = blockIdx.x*matrix_stride*matrix_stride;

  //Getting all nearest neighbours in the shared memory
  __shared__ double coordinates[MAX_NN][2];
  int index = tree[blockIdx.x*blockDim.x+threadIdx.x];
  coordinates[threadIdx.x][0] = all_coordinates[index*2] - all_coordinates[blockIdx.x*2];
  coordinates[threadIdx.x][1] = all_coordinates[index*2+1] - all_coordinates[blockIdx.x*2+1];
  __syncthreads();

  row_vector_from_polynomial(coordinates[threadIdx.x][0],coordinates[threadIdx.x][1],matrix_stride, pdeg,&A[offsetA+threadIdx.x*matrix_stride+blockDim.x],&A[offsetA+blockDim.x*matrix_stride+threadIdx.x]);

  //Running over all neighbours in this point.
  for(int i = 0; i < blockDim.x; i++)
    {
      double x = coordinates[i][0] - coordinates[threadIdx.x][0];
      double y = coordinates[i][1] - coordinates[threadIdx.x][1];
      //Setting changing part of coefficient matrix
      A[offsetA+threadIdx.x*matrix_stride+i] = (*rbf)(x,y,shapes[blockIdx.x]);
    }
};

/*
  This is a Gaussian hard-wired version of the matrix weight set. Mostly
  for performance comparison.
*/

__global__ void cuFD_ga_weights_matrix_part(int* tree, double *all_coordinates, double *shapes,int matrix_stride, int pdeg, double* A);

/*
  This sets the vector part of the weight calculation. Now you also
  need to provide the desired derivative. The current derivative order
  scheme is:

  1: d/dx
  2: d/dy
  3: d^2/dxx
  4: d^2/dyy
  5: d^2/dxy
  6: d^2/dxx + d^2/dyy 
  7: d^2/dxx - d^2/dyy 
*/

template<class T> __global__ void cuFD_weights_vector_part(int* tree, double *all_coordinates, double *shapes,int matrix_stride, double* b, T *rbf, int derivative_order)
{
  //Getting all nearest neighbours in the shared memory
  __shared__ double coordinates[MAX_NN][2];
  int index = tree[blockIdx.x*blockDim.x+threadIdx.x];
  coordinates[threadIdx.x][0] = all_coordinates[index*2] - all_coordinates[blockIdx.x*2];
  coordinates[threadIdx.x][1] = all_coordinates[index*2+1] - all_coordinates[blockIdx.x*2+1];
  __syncthreads();

  int offsetb = blockIdx.x*matrix_stride;
  double x = coordinates[threadIdx.x][0];
  double y = coordinates[threadIdx.x][1];
  b[offsetb+threadIdx.x] = rbf->D(-x,-y,shapes[blockIdx.x],derivative_order);

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

/*
  This version hard-wires the derivative but leaves the RBF to the 
  template.
*/

template<class T> __global__ void cuFD_weights_vector_part_dx(int* tree, double *all_coordinates, double *shapes,int matrix_stride, double* b, T *rbf)
{
  //Getting all nearest neighbours in the shared memory
  __shared__ double coordinates[MAX_NN][2];
  int index = tree[blockIdx.x*blockDim.x+threadIdx.x];
  coordinates[threadIdx.x][0] = all_coordinates[index*2] - all_coordinates[blockIdx.x*2];
  coordinates[threadIdx.x][1] = all_coordinates[index*2+1] - all_coordinates[blockIdx.x*2+1];
  __syncthreads();

  int offsetb = blockIdx.x*matrix_stride;
  double x = coordinates[threadIdx.x][0];
  double y = coordinates[threadIdx.x][1];
  b[offsetb+threadIdx.x] = rbf->D(-x,-y,shapes[blockIdx.x],derivative_order);

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
	}
    }
}

/*
  This is a agaibn a ga-rbf hard-wired version of the vector part. 
  In addition, it is also hard-wired to the x derivative. This
  is all for performance comparison.
*/

__global__ void cuFD_ga_dx_weights_vector_part(int* tree, double *all_coordinates, double *shapes,int matrix_stride, double* b);


/*
  The same as above, but this gives the opportunity to multiply
  the derivative operator with a given factor. 
*/

template<class T> __global__ void cuFD_weights_vector_part(int* tree, double *all_coordinates, double *shapes,int matrix_stride, double* b, T *rbf, int derivative_order, double factor)
{
  //Getting all nearest neighbours in the shared memory
  __shared__ double coordinates[MAX_NN][2];
  int index = tree[blockIdx.x*blockDim.x+threadIdx.x];
  coordinates[threadIdx.x][0] = all_coordinates[index*2] - all_coordinates[blockIdx.x*2];
  coordinates[threadIdx.x][1] = all_coordinates[index*2+1] - all_coordinates[blockIdx.x*2+1];
  __syncthreads();

  int offsetb = blockIdx.x*matrix_stride;
  double x = coordinates[threadIdx.x][0];
  double y = coordinates[threadIdx.x][1];
  b[offsetb+threadIdx.x] = factor*rbf->D(-x,-y,shapes[blockIdx.x],derivative_order);

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


/**
   differentiate
**/

/*
  This evaluates a findif product 
  given a set of weights and an input function.
  Some speed-up here if the number of neighbours is a power of 2, 
  if it's not though, wrong result from this kernel.
*/

__global__ void cuFD_differentiate_product_pow2(double *f, double *w, double *d);

/*
  Same as above but the save version if nn is not a power of 2. Hence, slightly
  slower. 
*/

__global__ void cuFD_differentiate_product(double *f, double *w, double *d);

/**
   opitmise
**/

/*
  This routine calculates the parts of A and b in the LSE which are
  constant under a changing shape parameter. 
*/

__global__ void cuFD_optimise_const_part(int* tree, double *all_coordinates, int matrix_stride, int pdeg, double* A, double *b, int derivative_order);

/*
  As before the version which lets you mulitply the derivative operator 
  with a factor.
*/

__global__ void cuFD_optimise_const_part(int* tree, double *all_coordinates, int matrix_stride, int pdeg, double* A, double *b, int derivative_order, double factor);


/*
  This kernel calculates the parts of the LSE which change under a change
  of shape parameter. 
*/

template<class T> __global__ void cuFD_optimise_shape_dependent_part(int* tree, double *all_coordinates, double *shapes,int matrix_stride, double* A, double *b, T *rbf, int derivative_order)
{
  //Getting all nearest neighbours in the shared memory
  __shared__ double coordinates[MAX_NN][2];
  int index = tree[blockIdx.x*blockDim.x+threadIdx.x];
  coordinates[threadIdx.x][0] = all_coordinates[index*2] - all_coordinates[blockIdx.x*2];
  coordinates[threadIdx.x][1] = all_coordinates[index*2+1] - all_coordinates[blockIdx.x*2+1];
  __syncthreads();

  int offsetb = blockIdx.x*matrix_stride;
  int offsetA = offsetb*matrix_stride;

  //Running over all neighbours in this point.
  for(int i = 0; i < blockDim.x; i++)
    {
      double x = coordinates[i][0] - coordinates[threadIdx.x][0];
      double y = coordinates[i][1] - coordinates[threadIdx.x][1];
      //Setting changing part of coefficient matrix
      A[offsetA+threadIdx.x*matrix_stride+i] = (*rbf)(-x,-y,shapes[blockIdx.x]);

    }
  //Setting changing part of result vector
  double x = coordinates[threadIdx.x][0];
  double y = coordinates[threadIdx.x][1];
  b[offsetb+threadIdx.x] = rbf->D(-x,-y,shapes[blockIdx.x],derivative_order);
};

/*
  And again an additional version which lets you multiple the derivative
  operator with a constant.
*/

template<class T> __global__ void cuFD_optimise_shape_dependent_part(int* tree, double *all_coordinates, double *shapes,int matrix_stride, double* A, double *b, T *rbf, int derivative_order, double factor)
{
  //Getting all nearest neighbours in the shared memory
  __shared__ double coordinates[MAX_NN][2];
  int index = tree[blockIdx.x*blockDim.x+threadIdx.x];
  coordinates[threadIdx.x][0] = all_coordinates[index*2] - all_coordinates[blockIdx.x*2];
  coordinates[threadIdx.x][1] = all_coordinates[index*2+1] - all_coordinates[blockIdx.x*2+1];
  __syncthreads();

  int offsetb = blockIdx.x*matrix_stride;
  int offsetA = offsetb*matrix_stride;

  //Running over all neighbours in this point.
  for(int i = 0; i < blockDim.x; i++)
    {
      double x = coordinates[i][0] - coordinates[threadIdx.x][0];
      double y = coordinates[i][1] - coordinates[threadIdx.x][1];
      //Setting changing part of coefficient matrix
      A[offsetA+threadIdx.x*matrix_stride+i] = (*rbf)(-x,-y,shapes[blockIdx.x]);

    }
  //Setting changing part of result vector
  double x = coordinates[threadIdx.x][0];
  double y = coordinates[threadIdx.x][1];
  b[offsetb+threadIdx.x] = factor*rbf->D(-x,-y,shapes[blockIdx.x],derivative_order);

};

/*
  Same as above but lets you multiply the derivative operator with a 
  constant.
*/

__global__ void cuFD_optimise_shape_dependent_part(int* tree, double *coordinates, double *shapes,int matrix_stride, double* A, double *b, char rbf_type, double rbf_order, int derivative_order, double factor);

/*
  This kernel calculates the function values 
  at each neighbour position collevetively. Has to be called only once.
*/

__global__ void cuFD_optimise_func_eval(int *tree, double *func, double *nn_func);

/*
  This kernel calculates the current derivative from the weights 
  and calculates the absolute error wrt the reference functions.
*/

__global__ void cuFD_optimise_calculate_and_compare(double *f, double *w, double *truth, double *err, int matrix_stride);

/*
  Same as above but again slightly faster if the number of nearest neighbours
  is a power 2. Again, wrong result if not.
*/

__global__ void cuFD_optimise_calculate_and_compare_pow2(double *f, double *w, double *truth, double *err, int matrix_stride);


#endif /* CUDA_FD_KERNELS_H  */
