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
  You need to provide RBF type and order.
  Refer to cuRBFs.h for valid selections.
*/

__global__ void cuFD_weights_matrix_part(int* tree, double *coordinates, double *shapes,int matrix_stride, int pdeg, double* A, char rbf_type, double rbf_order);

/*
  This sets the vector part of the weight calculation. Now you also
  need to provide the desied derivative. The current derivative order
  scheme is:

  1: d/dx
  2: d/dy
  3: d^2/dxx
  4: d^2/dyy
  5: d^2/dxy
  6: d^2/dxx + d^2/dyy 
  7: d^2/dxx - d^2/dyy 
*/

__global__ void cuFD_weights_vector_part(int* tree, double *coordinates, double *shapes,int matrix_stride, double* b, char rbf_type, double rbf_order, int derivative_order);

/*
  The same as above, but this gives the opportunity to multiply
  the derivative operator with a given factor. 
*/

__global__ void cuFD_weights_vector_part(int* tree, double *coordinates, double *shapes,int matrix_stride, double* b, char rbf_type, double rbf_order, int derivative_order, double factor);

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

__global__ void cuFD_optimise_const_part(int* tree, double *all_coordinates, int matrix_stride, int pdeg, double* A, double *b, char rbf_type, double rbf_order, int derivative_order);

/*
  This kernel calculates the parts of the LSE which change under a change
  of shape parameter. 
*/

__global__ void cuFD_optimise_shape_dependent_part(int* tree, double *coordinates, double *shapes,int matrix_stride, double* A, double *b, char rbf_type, double rbf_order, int derivative_order);

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
