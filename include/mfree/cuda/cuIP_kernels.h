/*** /cuda/cuda_cuIP_kernels.h
     These are the CUDA kernels which are associated
     with the device interpolation routines of libmfree.

Julian Merten
INAF OAS Bologna
Jul 2018
julian.merten@inaf.it
http://www.julianmerten.net
***/


#ifndef    CUDA_CUIP_KERNELS_H
#define    CUDA_CUIP_KERNELS_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <saw2/cuda/cuda_manager.h>
#include <saw2/cuda/cuda_device_functions.h>

/**
  The first set of kernels calculates the linear system of 
  equation the solution to which are the interpolation weights.
**/

/*
  This kernel calculates the coefficient matrix.
  The kernel needs the nearest neighbour map for a set of nodes
  to which is interpolated to, the coordinates of the interpolant, the shape
  parameter for each node (only relevant if RBF with shape is used),
  the stride in index map, the degree of polynomial support, a
  radial basis function and a pointer to the coefficient matrix. 
*/

template<class T> __global__ cuIP_matrix_part(int *index_map, double *interpolation_coordinates, double *interpolant_coordinates, double *shapes, int matrix_stride, int pdeg, double *A, T *rbf)
{
  int offsetA = blockIdx.x*matrix_stride*matrix_stride;
  
  //Getting all nearest neighbours in the shared memory
  __shared__ double coordinates[SHAPE_OPT_MAX_NN][2];
  int index = index_map[blockIdx.x*blockDim.x+threadIdx.x];
  coordinates[threadIdx.x][0] = interpolant_coordinates[index*2] - interpolation_coordinates[blockIdx.x*2];
  coordinates[threadIdx.x][1] = interpolant_coordinates[index*2+1] - interpolation_coordinates[blockIdx.x*2+1];
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
  This calculates the result vector of the interpolation LSE.
  It needs the NN index map and the interpolant function values, together
  with the stride in the index map, the polynomial degree, and a pointer
  to the result vector.
*/

__global__ cuIP_vector_part(int* index_map, double *interpolant,int matrix_stride, double* b);

/**
   The second set of kernels calculates the scalar product between
   interpolant values and interpolation weights to get the interpolation
   result.
**/

/*
  The first kernel can be used if the numberof nearest neighbours is
  a power of 2. If not, the kernel will provide wrong results.
*/

__global__ void cuIP_product_pow2(int *tree, double *interpolation_coordinates, double *interpolant_coordinates, double *shapes, double *w, double *i, int matrix_stride);

/*
  This is the more general version of the routine where the number
  of nearest neighbours does not have to be a power of 2. Slightly 
  slower than the version above. 
*/

__global__ void cuIP_product(int *tree, double *interpolation_coordinates, double *interpolant_coordinates, double *shapes, double *w, double *i, int matrix_stride);

/**
   The final set of kernels splits the building of result vector
   and coefficient matrix into those parts which are dependent of
   the shaper parameter and those which are not. This is used in conjuction
   with the shape parameter optimisation routines. 
   This block also provides kernels to compare the outcome of an
   interpolation with a reference result. 
**/

/*
  The shape-parameter dependent part of IP LSE building.
*/


template<class T> __global__ void cuIP_optimise_shape_dependent_part(int* tree, double *interpolation_coordinates, double *interpolant_coordinates, double *shapes, int matrix_stride, double* A, T *rbf)
{
  int offsetA = blockIdx.x*matrix_stride*matrix_stride;

  //Getting all nearest neighbours in the shared memory
  __shared__ double coordinates[SHAPE_OPT_MAX_NN][2];
  int index = tree[blockIdx.x*blockDim.x+threadIdx.x];
  coordinates[threadIdx.x][0] = interpolant_coordinates[index*2] - interpolation_coordinates[blockIdx.x*2];
  coordinates[threadIdx.x][1] = interpolant_coordinates[index*2+1] - interpolation_coordinates[blockIdx.x*2+1];
  __syncthreads();

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
  The shape-independent part of IP LSE building.
*/

__global__ void cuIP_optimise_const_part(int* tree, double *interpolation_coordinates, double *interpolant, double *interpolant_coordinates, int matrix_stride, int pdeg, double* A, double *b);

/* 
   This last of kernels calculates the interpolation and compares
   it to a pre-defined reference result.
   Again, the distinction between number of nearest neighbours as a power of 2
   and the general (slightly slower) case is made. 
*/

__global__ void cuIP_optimise_calculate_IP_and_compare_pow2(int *tree, double *interpolation_coordinates, double *interpolant_coordinates, double *shapes, double *w, double *truth, double *err, int matrix_stride);

__global__ void cuIP_optimise_calculate_IP_and_compare(int *tree, double *interpolation_coordinates, double *interpolant_coordinates, double *shapes, double *w, double *truth, double *err, int matrix_stride);


#endif /* CUDA_CUIP_KERNELS_H */
