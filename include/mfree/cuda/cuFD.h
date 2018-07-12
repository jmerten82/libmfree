/*** /cuda/cuFD.h
     A collection of host drivers which 
     perform RBF FD operations.

Julian Merten
INAF OA Bologna
Jul 2018
julian.merten@oabo.inaf.it
http://www.julianmerten.net
***/

#ifndef    CUDA_FD_H
#define    CUDA_FD_H

#include <stdexcept>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <omp.h>
#include <mfree/cuda/cuda_manager.h>
#include <mfree/cuda/cuFD_kernels.h>
#include <mfree/test_functions.h>

using namespace std;

/***
   This is the main interface to device FD operations via CUDA host 
   drivers. A CUDA manager is needed in all routines for device
   and device memory management. The routines fall in the following
   categories

   weights: This is for calculating FD weights on the device for all
   or specific derivatives and given a radial basis functions
   and eventually shapes for it. 

   optimise: This optimises shapes for a given RBF using a predefined
   function as reference. 

   differentiate: Calculates the derivatives if shapes are available
   on the device.

   For all these routines, the derivative selection follows this scheme. 
   
   1: d/dx
   2: d/dy
   3: d^2/dxx 
   4: d^2/dyy
   5: d^2/dxy
   6: laplace
   7: neg_laplace
***/

/**
   weights
**/

/*
  This sets the FD weights for a selected derivative and given an RBF with 
  shapes. The derivative operator can be multiplied by a factor if desired.
*/ 

template<class T> void cuFD_weights_set(T *rbf, vector<double> shapes, cuda_manager *cuman, int derivative_selection, double factor = 1.)
{

  vector<string> pointer_pointers;
  if(derivative_selection == 1)
    {
      pointer_pointers.push_back("dx_shapes");
      pointer_pointers.push_back("dx");
    }
  else if(derivative_selection == 2)
    {
      pointer_pointers.push_back("dy_shapes");
      pointer_pointers.push_back("dy");
    }
  else if(derivative_selection == 3)
    {
      pointer_pointers.push_back("dxx_shapes");
      pointer_pointers.push_back("dxx");
    }
  else if(derivative_selection == 4)
    {
      pointer_pointers.push_back("dyy_shapes");
      pointer_pointers.push_back("dyy");
    }
  else if(derivative_selection == 5)
    {
      pointer_pointers.push_back("dxy_shapes");
      pointer_pointers.push_back("dxy");
    }
  else if(derivative_selection == 6)
    {
      pointer_pointers.push_back("laplace_shapes");
      pointer_pointers.push_back("laplace");
    }
  else if(derivative_selection == 7)
    {
      pointer_pointers.push_back("neg_laplace_shapes");
      pointer_pointers.push_back("neg_laplace");
    }
  else
    {
      throw invalid_argument("cuFD_weights: Invalid derivative selection.");
    }

  //Problem size
  int num_nodes = cuman->n("nodes");
  int nn = cuman->n("nn");
  int pdeg = cuman->n("pdeg");

  //Checking if device resources are allocated and parameters make sense. 
  if(shapes.size() != num_nodes)
    {
      throw invalid_argument("cuFD: Shape vector and device CUDA_MAN allocations do not match.");
    }
  if(pdeg < 0)
    {
      throw invalid_argument("cuFD: Invalid polynomial degree.");
    }

  //Calculating important numbers
  int polynomial = (pdeg+1)*(pdeg+2)/2;
  int matrix_stride = nn+polynomial;

  //Setting device to first available.
  checkCudaErrors(cudaSetDevice(cuman->device_query().second));

  //Allocating necessary cublas structures
  cublasHandle_t handle;
  cublasCreate(&handle);
  int *d_pivotArray;
  int *d_infoArray;
  int info;
  checkCudaErrors(cudaMalloc((void **)&d_pivotArray, sizeof(int)*num_nodes*matrix_stride));
  checkCudaErrors(cudaMalloc((void **)&d_infoArray, sizeof(int)*num_nodes));

  //Getting shapes on device
  checkCudaErrors(cudaMemcpy(cuman->FD_device_pointer(pointer_pointers[0]),&shapes[0],num_nodes*sizeof(double),cudaMemcpyHostToDevice));

  //Allocating working memory for weight creation
  double *d_A; //This will hold all coefficient matrices for all nodes
  double *d_b; //This will hold all result vectors for all nodes
  checkCudaErrors(cudaMalloc((void **)&d_A, sizeof(double)*matrix_stride*matrix_stride*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(double)*matrix_stride*num_nodes));

  //Creating pointer array on host for coefficient matrices and copying to device
  double **d_A_pointers;
  double *A_pointers[num_nodes];
  for(int i = 0; i < num_nodes; i++)
    {
      A_pointers[i] = d_A+i*matrix_stride*matrix_stride;
    }
  checkCudaErrors(cudaMalloc((void **)&d_A_pointers,num_nodes*sizeof(double*)));
  checkCudaErrors(cudaMemcpy(d_A_pointers,A_pointers,num_nodes*sizeof(double *),cudaMemcpyHostToDevice));

  //Setting lower right block of A to 0.
  cuFD_weights_set_zeros<<<num_nodes,polynomial>>>(matrix_stride,polynomial,nn,d_A);

  //Calculating coefficient matrix and inverting already since const.
  cuFD_weights_matrix_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer(pointer_pointers[0]),matrix_stride,pdeg,d_A,rbf);
  cublasDgetrfBatched(handle,matrix_stride,d_A_pointers,matrix_stride, d_pivotArray, d_infoArray,num_nodes);

  //Creating pointer array for result vector
  double **d_b_pointers;
  double *b_pointers[num_nodes];
  for(int i = 0; i < num_nodes; i++)
    {
      b_pointers[i] = d_b+i*matrix_stride;
    }
  checkCudaErrors(cudaMalloc((void **)&d_b_pointers,num_nodes*sizeof(double*)));
  checkCudaErrors(cudaMemcpy(d_b_pointers,b_pointers,num_nodes*sizeof(double *),cudaMemcpyHostToDevice));

  //Calculating result vectors and solving system
  if(factor == 1.)
    {
      cuFD_weights_ga_vector_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer(pointer_pointers[0]),matrix_stride,d_b,rbf,derivative_selection); 
    }
  else
    {
      cuFD_weights_ga_vector_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer(pointer_pointers[0]),matrix_stride,d_b,rbf,derivative_selection,factor);
    }

  cublasDgetrsBatched(handle,CUBLAS_OP_N,matrix_stride,1,(const double**)d_A_pointers,matrix_stride,d_pivotArray,d_b_pointers,matrix_stride,&info,num_nodes);

  //Copying weights into permanent memory location
  checkCudaErrors(cudaMemcpy2D(cuman->FD_device_pointer(pointer_pointers[1]),sizeof(double)*nn,d_b,sizeof(double)*matrix_stride,sizeof(double)*nn,num_nodes,cudaMemcpyDeviceToDevice));

  //Sending result to all devices
  cuman->distribute_FD_weights(pointer_pointers[1]);

  if(!cuman->FD_weights_status(pointer_pointers[1]))
    {
      cuman->switch_FD_weights_status(pointer_pointers[1]);
    }

  //Destroying cublas handle
  cublasDestroy(handle);

  //FREE DEVICE MEMORY WE ALLOCATED
  checkCudaErrors(cudaFree(d_pivotArray));
  checkCudaErrors(cudaFree(d_infoArray));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_b));
  checkCudaErrors(cudaFree(d_A_pointers));
  checkCudaErrors(cudaFree(d_b_pointers));
};

/*
  This sets the FD weights for all possible derivatives and given an RBF with 
  shapes. The derivative operator can be multiplied by a factor if desired.
*/ 

template<class T> void cuFD_weights_set(T *rbf, vector<vector<double> > shapes, cuda_manager *cuman, double factor = 1.)
{
  //Problem size
  int num_nodes = cuman->n("nodes");
  int nn = cuman->n("nn");
  int pdeg = cuman->n("pdeg");

  //Checking if device resources are allocated and parameters make sense. 
  if(shapes.size() != num_nodes)
    {
      throw invalid_argument("cuFD: Shape vector and device CUDA_MAN allocations do not match.");
    }
  if(pdeg < 0)
    {
      throw invalid_argument("cuFD: Invalid polynomial degree.");
    }

  //Calculating important numbers
  int polynomial = (pdeg+1)*(pdeg+2)/2;
  int matrix_stride = nn+polynomial;

  //Setting device to first available.
  checkCudaErrors(cudaSetDevice(cuman->device_query().second));

  //Allocating necessary cublas structures
  cublasHandle_t handle;
  cublasCreate(&handle);
  int *d_pivotArray1, *d_pivotArray2, *d_pivotArray3, *d_pivotArray4, *d_pivotArray5, *d_pivotArray6, *d_pivotArray7;
  int *d_infoArray1, *d_infoArray2, *d_infoArray3, *d_infoArray4, *d_infoArray5, *d_infoArray6, *d_infoArray7;
  int info;
  checkCudaErrors(cudaMalloc((void **)&d_pivotArray1, sizeof(int)*num_nodes*matrix_stride));
  checkCudaErrors(cudaMalloc((void **)&d_infoArray1, sizeof(int)*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_pivotArray2, sizeof(int)*num_nodes*matrix_stride));
  checkCudaErrors(cudaMalloc((void **)&d_infoArray2, sizeof(int)*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_pivotArray3, sizeof(int)*num_nodes*matrix_stride));
  checkCudaErrors(cudaMalloc((void **)&d_infoArray3, sizeof(int)*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_pivotArray4, sizeof(int)*num_nodes*matrix_stride));
  checkCudaErrors(cudaMalloc((void **)&d_infoArray4, sizeof(int)*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_pivotArray5, sizeof(int)*num_nodes*matrix_stride));
  checkCudaErrors(cudaMalloc((void **)&d_infoArray5, sizeof(int)*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_pivotArray6, sizeof(int)*num_nodes*matrix_stride));
  checkCudaErrors(cudaMalloc((void **)&d_infoArray6, sizeof(int)*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_pivotArray7, sizeof(int)*num_nodes*matrix_stride));
  checkCudaErrors(cudaMalloc((void **)&d_infoArray7, sizeof(int)*num_nodes));

  //Getting shapes on device
  checkCudaErrors(cudaMemcpy(cuman->FD_device_pointer("dx_shapes"),&shapes[0][0],num_nodes*sizeof(double),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cuman->FD_device_pointer("dy_shapes"),&shapes[1][0],num_nodes*sizeof(double),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cuman->FD_device_pointer("dxx_shapes"),&shapes[2][0],num_nodes*sizeof(double),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cuman->FD_device_pointer("dyy_shapes"),&shapes[3][0],num_nodes*sizeof(double),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cuman->FD_device_pointer("dxy_shapes"),&shapes[4][0],num_nodes*sizeof(double),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cuman->FD_device_pointer("laplace_shapes"),&shapes[5][0],num_nodes*sizeof(double),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(cuman->FD_device_pointer("neg_laplace_shapes"),&shapes[6][0],num_nodes*sizeof(double),cudaMemcpyHostToDevice));

  //Allocating working memory for weight creation
  double *d_A1, *d_A2, *d_A3, *d_A4, *d_A5, *d_A6, *d_A7; //This will hold all coefficient matrices for all nodes
  double *d_b1, *d_b2, *d_b3, *d_b4, *d_b5, *d_b6, *d_b7; //This will hold all result vectors for all nodes
  checkCudaErrors(cudaMalloc((void **)&d_A1, sizeof(double)*matrix_stride*matrix_stride*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_A2, sizeof(double)*matrix_stride*matrix_stride*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_A3, sizeof(double)*matrix_stride*matrix_stride*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_A4, sizeof(double)*matrix_stride*matrix_stride*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_A5, sizeof(double)*matrix_stride*matrix_stride*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_A6, sizeof(double)*matrix_stride*matrix_stride*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_A7, sizeof(double)*matrix_stride*matrix_stride*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_b1, sizeof(double)*matrix_stride*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_b2, sizeof(double)*matrix_stride*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_b3, sizeof(double)*matrix_stride*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_b4, sizeof(double)*matrix_stride*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_b5, sizeof(double)*matrix_stride*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_b6, sizeof(double)*matrix_stride*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_b7, sizeof(double)*matrix_stride*num_nodes));

  //Creating pointer array on host for coefficient matrices and copying to device
  double **d_A_pointers1, **d_A_pointers2, **d_A_pointers3, **d_A_pointers4, **d_A_pointers5, **d_A_pointers6, **d_A_pointers7;
  double *A_pointers1[num_nodes], *A_pointers2[num_nodes], *A_pointers3[num_nodes], *A_pointers4[num_nodes], *A_pointers5[num_nodes], *A_pointers6[num_nodes], *A_pointers7[num_nodes];
  for(int i = 0; i < num_nodes; i++)
    {
      A_pointers1[i] = d_A1+i*matrix_stride*matrix_stride;
      A_pointers2[i] = d_A2+i*matrix_stride*matrix_stride;
      A_pointers3[i] = d_A3+i*matrix_stride*matrix_stride;
      A_pointers4[i] = d_A4+i*matrix_stride*matrix_stride;
      A_pointers5[i] = d_A5+i*matrix_stride*matrix_stride;
      A_pointers6[i] = d_A6+i*matrix_stride*matrix_stride;
      A_pointers7[i] = d_A7+i*matrix_stride*matrix_stride;
    }
  checkCudaErrors(cudaMalloc((void **)&d_A_pointers1,num_nodes*sizeof(double*)));
  checkCudaErrors(cudaMemcpy(d_A_pointers1,A_pointers1,num_nodes*sizeof(double *),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void **)&d_A_pointers2,num_nodes*sizeof(double*)));
  checkCudaErrors(cudaMemcpy(d_A_pointers2,A_pointers2,num_nodes*sizeof(double *),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void **)&d_A_pointers3,num_nodes*sizeof(double*)));
  checkCudaErrors(cudaMemcpy(d_A_pointers3,A_pointers3,num_nodes*sizeof(double *),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void **)&d_A_pointers4,num_nodes*sizeof(double*)));
  checkCudaErrors(cudaMemcpy(d_A_pointers4,A_pointers4,num_nodes*sizeof(double *),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void **)&d_A_pointers5,num_nodes*sizeof(double*)));
  checkCudaErrors(cudaMemcpy(d_A_pointers5,A_pointers5,num_nodes*sizeof(double *),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void **)&d_A_pointers6,num_nodes*sizeof(double*)));
  checkCudaErrors(cudaMemcpy(d_A_pointers6,A_pointers6,num_nodes*sizeof(double *),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void **)&d_A_pointers7,num_nodes*sizeof(double*)));
  checkCudaErrors(cudaMemcpy(d_A_pointers7,A_pointers7,num_nodes*sizeof(double *),cudaMemcpyHostToDevice));

  //Setting lower right block of A to 0.
  cuFD_weights_set_zeros<<<num_nodes,polynomial>>>(matrix_stride,polynomial,nn,d_A1);
  cuFD_weights_set_zeros<<<num_nodes,polynomial>>>(matrix_stride,polynomial,nn,d_A2);
  cuFD_weights_set_zeros<<<num_nodes,polynomial>>>(matrix_stride,polynomial,nn,d_A3);
  cuFD_weights_set_zeros<<<num_nodes,polynomial>>>(matrix_stride,polynomial,nn,d_A4);
  cuFD_weights_set_zeros<<<num_nodes,polynomial>>>(matrix_stride,polynomial,nn,d_A5);
  cuFD_weights_set_zeros<<<num_nodes,polynomial>>>(matrix_stride,polynomial,nn,d_A6);
  cuFD_weights_set_zeros<<<num_nodes,polynomial>>>(matrix_stride,polynomial,nn,d_A7);

  //Calculating coefficient matrix and inverting already since const.
  cuFD_weights_matrix_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("dx_shapes"),matrix_stride,pdeg,d_A1,rbf);
  cublasDgetrfBatched(handle,matrix_stride,d_A_pointers1,matrix_stride, d_pivotArray1, d_infoArray1,num_nodes);
cuFD_weights_matrix_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("dy_shapes"),matrix_stride,pdeg,d_A2,rbf);
  cublasDgetrfBatched(handle,matrix_stride,d_A_pointers2,matrix_stride, d_pivotArray2, d_infoArray2,num_nodes);
cuFD_weights_matrix_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("dxx_shapes"),matrix_stride,pdeg,d_A3,rbf);
  cublasDgetrfBatched(handle,matrix_stride,d_A_pointers3,matrix_stride, d_pivotArray3, d_infoArray3,num_nodes);
cuFD_weights_matrix_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("dyy_shapes"),matrix_stride,pdeg,d_A4,rbf);
  cublasDgetrfBatched(handle,matrix_stride,d_A_pointers4,matrix_stride, d_pivotArray4, d_infoArray4,num_nodes);
cuFD_weights_matrix_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("dxy_shapes"),matrix_stride,pdeg,d_A5,rbf);
  cublasDgetrfBatched(handle,matrix_stride,d_A_pointers5,matrix_stride, d_pivotArray5, d_infoArray5,num_nodes);
cuFD_weights_matrix_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("laplace_shapes"),matrix_stride,pdeg,d_A6,rbf);
  cublasDgetrfBatched(handle,matrix_stride,d_A_pointers6,matrix_stride, d_pivotArray6, d_infoArray6,num_nodes);
  cuFD_weights_matrix_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("neg_laplace_shapes"),matrix_stride,pdeg,d_A7,rbf);
 cublasDgetrfBatched(handle,matrix_stride,d_A_pointers7,matrix_stride, d_pivotArray7, d_infoArray7,num_nodes);

  //Creating pointer array for result vector
 double **d_b_pointers1, **d_b_pointers2, **d_b_pointers3, **d_b_pointers4, **d_b_pointers5, **d_b_pointers6, **d_b_pointers7;
 double *b_pointers1[num_nodes], *b_pointers2[num_nodes], *b_pointers3[num_nodes], *b_pointers4[num_nodes], *b_pointers5[num_nodes], *b_pointers6[num_nodes], *b_pointers7[num_nodes];
  for(int i = 0; i < num_nodes; i++)
    {
      b_pointers1[i] = d_b1+i*matrix_stride;
      b_pointers2[i] = d_b2+i*matrix_stride;
      b_pointers3[i] = d_b3+i*matrix_stride;
      b_pointers4[i] = d_b4+i*matrix_stride;
      b_pointers5[i] = d_b5+i*matrix_stride;
      b_pointers6[i] = d_b6+i*matrix_stride;
      b_pointers7[i] = d_b7+i*matrix_stride;
    }
  checkCudaErrors(cudaMalloc((void **)&d_b_pointers1,num_nodes*sizeof(double*)));
  checkCudaErrors(cudaMemcpy(d_b_pointers1,b_pointers1,num_nodes*sizeof(double *),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void **)&d_b_pointers2,num_nodes*sizeof(double*)));
  checkCudaErrors(cudaMemcpy(d_b_pointers2,b_pointers2,num_nodes*sizeof(double *),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void **)&d_b_pointers3,num_nodes*sizeof(double*)));
  checkCudaErrors(cudaMemcpy(d_b_pointers3,b_pointers3,num_nodes*sizeof(double *),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void **)&d_b_pointers4,num_nodes*sizeof(double*)));
  checkCudaErrors(cudaMemcpy(d_b_pointers4,b_pointers4,num_nodes*sizeof(double *),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void **)&d_b_pointers5,num_nodes*sizeof(double*)));
  checkCudaErrors(cudaMemcpy(d_b_pointers5,b_pointers5,num_nodes*sizeof(double *),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void **)&d_b_pointers6,num_nodes*sizeof(double*)));
  checkCudaErrors(cudaMemcpy(d_b_pointers6,b_pointers6,num_nodes*sizeof(double *),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMalloc((void **)&d_b_pointers7,num_nodes*sizeof(double*)));
  checkCudaErrors(cudaMemcpy(d_b_pointers7,b_pointers7,num_nodes*sizeof(double *),cudaMemcpyHostToDevice));

  //Calculating result vectors and solving system
  if(factor == 1.)
    {
      cuFD_weights_ga_vector_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("dx_shapes"),matrix_stride,d_b1,rbf,1); 
      cuFD_weights_ga_vector_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("dy_shapes"),matrix_stride,d_b2,rbf,2); 
      cuFD_weights_ga_vector_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("dxx_shapes"),matrix_stride,d_b3,rbf,3); 
      cuFD_weights_ga_vector_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("dyy_shapes"),matrix_stride,d_b4,rbf,4); 
      cuFD_weights_ga_vector_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("dxy_shapes"),matrix_stride,d_b5,rbf,5); 
      cuFD_weights_ga_vector_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("laplace_shapes"),matrix_stride,d_b6,rbf,6); 
      cuFD_weights_ga_vector_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("neg_laplace_shapes"),matrix_stride,d_b7,rbf,7); 
    }
  else
    {
      cuFD_weights_vector_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("dx_shapes"),matrix_stride,d_b1,rbf,1,factor); 
      cuFD_weights_vector_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("dy_shapes"),matrix_stride,d_b2,rbf,2,factor); 
      cuFD_weights_vector_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("dxx_shapes"),matrix_stride,d_b3,rbf,3,factor); 
      cuFD_weights_vector_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("dyy_shapes"),matrix_stride,d_b4,rbf,4,factor); 
      cuFD_weights_vector_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("dxy_shapes"),matrix_stride,d_b5,rbf,5,factor); 
      cuFD_weights_vector_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("laplace_shapes"),matrix_stride,d_b6,rbf,6,factor); 
      cuFD_weights_vector_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("neg_laplace_shapes"),matrix_stride,d_b7,rbf,7,factor); 
    }

  cublasDgetrsBatched(handle,CUBLAS_OP_N,matrix_stride,1,(const double**)d_A_pointers1,matrix_stride,d_pivotArray1,d_b_pointers1,matrix_stride,&info,num_nodes);
  cublasDgetrsBatched(handle,CUBLAS_OP_N,matrix_stride,1,(const double**)d_A_pointers2,matrix_stride,d_pivotArray2,d_b_pointers2,matrix_stride,&info,num_nodes);
  cublasDgetrsBatched(handle,CUBLAS_OP_N,matrix_stride,1,(const double**)d_A_pointers3,matrix_stride,d_pivotArray3,d_b_pointers3,matrix_stride,&info,num_nodes);
  cublasDgetrsBatched(handle,CUBLAS_OP_N,matrix_stride,1,(const double**)d_A_pointers4,matrix_stride,d_pivotArray4,d_b_pointers4,matrix_stride,&info,num_nodes);
  cublasDgetrsBatched(handle,CUBLAS_OP_N,matrix_stride,1,(const double**)d_A_pointers5,matrix_stride,d_pivotArray5,d_b_pointers5,matrix_stride,&info,num_nodes);
  cublasDgetrsBatched(handle,CUBLAS_OP_N,matrix_stride,1,(const double**)d_A_pointers6,matrix_stride,d_pivotArray6,d_b_pointers6,matrix_stride,&info,num_nodes);
  cublasDgetrsBatched(handle,CUBLAS_OP_N,matrix_stride,1,(const double**)d_A_pointers7,matrix_stride,d_pivotArray7,d_b_pointers7,matrix_stride,&info,num_nodes);
  
  //Copying weights into permanent memory location
  checkCudaErrors(cudaMemcpy2D(cuman->FD_device_pointer("dx"),sizeof(double)*nn,d_b1,sizeof(double)*matrix_stride,sizeof(double)*nn,num_nodes,cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy2D(cuman->FD_device_pointer("dy"),sizeof(double)*nn,d_b2,sizeof(double)*matrix_stride,sizeof(double)*nn,num_nodes,cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy2D(cuman->FD_device_pointer("dxx"),sizeof(double)*nn,d_b3,sizeof(double)*matrix_stride,sizeof(double)*nn,num_nodes,cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy2D(cuman->FD_device_pointer("dyy"),sizeof(double)*nn,d_b4,sizeof(double)*matrix_stride,sizeof(double)*nn,num_nodes,cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy2D(cuman->FD_device_pointer("dxy"),sizeof(double)*nn,d_b5,sizeof(double)*matrix_stride,sizeof(double)*nn,num_nodes,cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy2D(cuman->FD_device_pointer("laplace"),sizeof(double)*nn,d_b6,sizeof(double)*matrix_stride,sizeof(double)*nn,num_nodes,cudaMemcpyDeviceToDevice));
  checkCudaErrors(cudaMemcpy2D(cuman->FD_device_pointer("neg_laplace"),sizeof(double)*nn,d_b7,sizeof(double)*matrix_stride,sizeof(double)*nn,num_nodes,cudaMemcpyDeviceToDevice));

  //Sending result to all devices
  cuman->distribute_FD_weights();

  if(!cuman->FD_weights_status("dx"))
    {
      cuman->switch_FD_weights_status("dx");
    }
  if(!cuman->FD_weights_status("dy"))
    {
      cuman->switch_FD_weights_status("dy");
    }
  if(!cuman->FD_weights_status("dxx"))
    {
      cuman->switch_FD_weights_status("dxx");
    }
  if(!cuman->FD_weights_status("dyy"))
    {
      cuman->switch_FD_weights_status("dyy");
    }
  if(!cuman->FD_weights_status("dxy"))
    {
      cuman->switch_FD_weights_status("dxy");
    }
  if(!cuman->FD_weights_status("laplace"))
    {
      cuman->switch_FD_weights_status("laplace");
    }
  if(!cuman->FD_weights_status("neg_laplace"))
    {
      cuman->switch_FD_weights_status("neg_laplace");
    }


  //Destroying cublas handle
  cublasDestroy(handle);

  //FREE DEVICE MEMORY WE ALLOCATED
  checkCudaErrors(cudaFree(d_pivotArray1));
  checkCudaErrors(cudaFree(d_infoArray1));
  checkCudaErrors(cudaFree(d_pivotArray2));
  checkCudaErrors(cudaFree(d_infoArray2));
  checkCudaErrors(cudaFree(d_pivotArray3));
  checkCudaErrors(cudaFree(d_infoArray3));
  checkCudaErrors(cudaFree(d_pivotArray4));
  checkCudaErrors(cudaFree(d_infoArray4));
  checkCudaErrors(cudaFree(d_pivotArray5));
  checkCudaErrors(cudaFree(d_infoArray5));
  checkCudaErrors(cudaFree(d_pivotArray6));
  checkCudaErrors(cudaFree(d_infoArray6));
  checkCudaErrors(cudaFree(d_pivotArray7));
  checkCudaErrors(cudaFree(d_infoArray7));
  checkCudaErrors(cudaFree(d_A1));
  checkCudaErrors(cudaFree(d_A2));
  checkCudaErrors(cudaFree(d_A3));
  checkCudaErrors(cudaFree(d_A4));
  checkCudaErrors(cudaFree(d_A5));
  checkCudaErrors(cudaFree(d_A6));
  checkCudaErrors(cudaFree(d_A7));
  checkCudaErrors(cudaFree(d_b1));
  checkCudaErrors(cudaFree(d_b2));
  checkCudaErrors(cudaFree(d_b3));
  checkCudaErrors(cudaFree(d_b4));
  checkCudaErrors(cudaFree(d_b5));
  checkCudaErrors(cudaFree(d_b6));
  checkCudaErrors(cudaFree(d_b7));
  checkCudaErrors(cudaFree(d_A_pointers1));
  checkCudaErrors(cudaFree(d_A_pointers2));
  checkCudaErrors(cudaFree(d_A_pointers3));
  checkCudaErrors(cudaFree(d_A_pointers4));
  checkCudaErrors(cudaFree(d_A_pointers5));
  checkCudaErrors(cudaFree(d_A_pointers6));
  checkCudaErrors(cudaFree(d_A_pointers7));
  checkCudaErrors(cudaFree(d_b_pointers1));
  checkCudaErrors(cudaFree(d_b_pointers2));
  checkCudaErrors(cudaFree(d_b_pointers3));
  checkCudaErrors(cudaFree(d_b_pointers4));
  checkCudaErrors(cudaFree(d_b_pointers5));
  checkCudaErrors(cudaFree(d_b_pointers6));
  checkCudaErrors(cudaFree(d_b_pointers7));
};

/**
   differentiate
**/

/*
  This sets the function that you want to differentiate.
  It basically takes the current tree of the system and evaluates 
  the function at all relevantpositions. 
  This makes subsequent summations very quick and easy.
*/

void cuFD_differentiate_set(vector<double> function, cuda_manager *cuman);

/*
  This differentiates the device memory function which was set earlier with
  set and returns the derivative.
*/

vector<double> cuFD_differentiate_and_return(int derivative_selection, cuda_manager *cuman);

/*
  This differentiates the function in device memory and writes it
  to derivative in device memory.
*/

void cuFD_differentiate(int derivative_selection, cuda_manager *cuman);

/*
  This calculates all seven derivatives and returns them.
*/

vector<vector<double> > cuFD_differentiate(cuda_manager *cuman);

/**
   optimise
**/

/*
  This optimises the shape parameters for all nodes using a given test function
  and given a derivative selection. As usual the derivative key is

   1: d/dx
   2: d/dy
   3: d^2/dxx 
   4: d^2/dyy
   5: d^2/dxy
   6: laplace
   7: neg_laplace

   Device selection lets you pick a device managed by the cuda_manager.
*/


template <class T> vector<double> cuFD_optimise_shapes(T *rbf, int derivative_order, test_function *reference, vector<double> *final_errors, cuda_manager *cuman, double shape_start, double shape_stop, int brackets, int iterations, bool square_switch, double factor = 1., int device_selection = 0)
{

  //Basic numbers/Problem size
  int num_nodes = cuman->n("nodes");
  int nn = cuman->n("nn");
  int pdeg = cuman->n("pdeg");

  //Some simple checks on the input
  if(pdeg < 0)
    {
      throw runtime_error("cuFD: Invalid polynomial degree.");
    }
  if(shape_start > shape_stop || brackets < 0 || iterations < 0)
    {
      throw invalid_argument("cuFD: Invalid optimisation bracket.");
    }
  if(num_nodes <= 0 || nn <= 0 || cuman->n() <= 0)
    {
      throw invalid_argument("cuFD: Nothing to optimise or to optimise on.");
    }

  if(derivative_order < 1 || derivative_order > 7)
    {
      throw invalid_argument("cuFD: Invalid derivative selection for optimise.");
    }

  //More Basic numbers/Problem size
  int polynomial = (pdeg+1)*(pdeg+2)/2;
  int matrix_stride = nn+polynomial;

  //Allocating for final result
  vector<double> h_shapes(num_nodes,0.);
  final_errors->clear();
  final_errors->resize(num_nodes);

  //Setting device to first available.
  checkCudaErrors(cudaSetDevice(cuman->device_query(device_selection).second));



//This is annoying, but getting coordinates off device to evaluate the test function
  vector<double> h_coordinates(num_nodes*2,0.);
  cudaMemcpy(&h_coordinates[0],cuman->coordinate_pointer(),num_nodes*2*sizeof(double),cudaMemcpyDeviceToHost);

  //Creating the function values on host
  vector<double> h_function;
  for(int i = 0; i < num_nodes; i++)
    {
      vector<double> current_xy;
      current_xy.push_back(h_coordinates[i*2]);
      current_xy.push_back(h_coordinates[i*2+1]);
      h_function.push_back((*reference)(current_xy));
    }
 //Setting reference
  string selector;
    switch(derivative_order)
      {
      case 1: selector = "x";
      case 2: selector = "y";
      case 3: selector = "xx";
      case 4: selector = "yy";
      case 5: selector = "xy";
      case 6: selector = "Laplace";
      case 7: selector = "Neg_Laplace";
      }

  vector<double> h_truth;
  for(int i = 0; i < num_nodes; i++)
    {
      vector<double> c_pair;
      c_pair.push_back(h_coordinates[i*2]);
      c_pair.push_back(h_coordinates[i*2+1]);
      h_truth.push_back(reference->D(c_pair,selector)*factor);
    }

  //In-routine device memory pointers
  double *d_function;
  double *d_truth;
  double *d_err;
  double *d_A; //This will hold all coefficient matrices for all nodes
  double *d_b; //This will hold all result vectors for all nodes
  double *d_A_save; //This will save the constant part of the coefficient matrix which will only have to be calculated once. 
  double *d_b_save; //This will save the constant part of the result vector which will only have to be calculated once.

  //In-routine device memory allocations
  checkCudaErrors(cudaMalloc((void **)&d_function, sizeof(double)*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_truth, sizeof(double)*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_err, sizeof(double)*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_A, sizeof(double)*matrix_stride*matrix_stride*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(double)*matrix_stride*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_A_save, sizeof(double)*matrix_stride*matrix_stride*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_b_save, sizeof(double)*matrix_stride*num_nodes));

  //Device memory pointers to linear system arrays, created on host and copied to device
  double **d_A_pointers;
  double **d_b_pointers;
  double *A_pointers[num_nodes];
  double *b_pointers[num_nodes];
  for(int i = 0; i < num_nodes; i++)
    {
      A_pointers[i] = d_A+i*matrix_stride*matrix_stride;
      b_pointers[i] = d_b+i*matrix_stride;
    }
  checkCudaErrors(cudaMalloc((void **)&d_A_pointers,num_nodes*sizeof(double*)));
  checkCudaErrors(cudaMalloc((void **)&d_b_pointers,num_nodes*sizeof(double*)));
  checkCudaErrors(cudaMemcpy(d_function,&h_function[0],sizeof(double)*h_function.size(),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_truth,&h_truth[0],sizeof(double)*h_truth.size(),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_pointers,A_pointers,num_nodes*sizeof(double *),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_b_pointers,b_pointers,num_nodes*sizeof(double *),cudaMemcpyHostToDevice));

  //Before entering shape loop, calculating the shape-independent parts of the linear system array

  //Setting lower right block of A to 0.
  cuFD_weights_set_zeros<<<num_nodes,polynomial>>>(matrix_stride,polynomial,nn,d_A_save);

  if(factor == 1.0)
    {
      cuFD_optimise_const_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),matrix_stride,pdeg,d_A_save, d_b_save,derivative_order);
    }
  else
    {
      cuFD_optimise_const_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),matrix_stride,pdeg,d_A_save, d_b_save,derivative_order,factor);
    }

  //Evaluating the input function on device for weights
  cuFD_optimise_func_eval<<<num_nodes,nn>>>(cuman->index_map_pointer(), d_function, cuman->FD_device_pointer("function"));

  //Allocating necessary cublas structures
  cublasHandle_t handle;
  cublasCreate(&handle);
  int *d_pivotArray;
  int *d_infoArray;
  int info;
  checkCudaErrors(cudaMalloc((void **)&d_pivotArray, sizeof(int)*num_nodes*matrix_stride));
  checkCudaErrors(cudaMalloc((void **)&d_infoArray, sizeof(int)*num_nodes));
  
  //Allocating quantities tracking the current errors on derivative
  vector<double> all_errors(num_nodes*iterations,-1.); //Vector containing all errors per node for all iterations per bracket
  vector<double> all_shapes(num_nodes*iterations,0.);
  vector<double> min_shape(num_nodes, shape_start);
  vector<double> max_shape(num_nodes, shape_stop);
  vector<double> increment(num_nodes,0.);


  //Start loop over shapes
  for(int current_bracket = 0; current_bracket < brackets; current_bracket++)
    {
      //Setting limits according to full iterations
      for(int i = 0; i < num_nodes; i++)
	{
	  increment[i] = (max_shape[i] - min_shape[i])/(iterations);
	}
      for(int current_iteration = 0; current_iteration < iterations; current_iteration++)
	{
	  //Figure out which is current shape for all nodes.
	  for(int i = 0; i < num_nodes; i++)
	    {
	      double e1 = current_iteration*increment[i] + min_shape[i];
	      if(!square_switch)
		{
		  e1 *= e1;
		}
	      all_shapes[i*iterations+current_iteration] = e1;
	    }
	  
	  //Transfer shapes to GPU
	  //CAREFUL HERE...JUST USING dx shapes aray as place holder, THIS DOES NOT MEAN THAT IT WILL CONTAIN BEST SHAPES IN THE END. THIS IS A WEIGHT OPTIMISATION ROUTINE
	  checkCudaErrors(cudaMemcpy2D(cuman->FD_device_pointer("dx_shapes"),sizeof(double),&all_shapes[current_iteration],iterations*sizeof(double),sizeof(double),num_nodes,cudaMemcpyHostToDevice));
	  
	  //Fill current version of A and b with saves ones
	  //Copy static part into dynamical one on device. 
	  checkCudaErrors(cudaMemcpy(d_A,d_A_save,sizeof(double)*matrix_stride*matrix_stride*num_nodes,cudaMemcpyDeviceToDevice));
	  checkCudaErrors(cudaMemcpy(d_b,d_b_save,sizeof(double)*matrix_stride*num_nodes,cudaMemcpyDeviceToDevice));

	  if(factor == 1.0)
	    {
	      cuFD_optimise_shape_dependent_part_dx<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("dx_shapes"),matrix_stride,d_A,d_b,rbf,derivative_order);
	    }
	  else
	    {
	      cuFD_optimise_shape_dependent_part_dx<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("dx_shapes"),matrix_stride,d_A,d_b,rbf,derivative_order,factor);
	    }

	  //Use cuBLAS batch mode to solve all systems	  
	  
	  cublasDgetrfBatched(handle,matrix_stride,d_A_pointers,matrix_stride, d_pivotArray, d_infoArray,num_nodes);
	  cublasDgetrsBatched(handle,CUBLAS_OP_N,matrix_stride,1,(const double**)d_A_pointers,matrix_stride,d_pivotArray,d_b_pointers,matrix_stride,&info,num_nodes);
	  //Final kernel call
	  //Now it depends if nn is power of 2
	  if(nn == 2 || nn == 4 || nn == 8 || nn == 16 || nn == 32 || nn == 64 || nn == 128 || nn == 256)
	    {
	      cuFD_optimise_calculate_and_compare_pow2<<<num_nodes,nn>>>(cuman->FD_device_pointer("function"),d_b,d_truth,d_err, matrix_stride);
	    }
	  else
	    {
	      cuFD_optimise_calculate_and_compare<<<num_nodes,nn>>>(cuman->FD_device_pointer("function"),d_b,d_truth,d_err, matrix_stride);
	    }

//Copy current errors to host before moving to next shape params
	  checkCudaErrors(cudaMemcpy2D(&all_errors[current_iteration],iterations*sizeof(double),d_err,sizeof(double),sizeof(double),num_nodes,cudaMemcpyDeviceToHost));	  
	} //End shape iteration loop

      //Figuring out new shapes
      vector<int> best_eps_index;
      for(int i = 0; i < num_nodes; i++)
	{
	  vector<double>::iterator best_eps_error;
	  best_eps_error = min_element(all_errors.begin()+i*iterations,all_errors.begin()+(i+1)*iterations);
	  best_eps_index.push_back(distance(all_errors.begin()+i*iterations,best_eps_error));
	  if(best_eps_index[i] == 0)
	    {
	      if(square_switch)
		{
		  min_shape[i] = all_shapes[i*iterations];
		  max_shape[i] = all_shapes[i*iterations+1];
		}
	      else
		{
		  min_shape[i] = sqrt(all_shapes[i*iterations]);
		  max_shape[i] = sqrt(all_shapes[i*iterations+1]);
		}		  
	    }
	  else if(best_eps_index[i] == iterations-1)
	    {
	      if(square_switch)
		{
		  min_shape[i] = all_shapes[i*iterations+best_eps_index[i]-1];
		  max_shape[i] = all_shapes[i*iterations+best_eps_index[i]];
		}
	      else
		{
		  min_shape[i] = sqrt(all_shapes[i*iterations+best_eps_index[i]-1]);
		  max_shape[i] = sqrt(all_shapes[i*iterations+best_eps_index[i]]);
		}
	    }
	  else
	    {
	      if(square_switch)
		{
		  min_shape[i] = all_shapes[i*iterations+best_eps_index[i]-1];
		  max_shape[i] = all_shapes[i*iterations+best_eps_index[i]+1];
		}
	      else
		{
		  min_shape[i] = sqrt(all_shapes[i*iterations+best_eps_index[i]-1]);
		  max_shape[i] = sqrt(all_shapes[i*iterations+best_eps_index[i]+1]);
		}
	    }
	}
      if(current_bracket == brackets-1)
	{
	  for(int i = 0; i < num_nodes; i++)
	    {
	      if(square_switch)
		{
		  h_shapes[i] = all_shapes[i*iterations+best_eps_index[i]];
		}
	      else
		{
		  h_shapes[i] = sqrt(all_shapes[i*iterations+best_eps_index[i]]);
		}
	      (*final_errors)[i] = all_errors[i*iterations+best_eps_index[i]];
	    }
	}
    }//End of bracket loop

  //Clearing device memory
  cublasDestroy(handle);
  checkCudaErrors(cudaFree(d_function));
  checkCudaErrors(cudaFree(d_truth));
  checkCudaErrors(cudaFree(d_err));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_b));
  checkCudaErrors(cudaFree(d_A_save));
  checkCudaErrors(cudaFree(d_b_save));
  checkCudaErrors(cudaFree(d_A_pointers));
  checkCudaErrors(cudaFree(d_b_pointers));
  checkCudaErrors(cudaFree(d_pivotArray));
  checkCudaErrors(cudaFree(d_infoArray));
  
  return h_shapes;
};

#endif /* CUDA_FD_H */
