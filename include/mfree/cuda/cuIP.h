/*** /cuda/cuda_cuIP.h
     This collects CUDA functions
     related to interpolation 
     on the devices.

Julian Merten
INAF OA Bologna
August 2018
julian.merten@inaf.it
http://www.julianmerten.net
***/

#ifndef    CUDA_CUIP_H
#define    CUDA_CUIP_H

#include <stdexcept>
#include <flann/flann.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <mfree/cuda/cuda_manager.h>
#include <mfree/cuda/cuIP_kernels.h>
#include <mfree/test_functions.h>

using namespace std;

/**
   Compared to differentiation, interpolation on the GPU is 
   much less general to implement since in support grid is
   not the output grid. Currently the decision is hence to provide
   very hig-level routines. In fact there is only one for shape 
   optimsiation and one for the actual interpolation. This 
   also means that the CUDA manager stores no information for the
   interpolation besides the support coordinates. 
**/

/*
  This performs and returns an actual interpolation,
*/

template<class T> vector<double> cuIP_interpolate(vector<double> interpolant, vector<double> interpolation_coordinates, vector<double> shapes, cuda_manager *saw2_manager, T *rbf)
{

  //Numbers
  int num_nodes = saw2_manager->n("nodes");
  int nn = saw2_manager->n("nn");
  int pdeg = saw2_manager->n("pdeg");
  int polynomial = (pdeg+1)*(pdeg+2)/2;
  int matrix_stride = nn+polynomial;
  int num_nodes_interpolation = shapes.size();

  //Quick checks 

  if(num_nodes != interpolant.size())
    {
      throw invalid_argument("cuIP: Interpolant and underlying mesh not consistent.");
    }

  if(interpolation_coordinates.size() != 2*num_nodes_interpolation)
    {
      throw invalid_argument("cuIP: Coordinates and shapes size not consistent.");
    }


  //Setting device to first one in line
  checkCudaErrors(cudaSetDevice(saw2_manager->device_query(0).second));

  //Get mesh coordinates off device
  vector<double> coordinates(num_nodes*2,0.);
  checkCudaErrors(cudaMemcpy(&coordinates[0],saw2_manager->coordinate_pointer(),coordinates.size()*sizeof(double),cudaMemcpyDeviceToHost));

  //Build tree for output node coordinates
  vector<int> h_interpolation_tree;
  vector<double> interpolation_distances;
  h_interpolation_tree.resize(nn*num_nodes_interpolation);
  interpolation_distances.resize(nn*num_nodes_interpolation);
  flann::Matrix<double> flann_dataset(&coordinates[0],num_nodes,2);
  flann::Matrix<double> flann_dataset_interpolation(&interpolation_coordinates[0],num_nodes_interpolation,2);
  flann::Matrix<int> flann_tree(&h_interpolation_tree[0],num_nodes_interpolation,nn);
  flann::Matrix<double> flann_distances(&interpolation_distances[0],num_nodes_interpolation,nn);
  flann::Index<flann::L2<double> > index(flann_dataset, flann::KDTreeIndexParams(4));
  index.buildIndex();
  index.knnSearch(flann_dataset_interpolation, flann_tree, flann_distances, nn, flann::SearchParams(128));

  //Allocate auxilliary memory on device and copying it
  cudaError_t error;
  int *d_interpolation_tree;
  double *d_interpolation_coordinates;
  double *d_interpolant;
  double *d_shapes;
  double *d_interpolation;
  error = cudaMalloc((void**)&d_interpolation_tree,sizeof(int)*h_interpolation_tree.size());
  if(error != cudaSuccess)
    {
      throw invalid_argument("cuIP: Could not allocate tree memory.");
    }
  error = cudaMalloc((void**)&d_interpolation_coordinates,sizeof(double)*2*num_nodes_interpolation);
  if(error != cudaSuccess)
    {
      throw invalid_argument("cuIP: Could not allocate coordinate memory.");
    }
  error = cudaMalloc((void**)&d_interpolant,sizeof(double)*num_nodes);
  if(error != cudaSuccess)
    {
      throw invalid_argument("cuIP: Could not allocate interpolant memory.");
    }
  error = cudaMalloc((void**)&d_shapes,sizeof(double)*num_nodes_interpolation);
  if(error != cudaSuccess)
    {
      throw invalid_argument("cuIP: Could not allocate shapes memory.");
    }
  error = cudaMalloc((void**)&d_interpolation,sizeof(double)*num_nodes_interpolation);
  if(error != cudaSuccess)
    {
      throw invalid_argument("cuIP: Could not allocate result memory.");
    }
  checkCudaErrors(cudaMemcpy(d_interpolation_tree,&h_interpolation_tree[0],sizeof(int)*h_interpolation_tree.size(),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_interpolation_coordinates,&interpolation_coordinates[0],sizeof(double)*interpolation_coordinates.size(),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_interpolant,&interpolant[0],sizeof(double)*interpolant.size(),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_shapes,&shapes[0],sizeof(double)*shapes.size(),cudaMemcpyHostToDevice));

  //Allocating cublas helpers
  cublasHandle_t handle;
  cublasCreate(&handle);
  int *d_pivotArray;
  int *d_infoArray;
  int info;
  checkCudaErrors(cudaMalloc((void **)&d_pivotArray, sizeof(int)*num_nodes*matrix_stride));
  checkCudaErrors(cudaMalloc((void **)&d_infoArray, sizeof(int)*num_nodes));

  //Allocating working memory for weight creation
  double *d_A; //This will hold all coefficient matrices for all nodes
  double *d_b; //This will hold all result vectors for all nodes
  checkCudaErrors(cudaMalloc((void **)&d_A, sizeof(double)*matrix_stride*matrix_stride*num_nodes_interpolation));
  checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(double)*matrix_stride*num_nodes_interpolation));

  //Creating pointer arrays on host for LSE and copying to device
  double **d_A_pointers, **d_b_pointers;
  double *A_pointers[num_nodes_interpolation], *b_pointers[num_nodes_interpolation];
  for(int i = 0; i < num_nodes_interpolation; i++)
    {
      A_pointers[i] = d_A+i*matrix_stride*matrix_stride;
      b_pointers[i] = d_b+i*matrix_stride;
    }
  checkCudaErrors(cudaMalloc((void **)&d_A_pointers,num_nodes_interpolation*sizeof(double*)));
  checkCudaErrors(cudaMalloc((void **)&d_b_pointers,num_nodes_interpolation*sizeof(double*)));
  checkCudaErrors(cudaMemcpy(d_A_pointers,A_pointers,num_nodes_interpolation*sizeof(double *),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_b_pointers,b_pointers,num_nodes_interpolation*sizeof(double *),cudaMemcpyHostToDevice));

  //Build linear systems
  cuIP_matrix_part<<<num_nodes_interpolation,nn>>>(d_interpolation_tree,d_interpolation_coordinates, saw2_manager->coordinate_pointer(),d_shapes,matrix_stride,pdeg,d_A,rbf);
  cuIP_vector_part<<<num_nodes_interpolation,nn>>>(d_interpolation_tree, d_interpolant, matrix_stride, d_b);

  //Solve for weights
  cublasDgetrfBatched(handle,matrix_stride,d_A_pointers,matrix_stride, d_pivotArray, d_infoArray,num_nodes_interpolation);
  cublasDgetrsBatched(handle,CUBLAS_OP_N,matrix_stride,1,(const double**)d_A_pointers,matrix_stride,d_pivotArray,d_b_pointers,matrix_stride,&info,num_nodes_interpolation);

  //Calculating interpolation

  if(nn == 2 || nn == 4 || nn == 8 || nn == 16 || nn == 32 || nn == 64 || nn == 128 || nn == 256)
    {
      cuIP_product_pow2<<<num_nodes_interpolation,nn>>>(d_interpolation_tree, d_interpolation_coordinates,saw2_manager->coordinate_pointer(),d_shapes,d_b,d_interpolation,matrix_stride,rbf);
    }
  else
    {
      cuIP_product<<<num_nodes_interpolation,nn>>>(d_interpolation_tree, d_interpolation_coordinates,saw2_manager->coordinate_pointer(),d_shapes,d_b,d_interpolation,matrix_stride,rbf);
    }

  vector<double> result(num_nodes_interpolation,0.);
  checkCudaErrors(cudaMemcpy(&result[0],d_interpolation,sizeof(double)*num_nodes_interpolation,cudaMemcpyDeviceToHost));

  //Deallocating auxilliary device memory
  cublasDestroy(handle);
  checkCudaErrors(cudaFree(d_interpolation_tree));
  checkCudaErrors(cudaFree(d_interpolation_coordinates));
  checkCudaErrors(cudaFree(d_interpolant));
  checkCudaErrors(cudaFree(d_shapes));
  checkCudaErrors(cudaFree(d_interpolation));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_b));
  checkCudaErrors(cudaFree(d_A_pointers));
  checkCudaErrors(cudaFree(d_b_pointers));
  checkCudaErrors(cudaFree(d_pivotArray));
  checkCudaErrors(cudaFree(d_infoArray));

  return result;
};

template<class T> vector<double> cuIP_optimise_shapes(test_function *reference, vector<double> interpolation_coordinates, vector<double> *final_errors, cuda_manager *saw2_manager, double shape_start, double shape_stop, int brackets, int iterations, T *rbf)
{

  //numbers
  int nn = saw2_manager->n("nn");
  int pdeg = saw2_manager->n("pdeg");
  int num_interpolation_nodes = interpolation_coordinates.size()/2;
  int num_nodes = saw2_manager->n("nodes");
  int polynomial = (pdeg+2)*(pdeg+1)/2;
  int matrix_stride = nn+polynomial;

  //Allocating final result
  vector<double> h_shapes(num_interpolation_nodes,0.);
  final_errors->clear();
  final_errors->resize(num_nodes);

  //Setting device to first one available
  checkCudaErrors(cudaSetDevice(saw2_manager->device_query(0).second));

  //Getting coordinates off device
  vector<double> h_coordinates(num_nodes*2,0.);
  cudaMemcpy(&h_coordinates[0],saw2_manager->coordinate_pointer(),num_nodes*2*sizeof(double),cudaMemcpyDeviceToHost);

  //Creating the interpolation tree
  vector<int> h_interpolation_tree;
  vector<double> interpolation_distances;
  h_interpolation_tree.resize(nn*num_interpolation_nodes);
  interpolation_distances.resize(nn*num_interpolation_nodes);
  flann::Matrix<double> flann_dataset(&h_coordinates[0],num_nodes,2);
  flann::Matrix<double> flann_dataset_interpolation(&interpolation_coordinates[0],num_interpolation_nodes,2);
  flann::Matrix<int> flann_tree(&h_interpolation_tree[0],num_interpolation_nodes,nn);
  flann::Matrix<double> flann_distances(&interpolation_distances[0],num_interpolation_nodes,nn);
  flann::Index<flann::L2<double> > index(flann_dataset, flann::KDTreeIndexParams(4));
  index.buildIndex();
  index.knnSearch(flann_dataset_interpolation, flann_tree, flann_distances, nn, flann::SearchParams(128));

  //Creating the inteprolant values on host
  vector<double> h_interpolant;
  for(int i = 0; i < num_nodes; i++)
    {
      vector<double> current_xy;
      current_xy.push_back(h_coordinates[i*2]);
      current_xy.push_back(h_coordinates[i*2+1]);
      h_interpolant.push_back((*reference)(current_xy));
    }

  //Setting interpolation reference
  vector<double> h_truth;
  for(int i = 0; i < num_interpolation_nodes; i++)
    {
      vector<double> c_pair;
      c_pair.push_back(interpolation_coordinates[i*2]);
      c_pair.push_back(interpolation_coordinates[i*2+1]);
      h_truth.push_back((*reference)(c_pair));
    }

  //In-routine device memory pointers
  cudaError_t error;
  int *d_interpolation_tree;
  double *d_interpolation_coordinates;
  double *d_coordinates; //Interpolant coordinates
  double *d_interpolant;
  double *d_truth;
  double *d_err;
  double *d_shapes;
  double *d_A; //This will hold all coefficient matrices for all nodes
  double *d_b; //This will hold all result vectors for all nodes
  double *d_A_save; //This will save the constant part of the coefficient matrix which will only have to be calculated once. 
  
  //In-routine device memory allocations
  checkCudaErrors(cudaMalloc((void **)&d_interpolation_coordinates, sizeof(double)*interpolation_coordinates.size()));
  checkCudaErrors(cudaMalloc((void **)&d_interpolation_tree, sizeof(double)*h_interpolation_tree.size()));
  checkCudaErrors(cudaMalloc((void **)&d_interpolant, sizeof(double)*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_coordinates, sizeof(double)*2*num_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_truth, sizeof(double)*num_interpolation_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_err, sizeof(double)*num_interpolation_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_shapes, sizeof(double)*num_interpolation_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_A, sizeof(double)*matrix_stride*matrix_stride*num_interpolation_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_b, sizeof(double)*matrix_stride*num_interpolation_nodes));
  checkCudaErrors(cudaMalloc((void **)&d_A_save, sizeof(double)*matrix_stride*matrix_stride*num_interpolation_nodes));
 //Device memory pointers to linear system arrays, created on host and copied to device
  double **d_A_pointers;
  double **d_b_pointers;
  double *A_pointers[num_interpolation_nodes];
  double *b_pointers[num_interpolation_nodes];
  for(int i = 0; i < num_interpolation_nodes; i++)
    {
      A_pointers[i] = d_A+i*matrix_stride*matrix_stride;
      b_pointers[i] = d_b+i*matrix_stride;
    }
  checkCudaErrors(cudaMalloc((void **)&d_A_pointers,num_interpolation_nodes*sizeof(double*)));
  checkCudaErrors(cudaMalloc((void **)&d_b_pointers,num_interpolation_nodes*sizeof(double*)));
  checkCudaErrors(cudaMemcpy(d_interpolation_coordinates,&interpolation_coordinates[0],sizeof(double)*interpolation_coordinates.size(),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_interpolation_tree,&h_interpolation_tree[0],sizeof(double)*h_interpolation_tree.size(),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_coordinates,&h_coordinates[0],sizeof(double)*h_coordinates.size(),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_interpolant,&h_interpolant[0],sizeof(double)*h_interpolant.size(),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_truth,&h_truth[0],sizeof(double)*h_truth.size(),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_A_pointers,A_pointers,num_interpolation_nodes*sizeof(double *),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_b_pointers,b_pointers,num_interpolation_nodes*sizeof(double *),cudaMemcpyHostToDevice));

  //Before entering shape loop, calculating the shape-independent parts of the linear system array
  cuIP_optimise_const_part<<<num_interpolation_nodes,nn>>>(d_interpolation_tree,d_interpolation_coordinates, d_interpolant,saw2_manager->coordinate_pointer(), matrix_stride, pdeg,d_A_save,d_b);

  //Allocating necessary cublas structures
  cublasHandle_t handle;
  cublasCreate(&handle);
  int *d_pivotArray;
  int *d_infoArray;
  int info;
  checkCudaErrors(cudaMalloc((void **)&d_pivotArray, sizeof(int)*num_interpolation_nodes*matrix_stride));
  checkCudaErrors(cudaMalloc((void **)&d_infoArray, sizeof(int)*num_interpolation_nodes));
  
  //Allocating quantities tracking the current errors on derivative
  vector<double> all_errors(num_interpolation_nodes*iterations,-1.); //Vector containing all errors per node for all iterations per bracket
  vector<double> all_shapes(num_interpolation_nodes*iterations,0.);
  vector<double> min_shape(num_interpolation_nodes, shape_start);
  vector<double> max_shape(num_interpolation_nodes, shape_stop);
  vector<double> increment(num_interpolation_nodes,0.);

  //Start loop over shapes
  for(int current_bracket = 0; current_bracket < brackets; current_bracket++)
    {
      //Setting limits according to full iterations
      for(int i = 0; i < num_interpolation_nodes; i++)
	{
	  increment[i] = (max_shape[i] - min_shape[i])/(iterations);
	}
      for(int current_iteration = 0; current_iteration < iterations; current_iteration++)
	{
	  //Figure out which is current shape for all nodes.
	  for(int i = 0; i < num_interpolation_nodes; i++)
	    {
	      double e1 = current_iteration*increment[i] + min_shape[i];
	      all_shapes[i*iterations+current_iteration] = e1*e1;
	    }

	  //Transfer shapes to GPU
	  //CAREFUL HERE...JUST USING dx shapes aray as place holder, THIS DOES NOT MEAN THAT IT WILL CONTAIN BEST SHAPES IN THE END. THIS IS A WEIGHT OPTIMISATION ROUTINE
	  checkCudaErrors(cudaMemcpy2D(d_shapes,sizeof(double),&all_shapes[current_iteration],iterations*sizeof(double),sizeof(double),num_interpolation_nodes,cudaMemcpyHostToDevice));
	  
	  //Fill current version of A and b with saves ones
	  //Copy static part into dynamical one on device. 
	  checkCudaErrors(cudaMemcpy(d_A,d_A_save,sizeof(double)*matrix_stride*matrix_stride*num_interpolation_nodes,cudaMemcpyDeviceToDevice));

	  //Filling shape-dependent part of LSE
	  cuIP_optimise_shape_dependent_part<<<num_interpolation_nodes,nn>>>(d_interpolation_tree,d_interpolation_coordinates,d_coordinates,d_shapes,matrix_stride,d_A,rbf);

	  //Use cuBLAS batch mode to solve all systems	  
	  cublasDgetrfBatched(handle,matrix_stride,d_A_pointers,matrix_stride, d_pivotArray, d_infoArray,num_interpolation_nodes);
	  cublasDgetrsBatched(handle,CUBLAS_OP_N,matrix_stride,1,(const double**)d_A_pointers,matrix_stride,d_pivotArray,d_b_pointers,matrix_stride,&info,num_interpolation_nodes);

	  //Final kernel call
	  //Now it depends if nn is power of 2
	  if(nn == 2 || nn == 4 || nn == 8 || nn == 16 || nn == 32 || nn == 64 || nn == 128 || nn == 256)
	    {
	      cuIP_optimise_calculate_IP_and_compare_pow2<<<num_interpolation_nodes,nn>>>(d_interpolation_tree,d_interpolation_coordinates,d_coordinates,d_shapes,d_b,d_truth,d_err, matrix_stride,rbf);
	    }
	  else
	    {
	      cuIP_optimise_calculate_IP_and_compare<<<num_interpolation_nodes,nn>>>(d_interpolation_tree,d_interpolation_coordinates,d_coordinates,d_shapes,d_b,d_truth,d_err, matrix_stride,rbf);
	    }

	  //Copy current errors to host before moving to next shape params
	  checkCudaErrors(cudaMemcpy2D(&all_errors[current_iteration],iterations*sizeof(double),d_err,sizeof(double),sizeof(double),num_interpolation_nodes,cudaMemcpyDeviceToHost));	 
	}//End of shape iteration loop

     //Figuring out new shapes
      int best_eps_index;
      for(int i = 0; i < num_interpolation_nodes; i++)
	{
	  vector<double>::iterator best_eps_error;
	  best_eps_error = min_element(all_errors.begin()+i*iterations,all_errors.begin()+(i+1)*iterations);
	  best_eps_index = distance(all_errors.begin()+i*iterations,best_eps_error);
	  if(best_eps_index == 0)
	    {
	      min_shape[i] = sqrt(all_shapes[i*iterations]);
	      max_shape[i] = sqrt(all_shapes[i*iterations+1]);		  
	    }
	  else if(best_eps_index == iterations-1)
	    {
	      min_shape[i] = sqrt(all_shapes[i*iterations+best_eps_index-1]);
	      max_shape[i] = sqrt(all_shapes[i*iterations+best_eps_index]);
	    }
	  else
	    {
	      min_shape[i] = sqrt(all_shapes[i*iterations+best_eps_index-1]);
	      max_shape[i] = sqrt(all_shapes[i*iterations+best_eps_index+1]);
	    }
	}
      if(current_bracket == brackets-1)
	{
	  for(int i = 0; i < num_interpolation_nodes; i++)
	    {
	      h_shapes[i] = sqrt(all_shapes[i*iterations+best_eps_index]);
	      (*final_errors)[i] = all_errors[i*iterations+best_eps_index];
	    }
	}
    } //End of bracket loop

  //Clearing device memory
  cublasDestroy(handle);
  checkCudaErrors(cudaFree(d_interpolation_coordinates));
  checkCudaErrors(cudaFree(d_interpolation_tree));
  checkCudaErrors(cudaFree(d_interpolant));
  checkCudaErrors(cudaFree(d_coordinates));
  checkCudaErrors(cudaFree(d_truth));
  checkCudaErrors(cudaFree(d_err));
  checkCudaErrors(cudaFree(d_shapes));
  checkCudaErrors(cudaFree(d_A));
  checkCudaErrors(cudaFree(d_b));
  checkCudaErrors(cudaFree(d_A_save));
  checkCudaErrors(cudaFree(d_A_pointers));
  checkCudaErrors(cudaFree(d_b_pointers));
  checkCudaErrors(cudaFree(d_pivotArray));
  checkCudaErrors(cudaFree(d_infoArray));



  return h_shapes;

};


#endif /* CUDA_CUIP_H */
