/*** /cuda/cuFD.cu
     A collection of host drivers which 
     perform RBF FD operations.

Julian Merten
INAF OA Bologna
Jul 2018
julian.merten@oabo.inaf.it
http://www.julianmerten.net
***/

#include <mfree/cuda/cuFD.h>

void cuFD_differentiate_set(vector<double> function, cuda_manager *cuman)
{
  int num_nodes = cuman->n("nodes");
  int nn = cuman->n("nn");

  //Very simple error checks
  if(function.size() < num_nodes)
    {
      throw invalid_argument("cuFD: Given function too short for managed grid.");
    }

  //Setting device to first one
  checkCudaErrors(cudaSetDevice(cuman->device_query(0).second));

  //Getting f on device. 
  double *d_f;
  checkCudaErrors(cudaMalloc((void **)&d_f,sizeof(double)*num_nodes));
  checkCudaErrors(cudaMemcpy(d_f,&function[0],num_nodes*sizeof(double),cudaMemcpyHostToDevice));

  //Running kernel to set function
  cuFD_optimise_func_eval<<<num_nodes,nn>>>(cuman->index_map_pointer(),d_f,cuman->FD_device_pointer("function"));

  //Distributing function to all devices
  cuman->distribute_FD("function");

  //Free allocated memory
  checkCudaErrors(cudaFree(d_f));

}

vector<double> cuFD_differentiate_and_return(int derivative_selection, cuda_manager *cuman)
{
  string selection;
  switch(derivative_selection)
    {
    case 1: selection = "dx";
    case 2: selection = "dy";
    case 3: selection = "dxx";
    case 4: selection = "dxy";
    case 5: selection = "dxy";
    case 6: selection = "laplace";
    case 7: selection = "neg_laplace";
    }

  //numbers
  int num_nodes = cuman->n("nodes");
  int nn = cuman->n("nn");
  
  //Setting device to first one
  checkCudaErrors(cudaSetDevice(cuman->device_query(0).second));

  if(nn == 2 || nn == 4 || nn == 8 || nn == 16 || nn == 32 || nn == 64 || nn == 128)
    {
      cuFD_differentiate_product_pow2<<<num_nodes,nn>>>(cuman->FD_device_pointer("function"),cuman->FD_device_pointer(selection),cuman->FD_device_pointer("derivative"));
    }
  else
    {
      cuFD_differentiate_product_pow2<<<num_nodes,nn>>>(cuman->FD_device_pointer("function"),cuman->FD_device_pointer(selection),cuman->FD_device_pointer("derivative"));
    }

  //Sending result around to all devices
  cuman->distribute_FD("derivative");

  //Getting derivative off device
  vector<double> result(num_nodes,0.);
  checkCudaErrors(cudaMemcpy(&result[0],cuman->FD_device_pointer("derivative"),num_nodes*sizeof(double),cudaMemcpyDeviceToHost));

  return result;

}


void cuFD_differentiate(int derivative_selection, cuda_manager *cuman)
{
  string selection;
  switch(derivative_selection)
    {
    case 1: selection = "dx";
    case 2: selection = "dy";
    case 3: selection = "dxx";
    case 4: selection = "dxy";
    case 5: selection = "dxy";
    case 6: selection = "laplace";
    case 7: selection = "neg_laplace";
    }

  //numbers
  int num_nodes = cuman->n("nodes");
  int nn = cuman->n("nn");

  //Setting device to first one
  checkCudaErrors(cudaSetDevice(cuman->device_query(0).second));

  //Calculating derivatives in the fastest way depending on nearest neighbours
  if(nn == 2 || nn == 4 || nn == 8 || nn == 16 || nn == 32 || nn == 64 || nn == 128)
    {
      cuFD_differentiate_product_pow2<<<num_nodes,nn>>>(cuman->FD_device_pointer("function"),cuman->FD_device_pointer(selection),cuman->FD_device_pointer("derivative"));
    }
  else
    {
      cuFD_differentiate_product<<<num_nodes,nn>>>(cuman->FD_device_pointer("function"),cuman->FD_device_pointer(selection),cuman->FD_device_pointer("derivative"));
    }
  cuman->distribute_FD("derivative");
}


vector<vector<double> > cuFD_differentiate(cuda_manager *cuman)
{

  //Numbers
  int num_nodes = cuman->n("nodes");
  int nn = cuman->n("nn");

  //Setting device to first one in line
  checkCudaErrors(cudaSetDevice(cuman->device_query(0).second));

  //Setting up output
  vector<vector<double> > result(7,vector<double>(num_nodes,0.));

  if(nn == 2 || nn == 4 || nn == 8 || nn == 16 || nn == 32 || nn == 64 || nn == 128)
    {
     cuFD_differentiate_product_pow2<<<num_nodes,nn>>>(cuman->FD_device_pointer("function"),cuman->FD_device_pointer("dx"),cuman->FD_device_pointer("derivative"));
     checkCudaErrors(cudaMemcpy(&result[0][0],cuman->FD_device_pointer("derivative"),num_nodes*sizeof(double),cudaMemcpyDeviceToHost));
     cuFD_differentiate_product_pow2<<<num_nodes,nn>>>(cuman->FD_device_pointer("function"),cuman->FD_device_pointer("dy"),cuman->FD_device_pointer("derivative"));
     checkCudaErrors(cudaMemcpy(&result[1][0],cuman->FD_device_pointer("derivative"),num_nodes*sizeof(double),cudaMemcpyDeviceToHost));
     cuFD_differentiate_product_pow2<<<num_nodes,nn>>>(cuman->FD_device_pointer("function"),cuman->FD_device_pointer("dxx"),cuman->FD_device_pointer("derivative"));
     checkCudaErrors(cudaMemcpy(&result[2][0],cuman->FD_device_pointer("derivative"),num_nodes*sizeof(double),cudaMemcpyDeviceToHost));
     cuFD_differentiate_product_pow2<<<num_nodes,nn>>>(cuman->FD_device_pointer("function"),cuman->FD_device_pointer("dyy"),cuman->FD_device_pointer("derivative"));
     checkCudaErrors(cudaMemcpy(&result[3][0],cuman->FD_device_pointer("derivative"),num_nodes*sizeof(double),cudaMemcpyDeviceToHost));
     cuFD_differentiate_product_pow2<<<num_nodes,nn>>>(cuman->FD_device_pointer("function"),cuman->FD_device_pointer("dxy"),cuman->FD_device_pointer("derivative"));
     checkCudaErrors(cudaMemcpy(&result[4][0],cuman->FD_device_pointer("derivative"),num_nodes*sizeof(double),cudaMemcpyDeviceToHost));
     cuFD_differentiate_product_pow2<<<num_nodes,nn>>>(cuman->FD_device_pointer("function"),cuman->FD_device_pointer("laplace"),cuman->FD_device_pointer("derivative"));
     checkCudaErrors(cudaMemcpy(&result[5][0],cuman->FD_device_pointer("derivative"),num_nodes*sizeof(double),cudaMemcpyDeviceToHost));
     cuFD_differentiate_product_pow2<<<num_nodes,nn>>>(cuman->FD_device_pointer("function"),cuman->FD_device_pointer("neg_laplace"),cuman->FD_device_pointer("derivative"));
     checkCudaErrors(cudaMemcpy(&result[6][0],cuman->FD_device_pointer("derivative"),num_nodes*sizeof(double),cudaMemcpyDeviceToHost));
    }

 else
   {
     cuFD_differentiate_product<<<num_nodes,nn>>>(cuman->FD_device_pointer("function"),cuman->FD_device_pointer("dx"),cuman->FD_device_pointer("derivative"));
     checkCudaErrors(cudaMemcpy(&result[0][0],cuman->FD_device_pointer("derivative"),num_nodes*sizeof(double),cudaMemcpyDeviceToHost));
     cuFD_differentiate_product<<<num_nodes,nn>>>(cuman->FD_device_pointer("function"),cuman->FD_device_pointer("dy"),cuman->FD_device_pointer("derivative"));
     checkCudaErrors(cudaMemcpy(&result[1][0],cuman->FD_device_pointer("derivative"),num_nodes*sizeof(double),cudaMemcpyDeviceToHost));
     cuFD_differentiate_product<<<num_nodes,nn>>>(cuman->FD_device_pointer("function"),cuman->FD_device_pointer("dxx"),cuman->FD_device_pointer("derivative"));
     checkCudaErrors(cudaMemcpy(&result[2][0],cuman->FD_device_pointer("derivative"),num_nodes*sizeof(double),cudaMemcpyDeviceToHost));
     cuFD_differentiate_product<<<num_nodes,nn>>>(cuman->FD_device_pointer("function"),cuman->FD_device_pointer("dyy"),cuman->FD_device_pointer("derivative"));
     checkCudaErrors(cudaMemcpy(&result[3][0],cuman->FD_device_pointer("derivative"),num_nodes*sizeof(double),cudaMemcpyDeviceToHost));
     cuFD_differentiate_product<<<num_nodes,nn>>>(cuman->FD_device_pointer("function"),cuman->FD_device_pointer("dxy"),cuman->FD_device_pointer("derivative"));
     checkCudaErrors(cudaMemcpy(&result[4][0],cuman->FD_device_pointer("derivative"),num_nodes*sizeof(double),cudaMemcpyDeviceToHost));
     cuFD_differentiate_product<<<num_nodes,nn>>>(cuman->FD_device_pointer("function"),cuman->FD_device_pointer("laplace"),cuman->FD_device_pointer("derivative"));
     checkCudaErrors(cudaMemcpy(&result[5][0],cuman->FD_device_pointer("derivative"),num_nodes*sizeof(double),cudaMemcpyDeviceToHost));
     cuFD_differentiate_product<<<num_nodes,nn>>>(cuman->FD_device_pointer("function"),cuman->FD_device_pointer("neg_laplace"),cuman->FD_device_pointer("derivative"));
     checkCudaErrors(cudaMemcpy(&result[6][0],cuman->FD_device_pointer("derivative"),num_nodes*sizeof(double),cudaMemcpyDeviceToHost));
   }

  return result;

}

void cuFD_test_weight_functions(cuda_manager *cuman, int n)
{
  //CUDA events for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float elapsed_time = 0.;
  float increment = 0.;

  //Creating Gaussian RBF
  ga_rbf *rbf1 = new ga_rbf;
  phs4_rbf *rbf2 = new phs4_rbf;

  //Problem size
  int num_nodes = cuman->n("nodes");
  int nn = cuman->n("nn");
  int pdeg = cuman->n("pdeg");

  vector<double> shapes(num_nodes,1.0);
  int polynomial = (pdeg+1)*(pdeg+2)/2;
  int matrix_stride = nn+polynomial;

  //Setting device to first available.
  checkCudaErrors(cudaSetDevice(cuman->device_query().second));


  //Allocating necessary cublas structures
  cublasHandle_t handle;
  cublasCreate(&handle);
  int *d_pivotArray;
  int *d_infoArray;
  checkCudaErrors(cudaMalloc((void **)&d_pivotArray, sizeof(int)*num_nodes*matrix_stride));
  checkCudaErrors(cudaMalloc((void **)&d_infoArray, sizeof(int)*num_nodes));

  //Getting shapes on device
  checkCudaErrors(cudaMemcpy(cuman->FD_device_pointer("dx_shapes"),&shapes[0],num_nodes*sizeof(double),cudaMemcpyHostToDevice));

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

  //First timing...setting things to 0.

  //Setting lower right block of A to 0.
  //Calling kernel n+1 times and timing, first call is not timed
  for(int i = 0; i < n+1; i++)
    {
      cudaEventRecord(start);
      cuFD_weights_set_zeros<<<num_nodes,polynomial>>>(matrix_stride,polynomial,nn,d_A);  
      cudaEventRecord(stop);
      if(i != 0)
	{
	  cudaEventSynchronize(stop);
	  cudaEventElapsedTime(&increment,start,stop);
	  elapsed_time += increment;
	}
    }
  cout <<"Runtime for setting part of A to 0: " <<elapsed_time/((double) n) <<"msec." <<endl;
  elapsed_time = 0.;

  //Setting matrix part with RBF class implementation
  //Calling kernel n+1 times and timing, first call is not timed
  for(int i = 0; i < n+1; i++)
    {
      cudaEventRecord(start);
      cuFD_weights_matrix_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("dx_shapes"),matrix_stride,pdeg,d_A,rbf1);  
      cudaEventRecord(stop);
      if(i != 0)
	{
	  cudaEventSynchronize(stop);
	  cudaEventElapsedTime(&increment,start,stop);
	  elapsed_time += increment;
	}
    }
  cout <<"Runtime for setting matrix part via class: " <<elapsed_time/((double) n) <<"msec." <<endl;
  elapsed_time = 0.;

  //Setting matrix part with hard-wired implementation
  //Calling kernel n+1 times and timing, first call is not timed
  for(int i = 0; i < n+1; i++)
    {
      cudaEventRecord(start);
      cuFD_ga_weights_matrix_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("dx_shapes"),matrix_stride,pdeg,d_A);  
      cudaEventRecord(stop);
      if(i != 0)
	{
	  cudaEventSynchronize(stop);
	  cudaEventElapsedTime(&increment,start,stop);
	  elapsed_time += increment;
	}
    }
  cout <<"Runtime for setting matrix part via hard-wired imeplentation: " <<elapsed_time/((double) n) <<"msec." <<endl;
  elapsed_time = 0.;

  //Setting matrix part with phs4 implementation
  //Calling kernel n+1 times and timing, first call is not timed
  for(int i = 0; i < n+1; i++)
    {
      cudaEventRecord(start);
      cuFD_weights_matrix_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("dx_shapes"),matrix_stride,pdeg,d_A,rbf2);
      cudaEventRecord(stop);
      if(i != 0)
	{
	  cudaEventSynchronize(stop);
	  cudaEventElapsedTime(&increment,start,stop);
	  elapsed_time += increment;
	}
    }
  cout <<"Runtime for setting matrix part via phs4: " <<elapsed_time/((double) n) <<"msec." <<endl;
  elapsed_time = 0.;

  //Setting vector part with regular class implementation
  //Calling kernel n+1 times and timing, first call is not timed
  for(int i = 0; i < n+1; i++)
    {
      cudaEventRecord(start);
      cuFD_weights_vector_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("dx_shapes"),matrix_stride,d_b,rbf1,1); 
      cudaEventRecord(stop);
      if(i != 0)
	{
	  cudaEventSynchronize(stop);
	  cudaEventElapsedTime(&increment,start,stop);
	  elapsed_time += increment;
	}
    }
  cout <<"Runtime for setting vector part via class: " <<elapsed_time/((double) n) <<"msec." <<endl;
  elapsed_time = 0.;

  //Setting vector part with regular class implementation
  //Calling kernel n+1 times and timing, first call is not timed
  for(int i = 0; i < n+1; i++)
    {
      cudaEventRecord(start);
      cuFD_ga_dx_weights_vector_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("dx_shapes"),matrix_stride,d_b); 
      cudaEventRecord(stop);
      if(i != 0)
	{
	  cudaEventSynchronize(stop);
	  cudaEventElapsedTime(&increment,start,stop);
	  elapsed_time += increment;
	}
    }
  cout <<"Runtime for setting vector part via hard-wired implementation: " <<elapsed_time/((double) n) <<"msec." <<endl;
  elapsed_time = 0.;

  //Setting vector part with regular class implementation
  //Calling kernel n+1 times and timing, first call is not timed
  for(int i = 0; i < n+1; i++)
    {
      cudaEventRecord(start);
      cuFD_weights_vector_part<<<num_nodes,nn>>>(cuman->index_map_pointer(),cuman->coordinate_pointer(),cuman->FD_device_pointer("dx_shapes"),matrix_stride,d_b,rbf2,1); 
      cudaEventRecord(stop);
      if(i != 0)
	{
	  cudaEventSynchronize(stop);
	  cudaEventElapsedTime(&increment,start,stop);
	  elapsed_time += increment;
	}
    }
  cout <<"Runtime for setting vector part via phs4: " <<elapsed_time/((double) n) <<"msec." <<endl;
  elapsed_time = 0.;


}
