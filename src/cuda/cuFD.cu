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
