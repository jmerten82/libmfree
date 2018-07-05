/*** /cuda/cuda_manager.cu
     This provides the base class
     which handles CUDA device and mainly
     memory management. 

Julian Merten
INAF OA Bologna
July 2018
julian.merten@inaf.it
http://www.julianmerten.net
***/

#include <mfree/cuda/cuda_manager.h>

using namespace std;

__constant__ double CONSTblock[CONST_MEM_BLOCK_SIZE];

cuda_manager::cuda_manager(double compute, double cuda)
{
  specs.resize(2);
  specs[0] = compute;
  specs[1] = cuda;


  //Counting CUDA devices
  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if(error_id != cudaSuccess)
    {
      throw invalid_argument("CUDA_MAN: DeviceCount failed.");
    }

  //Querying CUDA runtime info
  cuda_runtime_environment.resize(2);
  cuda_runtime_environment_h.resize(2);
  error_id = cudaDriverGetVersion(&cuda_runtime_environment[0]);
  if(error_id != cudaSuccess)
    {
      throw invalid_argument("CUDA_MAN: Driver query failed.");
    }
  error_id = cudaRuntimeGetVersion(&cuda_runtime_environment[1]);
  if(error_id != cudaSuccess)
    {
      throw invalid_argument("CUDA_MAN: Driver query failed.");
    }
  stringstream h_version;
  double out;
  h_version <<cuda_runtime_environment[0]/1000 <<"." <<(cuda_runtime_environment[0]%100)/10;
  h_version >>out;
  cuda_runtime_environment_h[0] = out;
  h_version.clear();
  h_version <<cuda_runtime_environment[1]/1000 <<"." <<(cuda_runtime_environment[1]%100)/10;
  h_version >>out;
  cuda_runtime_environment_h[1] = out;


  //This is the case where nothing needs to be done. CUDA RT too old. 
  if(cuda_runtime_environment_h[1] < specs[1])
    {
      my_devices.resize(0);
    }

  else
    {

      //Querying CUDA device propoerties
      for(int dev = 0; dev < deviceCount; dev++)
	{
	  cudaDeviceProp deviceProp;
	  
	  error_id = cudaSetDevice(dev);
	  if(error_id != cudaSuccess)
	    {
	      throw invalid_argument("CUDA_MAN: Setting device failed.");
	    }
	  error_id = cudaDeviceReset();
	  if(error_id != cudaSuccess)
	    {
	      throw invalid_argument("CUDA_MAN: Flushing device failed.");
	    }
	  
	  error_id = cudaGetDeviceProperties(&deviceProp, dev);
	  if(error_id != cudaSuccess)
	    {
	      throw invalid_argument("CUDA_MAN: Querying device failed.");
	    }
	  //Checking if device fulfills requirements
	  stringstream cc;
	  double cc_h;
	  stringstream cc_num;
	  cc_num <<deviceProp.major <<"." <<deviceProp.minor;
	  cc_num >>cc_h;
	  if(cc_h >= specs[0])
	    {
	      pair<cudaDeviceProp,int> devPair;
	      devPair.first = deviceProp;
	      devPair.second = dev;
	      my_devices.push_back(devPair);
	    }
	}    
    }//Case if CUDA RT constraint is fulfilled

  num_nodes = 0;
  nn = 0;
  pdeg = -1;
  FD_allocations = false;
  for(int i = 0; i < my_devices.size(); i++)
    {
      FD_pointer_array another_new_struct;
      FD_pointers.push_back(another_new_struct);
      int *p = NULL;
      double *p2 = NULL;
      index_maps.push_back(p);
      coordinates.push_back(p2);
    }

  FD_weights = vector<bool>(7,false);
}

string cuda_manager::report(bool to_console)
{

  cudaError_t error;
  int current_device;
  error = cudaGetDevice(&current_device);
  if(error != cudaSuccess)
    {
      throw invalid_argument("CUDA_MAN: Get device call failed.");
    }

  ostringstream info_stream;

  info_stream <<"This is a libsaw2 cuda manager." <<endl;
  info_stream <<"CUDA-Driver version: " <<cuda_runtime_environment_h[0] <<endl;
  info_stream <<"CUDA-Runtime version: " <<cuda_runtime_environment_h[1] <<endl;
  info_stream <<"Number of CUDA devices: " <<my_devices.size() <<endl;
  info_stream <<"Currently active device: " << current_device <<endl;
  info_stream <<endl;

  unsigned long long perf, mem;
  double cc; 
  int sm_per_multiproc;

  for(unsigned int i = 0; i < my_devices.size(); i++)
    {
      stringstream cc_num;
      sm_per_multiproc = _ConvertSMVer2Cores(my_devices[i].first.major, my_devices[i].first.minor);
      perf = (unsigned long long) my_devices[i].first.multiProcessorCount * sm_per_multiproc * my_devices[i].first.clockRate;
      cc_num <<my_devices[i].first.major <<"." <<my_devices[i].first.minor;
      cc_num >>cc;
      mem = my_devices[i].first.totalGlobalMem;
      info_stream <<"Device " <<i <<": " <<my_devices[i].first.name <<" | " <<"Compute capability: " <<cc <<" | " <<"Total global memory: " <<mem/1048576.0f <<" Mbyte" <<" | " <<"Peak performance: " <<perf*1e-6 <<" GFLOPS" <<endl;
      info_stream <<endl;
    }

  if(to_console)
    {
      cout <<info_stream.str() <<endl;
    }

  return info_stream.str();

}

bool cuda_manager::check()
{

  return (my_devices.size() > 0);

}

vector<int> cuda_manager::criterion_query(string criterion, int size)
{

  if(size > my_devices.size())
    {
      size = my_devices.size();
    }

  vector<int> output;
  vector<pair<double,int> > list;

  if(my_devices.size() == 1)
    {
      output.push_back(my_devices[0].second);
    }
  else
    {
      if(criterion == "performance")
	{
	  int sm_per_multiproc  = 0;
	  unsigned long long current_compute_perf = 0;
	  for(unsigned int i = 0; i < my_devices.size(); i++)
	    {
	      sm_per_multiproc = _ConvertSMVer2Cores(my_devices[i].first.major, my_devices[i].first.minor);
	      current_compute_perf = (unsigned long long) my_devices[i].first.multiProcessorCount * sm_per_multiproc * my_devices[i].first.clockRate;
	      pair<double, int> current_pair((double) current_compute_perf,my_devices[i].second);
	      list.push_back(current_pair);
	    }
	}

      if(criterion == "compute_capability")
	{
	  double current_cc = 0.;

	  
	  for(unsigned int i = 0; i < my_devices.size(); i++)
	    {
	      stringstream cc_num;
	      cc_num <<my_devices[i].first.major <<"." <<my_devices[i].first.minor;
	      cc_num >>current_cc;
	      pair<double, int> current_pair(current_cc,my_devices[i].second);
	      list.push_back(current_pair);
	    }
	}

      if(criterion == "memory")
	{
	  unsigned long long current_mem = 0; 
	  for(unsigned int i = 0; i < my_devices.size(); i++)
	    {
	      current_mem = my_devices[i].first.totalGlobalMem;
	      pair<double, int> current_pair(current_mem,my_devices[i].second);
	      list.push_back(current_pair);
	    }
	}
      //Sorting list
      sort(list.begin(),list.end());
      reverse(list.begin(),list.end());
      int counter = 0;
      while(counter < size)
	{
	  output.push_back(list[counter].second);
	  counter++;
	}
    } //end of case my_devices.size() != 1

  return output;
}

pair<cudaDeviceProp,int> cuda_manager::device_query(int index)
{
  if(index < my_devices.size())
    {
      return my_devices[index]; 
    }
  else
    {
      throw invalid_argument("CUDA_MAN: Invalid device selection, index out of range.");
    }
} 

void cuda_manager::reset(mesh_free_2D *input, int pdeg_in)
{

  cudaError_t error;
  deallocate_FD_builder_memory();

  if(num_nodes != 0)
    {
      for(int i = 0; i < my_devices.size(); i++)
	{
	  error = cudaSetDevice(my_devices[i].second);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Cannot set device.");
	    }
	  cudaFree(index_maps[i]);
	  cudaFree(coordinates[i]);
	}
    }
  num_nodes = input->return_grid_size();
  cout <<"Setting all devices to handle " <<num_nodes <<" nodes." <<endl;
  vector<int> neighbours = input->neighbours();
  vector<double> mesh_coordinates;
  for(int i = 0; i < num_nodes; i++)
    {
      mesh_coordinates.push_back((*input)(i,0));
      mesh_coordinates.push_back((*input)(i,1));
    }

  nn = neighbours.size()/num_nodes;
  pdeg = pdeg_in;

  for(int i = 0; i < my_devices.size(); i++)
    {
      error = cudaSetDevice(my_devices[i].second);
      if(error != cudaSuccess)
	{
	  throw runtime_error("CUDA MAN: Cannot set device.");
	}
      error = cudaMalloc((void**)&index_maps[i],sizeof(int)*num_nodes*MAX_NN);
      if(error != cudaSuccess)
	{
	  throw runtime_error("CUDA MAN: Could not allocate device memory for index map.");
	}
      error = cudaMalloc((void**)&coordinates[i],sizeof(double)*2*num_nodes);
      if(error != cudaSuccess)
	{
	  throw runtime_error("CUDA MAN: Could not allocate device memory for index map.");
	}
      error = cudaMemcpy(index_maps[i],&neighbours[0],sizeof(int)*num_nodes*nn,cudaMemcpyHostToDevice);
      if(error != cudaSuccess)
	{
	  throw runtime_error("CUDA MAN: Could not copy index map onto device.");      
	}
      error = cudaMemcpy(coordinates[i],&mesh_coordinates[0],sizeof(double)*2*num_nodes,cudaMemcpyHostToDevice);
      if(error != cudaSuccess)
	{
	  throw runtime_error("CUDA MAN: Could not copy coordinates onto device.");      
	}

      size_t free, total;
      error = cudaMemGetInfo(&free,&total);
      if(error != cudaSuccess)
	{
	  throw runtime_error("CUDA MAN: Could not query memory status");
	}
      cout <<"Device " <<my_devices[i].second <<" set."<<"Device's memory status: " <<free <<"/" <<total <<" Bytes free." <<endl;
    }
}

void cuda_manager::reset()
{
  cudaError_t error;
  deallocate_FD_builder_memory();
  
  if(num_nodes != 0)
    {
      for(int i = 0; i < my_devices.size(); i++)
	{
	  error = cudaSetDevice(my_devices[i].second);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Cannot set device.");
	    }
	  cudaFree(index_maps[i]);
	  cudaFree(coordinates[i]);
	}
    }
  num_nodes = 0;
  nn = 0;
  pdeg = -1;
  
  cout <<"ALl device memory clear." <<endl;
  size_t free, total;
  for(int i = 0; i < my_devices.size(); i++)
    {
      error = cudaSetDevice(my_devices[i].second);
      if(error != cudaSuccess)
	{
	  throw runtime_error("CUDA MAN: Cannot set device.");
	}
      error = cudaMemGetInfo(&free,&total);
      if(error != cudaSuccess)
	{
	  throw runtime_error("CUDA MAN: Could not query memory status");
	}
      cout <<"Device " <<my_devices[i].second <<" set."<<"Device's memory status: " <<free <<"/" <<total <<" Bytes free." <<endl;
    }
}

int cuda_manager::n(string selection)
{

  if(selection == "nn")
    {
      return nn;
    }
  else if(selection == "nodes")
    {
      return num_nodes;
    }
  else if(selection == "pdeg")
    {
      return pdeg;
    }
  else
    {
      return my_devices.size();
    }
}

double* cuda_manager::coordinate_pointer()
{
  cudaError_t  error;

  if(num_nodes == 0)
    {
      throw invalid_argument("CUDA MAN: No coordinates allocated. Zero nodes.");
    }

  int current_device;
  error = cudaGetDevice(&current_device);
  if(error != cudaSuccess)
    {
      throw runtime_error("CUDA_MAN: Could not set device.");
    }
  int index = -1;
  for(int i = 0; i < my_devices.size(); i++)
    {
      if(my_devices[i].second == current_device)
	{
	  index = i;
	  break;
	}
    }
  if(index < 0)
    {
      throw invalid_argument("CUDA_MAN: Invalid coordinate pointer requested. Possibly, device not under management.");
    }

  return coordinates[index];
}

int* cuda_manager::index_map_pointer()
{
  cudaError_t  error;

  if(num_nodes == 0)
    {
      throw invalid_argument("CUDA MAN: No index maps allocated. Zero nodes.");
    }

  int current_device;
  error = cudaGetDevice(&current_device);
  if(error != cudaSuccess)
    {
      throw runtime_error("CUDA MAN: Could not set device.");
    }
  int index = -1;
  for(int i = 0; i < my_devices.size(); i++)
    {
      if(my_devices[i].second == current_device)
	{
	  index = i;
	  break;
	}
    }
  if(index < 0)
    {
      throw invalid_argument("CUDA_MAN: Invalid index map  pointer requested. Possibly, device not under management.");
    }
  return index_maps[index];
}

double* cuda_manager::const_memory_block_pointer(string symbol)
{
  cudaError_t error;

  double *const_mem_pointer;
  if(symbol == "")
    {
      error = cudaGetSymbolAddress((void**)&const_mem_pointer,CONSTblock);
    }
  else
    {
      error = cudaGetSymbolAddress((void**)&const_mem_pointer,symbol.c_str());
    }

  if(error != cudaSuccess)
    {
      throw runtime_error("CUDA_MAN: Could not retrieve const mem pointer.");
    }

  return const_mem_pointer;
}

void cuda_manager::allocate_FD_builder_memory()
{
  if(!FD_allocations)
    {
      long int nodes1_5 = num_nodes * MAX_NN;
      size_t num_bytes = (8*num_nodes+8*nodes1_5)*sizeof(double);
      cudaError_t error;
      
      for(int i = 0; i < my_devices.size(); i++)
	{
	  error = cudaSetDevice(my_devices[i].second);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not set device.");
	    }
	  cout <<"Allocating "<<num_bytes <<" Bytes of device memory on device: " <<my_devices[i].second <<"..." <<flush; 
	  
	  error = cudaMalloc((void**)&FD_pointers[i].dx_shapes,sizeof(double)*num_nodes);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not allocate dx_shapes memory");
	    }
	  error = cudaMalloc((void**)&FD_pointers[i].dy_shapes,sizeof(double)*num_nodes);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not allocate dy_shapes memory");
	    }
	  error = cudaMalloc((void**)&FD_pointers[i].dxx_shapes,sizeof(double)*num_nodes);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not allocate dxx_shapes memory");
	    }
	  error = cudaMalloc((void**)&FD_pointers[i].dyy_shapes,sizeof(double)*num_nodes);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not allocate dyy_shapes memory");
	    }
	  error = cudaMalloc((void**)&FD_pointers[i].dxy_shapes,sizeof(double)*num_nodes);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not allocate dxy_shapes memory");
	    }
	  error = cudaMalloc((void**)&FD_pointers[i].laplace_shapes,sizeof(double)*num_nodes);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not allocate laplace_shapes memory");
	    }
	  error = cudaMalloc((void**)&FD_pointers[i].neg_laplace_shapes,sizeof(double)*num_nodes);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not allocate neg_laplace_shapes memory");
	    }
	  error = cudaMalloc((void**)&FD_pointers[i].function,sizeof(double)*nodes1_5);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not allocate FD input function memory");
	    }
	  error = cudaMalloc((void**)&FD_pointers[i].derivative,sizeof(double)*num_nodes);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not allocate FD result memory");
	    }
	  error = cudaMalloc((void**)&FD_pointers[i].dx,sizeof(double)*nodes1_5);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not allocate dx memory");
	    }
	  error = cudaMalloc((void**)&FD_pointers[i].dy,sizeof(double)*nodes1_5);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not allocate dy memory");
	    }
	  error = cudaMalloc((void**)&FD_pointers[i].dxx,sizeof(double)*nodes1_5);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not allocate dxx memory");
	    }
	  error = cudaMalloc((void**)&FD_pointers[i].dyy,sizeof(double)*nodes1_5);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not allocate dyy memory");
	    }
	  error = cudaMalloc((void**)&FD_pointers[i].dxy,sizeof(double)*nodes1_5);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not allocate dxy memory");
	    }
	  error = cudaMalloc((void**)&FD_pointers[i].laplace,sizeof(double)*nodes1_5);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not allocate half_laplace memory");
	    }
	  error = cudaMalloc((void**)&FD_pointers[i].neg_laplace,sizeof(double)*nodes1_5);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not allocate neg_half_laplace memory");
	    }
	  
	  cout <<"done." <<endl;
	  size_t free, total;
	  error = cudaMemGetInfo(&free,&total);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not query memory status");
	    }
	  cout <<"Device memory status: " <<free <<"/" <<total <<" Bytes free." <<endl;
	}
      
      FD_allocations = true;
    }
  else
    {
      cout <<"FD memory already allocated on devices. Nothing to be done." <<endl;
    }
}

void cuda_manager::deallocate_FD_builder_memory()
{
  if(FD_allocations)
    {
      cudaError_t error;
      for(int i = 0; i < my_devices.size(); i++)
	{
	  error = cudaSetDevice(my_devices[i].second);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not set device.");
	    }
	  error = cudaFree(FD_pointers[i].dx_shapes);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not free dx_shapes.");
	    }
	  error = cudaFree(FD_pointers[i].dy_shapes);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not free dy_shapes.");
	    }
	  error = cudaFree(FD_pointers[i].dxx_shapes);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not free dxx_shapes.");
	    }
	  error = cudaFree(FD_pointers[i].dyy_shapes);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not free dyy_shapes.");
	    }
	  error = cudaFree(FD_pointers[i].dxy_shapes);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not free dxy_shapes.");
	    }
	  error = cudaFree(FD_pointers[i].laplace_shapes);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not free laplace_shapes.");
	    }
	  error = cudaFree(FD_pointers[i].neg_laplace_shapes);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not free neg_laplace_shapes.");
	    }
	  error = cudaFree(FD_pointers[i].dx);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not free dx.");
	    }
	  error = cudaFree(FD_pointers[i].dy);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not free dy.");
	    }
	  error = cudaFree(FD_pointers[i].dxx);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not free dxx.");
	    }
	  error = cudaFree(FD_pointers[i].dyy);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not free dyy.");
	    }
	  error = cudaFree(FD_pointers[i].dxy);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not free dxy.");
	    }
	  error = cudaFree(FD_pointers[i].laplace);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not free half_laplace.");
	    }
	  error = cudaFree(FD_pointers[i].neg_laplace);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not free neg_half_laplace.");
	    }
	  error = cudaFree(FD_pointers[i].function);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not free FD input function.");
	    }
	  error = cudaFree(FD_pointers[i].derivative);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not free FD result.");
	    }
	  size_t free, total;
	  error = cudaMemGetInfo(&free,&total);
	  if(error != cudaSuccess)
	    {
	      throw runtime_error("CUDA MAN: Could not query memory status");
	    }
	  cout <<"Cleared FD builder memory on device: " <<my_devices[i].second <<". Current device memory status: " <<free <<"/" <<total <<" Bytes free." <<endl;
	}
      FD_allocations = false;
      for(int i = 0; i < FD_weights.size(); i++)
	{
	  FD_weights[i] = false;
	}
    }
}

double* cuda_manager::FD_device_pointer(string selection)
{
  double *out = NULL;
  if(!FD_allocations)
    {
      throw invalid_argument("CUDA_MAN: Cannot request FD pointer. Not allocated.");
    }
  int current_device;
  cudaError_t error;
  error = cudaGetDevice(&current_device);
  if(error !=  cudaSuccess)
    {
      throw invalid_argument("CUDA MAN: Cannot get current device ID.");
    }

  int index = -1;
  for(int i = 0; i < my_devices.size(); i++)
    {
      if(my_devices[i].second == current_device)
	{
	  index = i;
	  break;
	}
    }
  if(index < 0)
    {
      throw invalid_argument("CUDA_MAN: Invalid LSE pointer requested. Possibly, device not under management.");
    }

  if(selection == "dx_shapes")
    {
      out = FD_pointers[current_device].dx_shapes;
    }
  if(selection == "dy_shapes")
    {
      out = FD_pointers[current_device].dy_shapes;
    }
  else if(selection == "dxx_shapes")
    {
      out = FD_pointers[current_device].dxx_shapes;
    }
  else if(selection == "dyy_shapes")
    {
      out = FD_pointers[current_device].dyy_shapes;
    }
  else if(selection == "dxy_shapes")
    {
      out = FD_pointers[current_device].dxy_shapes;
    }
  else if(selection == "laplace_shapes")
    {
      out = FD_pointers[current_device].laplace_shapes;
    }
  else if(selection == "neg_laplace_shapes")
    {
      out = FD_pointers[current_device].neg_laplace_shapes;
    }
  else if(selection == "dx")
    {
      out = FD_pointers[current_device].dx;
    }
  else if(selection == "dy")
    {
      out = FD_pointers[current_device].dy;
    }
  else if(selection == "dxx")
    {
      out = FD_pointers[current_device].dxx;
    }
  else if(selection == "dyy")
    {
      out = FD_pointers[current_device].dyy;
    }
  else if(selection == "dxy")
    {
      out = FD_pointers[current_device].dxy;
    }
  else if(selection == "laplace")
    {
      out = FD_pointers[current_device].laplace;
    }
  else if(selection == "neg_laplace")
    {
      out = FD_pointers[current_device].neg_laplace;
    }
  else if(selection == "function")
    {
      out = FD_pointers[current_device].function;
    }
  else if(selection == "derivative")
    {
      out = FD_pointers[current_device].derivative;
    }
  else
    {
      throw invalid_argument("CUDA MAN: Invalid selection for FD pointer return.");
    }
  return out;
}

bool cuda_manager::FD_weights_status(string selection)
{
  bool answer = false;
  if(selection == "dx")
    {
      answer = FD_weights[0];
    }
  else if(selection == "dy")
    {
      answer = FD_weights[1];
    }
  else if(selection == "dxx")
    {
      answer = FD_weights[2];
    }
  else if(selection == "dyy")
    {
      answer = FD_weights[3];
    }
  else if(selection == "dxy")
    {
      answer = FD_weights[4];
    }
  else if(selection == "laplace")
    {
      answer = FD_weights[5];
    }
  else if(selection == "neg_laplace")
    {
      answer = FD_weights[6];
    }
  else
    {
      throw invalid_argument("CUDA MAN: Invalid selection for FD_weights_status.");
    }
  return answer;
}

void cuda_manager::switch_FD_weights_status(string selection)
{
  if(selection == "dx")
    {
      FD_weights[0] = !FD_weights[0];
    }
  else if(selection == "dy")
    {
      FD_weights[1] = !FD_weights[1];
    }
  else if(selection == "dxx")
    {
      FD_weights[2] = !FD_weights[2];
    }
  else if(selection == "dyy")
    {
      FD_weights[3] = !FD_weights[3];
    }
  else if(selection == "dxy")
    {
      FD_weights[4] = !FD_weights[4];
    }
  else if(selection == "laplace")
    {
      FD_weights[5] = !FD_weights[5];
    }
  else if(selection == "neg_laplace")
    {
      FD_weights[6] = !FD_weights[6];
    }
  else
    {
      throw invalid_argument("CUDA MAN: Invalid selection for FD_weights_status switch.");
    }
}

void cuda_manager::distribute_FD_weights(string selection, int src)
{
  cudaError_t error;
  size_t copy_size = sizeof(double)*num_nodes * MAX_NN;
  if(selection == "dx")
    {
      if(!this->FD_weights_status(selection))
	{
	  throw invalid_argument("CUDA_MAN: Cannot distribute non-existing weights.");
	}
      for(int i = 0; i < my_devices.size(); i++)
	{
	  if(my_devices[i].second != my_devices[src].second)
	    {
	      error = cudaMemcpyPeer(FD_pointers[i].dx,my_devices[i].second,FD_pointers[src].dx,my_devices[src].second,copy_size);
	      if(error != cudaSuccess)
		{
		  throw runtime_error("CUDA_MAN: Could not distribute dx.");
		}
	    }
	}
    }
  else if(selection == "dy")
    {
      if(!this->FD_weights_status(selection))
	{
	  throw invalid_argument("CUDA_MAN: Cannot distribute non-existing weights.");
	}
      for(int i = 0; i < my_devices.size(); i++)
	{
	  if(my_devices[i].second != my_devices[src].second)
	    {
	      error = cudaMemcpyPeer(FD_pointers[i].dy,my_devices[i].second,FD_pointers[src].dy,my_devices[src].second,copy_size);
	      if(error != cudaSuccess)
		{
		  throw runtime_error("CUDA_MAN: Could not distribute dy.");
		}
	    }
	}
    }
  else if(selection == "dxx")
    {
      if(!this->FD_weights_status(selection))
	{
	  throw invalid_argument("CUDA_MAN: Cannot distribute non-existing weights.");
	}
      for(int i = 0; i < my_devices.size(); i++)
	{
	  if(my_devices[i].second != my_devices[src].second)
	    {
	      error = cudaMemcpyPeer(FD_pointers[i].dxx,my_devices[i].second,FD_pointers[src].dxx,my_devices[src].second,copy_size);
	      if(error != cudaSuccess)
		{
		  throw runtime_error("CUDA_MAN: Could not distribute dxx.");
		}
	    }
	}
    }
  else if(selection == "dyy")
    {
      if(!this->FD_weights_status(selection))
	{
	  throw invalid_argument("CUDA_MAN: Cannot distribute non-existing weights.");
	}
      for(int i = 0; i < my_devices.size(); i++)
	{
	  if(my_devices[i].second != my_devices[src].second)
	    {
	      error = cudaMemcpyPeer(FD_pointers[i].dyy,my_devices[i].second,FD_pointers[src].dyy,my_devices[src].second,copy_size);
	      if(error != cudaSuccess)
		{
		  throw runtime_error("CUDA_MAN: Could not distribute dyy.");
		}
	    }
	}
    }
  else if(selection == "dxy")
    {
      if(!this->FD_weights_status(selection))
	{
	  throw invalid_argument("CUDA_MAN: Cannot distribute non-existing weights.");
	}
      for(int i = 0; i < my_devices.size(); i++)
	{
	  if(my_devices[i].second != my_devices[src].second)
	    {
	      error = cudaMemcpyPeer(FD_pointers[i].dxy,my_devices[i].second,FD_pointers[src].dxy,my_devices[src].second,copy_size);
	      if(error != cudaSuccess)
		{
		  throw runtime_error("CUDA_MAN: Could not distribute dxy.");
		}
	    }
	}
    }
  else if(selection == "laplace")
    {
      if(!this->FD_weights_status(selection))
	{
	  throw invalid_argument("CUDA_MAN: Cannot distribute non-existing weights.");
	}
      for(int i = 0; i < my_devices.size(); i++)
	{
	  if(my_devices[i].second != my_devices[src].second)
	    {
	      error = cudaMemcpyPeer(FD_pointers[i].laplace,my_devices[i].second,FD_pointers[src].laplace,my_devices[src].second,copy_size);
	      if(error != cudaSuccess)
		{
		  throw runtime_error("CUDA_MAN: Could not distribute laplace.");
		}
	    }
	}
    }
  else if(selection == "neg_laplace")
    {
      if(!this->FD_weights_status(selection))
	{
	  throw invalid_argument("CUDA_MAN: Cannot distribute non-existing weights.");
	}
      for(int i = 0; i < my_devices.size(); i++)
	{
	  if(my_devices[i].second != my_devices[src].second)
	    {
	      error = cudaMemcpyPeer(FD_pointers[i].neg_laplace,my_devices[i].second,FD_pointers[src].neg_laplace,my_devices[src].second,copy_size);
	      if(error != cudaSuccess)
		{
		  throw runtime_error("CUDA_MAN: Could not distribute neg_laplace.");
		}
	    }
	}
    }
  else
    {
      if(!this->FD_weights_status("dx") || !this->FD_weights_status("dy") || !this->FD_weights_status("dxx") || !this->FD_weights_status("dyy") || !this->FD_weights_status("dxy") || !this->FD_weights_status("laplace") || !this->FD_weights_status("neg_laplace"))
	{
	  throw invalid_argument("CUDA_MAN: Cannot distribute non-existing weights.");
	}
      for(int i = 0; i < my_devices.size(); i++)
	{
	  if(my_devices[i].second != my_devices[src].second)
	    {
	      error = cudaMemcpyPeer(FD_pointers[i].dx,my_devices[i].second,FD_pointers[src].dx,my_devices[src].second,copy_size);
	      if(error != cudaSuccess)
		{
		  throw runtime_error("CUDA_MAN: Could not distribute dx.");
		}
	      error = cudaMemcpyPeer(FD_pointers[i].dy,my_devices[i].second,FD_pointers[src].dy,my_devices[src].second,copy_size);
	      if(error != cudaSuccess)
		{
		  throw runtime_error("CUDA_MAN: Could not distribute dy.");
		}
	      error = cudaMemcpyPeer(FD_pointers[i].dxx,my_devices[i].second,FD_pointers[src].dxx,my_devices[src].second,copy_size);
	      if(error != cudaSuccess)
		{
		  throw runtime_error("CUDA_MAN: Could not distribute dxx.");
		}
	      error = cudaMemcpyPeer(FD_pointers[i].dyy,my_devices[i].second,FD_pointers[src].dyy,my_devices[src].second,copy_size);
	      if(error != cudaSuccess)
		{
		  throw runtime_error("CUDA_MAN: Could not distribute dyy.");
		}
	      error = cudaMemcpyPeer(FD_pointers[i].dxy,my_devices[i].second,FD_pointers[src].dxy,my_devices[src].second,copy_size);
	      if(error != cudaSuccess)
		{
		  throw runtime_error("CUDA_MAN: Could not distribute dxy.");
		}
	      error = cudaMemcpyPeer(FD_pointers[i].laplace,my_devices[i].second,FD_pointers[src].laplace,my_devices[src].second,copy_size);
	      if(error != cudaSuccess)
		{
		  throw runtime_error("CUDA_MAN: Could not distribute laplace.");
		}
	      error = cudaMemcpyPeer(FD_pointers[i].neg_laplace,my_devices[i].second,FD_pointers[src].neg_laplace,my_devices[src].second,copy_size);
	      if(error != cudaSuccess)
		{
		  throw runtime_error("CUDA_MAN: Could not distribute neg_laplace.");
		}
	    }
	}
    }
}

void cuda_manager::distribute_FD(string selection, int src)
{
  cudaError_t error;
  size_t copy_size1 = sizeof(double)*num_nodes;
  size_t copy_size2 = copy_size1 * MAX_NN;

  if(selection == "")
    {
      for(int i = 0; i < my_devices.size(); i++)
	{
	  if(my_devices[i].second != my_devices[src].second)
	    {
	      error = cudaMemcpyPeer(FD_pointers[i].function,my_devices[i].second,FD_pointers[src].function,my_devices[src].second,copy_size2);
	      if(error != cudaSuccess)
		{
		  throw runtime_error("CUDA_MAN: Could not distribute FD function.");
		}
	      error = cudaMemcpyPeer(FD_pointers[i].derivative,my_devices[i].second,FD_pointers[src].derivative,my_devices[src].second,copy_size1);
	      if(error != cudaSuccess)
		{
		  throw runtime_error("CUDA_MAN: Could not distribute FD derivative.");
		}
	    }
	}
    
    }

  else if(selection == "function")
    {
      for(int i = 0; i < my_devices.size(); i++)
	{
	  if(my_devices[i].second != my_devices[src].second)
	    {
	      error = cudaMemcpyPeer(FD_pointers[i].function,my_devices[i].second,FD_pointers[src].function,my_devices[src].second,copy_size2);
	      if(error != cudaSuccess)
		{
		  throw runtime_error("CUDA_MAN: Could not distribute FD function.");
		}
	    }
	}
    }

  else if(selection == "derivative")
    {
      for(int i = 0; i < my_devices.size(); i++)
	{
	  if(my_devices[i].second != my_devices[src].second)
	    {
	      error = cudaMemcpyPeer(FD_pointers[i].derivative,my_devices[i].second,FD_pointers[src].derivative,my_devices[src].second,copy_size1);
	      if(error != cudaSuccess)
		{
		  throw runtime_error("CUDA_MAN: Could not distribute FD derivative.");
		}
	    }
	}
    }
  else
    {
      throw invalid_argument("CUDA_MAN: Invalid selection for distribute FD.");
    }
}

