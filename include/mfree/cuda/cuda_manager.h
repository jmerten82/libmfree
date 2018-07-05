/*** /cuda/cuda_manager.h
     This provides the base class
     which handles CUDA device and mainly
     memory management. 

Julian Merten
INAF OA Bologna
July 2018
julian.merten@inaf.it
http://www.julianmerten.net
***/

#ifndef    CUDA_MANAGER_H
#define    CUDA_MANAGER_H


/*
  This hardwires certain quantities, usually
  related to the parallel calculation splitting.
*/

#define MAX_NN 128
#define MAX_PDEG 10
#define CONST_MEM_BLOCK_SIZE 8100
#define ARRAY_CPY_THREAD_SIZE 128

#include <vector>
#include <iostream>
#include <sstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_timer.h>
#include <mfree/mesh_free_differentiate.h>

using namespace std;

/*
  General definition of device constant memory blocks.
*/

extern __constant__ double CONSTblock[CONST_MEM_BLOCK_SIZE];

/*
  This structure holds device pointers which are related to 
  finite differencing on the GPU.
*/

typedef struct FD_pointer_array
{

  //Data
  double *function;

  //Shapes
  double *dx_shapes;
  double *dy_shapes;
  double *dxx_shapes;
  double *dyy_shapes;
  double *dxy_shapes;
  double *laplace_shapes;
  double *neg_laplace_shapes;

  //Weights
  double *dx;
  double *dy;
  double *dxx;
  double *dyy;
  double *dxy;
  double *laplace;
  double *neg_laplace;

  //Results
  double *derivative;

} FD_pointer_array;


class cuda_manager
{

 protected:

  /*
    This will hold information on the current CUDA driver and runtime
    environment.
  */

  vector<int> cuda_runtime_environment;

  /*
    This holds the same information as above, but decoding the integer
    scheme of CUDA runtime integers and saving it human-readable. 
  */

  vector<double> cuda_runtime_environment_h;


  /*
    This saves all information of all available devices on the system by
    just saving a number of CUDA device property objects, together with
    their deviceID on the system.
  */
  
  vector<pair<cudaDeviceProp,int> > my_devices;

  /*
    These are the expectations on compute capability and cuda version
    this manager was created with. They can be used for compatibility tests 
    further down the line.
  */

  vector<double> specs;

/*
    This is the number of mesh-free nodes that the devices are
    currently working on. 
  */

  int num_nodes;

  /*
    The number of nearest neighbours connecting the nodes in the
    index map.
  */

  int nn;

  /*
    The degree of polynomial support for RBF operations.
  */

  int pdeg;

  /*
    This holds pointers to the coordinates saved on all nodes. 
  */

  vector<double*> coordinates;

  /*
    A vector of pointers to neighbour trees on all devices.
  */

  vector<int*> index_maps;

/*
    This holds all relevant pointers on all devices related to
    finite differncing. 
  */

  vector<FD_pointer_array> FD_pointers;

  /*
    This flag indicates if findif related memory has been allocated on 
    the devices. 
  */

  bool FD_allocations;
  
  /*
    Set of flags if the findif weights on the devices have been created.
    The order is:
    first_derivatives
    second_derivatives
  */

  vector<bool> FD_weights;

public:

  /*
    Standard constructor which needs the minimum current compute capability
    and cuda version.
  */

  cuda_manager(double compute, double cuda);

  /*
    This provides all information of the current CUDA device environment and
    writes it into a single string. If the flag is set it also writes it 
    to standard out.
  */
  
  string report(bool to_console = false);

  /*
    This checks if any device code can be run on this machine.
  */
  
  bool check();

  /*
    This returns a list of deviceIDs, ordered by a a given criterion.
    The currently implemented criterions are:

    compute_capability
    performance
    memory

    If the requested size is larger than the number of available devices,
    the list will only be an ordered list of the available GPU resources. 

  */

  vector<int> criterion_query(string criterion = "compute_capability",int size = 1);

/*
    This returns the pair of cuda device property and respective
    DeviceID, provided an index for the pair vector.
  */

  pair<cudaDeviceProp,int> device_query(int index = 0);

  /*
    This flushes all device allocations and resets the 
    manager to deal with a new number of nodes according 
    to the mesh-free domain used here. It also immediately
    copies the tree on all devices. The degree of polynomial
    support for later FD and IP operations is also set at this 
    stage. 
  */

  void reset(mesh_free_2D *reference, int pdeg);

  /*
    This deallocates all memory on all devices and sets the
    number of nodes to 0.
  */

  void reset();

  /*
    This returns key numbers of the CUDA manager, inlcuding the total
    number of managed devices, the current number of nodes,
    the number of nearest neighbours per node and the degree of 
    polynomial support. Selections are:
    devices
    nodes
    nn
    pdeg
  */

  int n(string selection = "devices");

  /*
    This returns a pointer to the grid coordinates on the relevant device.
  */

  double* coordinate_pointer();

  /*
    This returns the relevant pointer to the index map on the device.
  */

  int* index_map_pointer();

  /*
    Returns a pointer to the constant memory block which is 
    defined in this header. If the symbol is not the usual 
    CONSTblock, it can be set by hand here. 
  */

  double* const_memory_block_pointer(string symbol = "");

/*
    This allocates device memory for finite differencing.
  */

  void allocate_FD_builder_memory();

  /*
    Deallocates all FD  memory on all relevant devices. 
  */

  void  deallocate_FD_builder_memory();

  /*
    This returns a device memory pointer related to FD operations.
    Selections are:
    dx_shapes
    dy_shapes
    dxx_shapes
    dyy_shapes
    dxy_shapes
    laplace_shapes
    neg_laplace_shapes
    dx
    dy
    dxy
    laplace
    neg_laplace
    function
    derivative
  */

  double* FD_device_pointer(string selection);

  /*
    Returns if certain FD weights have been created on the devices.
    Selections are:
    dx
    dy
    dxx
    dyy
    dxy
    laplace
    neg_laplace
  */

  bool FD_weights_status(string selection);

  /*
    Switches the selected FD weights status. Selection are
    dx
    dy
    dxx
    dyy
    dxy
    laplace
    neg_laplace
  */

  void switch_FD_weights_status(string selection);

  /*
    This sends the current data stored in the FD weights of device src
    to all other devices. If no selection is made, both first and
    second derivatives are sent around. Selections are:
    dx
    dy
    dxx
    dyy
    dxy
    laplace
    neg_laplace


  */

  void distribute_FD_weights(string selection = "", int src = 0);

  /*
    This distributes some additional FD related quantities between the 
    devices. Selection are
    function
    derivative
  */

  void distribute_FD(string selection = "", int src = 0);
};



#endif /* CUDA_MANAGER_H */
