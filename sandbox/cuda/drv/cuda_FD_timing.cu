/*** cuda_FD.cpp
     This just times the set weights routines of the cuFD routines.

Julian Merten
INAF OA Bologna
July 2018
julian.merten@inaf.it
http://www.julianmerten.net
***/

#include <iostream>
#include <tclap/CmdLine.h>

#include <mfree/grid_utils.h>
#include <mfree/mesh_free_differentiate.h>

#include <mfree/cuda/cuda_manager.h>
#include <mfree/cuda/cuFD.h>

int main(int argc, char* argv[])
{
  TCLAP::CmdLine cmd("cuda_FD", ' ',"0.1");
  TCLAP::ValueArg<int> nnArg("n","nn","Number of nearest neighbours.",false,32,"int",cmd);
  TCLAP::ValueArg<int> dimArg("d","dim","Number of nodes.",false,1000,"int",cmd);  
  TCLAP::ValueArg<int> pdegArg("p","pdeg","Order of polynomial support.",false,4,"int",cmd);   

 cmd.parse( argc, argv );


 int dim = dimArg.getValue();
 int nn = nnArg.getValue();
 int pdeg = pdegArg.getValue();

 //Setting up test mesh
 coordinate_grid helper("random", dim, 0, 41);
 vector<double> coordinates;
 coordinates = helper();

 //Calling CUDA kernel for FD testing.
 mesh_free_2D mf(&coordinates);
 mf.build_tree(nn);

 //Setting up CUDA manager
 cuda_manager cm(3.0,7.5);
 string cuda_info = cm.report(true);

 //Setting cuda manager up for current mesh-free domain
 cout <<"Setting up the devices for FD given a mesh-free domain." <<endl;
 cm.reset(&mf,pdeg);
 cout <<endl;

 cout <<"Testing weight kernels" <<endl <<endl;

 cuFD_test_weight_functions(&cm);


 return 0;

}
