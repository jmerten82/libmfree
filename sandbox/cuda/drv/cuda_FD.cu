/*** cuda_FD.cu
     This routine tests basic FD capabilities. 

Julian Merten
INAF OA Bologna
Jul 2018
julian.merten@oabo.inaf.it
http://www.julianmerten.net
***/

#include <iostream>
#include <tclap/CmdLine.h>

#include <mfree/rwfits.h>
#include <mfree/grid_utils.h>
#include <mfree/mesh_free_differentiate.h>
#include <mfree/test_functions.h>
#include <mfree/test_functions_implementation.h>
#include <mfree/radial_basis_function.h>
#include <mfree/rbf_implementation.h>
#include <mfree/rbf_polyharmonic_splines.h>
#include <mfree/rbf_wendland.h>

#include <mfree/cuda/cuda_manager.h>
#include <mfree/cuda/cuFD.h>
#include <mfree/cuda/cuRBFs.h>


int main(int argc, char* argv[])
{
  TCLAP::CmdLine cmd("cuda_FD", ' ',"0.1");
  TCLAP::ValueArg<int> nnArg("n","nn","Number of nearest neighbours.",false,32,"int",cmd);
  TCLAP::ValueArg<int> dimArg("d","dim","Number of nodes.",false,1000,"int",cmd);  
  TCLAP::ValueArg<int> pdegArg("p","pdeg","Order of polynomial support.",false,4,"int",cmd);
  TCLAP::ValueArg<double> shapeArg("e","eps","The RBF shape parameter.",false,1.0,"double",cmd);  
  TCLAP::ValueArg<string> fileArg("f","file","The filename for the FITS output checks.",false,"../../results/cuda_FD_output.fits","string",cmd);
  TCLAP::ValueArg<string> rbfArg("r","rbf","The RBF to use.",false,"ga","string",cmd);
 cmd.parse( argc, argv );

 int dim = dimArg.getValue();
 int nn = nnArg.getValue();
 int pdeg = pdegArg.getValue();
 string filename = fileArg.getValue();
 string rbf_name = rbfArg.getValue();
 double epsilon = shapeArg.getValue();

 int rbf_number = 0;
 if(rbf_name == "ga")
   {
     rbf_number = 1;
   }
 else if(rbf_name == "wendland")
   {
     rbf_number = 2;
   }
 else if(rbf_name == "phs3")
   {
     rbf_number = 3;
   }
 else if(rbf_name == "phs4")
   {
     rbf_number = 4;
   }

 //Stopwatch for benchmarking
 StopWatchLinux sw;

 //Creating random domain
 coordinate_grid helper("random", dim, 0, 41);
 vector<double> coordinates;
 coordinates = helper();

 //Defining a test function and its derivatives
 bengts_function ref_function;
 vector<double> f, dx_f, dy_f, dxx_f, dyy_f, dxy_f, laplace_f, neg_laplace_f;
 for(int i = 0; i < dim; i++)
   {
     vector<double> c;
     c.push_back(coordinates[2*i]);
     c.push_back(coordinates[2*i+1]);
     f.push_back(ref_function(c));
     dx_f.push_back(ref_function.D(c,"x"));
     dy_f.push_back(ref_function.D(c,"y"));
     dxx_f.push_back(ref_function.D(c,"xx"));
     dyy_f.push_back(ref_function.D(c,"yy"));
     dxy_f.push_back(ref_function.D(c,"xy"));
     laplace_f.push_back(ref_function.D(c,"Laplace"));
     neg_laplace_f.push_back(ref_function.D(c,"Neg_Laplace"));
    }

 //Creating mesh free domain and writing out test function with derivatives
 mesh_free_2D mf(&coordinates);
 mf.build_tree(nn);

 //Creating non-device RBFs
 gaussian_rbf ga(0.,0.,0.,epsilon);
 wendland_C2 wc2(0.,0.,0.,epsilon);
 phs_third_order phs3;
 phs_fourth_order phs4;

 vector<vector<double> > current_output(7,vector<double>());

 sw.start();
 switch(rbf_number)
   {
   case 1: cout <<mf.differentiate(&f,"x",pdeg,&ga,&current_output[0]) <<endl;
     break;
   case 2: cout <<mf.differentiate(&f,"x",pdeg,&wc2,&current_output[0]) <<endl;
     break;
   case 3: cout <<mf.differentiate(&f,"x",pdeg,&phs3,&current_output[0]) <<endl;
     break;
   case 4: cout <<mf.differentiate(&f,"x",pdeg,&phs4,&current_output[0]) <<endl;
     break;
   }
 switch(rbf_number)
   {
   case 1: cout <<mf.differentiate(&f,"y",pdeg,&ga,&current_output[1]) <<endl;
     break;
   case 2: cout <<mf.differentiate(&f,"y",pdeg,&wc2,&current_output[1]) <<endl;
     break;
   case 3: cout <<mf.differentiate(&f,"y",pdeg,&phs3,&current_output[1]) <<endl;
     break;
   case 4: cout <<mf.differentiate(&f,"y",pdeg,&phs4,&current_output[1]) <<endl;
     break;
   }
 switch(rbf_number)
   {
   case 1: cout <<mf.differentiate(&f,"xx",pdeg,&ga,&current_output[2]) <<endl;
     break;
   case 2: cout <<mf.differentiate(&f,"xx",pdeg,&wc2,&current_output[2]) <<endl;
     break;
   case 3: cout <<mf.differentiate(&f,"xx",pdeg,&phs3,&current_output[2]) <<endl;
     break;
   case 4: cout <<mf.differentiate(&f,"xx",pdeg,&phs4,&current_output[2]) <<endl;
     break;
   }
 switch(rbf_number)
   {
   case 1: cout <<mf.differentiate(&f,"yy",pdeg,&ga,&current_output[3]) <<endl;
     break;
   case 2: cout <<mf.differentiate(&f,"yy",pdeg,&wc2,&current_output[3]) <<endl;
     break;
   case 3: cout <<mf.differentiate(&f,"yy",pdeg,&phs3,&current_output[3]) <<endl;
     break;
   case 4: cout <<mf.differentiate(&f,"yy",pdeg,&phs4,&current_output[3]) <<endl;
     break;
   }
 switch(rbf_number)
   {
   case 1: cout <<mf.differentiate(&f,"xy",pdeg,&ga,&current_output[4]) <<endl;
     break;
   case 2: cout <<mf.differentiate(&f,"xy",pdeg,&wc2,&current_output[4]) <<endl;
     break;
   case 3: cout <<mf.differentiate(&f,"xy",pdeg,&phs3,&current_output[4]) <<endl;
     break;
   case 4: cout <<mf.differentiate(&f,"xy",pdeg,&phs4,&current_output[4]) <<endl;
     break;
   }
switch(rbf_number)
   {
   case 1: cout <<mf.differentiate(&f,"Laplace",pdeg,&ga,&current_output[5]) <<endl;
     break;
   case 2: cout <<mf.differentiate(&f,"Laplace",pdeg,&wc2,&current_output[5]) <<endl;
     break;
   case 3: cout <<mf.differentiate(&f,"Laplace",pdeg,&phs3,&current_output[5]) <<endl;
     break;
   case 4: cout <<mf.differentiate(&f,"Laplace",pdeg,&phs4,&current_output[5]) <<endl;
     break;
   }
 switch(rbf_number)
   {
   case 1: cout <<mf.differentiate(&f,"Neg_Laplace",pdeg,&ga,&current_output[6]) <<endl;
     break;
   case 2: cout <<mf.differentiate(&f,"Neg_Laplace",pdeg,&wc2,&current_output[6]) <<endl;
     break;
   case 3: cout <<mf.differentiate(&f,"Neg_Laplace",pdeg,&phs3,&current_output[6]) <<endl;
     break;
   case 4: cout <<mf.differentiate(&f,"Neg_Laplace",pdeg,&phs4,&current_output[6]) <<endl;
     break;
   }
 sw.stop();

 cout <<"Seven derivatives with libmfree took: " <<sw.getTime() <<"msec." <<endl;
 sw.reset(); 

 //wriring the CPU results to FITS
 voronoi_to_fits(&mf,&current_output[0],filename);
 voronoi_to_fits(&mf,&current_output[1],filename,"CPU_dy");
 voronoi_to_fits(&mf,&current_output[2],filename,"CPU_dxx");
 voronoi_to_fits(&mf,&current_output[3],filename,"CPU_dyy");
 voronoi_to_fits(&mf,&current_output[4],filename,"CPU_dxy"); 
 voronoi_to_fits(&mf,&current_output[5],filename,"CPU_Laplace");
 voronoi_to_fits(&mf,&current_output[6],filename,"CPU_negLaplace");

 //Initialising cuda manager
 cuda_manager cm(3.0,7.5);
 string cuda_info = cm.report(true);
 cm.reset(&mf,pdeg);
 cout <<"We are dealing with " <<cm.n() <<" devices." <<endl;
 cout <<"We will perform FD with " <<cm.n("nodes") <<" nodes." <<endl;
 cout <<"All FD operations will use " <<cm.n("nn") <<" nearest neighbours." <<endl;
 cout <<"The polynomial support will be of order " <<cm.n("pdeg") <<endl;
 cout <<endl;
 cout <<"Allocating device memory." <<endl;
 cm.allocate_FD_builder_memory();

 //Initilisaiing device RBFs
 ga_rbf *dga = new ga_rbf;
 wc2_rbf *dwc2 = new wc2_rbf;
 phs3_rbf *dphs3 = new phs3_rbf;
 phs4_rbf *dphs4 = new phs4_rbf;

 //Defining a unique shape parameter map
 vector<double> shapes(cm.n("nodes"),epsilon*epsilon);

 //Perforning the derivatives and timing them
 sw.start();
 cuFD_differentiate_set(f,&cm);
 //Looping over all possible derivatives
 for(int i = 1; i < 8; i++)
   {
     switch(rbf_number)
       {
       case 1: cuFD_weights_set(dga,shapes,&cm,i);
	 break;
       case 2: cuFD_weights_set(dwc2,shapes,&cm,i);
	 break;
       case 3: cuFD_weights_set(dphs3,shapes,&cm,i);
	 break;
       case 4: cuFD_weights_set(dphs4,shapes,&cm,i);
	 break;
       }
   }
 vector<vector<double> > current_device_output = cuFD_differentiate(&cm); 
 sw.stop();
 cout <<"Taking seven device derivatives took: " <<sw.getTime() <<"msec" <<endl;

 voronoi_to_fits(&mf,&current_device_output[0],filename,"GPU_dx");
 voronoi_to_fits(&mf,&current_device_output[1],filename,"GPU_dy");
 voronoi_to_fits(&mf,&current_device_output[2],filename,"GPU_dxx");
 voronoi_to_fits(&mf,&current_device_output[3],filename,"GPU_dyy");
 voronoi_to_fits(&mf,&current_device_output[4],filename,"GPU_dxy"); 
 voronoi_to_fits(&mf,&current_device_output[5],filename,"GPU_Laplace");
 voronoi_to_fits(&mf,&current_device_output[6],filename,"GPU_negLaplace");


 return 0;
}
