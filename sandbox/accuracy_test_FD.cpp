/*** accuracy_test_FD.cpp
     This sandbox routine tests in detail the accuracy of RBF FD.
     
     Julian Merten
     University of Oxford
     Jul 2017
     http://www.julianmerten.net
***/

#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <mfree/radial_basis_function.h>
#include <mfree/rbf_implementation.h>
#include <mfree/rbf_polyharmonic_splines.h>
#include <mfree/test_functions.h>
#include <mfree/test_functions_implementation.h>
#include <mfree/mesh_free.h>
#include <mfree/mesh_free_differentiate.h>
#include <mfree/rwfits.h>
#include <mfree/grid_utils.h>

using namespace std;

int main()
{

  string filename_out = "./data/accuracy_test_laplacian_std_gaussian.txt";

  //Define RBF, polynomial degree, number of grid points and test function to be tested


  unsigned int pdeg = 6;
  gaussian_rbf rbf;
  int N = 1000;
  nfw_lensing_potential test_function;
  string derivative = "Laplace";


  //Create random distribution of points

  coordinate_grid std_grid("random",N,0,42);
  coordinate_grid refined_grid("random",N,2,42);
  vector<double> std_coords = std_grid();
  vector<double> refined_coords = refined_grid();

  int size_std = std_coords.size() / 2.;
  int refined_std = refined_coords.size() / 2.;

  //Evaluate test function and its derivatives on this grid

  vector<double> f_x, Df_x, current_point;
  current_point.resize(2); 

  for(unsigned int i = 0; i < size_std; i++)
    {
      current_point[0] = std_coords[i*2];
      current_point[1] = std_coords[i*2+1];
      f_x.push_back(test_function(current_point));
      Df_x.push_back(test_function.D(current_point,derivative));
    }

  //Creating the 2D mesh_free domain

  mesh_free_2D domain(&std_coords);
  domain.build_tree();


  //Creating output file that will carry results
  ofstream output(filename_out.c_str());

  output <<"#accuracy tests for RBF FS" <<endl;
  output <<"#shape  pdeg  condition  mean_abs_err  mean_rel_err  max_abs_err  max_rel_err max_err_x max_err_y" <<endl;

  double shape = 0.01;

  vector<double> diff, rel_diff, result;
  double current_condition;
  diff.resize(size_std);
  rel_diff.resize(size_std);

  for(unsigned int i = 0; i < 200; i++)
    {
      rbf.set_epsilon(shape);
      for(unsigned int j = 0; j < 11; j++)
	{
	  cout <<"shape: " <<shape <<"\t" <<"pdeg: " <<j <<endl;
	  current_condition = domain.differentiate(&f_x,derivative,j,&rbf,&result);
	  //current_condition = domain.differentiate(&f_x,derivative,&rbf,&result);
	  for(unsigned int l = 0; l < size_std; l++)
	    {
	      double current = Df_x[l];
	      diff[l] = abs(result[l]-current);
	      rel_diff[l] = abs(diff[l]/current);
	    }

	  vector<double>::iterator it, rel_it;
	  int rel_pos;
	  it = max_element(diff.begin(),diff.end());
	  rel_it = max_element(rel_diff.begin(), rel_diff.end());
	  rel_pos = distance(rel_diff.begin(),rel_it);

	  output <<shape <<"  " <<j <<"  " <<current_condition <<"  " <<gsl_stats_mean(&diff[0],1,diff.size()) <<"  " <<gsl_stats_mean(&rel_diff[0],1,rel_diff.size()) <<"  " <<*it <<"  " <<*rel_it <<"  " <<std_coords[rel_pos*2] <<"  " <<std_coords[rel_pos*2+1] <<endl;  
	}
      shape += .01;
    }


  output.close();

  return 0;
}
