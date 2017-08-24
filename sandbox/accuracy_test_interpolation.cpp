/*** accuracy_test_interpolation.cpp
     This sandbox routine tests in detail the accuracy of RBF interpolation.
     
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
#include <mfree/rbf_wendland.h>
#include <mfree/test_functions.h>
#include <mfree/test_functions_implementation.h>
#include <mfree/mesh_free.h>
#include <mfree/mesh_free_differentiate.h>
#include <mfree/rwfits.h>
#include <mfree/grid_utils.h>

using namespace std;

int main()
{

  //Defining the basic environment of the test

  string filename_out = "./data/accuracy_test_interpol_wend.txt";
  unsigned int N_in = 600;
  unsigned int N_out = 1000;
  int seed_in = 42;
  int seed_out = 43;
  nfw_lensing_potential test_function;
  //gaussian_rbf rbf;
  //phs_first_order rbf;
  wendland_C2 rbf;

  //Creating the underlying random distributions of nodes

  coordinate_grid support_nodes("random",N_in,0,seed_in);
  coordinate_grid interpolant("random",N_out,0,seed_out);
  vector<double> nodes_in = support_nodes();
  vector<double> nodes_out = interpolant();

  //Creating the function values at support points and for interpolant comparison.

  vector<double> f_in, f_out, aux;
  aux.resize(2);

  for(unsigned int  i = 0; i < N_in; i++)
    {
      aux[0] = nodes_in[i*2];
      aux[1] = nodes_in[i*2+1];
      f_in.push_back(test_function(aux));
    }

  for(unsigned int  i = 0; i < N_out; i++)
    {
      aux[0] = nodes_out[i*2];
      aux[1] = nodes_out[i*2+1];
      f_out.push_back(test_function(aux));
    }

  //Creating mesh-free domain

  mesh_free_2D mfree(&nodes_in);

  //Creating output file that will record the result
  ofstream output(filename_out.c_str());
  output <<"#accuracy tests for RBF FS" <<endl;
  output <<"#shape  pdeg  condition  mean_abs_err  mean_rel_err  max_abs_err  max_rel_err max_err_x max_err_y" <<endl;

  //Error book keeping
  double shape = 0.001;
  vector<double> diff, rel_diff, result;
  double current_condition;
  diff.resize(N_out);
  rel_diff.resize(N_out);

  //Loop over shape parmaters and polynomial support degree
  for(unsigned int i = 0; i < 1000; i++)
    {
      rbf.set_epsilon(shape);
      for(unsigned int j = 0; j < 11; j++)
	{
	  cout <<"shape: " <<shape <<"\t" <<"pdeg: " <<j <<endl;
	  current_condition = mfree.interpolate(&nodes_out,&f_in,&result,&rbf,j,32);
	  //current_condition = mfree.interpolate(&nodes_out,&f_in,&result,&rbf,32);
	  for(unsigned int l = 0; l < N_out; l++)
	    {
	      double current = f_out[l];
	      diff[l] = abs(result[l]-current);
	      rel_diff[l] = abs(diff[l]/current);
	    }

	  vector<double>::iterator it, rel_it;
	  int rel_pos;
	  it = max_element(diff.begin(),diff.end());
	  rel_it = max_element(rel_diff.begin(), rel_diff.end());
	  rel_pos = distance(rel_diff.begin(),rel_it);

	  output <<shape <<"  " <<j <<"  " <<current_condition <<"  " <<gsl_stats_mean(&diff[0],1,diff.size()) <<"  " <<gsl_stats_mean(&rel_diff[0],1,rel_diff.size()) <<"  " <<*it <<"  " <<*rel_it <<"  " <<nodes_out[rel_pos*2] <<"  " <<nodes_out[rel_pos*2+1] <<endl;  
	}
      shape += .001;
    }


  output.close();


  return 0;

}
