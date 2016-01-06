/*** grid_optimimisation_tests.cpp
This specifically tests the optimisation routines to find the optmial
shape parameter  in the case of an RBF which needs one.
 
Julian Merten
Universiy of Oxford
Nov 2015
julian.merten@physics.ox.ac.uk
http://www.julianmerten.net
***/

#include <iostream>
#include <fstream>
#include <mfree/mesh_free.h>
#include <mfree/mesh_free_differentiate.h>
#include <mfree/grid_utils.h>
#include <mfree/radial_basis_function.h>
#include <mfree/rbf_implementation.h>
#include <mfree/test_functions.h>
#include <mfree/test_functions_implementation.h>
#include <mfree/rwfits.h>

using namespace std;

int main()
{

  //Creating random coordinates


  coordinate_grid coords1("random",900,2,42);

  //Creating the differentiable mesh-free object from the coords
  vector<double> construction_coordinates = coords1();
  mesh_free_2D mfree1(&construction_coordinates);
  mfree1.build_tree();

  //Creating a Gaussian RBF
  gaussian_rbf rbf1(1.0);

  //Creating a test function for grid optimisation
  //bengts_function test_function1;
  nfw_lensing_potential test_function1;

  vector<string> diffs;
  diffs.push_back("Laplace");
  diffs.push_back("x");

  vector<double> result = optimise_grid_differentiation(&mfree1,&rbf1,&test_function1,diffs,4,100,"./grid_optimisations_output");

  for(int i = 0; i < diffs.size(); i++)
    {
      cout <<"Final shape " <<diffs[i] <<": " <<result[i] <<endl;
    }

  //Visualising the output for the optimal shape
  int dim = mfree1.return_grid_size();
  vector<double> base_function;
  for(int i = 0; i < dim; i++)
    {
      vector<double> dummy;
      dummy.push_back(construction_coordinates[i*2]);
      dummy.push_back(construction_coordinates[i*2+1]);
      base_function.push_back(test_function1(dummy));
    }
  cubic_spline_rbf rbf2;

  voronoi_to_fits(&mfree1, &base_function,"./optimisation_test.fits");
  for(int i = 0; i < diffs.size(); i++)
    {
      rbf1.set_epsilon(result[i]);
      vector<double> differential;
      cout <<diffs[i] <<": " <<mfree1.differentiate(&base_function,diffs[i],&rbf1,&differential) <<endl;
      vector<double> real;
      for(int j = 0; j < dim; j++)
	{
	  vector<double> dummy;
	  dummy.push_back(construction_coordinates[j*2]);
	  dummy.push_back(construction_coordinates[j*2+1]);
	  real.push_back(test_function1.D(dummy,diffs[i]));
	}
      voronoi_to_fits(&mfree1, &real,"./optimisation_test.fits",diffs[i]);
      voronoi_to_fits(&mfree1, &differential,"./optimisation_test.fits",diffs[i]+"_num");
    }
      
  return 0;

}
