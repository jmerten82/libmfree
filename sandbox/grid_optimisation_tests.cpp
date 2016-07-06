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
#include <saw2/omp/tools.h>

using namespace std;

int main()
{

  int knn = 16;

  //Creating random coordinates


  coordinate_grid coords1("random",45,2,42);
  coordinate_grid coords2("regular",900,0);

  //Creating the differentiable mesh-free object from the coords
  vector<double> construction_coordinates = coords1();
  vector<double> output_coordinates = coords2();
  mesh_free_2D mfree1(&construction_coordinates);
  mesh_free_2D mfree2(&output_coordinates);

  mfree1.write_ASCII("./data/grid1.dat");
  mfree2.write_ASCII("./data/grid2.dat");

  //Creating a Gaussian RBF
  gaussian_rbf rbf1(1.0);

  //Creating a test function for grid optimisation
  //bengts_function test_function1;
  bengts_function test_function1;

  cout <<"Optimising interpolation..." <<flush;

  double result = optimise_grid_interpolation(&mfree1,&mfree2,&rbf1,&test_function1,knn,"",4,100,1e-3,"./data/optimise_interpolation.txt");

  double result2 = ga_omp_optimise_interpolation(&mfree1,&mfree2,&test_function1,2,knn,"",4,100,1.e-3);

  cout <<"Done." <<endl;
  cout <<"Final shape " <<result <<endl;
  cout <<"Final shape OMP " <<result2 <<endl;

  cout <<"Adaptively optimising interpolation..." <<flush;

  vector<double> adaptive_result = optimise_adaptive_grid_interpolation(&mfree1,&mfree2,&rbf1,&test_function1,knn,"",4,100,1e-3,"./data/adaptive_optimise_interpolation.txt");

  cout <<"Done." <<endl;

  cout <<adaptive_result.size() <<endl;

  vector<string> ds;
  ds.push_back("x");

  cout <<"Optimising differentiation..." <<flush;
  vector<double> result_diff = optimise_grid_differentiation(&mfree1,&rbf1, &test_function1,ds,4,100,1e-3,"./data/optimise_differentiation.txt");

  cout <<"Done." <<endl;
  cout <<"Final shape " <<flush;
  for(int i =0; i < result_diff.size(); i++)
    {
      cout <<"\t" <<result_diff[i] <<flush;
    }
  cout <<endl;

  cout <<"Adaptively optimising differentiation..." <<flush;

  vector<double> adaptive_result_diff = optimise_adaptive_grid_differentiation(&mfree1,&rbf1,&test_function1,ds[0],knn,4,100,1.e-3,"./data/adaptive_optimise_differentiation.txt");

  cout <<"Done." <<endl;



  int dim = mfree1.return_grid_size();
  vector<double> base_function;
  vector<double> base_function_diff;
  vector<double> interpolation_result;
  vector<double> adaptive_interpolation_result;
  vector<double> differentiation_result;
  vector<double> adaptive_differentiation_result;
  for(int i = 0; i < dim; i++)
    {
      vector<double> dummy;
      dummy.push_back(construction_coordinates[i*2]);
      dummy.push_back(construction_coordinates[i*2+1]);
      base_function.push_back(test_function1(dummy));
    }

  rbf1.set_epsilon(result);
  cout <<"Current condition: " <<mfree1.interpolate(&output_coordinates,&base_function,&interpolation_result,&rbf1) <<endl; 
  cout <<"Current adaptive condition: " <<mfree1.interpolate(&output_coordinates,&base_function,&adaptive_interpolation_result,&rbf1,&adaptive_result) <<endl;
  rbf1.set_epsilon(result_diff[0]);
  cout <<"Current diff condition:" <<mfree1.differentiate(&base_function,ds[0],&rbf1,&differentiation_result) <<endl;
  cout <<"Current adaptive diff condition:" <<mfree1.differentiate(&base_function,ds[0],&rbf1,&adaptive_result_diff,&adaptive_differentiation_result) <<endl;
  

  //Visualising the output for the optimal shape
  dim = mfree2.return_grid_size();
  int dim2 = mfree1.return_grid_size();
  vector<double> real_output_function;
  vector<double> real_output_diff_function;
  for(int i = 0; i < dim; i++)
    {
      vector<double> dummy;
      dummy.push_back(output_coordinates[i*2]);
      dummy.push_back(output_coordinates[i*2+1]);
      real_output_function.push_back(test_function1(dummy));
    }


  for(int i = 0; i < dim2; i++)
    {
      vector<double> dummy;
      dummy = mfree1(i);
      real_output_diff_function.push_back(test_function1.D(dummy,ds[0]));
    }

  vector<double> difference_map;
  double sum;
  difference_map.resize(real_output_function.size());
  
  voronoi_to_fits(&mfree2, &real_output_function,"./data/optimisation_interpolation_test.fits");
  voronoi_to_fits(&mfree2, &interpolation_result,"./data/optimisation_interpolation_test.fits","interpolation");
  sum = 0.;
  for(int i = 0; i < difference_map.size(); i++)
    {
      difference_map[i] = (interpolation_result[i] - real_output_function[i]) / real_output_function[i];
      sum += abs(difference_map[i]);
    }
  sum *= 1./difference_map.size();
  cout <<"Average error on interpolation: " <<sum <<endl;
  voronoi_to_fits(&mfree2, &difference_map,"./data/optimisation_interpolation_test.fits","interpolation_diff");
  voronoi_to_fits(&mfree2, &adaptive_interpolation_result,"./data/optimisation_interpolation_test.fits","adaptive_interpolation");

  sum = 0.;
  for(int i = 0; i < difference_map.size(); i++)
    {
      difference_map[i] = (adaptive_interpolation_result[i] - real_output_function[i]) / real_output_function[i];
      sum += abs(difference_map[i]);
    }
  sum *= 1./difference_map.size();
  cout <<"Average error on adaptive interpolation: " <<sum <<endl;
  voronoi_to_fits(&mfree2, &difference_map,"./data/optimisation_interpolation_test.fits","adaptive_interpolation_diff");

  voronoi_to_fits(&mfree2, &adaptive_result,"./data/optimisation_interpolation_test.fits","shapes");

  voronoi_to_fits(&mfree1, &real_output_diff_function,"./data/optimisation_interpolation_test.fits","differentiation");
  voronoi_to_fits(&mfree1, &differentiation_result,"./data/optimisation_interpolation_test.fits","num_diff");

  sum = 0.;

  difference_map.resize(real_output_diff_function.size());
  for(int i = 0; i < difference_map.size(); i++)
    {
      difference_map[i] = (differentiation_result[i] - real_output_diff_function[i]) / real_output_diff_function[i];
      sum += abs(difference_map[i]);
    }
  sum *= 1./difference_map.size();
  cout <<"Average error on differentiation: " <<sum <<endl;
  voronoi_to_fits(&mfree1, &difference_map,"./data/optimisation_interpolation_test.fits","differentiation_diff");

  voronoi_to_fits(&mfree1, &adaptive_differentiation_result,"./data/optimisation_interpolation_test.fits","adapt_num_diff");

  sum = 0.;

  difference_map.resize(real_output_diff_function.size());
  for(int i = 0; i < difference_map.size(); i++)
    {
      difference_map[i] = (adaptive_differentiation_result[i] - real_output_diff_function[i]) / real_output_diff_function[i];
      sum += abs(difference_map[i]);
    }
  sum *= 1./difference_map.size();
  cout <<"Average error on adaptive differentiation: " <<sum <<endl;
  voronoi_to_fits(&mfree1, &difference_map,"./data/optimisation_interpolation_test.fits","adapt_differentiation_diff");
  voronoi_to_fits(&mfree1, &adaptive_result_diff,"./data/optimisation_interpolation_test.fits","diff_shapes");


  

  return 0;

}
