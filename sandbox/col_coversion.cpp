#include <iostream>
#include <mfree/rwfits.h>
#include <mfree/grid_utils.h>
#include <mfree/mesh_free.h>
#include <mfree/mesh_free_differentiate.h>
#include <mfree/radial_basis_function.h>
#include <mfree/rbf_implementation.h>
#include <helper_timer.h>

using namespace std;

int main()
{
  //Stopwatch for benchmarking
  StopWatchLinux sw;


  int dim = 1000;
  int nn = 32;

  //Creating random domain
  coordinate_grid helper("random", dim, 0, 41);
  vector<double> coordinates;
  coordinates = helper();

  mesh_free_2D mf(&coordinates);
  mf.build_tree(nn);

  //Getting column tree with the two different methods
  vector<int> col_tree_old, length_counter_old, col_tree_new, length_counter_new;
  sw.start();
  int max_old = mf.neighbours_col(&col_tree_old, &length_counter_old);
  sw.stop();
  cout <<"Old method took: " <<sw.getTime() <<"msec" <<endl;
  sw.reset();
  sw.start();
  col_tree_new = mf.neighbours_col_ver2(&length_counter_new);
  sw.stop();
  cout <<"New method took: " <<sw.getTime() <<"msec" <<endl;
  sw.reset();

  vector<int> old_matrix(dim*dim,0);
  vector<int> new_matrix(dim*dim,0);
  int max_new = 2*nn;

  for(int i = 0; i < dim; i++)
    {
      for(int j = 0; j < length_counter_old[i]; j++)
	{
	  old_matrix[col_tree_old[i*max_old+j]*dim+i]++;
	}
    }

  for(int i = 0; i < dim; i++)
    {
      for(int j = 0; j < length_counter_new[i]; j++)
	{
	  new_matrix[col_tree_new[i*max_new+j]*dim+i]++;
	}
    }

 
  write_img_to_fits("./data/col_tests.fits",&old_matrix);
  write_img_to_fits("./data/col_tests.fits",&new_matrix,"new_method");

  gaussian_rbf ga;
  ga.set_epsilon(1.);

  sw.start();
  vector<double> old_weights = mf.create_finite_differences_weights_col("x",0,&ga, max_old);
  sw.stop();
  cout <<"Old weights took " <<sw.getTime() <<"msec" <<endl;
  sw.reset(); 

  sw.start();
  vector<double> new_weights = mf.create_finite_differences_weights_col_ver2("x",0,&ga);
  sw.stop();
  cout <<"New weights took " <<sw.getTime() <<"msec" <<endl;
  sw.reset();

  vector<int> old_weight_matrix(dim*dim,0);
  vector<int> new_weight_matrix(dim*dim,0);
  for(int i = 0; i < dim; i++)
    {
      for(int j = 0; j < length_counter_old[i]; j++)
	{
	  old_weight_matrix[col_tree_old[i*max_old+j]*dim+i] = old_weights[i*max_old+j];
	}
    }

  for(int i = 0; i < dim; i++)
    {
      for(int j = 0; j < length_counter_new[i]; j++)
	{
	  new_weight_matrix[col_tree_new[i*max_new+j]*dim+i] = new_weights[i*max_new+j];
	}
    }

 
  write_img_to_fits("./data/col_tests.fits",&old_matrix);
  write_img_to_fits("./data/col_tests.fits",&new_matrix,"new_method");
  write_img_to_fits("./data/col_tests.fits",&old_weight_matrix,"weights_old");
  write_img_to_fits("./data/col_tests.fits",&new_weight_matrix,"weights_new");

 











  return 0;

}
