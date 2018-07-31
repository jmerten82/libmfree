/*** grid_utils.h
This set of tools provides functionality for the SaWLens
workflow. This includes grid manipulations, averaging routines
and other useful utilities. 

Julian Merten
JPL/Caltech
May 2014
jmerten@caltech.edu
***/

#ifndef    GRID_UTILS_H
#define    GRID_UTILS_H

#include <cmath>
#include <ctime>
#include <stdexcept>
#include <fstream>
#include <string>
#include <algorithm>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_statistics.h>
#include <mfree/mesh_free.h>
#include <mfree/mesh_free_differentiate.h>
#include <mfree/radial_basis_function.h>
#include <mfree/test_functions.h>
#include <flann/flann.hpp>

using namespace std;

/*
  This routine fills the entries of the smaller grid
  with a random sampling from the bigger grid. The number of samples
  is defined by the size of the smaller grid. The seed for the random number
  generator can be given, if not it is seeded with the system time.
*/

void grid_sampling(mesh_free *big_grid, mesh_free *small_grid, int rng_seed = -1);


/**
   This is a designated class to create and manipulate coordinate 
   grids. It can deal with round, random and also regular grids.
**/

class coordinate_grid
{

 protected:

  /*
    Maintains the coordinate structure of the grid.
  */

  vector<double> coordinates;

 public:

  /*
    Standard constructor, creates an empty grid and allocates
    the coordinate vector.
  */

  coordinate_grid(int num_nodes);

  /*
    Constructor that sets up a grid already. Selection for type are
    random and regular. You also need to provide the number of
    grid nodes and the number refinements. One can also define a seed
    for the random number generator. If not, it is seeded with the system time.
  */

  coordinate_grid(string type, int num_nodes, int refinement_level = 0, int rng_seed = -1);

  /*
    Standard destructor.
  */

  ~coordinate_grid();

  /*
    Return the coordinate vector.
  */

  vector<double> operator() ();

  /*
    Adds a rectangular mask.
  */

  void add_mask(double x1, double x2, double y1, double y2);

  /*
    Adds a circular mask.
  */

  void add_mask(double x1, double y1, double r);

  /*
    Writes the coordinates into a file.
  */

  void write(string filename);

  /*
    Scales the full coorindate grid by a given uniform factor.
  */

  void scale(double factor); 
};

/*
  Build a unit grid with a given pixel size. If the flag is set it cuts out the
  the inner half of the grid.
*/

vector<double> build_unit_grid(double pixel_size, bool spare_centre); 

/*
  This helper function optimises the shape parameter of a mesh-free
  RBF to provide optimal derivative performance based on a provided 
  test function. You can provide a vector of derivative orders and 
  for each a shape parameter is provided in the output vector.
  If a verbose-mode filename is set, the resuls from the optimisation 
  process are written to ASCII.
*/

vector<double> optimise_grid_differentiation(mesh_free_differentiate *mesh_free_in,radial_basis_function *rbf_in, unsigned int pdeg,  test_function *test_function_in, vector<string> differentials, int refinements = 1, int steps = 100, double eps = 1e-3, string verbose_mode = "");

/*
  This is another version of the differentiation grid optimisation, which
  delivers an return an adaptive shape parameter, 
  which can vary from node to node. Only a single derivative is possible here.
  The number of nearest neighbours has to be defined here. 
*/

vector<double> optimise_adaptive_grid_differentiation(mesh_free_differentiate *mesh_free_in,radial_basis_function *rbf_in, unsigned int pdeg,  test_function *test_function_in, string differential, int knn = 16, int refinements = 1, int steps = 100, double eps = 1e-3, string verbose_mode = "");
  


/*
  This function is equivalent to the differentiation optimisation but optimises
  the inteprolation instead. Since this will depend on the current mesh-free
  layout and the target one, it needs to meshes to operate on. 
*/

double optimise_grid_interpolation(mesh_free *mesh_free_in, mesh_free *mesh_free_target, radial_basis_function *rbf_in, unsigned int pdeg,  test_function *test_function_in, int knn = 16, string test_function_switch = "", int refinements = 1, int steps = 100,double eps = 1e-3, string verbose_mode = "");

/*
  And also here this version of the function does the node-by-node optimisation
  with an adaptive shape parameter vectos as return.
*/

vector<double> optimise_adaptive_grid_interpolation(mesh_free *mesh_free_in, mesh_free *mesh_free_target, radial_basis_function *rbf_in, unsigned int pdeg,  test_function *test_function_in, int knn = 16, string test_function_switch = "", int refinements = 1, int steps = 100, double eps = 1e-3, string verbose_mode = "");

/*
  This compares a map to a reference and returns the absolute and relative
  errors, as well as the mean abolute and relative error, together with 
  the maximum absolute and relative error.
*/

vector<vector<double> > compare_maps(vector<double> *map, vector<double> *reference, double *max_abs, double *mean_abs, double *mean_relative, double *max_relative);




#endif    /*GRID_UTILS_H*/
