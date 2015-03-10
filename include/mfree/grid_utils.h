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
#include <gsl/gsl_rng.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_statistics.h>
#include <mfree/unstructured_grid.h>
#include <mfree/containers.h>
#include <flann/flann.hpp>

using namespace std;

/*
  This routine fills the entries of the smaller grid
  with a random sampling from the bigger grid. The number of samples
  is defined by the size of the smaller grid. The seed for the random number
  generator can be given, if not it is seeded with the system time.
*/

void grid_sampling(unstructured_grid *big_grid, unstructured_grid *small_grid, int rng_seed = -1);

/*
  This routines converts a function defined on a (big) grid to a (smaller) grid
  by searching the knn nearest neighbours of the small grid positions 
  on the larger grid and by performing an inverse distand-weighted 
  average of all these function values on the large grid. The covariance 
  matrix is also returned. The selection allows for different 
  averaging schemes. Current selections are:

  "unweighted"
  "inv_dist_weighted"

  The function returns the maxiumum distance between a point in the big grid
  and a new point on the small grid. Should be 0 if the grids overlap.
*/

double grid_conversion(unstructured_grid *big_grid, double *big_function, int stride_big, unstructured_grid *small_grid, vector<double> *small_function, vector<double> *covariance, int knn, string selection = "unweighted");

/*
  Simpler version of the above, just takes a large grid and a smaller grid 
  and translates the big_function defined on the large grid onto the small
  grid. Also here the function returns the maximum distance in this 
  transformation.
*/

double grid_conversion(unstructured_grid *big_grid, double *big_function, int stride_big, unstructured_grid *small_grid, vector<double> *small_function);

/*
  Create a combined grid out of four input coordinate vectors. Also
  Returns a flag vector indicating if each of the initial vector coordinates
  relates to the specific points. Also returns the four input vectors
  defined on the larger grid, where values which are not set on the larger
  grid are set to 0. Finally it outpus a redshift map of the grid.
*/



void grid_combination(vector<galaxy> *shear_field, vector<ccurve_estimator> *ccurve, vector<multiple_image_system> *msystems_in, vector<double> *shear1_covariance,vector<double> *shear2_covariance, vector<double> *coordinate_vector, vector<double> *shear1_out, vector<double>* shear2_out, vector<double> *ccurve_prefactor, vector<double> *shear1_covariance_out, vector<double> *shear2_covariance_out, vector<double> *redshifts);


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



#endif    /*GRID_UTILS_H*/
