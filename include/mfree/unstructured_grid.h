/*** unstructured_grid.h
Class which describes a completely un-structured, 1D, 2D or 3D mesh
by the coordinates of its nodes and their kD-tree. 
Derivatives on the grid are calculated via Radial-Basis
Function-based finite differencing stencils. Visualisation routines
and vector mappings are provided.

Julian Merten
Universiy of Oxford
Feb 2015
julian.merten@physics.ox.ac.uk
http://www.julianmerten.net
***/

#ifndef    UNSTRUC_GRID_H
#define    UNSTRUC_GRID_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_statistics.h>
#include <flann/flann.hpp>
#include <mfree/radial_basis_function.h>

using namespace std;


class unstructured_grid
{

 protected:

  /*
    The number of grid nodes.
  */

  int num_nodes;

  /*
    Dimensionality of the grid.
  */

  int dim;

  /*
    The grid coordinate vector as a single linear vector. Each node comes
    with a coordinate pair so the length of this vector is 2 * num_nodes. 
    The order is arranged such that the first node occupies the vector
    entries 0 (x-coordinate) and 1 (y-coordinate), the second node 2 and 3...
  */

  vector<double> coordinates;

  /*
    The kD-tree of the grid nodes. 
  */

  vector<int> kD_tree;

  /*
    The distances of each node to its neighbours.
  */

  vector<double> distances;

  /*
    A flag which is triggered once the grid nodes are updated. It indicates
    if the kD-tree needs to be updated.
  */

  bool kD_update;


 public:

  /*
    Standard constructor. Just sets some flags.
  */

  unstructured_grid();

  /*
    Standard destructor. Frees memory.
  */

  ~unstructured_grid();

  /*
    Returns the number of grid nodes. If dim is true, it returns the 
    number of grid dimensions.
  */

  int return_grid_size(bool dim = 0);


  /*
    Bracket operator which returns the two coordinates of the grid node
    at position node as a double vector of length 2.
  */

  vector<double>  operator() (int node);

  /*
    Bracket operator which returns only one component of the coordinate 
    at postion node. Set component to 0 for x, anything else for y.
  */

  double operator() (int node, int component);

  /*
    Set function which lets you assign a specific coordinate value, to be given
    as double vector of length >= 2, for a specific node with index node.
  */

  void set(int node, vector<double> new_coordinates);

  /*
    Resets the full coordinate structure of the current grid, and writes
    new coordinates according to the input vector.
  */

  void set(vector<double> *new_coordinates);

  /*
    Completely resets the grid to a new number of nodes
    and clears the coordinate and tree vectors.
  */

  void set(int num_nodes_in);

  /*
    Builds the kD-tree. The user has to specify the number of
    nearst neighbours that should be considered. 
  */

  void build_tree(int nearest_neighbours);

  /*
    Returns the neighbours of the selected node index if
    the kD tree needs not to be updated.
  */

  vector<int> neighbours(int node);

  /*
    Returns the full set of neighbours if
    the kD tree needs not to be updated.
  */

  vector<int> neighbours();

  /*
    Returns the column inverted set of neighbours, which is needed
    to perfrom summing operations which run over columns as the fast index.
    The length of this summation can be found by dividing the total size
    of this vector by the number of grid nodes. The length counter
    contains for each column the number of non-zero elements.
  */

  int neighbours_col(vector<int> *neighbours, vector<int> *length_counter);

  /*
    This routines uses a given radial basis function to interpolate an input function on the unstructured grid
    to a given output grid. This grid needs to be defined by coordinate tuples, where the start of each tuple is 
    separated by stride. The function returns the average condition of the weight coefficient matrix, together with 
    the function interpolant on the output grid. Also the number of nearest neighbours for the interpolation needs to 
    be specified. 
  */

  double interpolate(vector<double> *output_grid, int stride, vector<double> *input_function, radial_basis_function *RBF, int knn, vector<double> *output_function);


  /*
    Creates the finite differencing weights for specific differential 
    operators. Selection are
    x
    y
    xx
    yy
    xy
    xxx
    yyy
    xxy
    xyy
    Laplace
    Neg_Laplace
    Output will be the finite differencing weights in a vector
    of size (number of nearest ebighbours) x (number of nodes). Standard gsl
    routines are used to solve the linear system. The function
    returns the average condition number for the linear system in each
    grid pixel that created the finite differencing weights.
  */

  virtual double create_finite_differences_weights(string selection, vector<double> *weights, radial_basis_function *RBF) = 0;

  /*
    Creates the finite differencing weights for specific differential 
    operators. The difference to the routine before is that this
    is the column ordered version which is used for summing operations
    over the columns of a finite differencing matrix instead of the 
    usual row version.Selection are
    x
    y
    xx
    yy
    xy
    xxx
    yyy
    xxy
    xyy
    Laplace
    Neg_Laplace
    Output will be the finite differencing weights in a vector
    of size 
    (a little more than the number of nearest neighbours) x (number of nodes). 
    Standard gsl
    routines are used to solve the linear system. The function
    returns the average condition number for the linear system in each
    grid pixel that created the finite differencing weights.
    Max_length is the length of each column and is e.g. returned
    by the neighbours_col routine.
  */

  vector<double> create_finite_differences_weights_col(string selection, radial_basis_function *RBF, int max_length);
 

  /*
    Performs a derivative operation on a vector and returns the derivative. 
    The input vector must be of the same length as the grid has nodes and 
    the order must be identical. Selections for the derivatives are 
    x
    y
    xx
    yy
    xy
    xxx
    yyy
    xxy
    xyy
    Laplace
    Neg_Laplace
    and you need to pass also a radial basis function. The function
    returns the average condition number for the linear system in each
    grid pixel that created the finite differencing weights.
  */

  double differentiate(vector<double> *in, string selection, radial_basis_function *RBF, vector<double> *out);
};

class unstructured_grid_1D : public unstructured_grid
{

 public:

  /*
    Standard constructor. Just takes the number of nodes and initiates
    all node coordinates with (0,0).
  */

  unstructured_grid_1D(int num_nodes_in);

  /*
    Constructor which creates a grid out of already existing coordinate
    information. The constructor needs the number of nodes and a pointer 
    to a double array which contains the input coordinates. It expects the
    x and y coordinates to be saved in neighbouring pairs and allows
    the definition of a stride. The stride defines the distance in the
    array between subsequent first elements of the pairs. 
  */

  unstructured_grid_1D(int num_nodes_in, double *input_coordinates, int stride);

  /*
    Creates the finite differencing weights for specific differential 
    operators. Selection are
    x
    xx
    xxx
    Output will be the finite differencing weights in a vector
    of size (number of nearest ebighbours) x (number of nodes). Standard gsl
    routines are used to solve the linear system. The function
    returns the average condition number for the linear system in each
    grid pixel that created the finite differencing weights.
  */ 

  double create_finite_differences_weights(string selection, vector<double> *weights, radial_basis_function *RBF);
};

class unstructured_grid_2D : public unstructured_grid
{

 public:

  /*
    Standard constructor. Just takes the number of nodes and initiates
    all node coordinates with (0,0).
  */

  unstructured_grid_2D(int num_nodes_in);

  /*
    Constructor which creates a grid out of already existing coordinate
    information. The constructor needs the number of nodes and a pointer 
    to a double array which contains the input coordinates. It expects the
    x and y coordinates to be saved in neighbouring pairs and allows
    the definition of a stride. The stride defines the distance in the
    array between subsequent first elements of the pairs. 
  */

  unstructured_grid_2D(int num_nodes_in, double *input_coordinates, int stride);
  
  /*
    Performs a derivative operation on a vector and returns the derivative. 
    The input vector must be of the same length as the grid has nodes and 
    the order must be identical. Selections for the derivatives are 
    x
    y
    xx
    yy
    xy
    xxx
    yyy
    xxy
    xyy
    Laplace
    Neg_Laplace
    and you need to pass also a radial basis function. The function
    returns the average condition number for the linear system in each
    grid pixel that created the finite differencing weights.
  */

  double create_finite_differences_weights(string selection, vector<double> *weights, radial_basis_function *RBF);

  /*
    Creates the finite differencing weights for specific differential 
    operators. The difference to the routine before is that this
    is the column ordered version which is used for summing operations
    over the columns of a finite differencing matrix instead of the 
    usual row version.Selection are
    x
    y
    xx
    yy
    xy
    xxx
    yyy
    xxy
    xyy
    Laplace
    Neg_Laplace
    Output will be the finite differencing weights in a vector
    of size 
    (a little more than the number of nearest neighbours) x (number of nodes). 
    Standard gsl routines are used to solve the linear system. The function
    returns the weight_col vector directly.
    Max_length is the length of each column and is e.g. returned
    by the neighbours_col routine.
  */
};

class unstructured_grid_3D : public unstructured_grid
{

 public:


  /*
    Standard constructor. Just takes the number of nodes and initiates
    all node coordinates with (0,0).
  */

  unstructured_grid_3D(int num_nodes_in);

  /*
    Constructor which creates a grid out of already existing coordinate
    information. The constructor needs the number of nodes and a pointer 
    to a double array which contains the input coordinates. It expects the
    x and y coordinates to be saved in neighbouring pairs and allows
    the definition of a stride. The stride defines the distance in the
    array between subsequent first elements of the pairs. 
  */

  unstructured_grid_3D(int num_nodes_in, double *input_coordinates, int stride);
  
  /*
    UNFORTUNATELY NOT IMPLEMENTED YET: JUST RETURNS 0

    Performs a derivative operation on a vector and returns the derivative. 
    The input vector must be of the same length as the grid has nodes and 
    the order must be identical. Selections for the derivatives are 
    x
    y
    z
    xx
    yy
    zz
    xy
    xz
    yz
    Laplace
    and you need to pass also a radial basis function. The function
    returns the average condition number for the linear system in each
    grid pixel that created the finite differencing weights.
  */

  double create_finite_differences_weights(string selection, vector<double> *weights, radial_basis_function *RBF);
};


/**
   These auxilliary routines takes a given sparse neighbour and coefficient
   vector which is saved in "row by row" format and converts it into 
   "column by column" format. The routine also needs the total number of grid
   nodes. The int return value is the maximum number of entries per column.
**/

int findif_row_col_convert(int dim, vector<int> *knn_in, vector<double> *coefficients_in, vector<int> *knn_out, vector<double> *coefficients_out); 

int findif_row_col_convert(int dim, vector<int> *knn_in, vector<int> *knn_out, vector<int> *length_counter_out);

vector<double> findif_row_col_convert(int dim, int max_length, vector<int> *knn_in, vector<double> *coefficients_in);



#endif    /*UNSTRUC_GRID_H*/
