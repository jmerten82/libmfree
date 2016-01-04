/*** mesh_free.h
Class which describes a mesh free domain
by the coordinates of its nodes and their kD-tree. 
Interpolation is enabled by radial
basis functions.

Julian Merten
Universiy of Oxford
Aug 2015
julian.merten@physics.ox.ac.uk
http://www.julianmerten.net
***/

#ifndef    MESH_FREE_H
#define    MESH_FREE_H

#include <vector>
#include <cmath>
#include <string>
#include <fstream>
#include <stdexcept>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_statistics.h>
#include <flann/flann.hpp>
#include <mfree/radial_basis_function.h>
#include <mfree/rbf_implementation.h>

using namespace std;

/**
   The main base class for the mesh-free domain. The grids can have 
   any dimension. Operators for inter-grid operations are provided. 
   Interpolation functionality is provided via radial basis functions. 
**/  


class mesh_free
{

 protected:

  /*
    The number of nodes.
  */

  int num_nodes;

  /*
    Dimensionality of the domain.
  */

  int dim;

  /*
    The coordinate vector as a single linear vector. Each node comes
    with a coordinate pair so the length of this vector is 2 * num_nodes. 
    The order is arranged such that the first node occupies the vector
    entries 0 (x-coordinate) and 1 (y-coordinate), the second node 2 and 3...
  */

  vector<double> coordinates;

  /*
    The kD-tree of the nodes. 
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
    Standard constructor, assumes D = 2 as the standard case. Can also directly
    read an input vector with coordinates. These must come in tuples of D
    consecutive coordinates, separated by a stride. If the stride is 0, it
    will be set to D.
  */

  mesh_free(int dim = 2, vector<double> *input = NULL, int stride = 0);

  /*
    The copy constructor.
  */

  mesh_free(mesh_free &input);


  /*
    Standard destructor. Frees memory.
  */

  ~mesh_free();

  /*
    The plus operator concatenates the coordinate vectors of two domains with
    the same grid dimension. Throws error if input dimensions are not equal.
  */

  mesh_free operator + (mesh_free &input);

  /*
    The increment operator adds the nodes of another domain
    to the current domain, if they have the same dimension.
  */

  void operator += (mesh_free &input);

  /*
    The minus operator c = a-b compares the coordinate vectors of two domains a, b with
    the same grid dimension. It outputs a new domain c with all nodes which are in a but not in b. 
  */

  mesh_free operator - (mesh_free &input);


  /*
    The decrement operator subtracts all nodes from another domain from the
    current domain, if they are of the same dimensionality. 
  */

  void operator -= (mesh_free &input);

  /*
    The multiplication operator scales the coordinates of the current domain
    with a number.
  */

  void operator *= (double scale);

  /*
    This overloaded operator scales each dimension of the coordinates 
    with a separate number. The scale must have equal or larger dimension
    than the current domain. 
  */

  void operator *= (vector<double> scale);


  /*
    This decrement operator deletes the last n nodes from the domain.
  */

  void operator - (unsigned int n);

  /*
    Returns the number of nodes. If dim is true, it returns the 
    number of  dimensions.
  */

  int return_grid_size(bool dim = 0);


  /*
    Bracket operator which returns the coordinates of a node
    as a double vector of length D.
  */

  vector<double>  operator() (int node);

  /*
    Bracket operator which returns only one component of the coordinate 
    at postion node.
  */

  double operator() (int node, int component);

  /*
    Set function which lets you assign a specific coordinate value. 
    The length of the vector must >= D.
  */

  void set(int node, vector<double> new_coordinates);

  /*
    Resets the full coordinate structure of the current domain, and writes
    new coordinates according to the input vector.
  */

  void set(vector<double> *new_coordinates);

  /*
    Completely resets to a new number of nodes
    and clears the coordinate and tree vectors.
  */

  void set(int num_nodes_in);

  /*
    Simple routine that prints the main grid properties on cout.
    You can define an initial statement that will be printed
    in front of the output information.
  */

  void print_info(string statement = "");

  /*
    Another simple routine writing the grid to ASCII,
    by simply putting the coordinates of all dimensions per node
    on a single ASCII output line. If you enable column description
    there will be a small description of the output columns.
  */

  void write_ASCII(string filename, bool col_description = 0);

  /*
    Builds the kD-tree. The user has to specify the number of
    nearst neighbours that should be considered. 
  */

  void build_tree(int nearest_neighbours = 16);

  /*
    Returns the neighbours of the selected node index if
    the kD tree needs not to be updated. If tree needs to be updated
    it is created with a standard value of 16 nearest neighbours. This values
    can be changed. 
  */

  vector<int> neighbours(int node, int nearest_neighbours = 16);

  /*
    Returns the full set of neighbours if
    the kD tree needs not to be updated. If the tree needs
    to be updated it is created with a standard value of nearest neighbours of 16.
    This number can be changed. 
  */

  vector<int> neighbours(int nearest_neighbours = 16);

  /*
    Same as above but return the full set of distances to the 
    neighbours. 
  */

  vector<double> distances(int nearest_neighbours = 16);

  /*
    Returns the column inverted set of neighbours, which is needed
    to perform summing operations which run over columns as the fast index.
    The length of this summation can be found by dividing the total size
    of this vector by the number of grid nodes. The length counter
    contains for each column the number of non-zero elements.
  */

  int neighbours_col(vector<int> *neighbours, vector<int> *length_counter, int nearest_neighbours = 16);

  /*
    This routine takes another mesh-free structure, checks if dimensionalities
    match and embeds this strcuture in the existing grid. This means
    that it searches the knn nearest neighbours of all nodes of the input
    grid in the current grid and returns a vector with their indices. 
    If the respective vector is provided, also the L2 distances are
    written out. 
  */

  vector<int> embed(mesh_free *input, int knn = 16);

  /*
    Overloaded version of the embed function. This one takes directly 
    a coordinate vector. The problem with this approach is that it might 
    not be dimensionality safe. A stride between coorindate tuples can be
    provided. If not it assumes it to be the dimensionality of the current grid.
  */

  vector<int> embed(vector<double> *input, int knn = 16, int stride = 0);


  /*
    This routines uses a given radial basis function to interpolate an input function on a mesh free domain
    to a given output domain. This domain needs to be defined by coordinate tuples, where the start of each tuple is 
    separated by stride. The function returns the average condition of the weight coefficient matrix, together with 
    the function interpolant on the output grid. Also the number of nearest neighbours for the interpolation needs to 
    be specified. 
  */

  double interpolate(vector<double> *output_grid, vector<double> *input_function, vector<double> *output_function,  radial_basis_function *RBF, int knn = 16, int stride = 0);
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



#endif    /*MESH_FREE_H*/
