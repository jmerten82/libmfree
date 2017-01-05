/*** mesh_free_differentiate.h
 
Derivatives on the grid are calculated via Radial-Basis
Function-based finite differencing stencils.
Julian Merten
Universiy of Oxford
Feb 2015
julian.merten@physics.ox.ac.uk
http://www.julianmerten.net
***/

#ifndef    MESH_FREE_DIFF_H
#define    MESH_FREE_DIFF_H

#include <vector>
#include <cmath>
#include <stdexcept>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_statistics.h>
#include <flann/flann.hpp>
#include <mfree/radial_basis_function.h>
#include <mfree/mesh_free.h>
#include <mfree/polynomial_terms.h>

using namespace std;

/**
   This abstract base class cannot stand by its own but enables all 
   functionality for differentiating on a mesh-free domain. Derives from the
   more general mesh-free class mesh_free. It does not
   need a constructor since it is purely abstract. 
**/

class mesh_free_differentiate : public mesh_free
{

 public:

  /*
    Standard constuctor for inheritance reasons. 
  */

 mesh_free_differentiate(int dim = 2, vector<double> *input = NULL, int stride = 0) : mesh_free(dim,input,stride) {};

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
    This function is idential to the above but allows for an additional 
    polynomial support term of arbitrary degree. 
  */

  virtual double create_finite_differences_weights(string selection, unsigned pdeg, vector<double> *weights, radial_basis_function *RBF) = 0;

  /*
    The same function as above but with a shaped RBF and with a varying
    shape parameter.
  */

  virtual double create_finite_differences_weights(string selection, vector<double> *weights, radial_basis_function_shape *RBF, vector<double> *adaptive_shape_parameter) = 0;

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
    Again the same function as above but for a shaped RBF with varying 
    shape parameter.
  */

  vector<double> create_finite_differences_weights_col(string selection, radial_basis_function_shape *RBF, vector<double> *adaptive_shape_parameter, int max_length);



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

  /*
    This is another differentiate function which uses a RBF with shape
    and adaptive shape parameter.
  */

  double differentiate(vector<double> *in, string selection, radial_basis_function_shape *RBF, vector<double> *adaptive_shape_parameter, vector<double> *out);

  /*
    This version of the differentiation routine does not calculate derivatives
    on the existing mesh-free domain, but for a specific set of target
    coordinates that has to be provided.
  */

  virtual double differentiate(vector<double> *target_coordinates, vector<double> *in, string selection, radial_basis_function *RBF, vector<double> *out, int nn = 16) = 0;

  /*
    This should be the last version of this routine which now includes the 
    weight creation function which accounts for polynomial terms of arbitrary
    degree.
  */ 

  double differentiate(vector<double> *in, string selection, radial_basis_function *RBF, unsigned pdeg,  vector<double> *out);
};


/**
   Base class for 1D mesh free differentiation. This implements RBF 
   differtiation with a polynomial term up to third order. 
**/

class mesh_free_1D : public mesh_free_differentiate
{

 public:

  /*
    Standard constructor. Call mesh_free constrcutor with dim 1
  */

 mesh_free_1D(vector<double> *input = NULL, int stride = 0) : mesh_free_differentiate(1,input,stride) {};

  /*
    Copy constructor.
  */

 mesh_free_1D(mesh_free_1D &input);


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

  /*
    This version of the differentiation uses an RBF with shape parameter
    and a spatially varying one. 
  */

  double create_finite_differences_weights(string selection, vector<double> *weights, radial_basis_function_shape *RBF, vector<double> *adaptive_shape_parameter);

  /*
    This implementation is still somewhat experiemental and adds polynomial 
    support of arbitrary order to the creation of the weight. 
    The maximum polynomial order must be specified.
  */

  double create_finite_differences_weights(string selection, uint pdeg,  vector<double> *weights, radial_basis_function *RBF);

  /*
    This line is needed because of function hiding in derived classes. 
  */ 

  using mesh_free_differentiate::differentiate;

  /*
    This version of the differentiation routine does not calculate derivatives
    on the existing mesh-free domain, but for a specific set of target
    coordinates that has to be provided.
  */

  double differentiate(vector<double> *target_coordinates, vector<double> *in, string selection, radial_basis_function *RBF, vector<double> *out, int nn = 16);


};



/**
   Base class for 2D mesh free differentiation. This implements RBF 
   differtiation with a polynomial term up to third order. 
**/


class mesh_free_2D : public mesh_free_differentiate
{
  
 public:
  
  /*
    Standard constructor. Call mesh_free constrcutor with dim 1
  */
  
 mesh_free_2D(vector<double> *input = NULL, int stride = 0) : mesh_free_differentiate(2,input,stride) {};
  
  /*
    Copy constructor.
  */
  
  mesh_free_2D(mesh_free_2D &input);
  
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


  double create_finite_differences_weights(string selection, vector<double> *weights, radial_basis_function *RBF);

  /*
    This version of the differentiation uses an RBF with shape parameter
    and a spatially varying one. 
  */

  double create_finite_differences_weights(string selection, vector<double> *weights, radial_basis_function_shape *RBF, vector<double> *adaptive_shape_parameter);

  /*
    This implementation is still somewhat experiemental and adds polynomial 
    support of arbitrary order to the creation of the weight. 
    The maximum polynomial order must be specified.
  */

  double create_finite_differences_weights(string selection, uint pdeg,  vector<double> *weights, radial_basis_function *RBF);

  /*
    This line is needed because of function hiding in derived classes. 
  */ 

  using mesh_free_differentiate::differentiate;

  /*
    This version of the differentiation routine does not calculate derivatives
    on the existing mesh-free domain, but for a specific set of target
    coordinates that has to be provided.
  */

  double differentiate(vector<double> *target_coordinates, vector<double> *in, string selection, radial_basis_function *RBF, vector<double> *out, int nn = 16);

};


/**
   Base class for 3D mesh free differentiation. This implements RBF 
   differtiation with a polynomial term up to third order. 
**/

class mesh_free_3D : public mesh_free_differentiate
{
  
 public:
  
  /*
    Standard constructor. Call mesh_free constrcutor with dim 1
  */
  
 mesh_free_3D(vector<double> *input = NULL, int stride = 0) : mesh_free_differentiate(3,input,stride) {};
  
  /*
    Copy constructor.
  */
  
  mesh_free_3D(mesh_free_3D &input);
  
  /*
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

  /*
    This version of the differentiation uses an RBF with shape parameter
    and a spatially varying one. 
  */

  double create_finite_differences_weights(string selection, vector<double> *weights, radial_basis_function_shape *RBF, vector<double> *adaptive_shape_parameter);

  /*
    This implementation is still somewhat experiemental and adds polynomial 
    support of arbitrary order to the creation of the weight. 
    The maximum polynomial order must be specified.
  */

  double create_finite_differences_weights(string selection, uint pdeg,  vector<double> *weights, radial_basis_function *RBF);

  /*
    This line is needed because of function hiding in derived classes. 
  */ 

  using mesh_free_differentiate::differentiate;

  /*
    This version of the differentiation routine does not calculate derivatives
    on the existing mesh-free domain, but for a specific set of target
    coordinates that has to be provided.
  */

  double differentiate(vector<double> *target_coordinates, vector<double> *in, string selection, radial_basis_function *RBF, vector<double> *out, int nn = 16);

};



#endif    /*UNSTRUC_GRID_DIFF_H*/
