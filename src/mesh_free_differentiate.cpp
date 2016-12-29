/*** mesh_free_differentiate.cpp
 
Derivatives on the grid are calculated via Radial-Basis
Function-based finite differencing stencils.
Julian Merten
Universiy of Oxford
Feb 2015
julian.merten@physics.ox.ac.uk
http://www.julianmerten.net
***/

#include <mfree/mesh_free_differentiate.h>

using namespace std;

double mesh_free_differentiate::differentiate(vector<double> *in, string selection, radial_basis_function *RBF, vector<double> *out)
{

  //Checking the size of the input vector

  if(in->size() != num_nodes)
    {
      throw invalid_argument("UNSTRUC_GRID: Input vector for diff is invalid.");
    }

  //Creating the finite differences

  vector<double> weights;
  double condition = create_finite_differences_weights(selection, &weights, RBF);
  out->resize(num_nodes);
  int num_neighbours = kD_tree.size() / num_nodes;
  double value;
  int seed, pos;

  //Performing the finite differencing

  for(int i = 0; i < num_nodes; i++)
    {
      value = 0.;
      seed = i*num_neighbours;
      for(int j = 0; j < num_neighbours; j++)
	{
	  pos = kD_tree[seed+j];
	  value += (*in)[pos]*weights[seed+j];
	}
      (*out)[i] = value;
    }
  return condition;
}


double mesh_free_differentiate::differentiate(vector<double> *in, string selection, radial_basis_function_shape *RBF, vector<double> *adaptive_shape_parameter, vector<double> *out)
{

  //Checking the size of the input vectors

  if(in->size() != num_nodes)
    {
      throw invalid_argument("MFREE_DIFF: Input vector for diff is invalid.");
    }
  if(adaptive_shape_parameter->size() < num_nodes)
    {
      throw invalid_argument("MFREE_DIFF: Adaptive shape vector for diff is invalid.");
    }

  //Creating the finite differences

  vector<double> weights;
  double condition = create_finite_differences_weights(selection, &weights, RBF,adaptive_shape_parameter);
  out->resize(num_nodes);
  int num_neighbours = kD_tree.size() / num_nodes;
  double value;
  int seed, pos;

  //Performing the finite differencing

  for(int i = 0; i < num_nodes; i++)
    {
      value = 0.;
      seed = i*num_neighbours;
      for(int j = 0; j < num_neighbours; j++)
	{
	  pos = kD_tree[seed+j];
	  value += (*in)[pos]*weights[seed+j];
	}
      (*out)[i] = value;
    }
  return condition;



}


vector<double> mesh_free_differentiate::create_finite_differences_weights_col(string selection, radial_basis_function *RBF, int max_length)
{
  vector<double> aux;
  create_finite_differences_weights(selection, &aux, RBF);
  return findif_row_col_convert(num_nodes, max_length, &kD_tree, &aux);
}

vector<double> mesh_free_differentiate::create_finite_differences_weights_col(string selection, radial_basis_function_shape *RBF, vector<double> *adaptive_shape_parameter, int max_length)
{
  vector<double> aux;
  create_finite_differences_weights(selection, &aux, RBF,adaptive_shape_parameter);
  return findif_row_col_convert(num_nodes, max_length, &kD_tree, &aux);
}

mesh_free_1D::mesh_free_1D(mesh_free_1D &input)
{
  dim = input.return_grid_size(1);
  num_nodes = input.return_grid_size(0);
  distances = input.distances;
  kD_update = input.kD_update;
  kD_tree = input.kD_tree;
}
						      


double mesh_free_1D::create_finite_differences_weights(string selection, vector<double> *weights, radial_basis_function *RBF)
{
  //Checking if the tree is up to date
  if(kD_update)
    {
      throw invalid_argument("UNSTRUC_GRID: Tree needs to be updated befor FD.");
    }

  //Determining the size of the problem
  if(kD_tree.size() % num_nodes != 0)
    {
      throw logic_error("UNSTRUC_GRID: The kD-tree size is invalid.");
    }
  int num_neighbours = kD_tree.size() / num_nodes;

  //Allocating work space for linear system to get finite differencing 
  //weights
  gsl_matrix *A = gsl_matrix_calloc(num_neighbours+4,num_neighbours+4);
  gsl_matrix *V = gsl_matrix_calloc(A->size1,A->size2);
  gsl_vector *S = gsl_vector_calloc(A->size1);
  gsl_vector *work_dummy = gsl_vector_calloc(A->size1);
  gsl_vector *b = gsl_vector_calloc(A->size1);
  gsl_vector *x = gsl_vector_calloc(A->size1);
  vector<double> condition;

  //Setting the stage for the output weights
  weights->resize(kD_tree.size());

  //Allocating all helper quantities in the process
  double x_eval, x_1, value, x_node, min, max;
  int tree_position_seed, index1, index2;


  //Looping through all grid nodes
  for(int global = 0; global < num_nodes; global++)
    {
      //Getting evaluation coordinate
      x_eval = coordinates[global];
      //Resetting linear system
      gsl_matrix_set_identity(A);
      gsl_vector_set_all(b,0.);
      gsl_vector_set_all(x,0.);
      gsl_matrix_set_all(V,0.);
      gsl_vector_set_all(S,0.);
      gsl_vector_set_all(work_dummy,0.);
      tree_position_seed = global*num_neighbours;

      //Building up the linear system of equations

      for(int i = 0; i < num_neighbours; i++)
	{
	  //Getting node position and setting RBF to reference
	  x_node = coordinates[kD_tree[tree_position_seed+i]];
	  RBF->set_coordinate_offset(x_node);

	  //Setting main body of A matrix
	  for(int j = i+1; j < num_neighbours; j++)
	    {
	      //calculating the distance
	      index1 = kD_tree[tree_position_seed+i];
	      index2 = kD_tree[tree_position_seed+j];
	      x_1 = coordinates[index1];
	      x_1 -= coordinates[index2];
	      //Evaluating radial basis function
	      //This square root is not super efficient
	      //Should be implemented in r^2 manner
	      value = (* RBF)(x_1);
	      gsl_matrix_set(A,i,j,value);
	      gsl_matrix_set(A,j,i,value);
	    }

	  //Creating the missing entries in the coefficient matrix
	  //and the main part of the result vector
	  gsl_matrix_set(A,i,num_neighbours,1);
	  gsl_matrix_set(A,num_neighbours,i,1.);
	  gsl_matrix_set(A,i,num_neighbours+1,x_node);
	  gsl_matrix_set(A,num_neighbours+1,i,x_node);
	  gsl_matrix_set(A,i,num_neighbours+2,x_node*x_node);
	  gsl_matrix_set(A,num_neighbours+2,i,gsl_matrix_get(A,i,num_neighbours+2));
	  gsl_matrix_set(A,i,num_neighbours+3,x_node*x_node*x_node);
	  gsl_matrix_set(A,num_neighbours+3,i,gsl_matrix_get(A,i,num_neighbours+3));	   
      //Creating the data vector	  
	  if(selection == "x")
	    {
	      gsl_vector_set(b,i,RBF->Dx(x_eval));
	    }	  
	  else if(selection == "xx")
	    {
	      gsl_vector_set(b,i,RBF->Dxx(x_eval));
	    }	
	  else if(selection == "xxx")
	    {
	      gsl_vector_set(b,i,RBF->Dxxx(x_eval));
	    }
	}

      //Setting missing values in result vector
      if(selection == "x")
	{
	  gsl_vector_set(b,num_neighbours+1,1.);
	  gsl_vector_set(b,num_neighbours+2,2.*x_eval);
	  gsl_vector_set(b,num_neighbours+3,3.*x_eval*x_eval);
	}
      else if(selection == "xx")
	{
	  gsl_vector_set(b,num_neighbours+2,2.);
	  gsl_vector_set(b,num_neighbours+3,6.*x_eval);

	}
      else if(selection == "xxx")
	{
	  gsl_vector_set(b,num_neighbours+3,6.);
	}

      //Checking for the condition of the linear system via singular value
      //decomposition.
      gsl_linalg_SV_decomp(A, V, S, work_dummy);
      gsl_vector_minmax(S,&min,&max);
      condition.push_back(max/min);

      //Solving linear system using the SVD. 
      gsl_linalg_SV_solve(A, V, S, b, x);

      //Writing the output weights 
      for(int i = 0; i < num_neighbours; i++)
	{
	  (* weights)[tree_position_seed+i] = gsl_vector_get(x,i);
	}
    }
  //Cleanig up gsl work force
  gsl_matrix_free(A);
  gsl_vector_free(b);
  gsl_vector_free(x);
  gsl_matrix_free(V);
  gsl_vector_free(S);
  gsl_vector_free(work_dummy);


  return gsl_stats_mean(&condition[0], 1, num_nodes);
}

double mesh_free_1D::create_finite_differences_weights(string selection, vector<double> *weights, radial_basis_function_shape *RBF, vector<double> *adaptive_shape_parameter)
{
  //Checking if the tree is up to date
  if(kD_update)
    {
      throw invalid_argument("MFREE_DIFF: Tree needs to be updated befor FD.");
    }

  //Checking if shapes are valid
  if(adaptive_shape_parameter->size() < num_nodes)
    {
      throw invalid_argument("MFREE_DIFF: Adaptive shape parameter vector too short.");
    }


  //Determining the size of the problem
  if(kD_tree.size() % num_nodes != 0)
    {
      throw logic_error("UNSTRUC_GRID: The kD-tree size is invalid.");
    }
  int num_neighbours = kD_tree.size() / num_nodes;

  double shape_save = RBF->show_epsilon();

  //Allocating work space for linear system to get finite differencing 
  //weights
  gsl_matrix *A = gsl_matrix_calloc(num_neighbours+4,num_neighbours+4);
  gsl_matrix *V = gsl_matrix_calloc(A->size1,A->size2);
  gsl_vector *S = gsl_vector_calloc(A->size1);
  gsl_vector *work_dummy = gsl_vector_calloc(A->size1);
  gsl_vector *b = gsl_vector_calloc(A->size1);
  gsl_vector *x = gsl_vector_calloc(A->size1);
  vector<double> condition;

  //Setting the stage for the output weights
  weights->resize(kD_tree.size());

  //Allocating all helper quantities in the process
  double x_eval, x_1, value, x_node, min, max;
  int tree_position_seed, index1, index2;


  //Looping through all grid nodes
  for(int global = 0; global < num_nodes; global++)
    {
      RBF->set_epsilon((*adaptive_shape_parameter)[global]);

      //Getting evaluation coordinate
      x_eval = coordinates[global];
      //Resetting linear system
      gsl_matrix_set_identity(A);
      gsl_vector_set_all(b,0.);
      gsl_vector_set_all(x,0.);
      gsl_matrix_set_all(V,0.);
      gsl_vector_set_all(S,0.);
      gsl_vector_set_all(work_dummy,0.);
      tree_position_seed = global*num_neighbours;

      //Building up the linear system of equations

      for(int i = 0; i < num_neighbours; i++)
	{
	  //Getting node position and setting RBF to reference
	  x_node = coordinates[kD_tree[tree_position_seed+i]];
	  RBF->set_coordinate_offset(x_node);

	  //Setting main body of A matrix
	  for(int j = i+1; j < num_neighbours; j++)
	    {
	      //calculating the distance
	      index1 = kD_tree[tree_position_seed+i];
	      index2 = kD_tree[tree_position_seed+j];
	      x_1 = coordinates[index1];
	      x_1 -= coordinates[index2];
	      //Evaluating radial basis function
	      //This square root is not super efficient
	      //Should be implemented in r^2 manner
	      value = (* RBF)(x_1);
	      gsl_matrix_set(A,i,j,value);
	      gsl_matrix_set(A,j,i,value);
	    }

	  //Creating the missing entries in the coefficient matrix
	  //and the main part of the result vector
	  gsl_matrix_set(A,i,num_neighbours,1);
	  gsl_matrix_set(A,num_neighbours,i,1.);
	  gsl_matrix_set(A,i,num_neighbours+1,x_node);
	  gsl_matrix_set(A,num_neighbours+1,i,x_node);
	  gsl_matrix_set(A,i,num_neighbours+2,x_node*x_node);
	  gsl_matrix_set(A,num_neighbours+2,i,gsl_matrix_get(A,i,num_neighbours+2));
	  gsl_matrix_set(A,i,num_neighbours+3,x_node*x_node*x_node);
	  gsl_matrix_set(A,num_neighbours+3,i,gsl_matrix_get(A,i,num_neighbours+3));	   
      //Creating the data vector	  
	  if(selection == "x")
	    {
	      gsl_vector_set(b,i,RBF->Dx(x_eval));
	    }	  
	  else if(selection == "xx")
	    {
	      gsl_vector_set(b,i,RBF->Dxx(x_eval));
	    }	
	  else if(selection == "xxx")
	    {
	      gsl_vector_set(b,i,RBF->Dxxx(x_eval));
	    }
	}

      //Setting missing values in result vector
      if(selection == "x")
	{
	  gsl_vector_set(b,num_neighbours+1,1.);
	  gsl_vector_set(b,num_neighbours+2,2.*x_eval);
	  gsl_vector_set(b,num_neighbours+3,3.*x_eval*x_eval);
	}
      else if(selection == "xx")
	{
	  gsl_vector_set(b,num_neighbours+2,2.);
	  gsl_vector_set(b,num_neighbours+3,6.*x_eval);

	}
      else if(selection == "xxx")
	{
	  gsl_vector_set(b,num_neighbours+3,6.);
	}

      //Checking for the condition of the linear system via singular value
      //decomposition.
      gsl_linalg_SV_decomp(A, V, S, work_dummy);
      gsl_vector_minmax(S,&min,&max);
      condition.push_back(max/min);

      //Solving linear system using the SVD. 
      gsl_linalg_SV_solve(A, V, S, b, x);

      //Writing the output weights 
      for(int i = 0; i < num_neighbours; i++)
	{
	  (* weights)[tree_position_seed+i] = gsl_vector_get(x,i);
	}
    }
  //Cleanig up gsl work force
  gsl_matrix_free(A);
  gsl_vector_free(b);
  gsl_vector_free(x);
  gsl_matrix_free(V);
  gsl_vector_free(S);
  gsl_vector_free(work_dummy);

  RBF->set_epsilon(shape_save);

  return gsl_stats_mean(&condition[0], 1, num_nodes);
}

double mesh_free_1D::differentiate(vector<double> *target_coordinates, vector<double> *in, string selection, radial_basis_function *RBF, vector<double> *out, int nn)
{

  if(target_coordinates->size() < dim)
    {
      throw invalid_argument("MFREE_DIFF: Target coordinate vector invalid.");
    }

  if(in->size() < num_nodes)
    {
      throw invalid_argument("MFREE_DIFF: Input vector invalid.");
    }

  int num_out_nodes = target_coordinates->size() / dim;
  vector<int> output_tree;
  vector<double> output_distances;
  output_tree.resize(nn*num_out_nodes);
  output_distances.resize(nn*num_out_nodes);

  //Creating a tree for the target vector.

  flann::Matrix<double> flann_base(&coordinates[0],num_nodes,dim);
  flann::Matrix<double> flann_search(&(*target_coordinates)[0],num_out_nodes,dim);
  flann::Matrix<int> flann_tree(&output_tree[0],num_out_nodes,nn);
  flann::Matrix<double> flann_distances(&distances[0],num_out_nodes,nn);

  //Creating flann index
  flann::Index<flann::L2<double> > index(flann_base, flann::KDTreeIndexParams(4));
  index.buildIndex();
 //Performing flann nearest neighbours search
  index.knnSearch(flann_search, flann_tree, flann_distances, nn, flann::SearchParams(128));


  //Allocating work space for linear system to get finite differencing 
  //weights
  gsl_matrix *A = gsl_matrix_calloc(nn+4,nn+4);
  gsl_matrix *V = gsl_matrix_calloc(A->size1,A->size2);
  gsl_vector *S = gsl_vector_calloc(A->size1);
  gsl_vector *work_dummy = gsl_vector_calloc(A->size1);
  gsl_vector *b = gsl_vector_calloc(A->size1);
  gsl_vector *x = gsl_vector_calloc(A->size1);
  vector<double> condition;

  //Setting the stage for the output weights
  vector<double> output_weights;
  output_weights.resize(output_tree.size());

  //Allocating all helper quantities in the process
  double x_eval, x_1, value, x_node, min, max;
  int tree_position_seed, index1, index2;


  //Looping through all grid nodes
  for(int global = 0; global < num_out_nodes; global++)
    {
      //Getting evaluation coordinate
      x_eval = (*target_coordinates)[global];
      //Resetting linear system
      gsl_matrix_set_identity(A);
      gsl_vector_set_all(b,0.);
      gsl_vector_set_all(x,0.);
      gsl_matrix_set_all(V,0.);
      gsl_vector_set_all(S,0.);
      gsl_vector_set_all(work_dummy,0.);
      tree_position_seed = global*nn;

      //Building up the linear system of equations

      for(int i = 0; i < nn; i++)
	{
	  //Getting node position and setting RBF to reference
	  x_node = coordinates[output_tree[tree_position_seed+i]];
	  RBF->set_coordinate_offset(x_node);

	  //Setting main body of A matrix
	  for(int j = i+1; j < nn; j++)
	    {
	      //calculating the distance
	      index1 = output_tree[tree_position_seed+i];
	      index2 = output_tree[tree_position_seed+j];
	      x_1 = coordinates[index1];
	      x_1 -= coordinates[index2];
	      //Evaluating radial basis function
	      //This square root is not super efficient
	      //Should be implemented in r^2 manner
	      value = (* RBF)(x_1);
	      gsl_matrix_set(A,i,j,value);
	      gsl_matrix_set(A,j,i,value);
	    }

	  //Creating the missing entries in the coefficient matrix
	  //and the main part of the result vector
	  gsl_matrix_set(A,i,nn,1);
	  gsl_matrix_set(A,nn,i,1.);
	  gsl_matrix_set(A,i,nn+1,x_node);
	  gsl_matrix_set(A,nn+1,i,x_node);
	  gsl_matrix_set(A,i,nn+2,x_node*x_node);
	  gsl_matrix_set(A,nn+2,i,gsl_matrix_get(A,i,nn+2));
	  gsl_matrix_set(A,i,nn+3,x_node*x_node*x_node);
	  gsl_matrix_set(A,nn+3,i,gsl_matrix_get(A,i,nn+3));	   
      //Creating the data vector	  
	  if(selection == "x")
	    {
	      gsl_vector_set(b,i,RBF->Dx(x_eval));
	    }	  
	  else if(selection == "xx")
	    {
	      gsl_vector_set(b,i,RBF->Dxx(x_eval));
	    }	
	  else if(selection == "xxx")
	    {
	      gsl_vector_set(b,i,RBF->Dxxx(x_eval));
	    }
	}

      //Setting missing values in result vector
      if(selection == "x")
	{
	  gsl_vector_set(b,nn+1,1.);
	  gsl_vector_set(b,nn+2,2.*x_eval);
	  gsl_vector_set(b,nn+3,3.*x_eval*x_eval);
	}
      else if(selection == "xx")
	{
	  gsl_vector_set(b,nn+2,2.);
	  gsl_vector_set(b,nn+3,6.*x_eval);

	}
      else if(selection == "xxx")
	{
	  gsl_vector_set(b,nn+3,6.);
	}

      //Checking for the condition of the linear system via singular value
      //decomposition.
      gsl_linalg_SV_decomp(A, V, S, work_dummy);
      gsl_vector_minmax(S,&min,&max);
      condition.push_back(max/min);

      //Solving linear system using the SVD. 
      gsl_linalg_SV_solve(A, V, S, b, x);

      //Writing the output weights 
      for(int i = 0; i < nn; i++)
	{
	  output_weights[tree_position_seed+i] = gsl_vector_get(x,i);
	}
    }
  //Cleanig up gsl work force
  gsl_matrix_free(A);
  gsl_vector_free(b);
  gsl_vector_free(x);
  gsl_matrix_free(V);
  gsl_vector_free(S);
  gsl_vector_free(work_dummy);

  double output = gsl_stats_mean(&condition[0], 1, num_out_nodes);

  out->resize(num_out_nodes);
  int seed, pos;

  //Performing the finite differencing

  for(int i = 0; i < num_out_nodes; i++)
    {
      value = 0.;
      seed = i*nn;
      for(int j = 0; j < nn; j++)
	{
	  pos = output_tree[seed+j];
	  value += (*in)[pos]*output_weights[seed+j];
	}
      (*out)[i] = value;
    }


  return output;

}




mesh_free_2D::mesh_free_2D(mesh_free_2D &input)
{
  dim = input.return_grid_size(1);
  num_nodes = input.return_grid_size(0);
  distances = input.distances;
  kD_update = input.kD_update;
  kD_tree = input.kD_tree;
}

double mesh_free_2D::create_finite_differences_weights(string selection, vector<double> *weights, radial_basis_function *RBF)
{

  //Checking if the tree is up to date
  if(kD_update)
    {
      throw invalid_argument("UNSTRUC_GRID: Tree needs to be updated befor FD.");
    }

  //Determining the size of the problem
  if(kD_tree.size() % num_nodes != 0)
    {
      throw logic_error("UNSTRUC_GRID: The kD-tree size is invalid.");
    }
  int num_neighbours = kD_tree.size() / num_nodes;

  //Allocating work space for linear system to get finite differencing 
  //weights
  gsl_matrix *A = gsl_matrix_calloc(num_neighbours+10,num_neighbours+10);
  gsl_matrix *V = gsl_matrix_calloc(A->size1,A->size2);
  gsl_vector *S = gsl_vector_calloc(A->size1);
  gsl_vector *work_dummy = gsl_vector_calloc(A->size1);
  gsl_vector *b = gsl_vector_calloc(A->size1);
  gsl_vector *x = gsl_vector_calloc(A->size1);
  vector<double> condition;

  //Setting the stage for the output weights
  weights->resize(kD_tree.size());

  //Allocating all helper quantities in the process
  double x_eval, y_eval, x_1, y_1, value, x_node, y_node, min, max;
  int tree_position_seed, index1, index2;


  //Looping through all grid nodes
  for(int global = 0; global < num_nodes; global++)
    {
      //Getting evaluation coordinate
      x_eval = coordinates[global*2];
      y_eval = coordinates[global*2+1];
      //Resetting linear system
      gsl_matrix_set_identity(A);
      gsl_vector_set_all(b,0.);
      gsl_vector_set_all(x,0.);
      gsl_matrix_set_all(V,0.);
      gsl_vector_set_all(S,0.);
      gsl_vector_set_all(work_dummy,0.);
      tree_position_seed = global*num_neighbours;

      //Building up the linear system of equations

      for(int i = 0; i < num_neighbours; i++)
	{
	  //Getting node position and setting RBF to reference
	  x_node = coordinates[kD_tree[tree_position_seed+i]*2];
	  y_node = coordinates[kD_tree[tree_position_seed+i]*2+1];
	  RBF->set_coordinate_offset(x_node, y_node);

	  //Setting main body of A matrix
	  for(int j = i+1; j < num_neighbours; j++)
	    {
	      //calculating the distance
	      index1 = kD_tree[tree_position_seed+i];
	      index2 = kD_tree[tree_position_seed+j];
	      x_1 = coordinates[index1*2];
	      x_1 -= coordinates[index2*2];
	      x_1 *= x_1;   
	      y_1 = coordinates[index1*2+1];
	      y_1 -= coordinates[index2*2+1];
	      y_1 *= y_1;
	      x_1 += y_1;
	      //Evaluating radial basis function
	      //This square root is not super efficient
	      //Should be implemented in r^2 manner
	      value = (* RBF)(sqrt(x_1));
	      gsl_matrix_set(A,i,j,value);
	      gsl_matrix_set(A,j,i,value);
	    }

	  //Creating the missing entries in the coefficient matrix
	  //and the main part of the result vector
	  gsl_matrix_set(A,i,num_neighbours,1);
	  gsl_matrix_set(A,num_neighbours,i,1.);
	  gsl_matrix_set(A,i,num_neighbours+1,x_node);
	  gsl_matrix_set(A,num_neighbours+1,i,x_node);
	  gsl_matrix_set(A,i,num_neighbours+2,y_node);
	  gsl_matrix_set(A,num_neighbours+2,i,y_node);
	  gsl_matrix_set(A,i,num_neighbours+3,x_node*x_node);
	  gsl_matrix_set(A,num_neighbours+3,i,gsl_matrix_get(A,i,num_neighbours+3));
	  gsl_matrix_set(A,i,num_neighbours+4,y_node*y_node);
	  gsl_matrix_set(A,num_neighbours+4,i,gsl_matrix_get(A,i,num_neighbours+4));
	  gsl_matrix_set(A,i,num_neighbours+5,x_node*y_node);
	  gsl_matrix_set(A,num_neighbours+5,i,gsl_matrix_get(A,i,num_neighbours+5));
	  gsl_matrix_set(A,i,num_neighbours+6,x_node*x_node*x_node);
	  gsl_matrix_set(A,num_neighbours+6,i,gsl_matrix_get(A,i,num_neighbours+6));	  
	  gsl_matrix_set(A,i,num_neighbours+7,y_node*y_node*y_node);
	  gsl_matrix_set(A,num_neighbours+7,i,gsl_matrix_get(A,i,num_neighbours+7));
	  gsl_matrix_set(A,i,num_neighbours+8,gsl_matrix_get(A,i,num_neighbours+3)*y_node);
	  gsl_matrix_set(A,num_neighbours+8,i,gsl_matrix_get(A,i,num_neighbours+8));
	  gsl_matrix_set(A,i,num_neighbours+9,x_node*gsl_matrix_get(A,i,num_neighbours+4));
	  gsl_matrix_set(A,num_neighbours+9,i,gsl_matrix_get(A,i,num_neighbours+9));  
      //Creating the data vector	  
	  if(selection == "x")
	    {
	      gsl_vector_set(b,i,RBF->Dx(x_eval,y_eval));
	    }	  
	  else if(selection == "y")
	    {
	      gsl_vector_set(b,i,RBF->Dy(x_eval,y_eval));
	    }
	  else if(selection == "xx")
	    {
	      gsl_vector_set(b,i,RBF->Dxx(x_eval,y_eval));
	    }	
	  else if(selection == "yy")
	    {
	      gsl_vector_set(b,i,RBF->Dyy(x_eval,y_eval));
	    }
	  else if(selection == "xy")
	    {
	      gsl_vector_set(b,i,RBF->Dxy(x_eval,y_eval));
	    }
	  else if(selection == "xxx")
	    {
	      gsl_vector_set(b,i,RBF->Dxxx(x_eval,y_eval));
	    }
	  else if(selection == "yyy")
	    {
	      gsl_vector_set(b,i,RBF->Dyyy(x_eval,y_eval));
	    }
	  else if(selection == "xxy")
	    {
	      gsl_vector_set(b,i,RBF->Dxxy(x_eval,y_eval));
	    }
	  else if(selection == "xyy")
	    {
	      gsl_vector_set(b,i,RBF->Dxyy(x_eval,y_eval));
	    }
	  else if(selection == "Laplace")
	    {
	      gsl_vector_set(b,i,0.5*(RBF->Dxx(x_eval,y_eval)+RBF->Dyy(x_eval,y_eval)));
	    }
	  else if(selection == "Neg_Laplace")
	    {
	      gsl_vector_set(b,i,0.5*(RBF->Dxx(x_eval,y_eval)-RBF->Dyy(x_eval,y_eval)));
	    }
	}

      //Setting missing values in result vector
      if(selection == "x")
	{
	  gsl_vector_set(b,num_neighbours+1,1.);
	  gsl_vector_set(b,num_neighbours+3,2.*x_eval);
	  gsl_vector_set(b,num_neighbours+5,y_eval);
	  gsl_vector_set(b,num_neighbours+6,3.*x_eval*x_eval);
	  gsl_vector_set(b,num_neighbours+8,2.*x_eval*y_eval);
	  gsl_vector_set(b,num_neighbours+9,y_eval*y_eval);
	}
      else if(selection == "y")
	{
	  gsl_vector_set(b,num_neighbours+2,1.);
	  gsl_vector_set(b,num_neighbours+4,2.*y_eval);
	  gsl_vector_set(b,num_neighbours+5,x_eval);
	  gsl_vector_set(b,num_neighbours+7,3.*y_eval*y_eval);
	  gsl_vector_set(b,num_neighbours+8,x_eval*x_eval);
	  gsl_vector_set(b,num_neighbours+9,2.*x_eval*y_eval);
	}
      else if(selection == "xx")
	{
	  gsl_vector_set(b,num_neighbours+3,2.);
	  gsl_vector_set(b,num_neighbours+6,6.*x_eval);
	  gsl_vector_set(b,num_neighbours+8,2.*y_eval);
	}
      else if(selection == "yy")
	{
	  gsl_vector_set(b,num_neighbours+4,2.);
	  gsl_vector_set(b,num_neighbours+7,6.*y_eval);
	  gsl_vector_set(b,num_neighbours+9,2.*x_eval);
	}
      else if(selection == "xy")
	{
	  gsl_vector_set(b,num_neighbours+5,1.);
	  gsl_vector_set(b,num_neighbours+8,2.*x_eval);
	  gsl_vector_set(b,num_neighbours+9,2.*y_eval);
	}
      else if(selection == "xxx")
	{
	  gsl_vector_set(b,num_neighbours+6,6.);
	}
      else if(selection == "yyy")
	{
	  gsl_vector_set(b,num_neighbours+7,6.);
	}
      else if(selection == "xxy")
	{
	  gsl_vector_set(b,num_neighbours+8,2.);
	}
      else if(selection == "xyy")
	{
	  gsl_vector_set(b,num_neighbours+9,2.);
	}
      else if(selection == "Laplace")
	{
	  gsl_vector_set(b,num_neighbours+3,1.);
	  gsl_vector_set(b,num_neighbours+6,3.*x_eval);
	  gsl_vector_set(b,num_neighbours+8,1.*y_eval);
	  gsl_vector_set(b,num_neighbours+4,1.);
	  gsl_vector_set(b,num_neighbours+7,3.*y_eval);
	  gsl_vector_set(b,num_neighbours+9,1.*x_eval);
	}
      else if(selection == "Neg_Laplace")
	{
	  gsl_vector_set(b,num_neighbours+3,1.);
	  gsl_vector_set(b,num_neighbours+6,3.*x_eval);
	  gsl_vector_set(b,num_neighbours+8,1.*y_eval);
	  gsl_vector_set(b,num_neighbours+4,-1.);
	  gsl_vector_set(b,num_neighbours+7,-3.*y_eval);
	  gsl_vector_set(b,num_neighbours+9,-1.*x_eval);
	}

      //Checking for the condition of the linear system via singular value
      //decomposition.
      gsl_linalg_SV_decomp(A, V, S, work_dummy);
      gsl_vector_minmax(S,&min,&max);
      condition.push_back(max/min);

      //Solving linear system using the SVD. 
      gsl_linalg_SV_solve(A, V, S, b, x);

      //Writing the output weights 
      for(int i = 0; i < num_neighbours; i++)
	{
	  (* weights)[tree_position_seed+i] = gsl_vector_get(x,i);
	}
    }
  //Cleanig up gsl work force
  gsl_matrix_free(A);
  gsl_vector_free(b);
  gsl_vector_free(x);
  gsl_matrix_free(V);
  gsl_vector_free(S);
  gsl_vector_free(work_dummy);


  return gsl_stats_mean(&condition[0], 1, num_nodes);
}

double mesh_free_2D::create_finite_differences_weights(string selection, vector<double> *weights, radial_basis_function_shape *RBF, vector<double> *adaptive_shape_parameter)
{
  double shape_save = RBF->show_epsilon();

  if(adaptive_shape_parameter->size() < num_nodes)
    {
      throw invalid_argument("MFREE_DIFF: Adaptive shape parameter vector too short.");
    }


  //Checking if the tree is up to date
  if(kD_update)
    {
      throw invalid_argument("UNSTRUC_GRID: Tree needs to be updated befor FD.");
    }

  //Determining the size of the problem
  if(kD_tree.size() % num_nodes != 0)
    {
      throw logic_error("UNSTRUC_GRID: The kD-tree size is invalid.");
    }
  int num_neighbours = kD_tree.size() / num_nodes;

  //Allocating work space for linear system to get finite differencing 
  //weights
  gsl_matrix *A = gsl_matrix_calloc(num_neighbours+10,num_neighbours+10);
  gsl_matrix *V = gsl_matrix_calloc(A->size1,A->size2);
  gsl_vector *S = gsl_vector_calloc(A->size1);
  gsl_vector *work_dummy = gsl_vector_calloc(A->size1);
  gsl_vector *b = gsl_vector_calloc(A->size1);
  gsl_vector *x = gsl_vector_calloc(A->size1);
  vector<double> condition;

  //Setting the stage for the output weights
  weights->resize(kD_tree.size());

  //Allocating all helper quantities in the process
  double x_eval, y_eval, x_1, y_1, value, x_node, y_node, min, max;
  int tree_position_seed, index1, index2;


  //Looping through all grid nodes
  for(int global = 0; global < num_nodes; global++)
    {
      RBF->set_epsilon((*adaptive_shape_parameter)[global]);

      //Getting evaluation coordinate
      x_eval = coordinates[global*2];
      y_eval = coordinates[global*2+1];
      //Resetting linear system
      gsl_matrix_set_identity(A);
      gsl_vector_set_all(b,0.);
      gsl_vector_set_all(x,0.);
      gsl_matrix_set_all(V,0.);
      gsl_vector_set_all(S,0.);
      gsl_vector_set_all(work_dummy,0.);
      tree_position_seed = global*num_neighbours;

      //Building up the linear system of equations

      for(int i = 0; i < num_neighbours; i++)
	{
	  //Getting node position and setting RBF to reference
	  x_node = coordinates[kD_tree[tree_position_seed+i]*2];
	  y_node = coordinates[kD_tree[tree_position_seed+i]*2+1];
	  RBF->set_coordinate_offset(x_node, y_node);

	  //Setting main body of A matrix
	  for(int j = i+1; j < num_neighbours; j++)
	    {
	      //calculating the distance
	      index1 = kD_tree[tree_position_seed+i];
	      index2 = kD_tree[tree_position_seed+j];
	      x_1 = coordinates[index1*2];
	      x_1 -= coordinates[index2*2];
	      x_1 *= x_1;   
	      y_1 = coordinates[index1*2+1];
	      y_1 -= coordinates[index2*2+1];
	      y_1 *= y_1;
	      x_1 += y_1;
	      //Evaluating radial basis function
	      //This square root is not super efficient
	      //Should be implemented in r^2 manner
	      value = (* RBF)(sqrt(x_1));
	      gsl_matrix_set(A,i,j,value);
	      gsl_matrix_set(A,j,i,value);
	    }

	  //Creating the missing entries in the coefficient matrix
	  //and the main part of the result vector
	  gsl_matrix_set(A,i,num_neighbours,1);
	  gsl_matrix_set(A,num_neighbours,i,1.);
	  gsl_matrix_set(A,i,num_neighbours+1,x_node);
	  gsl_matrix_set(A,num_neighbours+1,i,x_node);
	  gsl_matrix_set(A,i,num_neighbours+2,y_node);
	  gsl_matrix_set(A,num_neighbours+2,i,y_node);
	  gsl_matrix_set(A,i,num_neighbours+3,x_node*x_node);
	  gsl_matrix_set(A,num_neighbours+3,i,gsl_matrix_get(A,i,num_neighbours+3));
	  gsl_matrix_set(A,i,num_neighbours+4,y_node*y_node);
	  gsl_matrix_set(A,num_neighbours+4,i,gsl_matrix_get(A,i,num_neighbours+4));
	  gsl_matrix_set(A,i,num_neighbours+5,x_node*y_node);
	  gsl_matrix_set(A,num_neighbours+5,i,gsl_matrix_get(A,i,num_neighbours+5));
	  gsl_matrix_set(A,i,num_neighbours+6,x_node*x_node*x_node);
	  gsl_matrix_set(A,num_neighbours+6,i,gsl_matrix_get(A,i,num_neighbours+6));	  
	  gsl_matrix_set(A,i,num_neighbours+7,y_node*y_node*y_node);
	  gsl_matrix_set(A,num_neighbours+7,i,gsl_matrix_get(A,i,num_neighbours+7));
	  gsl_matrix_set(A,i,num_neighbours+8,gsl_matrix_get(A,i,num_neighbours+3)*y_node);
	  gsl_matrix_set(A,num_neighbours+8,i,gsl_matrix_get(A,i,num_neighbours+8));
	  gsl_matrix_set(A,i,num_neighbours+9,x_node*gsl_matrix_get(A,i,num_neighbours+4));
	  gsl_matrix_set(A,num_neighbours+9,i,gsl_matrix_get(A,i,num_neighbours+9));  
      //Creating the data vector	  
	  if(selection == "x")
	    {
	      gsl_vector_set(b,i,RBF->Dx(x_eval,y_eval));
	    }	  
	  else if(selection == "y")
	    {
	      gsl_vector_set(b,i,RBF->Dy(x_eval,y_eval));
	    }
	  else if(selection == "xx")
	    {
	      gsl_vector_set(b,i,RBF->Dxx(x_eval,y_eval));
	    }	
	  else if(selection == "yy")
	    {
	      gsl_vector_set(b,i,RBF->Dyy(x_eval,y_eval));
	    }
	  else if(selection == "xy")
	    {
	      gsl_vector_set(b,i,RBF->Dxy(x_eval,y_eval));
	    }
	  else if(selection == "xxx")
	    {
	      gsl_vector_set(b,i,RBF->Dxxx(x_eval,y_eval));
	    }
	  else if(selection == "yyy")
	    {
	      gsl_vector_set(b,i,RBF->Dyyy(x_eval,y_eval));
	    }
	  else if(selection == "xxy")
	    {
	      gsl_vector_set(b,i,RBF->Dxxy(x_eval,y_eval));
	    }
	  else if(selection == "xyy")
	    {
	      gsl_vector_set(b,i,RBF->Dxyy(x_eval,y_eval));
	    }
	  else if(selection == "Laplace")
	    {
	      gsl_vector_set(b,i,0.5*(RBF->Dxx(x_eval,y_eval)+RBF->Dyy(x_eval,y_eval)));
	    }
	  else if(selection == "Neg_Laplace")
	    {
	      gsl_vector_set(b,i,0.5*(RBF->Dxx(x_eval,y_eval)-RBF->Dyy(x_eval,y_eval)));
	    }
	}

      //Setting missing values in result vector
      if(selection == "x")
	{
	  gsl_vector_set(b,num_neighbours+1,1.);
	  gsl_vector_set(b,num_neighbours+3,2.*x_eval);
	  gsl_vector_set(b,num_neighbours+5,y_eval);
	  gsl_vector_set(b,num_neighbours+6,3.*x_eval*x_eval);
	  gsl_vector_set(b,num_neighbours+8,2.*x_eval*y_eval);
	  gsl_vector_set(b,num_neighbours+9,y_eval*y_eval);
	}
      else if(selection == "y")
	{
	  gsl_vector_set(b,num_neighbours+2,1.);
	  gsl_vector_set(b,num_neighbours+4,2.*y_eval);
	  gsl_vector_set(b,num_neighbours+5,x_eval);
	  gsl_vector_set(b,num_neighbours+7,3.*y_eval*y_eval);
	  gsl_vector_set(b,num_neighbours+8,x_eval*x_eval);
	  gsl_vector_set(b,num_neighbours+9,2.*x_eval*y_eval);
	}
      else if(selection == "xx")
	{
	  gsl_vector_set(b,num_neighbours+3,2.);
	  gsl_vector_set(b,num_neighbours+6,6.*x_eval);
	  gsl_vector_set(b,num_neighbours+8,2.*y_eval);
	}
      else if(selection == "yy")
	{
	  gsl_vector_set(b,num_neighbours+4,2.);
	  gsl_vector_set(b,num_neighbours+7,6.*y_eval);
	  gsl_vector_set(b,num_neighbours+9,2.*x_eval);
	}
      else if(selection == "xy")
	{
	  gsl_vector_set(b,num_neighbours+5,1.);
	  gsl_vector_set(b,num_neighbours+8,2.*x_eval);
	  gsl_vector_set(b,num_neighbours+9,2.*y_eval);
	}
      else if(selection == "xxx")
	{
	  gsl_vector_set(b,num_neighbours+6,6.);
	}
      else if(selection == "yyy")
	{
	  gsl_vector_set(b,num_neighbours+7,6.);
	}
      else if(selection == "xxy")
	{
	  gsl_vector_set(b,num_neighbours+8,2.);
	}
      else if(selection == "xyy")
	{
	  gsl_vector_set(b,num_neighbours+9,2.);
	}
      else if(selection == "Laplace")
	{
	  gsl_vector_set(b,num_neighbours+3,1.);
	  gsl_vector_set(b,num_neighbours+6,3.*x_eval);
	  gsl_vector_set(b,num_neighbours+8,1.*y_eval);
	  gsl_vector_set(b,num_neighbours+4,1.);
	  gsl_vector_set(b,num_neighbours+7,3.*y_eval);
	  gsl_vector_set(b,num_neighbours+9,1.*x_eval);
	}
      else if(selection == "Neg_Laplace")
	{
	  gsl_vector_set(b,num_neighbours+3,1.);
	  gsl_vector_set(b,num_neighbours+6,3.*x_eval);
	  gsl_vector_set(b,num_neighbours+8,1.*y_eval);
	  gsl_vector_set(b,num_neighbours+4,-1.);
	  gsl_vector_set(b,num_neighbours+7,-3.*y_eval);
	  gsl_vector_set(b,num_neighbours+9,-1.*x_eval);
	}

      //Checking for the condition of the linear system via singular value
      //decomposition.
      gsl_linalg_SV_decomp(A, V, S, work_dummy);
      gsl_vector_minmax(S,&min,&max);
      condition.push_back(max/min);

      //Solving linear system using the SVD. 
      gsl_linalg_SV_solve(A, V, S, b, x);

      //Writing the output weights 
      for(int i = 0; i < num_neighbours; i++)
	{
	  (* weights)[tree_position_seed+i] = gsl_vector_get(x,i);
	}
    }
  //Cleanig up gsl work force
  gsl_matrix_free(A);
  gsl_vector_free(b);
  gsl_vector_free(x);
  gsl_matrix_free(V);
  gsl_vector_free(S);
  gsl_vector_free(work_dummy);

  RBF->set_epsilon(shape_save);

  return gsl_stats_mean(&condition[0], 1, num_nodes);
}

double mesh_free_2D::create_finite_differences_weights(string selection, uint pdeg,  vector<double> *weights, radial_basis_function *RBF)
{

  //Checking if the tree is up to date
  if(kD_update)
    {
      throw invalid_argument("UNSTRUC_GRID: Tree needs to be updated befor FD.");
    }

  //Determining the size of the problem
  if(kD_tree.size() % num_nodes != 0)
    {
      throw logic_error("UNSTRUC_GRID: The kD-tree size is invalid.");
    }
  int num_neighbours = kD_tree.size() / num_nodes;
  int polynomial = (pdeg+1)*(pdeg+2)/2.;




  //Allocating work space for linear system to get finite differencing 
  //weights
  gsl_matrix *A = gsl_matrix_calloc(num_neighbours+polynomial,num_neighbours+polynomial);
  gsl_matrix *V = gsl_matrix_calloc(A->size1,A->size2);
  gsl_vector *S = gsl_vector_calloc(A->size1);
  gsl_vector *work_dummy = gsl_vector_calloc(A->size1);
  gsl_vector *b = gsl_vector_calloc(A->size1);
  gsl_vector *x = gsl_vector_calloc(A->size1);
  vector<double> condition;

  //Setting the stage for the output weights
  weights->resize(kD_tree.size());

  //Allocating all helper quantities in the process
  double x_eval, y_eval, x_1, y_1, value, x_node, y_node, min, max;
  int tree_position_seed, index1, index2;


  //Looping through all grid nodes
  for(int global = 0; global < num_nodes; global++)
    {
      //Getting evaluation coordinate
      x_eval = coordinates[global*2];
      y_eval = coordinates[global*2+1];
      //Resetting linear system
      gsl_matrix_set_identity(A);
      gsl_vector_set_all(b,0.);
      gsl_vector_set_all(x,0.);
      gsl_matrix_set_all(V,0.);
      gsl_vector_set_all(S,0.);
      gsl_vector_set_all(work_dummy,0.);
      tree_position_seed = global*num_neighbours;

      //Building up the linear system of equations


      //Building the non-changing part of the result vector
      vector<double> missing_columns = polynomial_support_rhs_column_vector_2D(selection, pdeg);
      for(uint l = 0; l < missing_columns.size(); l++)
	{
	  gsl_vector_set(b,l+num_neighbours,missing_columns[l]);
	}

      //Main matrix part, looping over all neighbours. 
      for(int i = 0; i < num_neighbours; i++)
	{
	  //Getting node position and setting RBF to reference
	  x_node = coordinates[kD_tree[tree_position_seed+i]*2];
	  y_node = coordinates[kD_tree[tree_position_seed+i]*2+1];
	  RBF->set_coordinate_offset(x_eval, y_eval);

	  //Setting main body of A matrix
	  for(int j = i+1; j < num_neighbours; j++)
	    {
	      //calculating the distance
	      index1 = kD_tree[tree_position_seed+i];
	      index2 = kD_tree[tree_position_seed+j];
	      x_1 = coordinates[index1*2];
	      x_1 -= coordinates[index2*2];
	      x_1 *= x_1;   
	      y_1 = coordinates[index1*2+1];
	      y_1 -= coordinates[index2*2+1];
	      y_1 *= y_1;
	      x_1 += y_1;
	      //Evaluating radial basis function
	      //This square root is not super efficient
	      //Should be implemented in r^2 manner
	      value = (* RBF)(sqrt(x_1));
	      gsl_matrix_set(A,i,j,value);
	      gsl_matrix_set(A,j,i,value);
	    }

	  //Creating the missing entries in the coefficient matrix
	  //and the main part of the result vector

	  vector<double> missing_rows = row_vector_from_polynomial_2D(x_node-x_eval,y_node-y_eval, pdeg);
	  for(uint l = 0; l < missing_rows.size(); l++)
	    {
	      gsl_matrix_set(A,i,l+num_neighbours,missing_rows[l]);
	    }
 
	  //Creating the data vector	  
	  if(selection == "x")
	    {
	      gsl_vector_set(b,i,RBF->Dx(x_node,y_node));
	    }	  
	  else if(selection == "y")
	    {
	      gsl_vector_set(b,i,RBF->Dy(x_node,y_node));
	    }
	  else if(selection == "xx")
	    {
	      gsl_vector_set(b,i,RBF->Dxx(x_node,y_node));
	    }	
	  else if(selection == "yy")
	    {
	      gsl_vector_set(b,i,RBF->Dyy(x_node,y_node));
	    }
	  else if(selection == "xy")
	    {
	      gsl_vector_set(b,i,RBF->Dxy(x_node,y_node));
	    }
	  else if(selection == "xxx")
	    {
	      gsl_vector_set(b,i,RBF->Dxxx(x_node,y_node));
	    }
	  else if(selection == "yyy")
	    {
	      gsl_vector_set(b,i,RBF->Dyyy(x_node,y_node));
	    }
	  else if(selection == "xxy")
	    {
	      gsl_vector_set(b,i,RBF->Dxxy(x_node,y_node));
	    }
	  else if(selection == "xyy")
	    {
	      gsl_vector_set(b,i,RBF->Dxyy(x_node,y_node));
	    }
	  else if(selection == "Laplace")
	    {
	      gsl_vector_set(b,i,0.5*(RBF->Dxx(x_node,y_node)+RBF->Dyy(x_node,y_node)));
	    }
	  else if(selection == "Neg_Laplace")
	    {
	      gsl_vector_set(b,i,0.5*(RBF->Dxx(x_node,y_node)-RBF->Dyy(x_node,y_node)));
	    }
	}

      //Checking for the condition of the linear system via singular value
      //decomposition.
      gsl_linalg_SV_decomp(A, V, S, work_dummy);
      gsl_vector_minmax(S,&min,&max);
      condition.push_back(max/min);

      //Solving linear system using the SVD. 
      gsl_linalg_SV_solve(A, V, S, b, x);

      //Writing the output weights 
      for(int i = 0; i < num_neighbours; i++)
	{
	  (* weights)[tree_position_seed+i] = gsl_vector_get(x,i);
	}
    }
  //Cleanig up gsl work force
  gsl_matrix_free(A);
  gsl_vector_free(b);
  gsl_vector_free(x);
  gsl_matrix_free(V);
  gsl_vector_free(S);
  gsl_vector_free(work_dummy);


  return gsl_stats_mean(&condition[0], 1, num_nodes);




}



double mesh_free_2D::differentiate(vector<double> *target_coordinates, vector<double> *in, string selection, radial_basis_function *RBF, vector<double> *out, int nn)
{
  if(target_coordinates->size() < dim)
    {
      throw invalid_argument("MFREE_DIFF: Target coordinate vector invalid.");
    }

  if(in->size() < num_nodes)
    {
      throw invalid_argument("MFREE_DIFF: Input vector invalid.");
    }

  int num_out_nodes = target_coordinates->size() / dim;
  vector<int> output_tree;
  vector<double> output_distances;
  output_tree.resize(nn*num_out_nodes);
  output_distances.resize(nn*num_out_nodes);

  //Creating a tree for the target vector.

  flann::Matrix<double> flann_base(&coordinates[0],num_nodes,dim);
  flann::Matrix<double> flann_search(&(*target_coordinates)[0],num_out_nodes,dim);
  flann::Matrix<int> flann_tree(&output_tree[0],num_out_nodes,nn);
  flann::Matrix<double> flann_distances(&distances[0],num_out_nodes,nn);

  //Creating flann index
  flann::Index<flann::L2<double> > index(flann_base, flann::KDTreeIndexParams(4));
  index.buildIndex();
 //Performing flann nearest neighbours search
  index.knnSearch(flann_search, flann_tree, flann_distances, nn, flann::SearchParams(128));



  //Allocating work space for linear system to get finite differencing 
  //weights
  gsl_matrix *A = gsl_matrix_calloc(nn+10,nn+10);
  gsl_matrix *V = gsl_matrix_calloc(A->size1,A->size2);
  gsl_vector *S = gsl_vector_calloc(A->size1);
  gsl_vector *work_dummy = gsl_vector_calloc(A->size1);
  gsl_vector *b = gsl_vector_calloc(A->size1);
  gsl_vector *x = gsl_vector_calloc(A->size1);
  vector<double> condition;

  //Setting the stage for the output weights
  vector<double> output_weights;
  output_weights.resize(output_tree.size());

  //Allocating all helper quantities in the process
  double x_eval, y_eval, x_1, y_1, value, x_node, y_node, min, max;
  int tree_position_seed, index1, index2;


  //Looping through all grid nodes
  for(int global = 0; global < num_out_nodes; global++)
    {
      //Getting evaluation coordinate
      x_eval = (*target_coordinates)[global*2];
      y_eval = (*target_coordinates)[global*2+1];
      //Resetting linear system
      gsl_matrix_set_identity(A);
      gsl_vector_set_all(b,0.);
      gsl_vector_set_all(x,0.);
      gsl_matrix_set_all(V,0.);
      gsl_vector_set_all(S,0.);
      gsl_vector_set_all(work_dummy,0.);
      tree_position_seed = global*nn;

      //Building up the linear system of equations

      for(int i = 0; i < nn; i++)
	{
	  //Getting node position and setting RBF to reference
	  x_node = coordinates[output_tree[tree_position_seed+i]*2];
	  y_node = coordinates[output_tree[tree_position_seed+i]*2+1];
	  RBF->set_coordinate_offset(x_node, y_node);

	  //Setting main body of A matrix
	  for(int j = i+1; j < nn; j++)
	    {
	      //calculating the distance
	      index1 = output_tree[tree_position_seed+i];
	      index2 = output_tree[tree_position_seed+j];
	      x_1 = coordinates[index1*2];
	      x_1 -= coordinates[index2*2];
	      x_1 *= x_1;   
	      y_1 = coordinates[index1*2+1];
	      y_1 -= coordinates[index2*2+1];
	      y_1 *= y_1;
	      x_1 += y_1;
	      //Evaluating radial basis function
	      //This square root is not super efficient
	      //Should be implemented in r^2 manner
	      value = (* RBF)(sqrt(x_1));
	      gsl_matrix_set(A,i,j,value);
	      gsl_matrix_set(A,j,i,value);
	    }

	  //Creating the missing entries in the coefficient matrix
	  //and the main part of the result vector
	  gsl_matrix_set(A,i,nn,1);
	  gsl_matrix_set(A,nn,i,1.);
	  gsl_matrix_set(A,i,nn+1,x_node);
	  gsl_matrix_set(A,nn+1,i,x_node);
	  gsl_matrix_set(A,i,nn+2,y_node);
	  gsl_matrix_set(A,nn+2,i,y_node);
	  gsl_matrix_set(A,i,nn+3,x_node*x_node);
	  gsl_matrix_set(A,nn+3,i,gsl_matrix_get(A,i,nn+3));
	  gsl_matrix_set(A,i,nn+4,y_node*y_node);
	  gsl_matrix_set(A,nn+4,i,gsl_matrix_get(A,i,nn+4));
	  gsl_matrix_set(A,i,nn+5,x_node*y_node);
	  gsl_matrix_set(A,nn+5,i,gsl_matrix_get(A,i,nn+5));
	  gsl_matrix_set(A,i,nn+6,x_node*x_node*x_node);
	  gsl_matrix_set(A,nn+6,i,gsl_matrix_get(A,i,nn+6));	  
	  gsl_matrix_set(A,i,nn+7,y_node*y_node*y_node);
	  gsl_matrix_set(A,nn+7,i,gsl_matrix_get(A,i,nn+7));
	  gsl_matrix_set(A,i,nn+8,gsl_matrix_get(A,i,nn+3)*y_node);
	  gsl_matrix_set(A,nn+8,i,gsl_matrix_get(A,i,nn+8));
	  gsl_matrix_set(A,i,nn+9,x_node*gsl_matrix_get(A,i,nn+4));
	  gsl_matrix_set(A,nn+9,i,gsl_matrix_get(A,i,nn+9));  
      //Creating the data vector	  
	  if(selection == "x")
	    {
	      gsl_vector_set(b,i,RBF->Dx(x_eval,y_eval));
	    }	  
	  else if(selection == "y")
	    {
	      gsl_vector_set(b,i,RBF->Dy(x_eval,y_eval));
	    }
	  else if(selection == "xx")
	    {
	      gsl_vector_set(b,i,RBF->Dxx(x_eval,y_eval));
	    }	
	  else if(selection == "yy")
	    {
	      gsl_vector_set(b,i,RBF->Dyy(x_eval,y_eval));
	    }
	  else if(selection == "xy")
	    {
	      gsl_vector_set(b,i,RBF->Dxy(x_eval,y_eval));
	    }
	  else if(selection == "xxx")
	    {
	      gsl_vector_set(b,i,RBF->Dxxx(x_eval,y_eval));
	    }
	  else if(selection == "yyy")
	    {
	      gsl_vector_set(b,i,RBF->Dyyy(x_eval,y_eval));
	    }
	  else if(selection == "xxy")
	    {
	      gsl_vector_set(b,i,RBF->Dxxy(x_eval,y_eval));
	    }
	  else if(selection == "xyy")
	    {
	      gsl_vector_set(b,i,RBF->Dxyy(x_eval,y_eval));
	    }
	  else if(selection == "Laplace")
	    {
	      gsl_vector_set(b,i,0.5*(RBF->Dxx(x_eval,y_eval)+RBF->Dyy(x_eval,y_eval)));
	    }
	  else if(selection == "Neg_Laplace")
	    {
	      gsl_vector_set(b,i,0.5*(RBF->Dxx(x_eval,y_eval)-RBF->Dyy(x_eval,y_eval)));
	    }
	}

      //Setting missing values in result vector
      if(selection == "x")
	{
	  gsl_vector_set(b,nn+1,1.);
	  gsl_vector_set(b,nn+3,2.*x_eval);
	  gsl_vector_set(b,nn+5,y_eval);
	  gsl_vector_set(b,nn+6,3.*x_eval*x_eval);
	  gsl_vector_set(b,nn+8,2.*x_eval*y_eval);
	  gsl_vector_set(b,nn+9,y_eval*y_eval);
	}
      else if(selection == "y")
	{
	  gsl_vector_set(b,nn+2,1.);
	  gsl_vector_set(b,nn+4,2.*y_eval);
	  gsl_vector_set(b,nn+5,x_eval);
	  gsl_vector_set(b,nn+7,3.*y_eval*y_eval);
	  gsl_vector_set(b,nn+8,x_eval*x_eval);
	  gsl_vector_set(b,nn+9,2.*x_eval*y_eval);
	}
      else if(selection == "xx")
	{
	  gsl_vector_set(b,nn+3,2.);
	  gsl_vector_set(b,nn+6,6.*x_eval);
	  gsl_vector_set(b,nn+8,2.*y_eval);
	}
      else if(selection == "yy")
	{
	  gsl_vector_set(b,nn+4,2.);
	  gsl_vector_set(b,nn+7,6.*y_eval);
	  gsl_vector_set(b,nn+9,2.*x_eval);
	}
      else if(selection == "xy")
	{
	  gsl_vector_set(b,nn+5,1.);
	  gsl_vector_set(b,nn+8,2.*x_eval);
	  gsl_vector_set(b,nn+9,2.*y_eval);
	}
      else if(selection == "xxx")
	{
	  gsl_vector_set(b,nn+6,6.);
	}
      else if(selection == "yyy")
	{
	  gsl_vector_set(b,nn+7,6.);
	}
      else if(selection == "xxy")
	{
	  gsl_vector_set(b,nn+8,2.);
	}
      else if(selection == "xyy")
	{
	  gsl_vector_set(b,nn+9,2.);
	}
      else if(selection == "Laplace")
	{
	  gsl_vector_set(b,nn+3,1.);
	  gsl_vector_set(b,nn+6,3.*x_eval);
	  gsl_vector_set(b,nn+8,1.*y_eval);
	  gsl_vector_set(b,nn+4,1.);
	  gsl_vector_set(b,nn+7,3.*y_eval);
	  gsl_vector_set(b,nn+9,1.*x_eval);
	}
      else if(selection == "Neg_Laplace")
	{
	  gsl_vector_set(b,nn+3,1.);
	  gsl_vector_set(b,nn+6,3.*x_eval);
	  gsl_vector_set(b,nn+8,1.*y_eval);
	  gsl_vector_set(b,nn+4,-1.);
	  gsl_vector_set(b,nn+7,-3.*y_eval);
	  gsl_vector_set(b,nn+9,-1.*x_eval);
	}

      //Checking for the condition of the linear system via singular value
      //decomposition.
      gsl_linalg_SV_decomp(A, V, S, work_dummy);
      gsl_vector_minmax(S,&min,&max);
      condition.push_back(max/min);

      //Solving linear system using the SVD. 
      gsl_linalg_SV_solve(A, V, S, b, x);

      //Writing the output weights 
      for(int i = 0; i < nn; i++)
	{
	  output_weights[tree_position_seed+i] = gsl_vector_get(x,i);
	}
    }
  //Cleanig up gsl work force
  gsl_matrix_free(A);
  gsl_vector_free(b);
  gsl_vector_free(x);
  gsl_matrix_free(V);
  gsl_vector_free(S);
  gsl_vector_free(work_dummy);


  double output = gsl_stats_mean(&condition[0], 1, num_out_nodes);

  out->resize(num_out_nodes);
  int seed, pos;

  //Performing the finite differencing

  for(int i = 0; i < num_out_nodes; i++)
    {
      value = 0.;
      seed = i*nn;
      for(int j = 0; j < nn; j++)
	{
	  pos = output_tree[seed+j];
	  value += (*in)[pos]*output_weights[seed+j];
	}
      (*out)[i] = value;
    }


 return output;

}


mesh_free_3D::mesh_free_3D(mesh_free_3D &input)
{
  dim = input.return_grid_size(1);
  num_nodes = input.return_grid_size(0);
  distances = input.distances;
  kD_update = input.kD_update;
  kD_tree = input.kD_tree;
}


double mesh_free_3D::create_finite_differences_weights(string selection, vector<double> *weights, radial_basis_function *RBF)
{

  //Checking if the tree is up to date
  if(kD_update)
    {
      throw invalid_argument("UNSTRUC_GRID: Tree needs to be updated befor FD.");
    }

  //Determining the size of the problem
  if(kD_tree.size() % num_nodes != 0)
    {
      throw logic_error("UNSTRUC_GRID: The kD-tree size is invalid.");
    }
  int num_neighbours = kD_tree.size() / num_nodes;

  //Allocating work space for linear system to get finite differencing 
  //weights
  gsl_matrix *A = gsl_matrix_calloc(num_neighbours+10,num_neighbours+10);
  gsl_matrix *V = gsl_matrix_calloc(A->size1,A->size2);
  gsl_vector *S = gsl_vector_calloc(A->size1);
  gsl_vector *work_dummy = gsl_vector_calloc(A->size1);
  gsl_vector *b = gsl_vector_calloc(A->size1);
  gsl_vector *x = gsl_vector_calloc(A->size1);
  vector<double> condition;

  //Setting the stage for the output weights
  weights->resize(kD_tree.size());

  //Allocating all helper quantities in the process
  double x_eval, y_eval, z_eval, x_1, y_1, z_1,  value, x_node, y_node, z_node,  min, max;
  int tree_position_seed, index1, index2;


  //Looping through all grid nodes
  for(int global = 0; global < num_nodes; global++)
    {
      //Getting evaluation coordinate
      x_eval = coordinates[global*3];
      y_eval = coordinates[global*3+1];
      z_eval = coordinates[global*3+2];
      //Resetting linear system
      gsl_matrix_set_identity(A);
      gsl_vector_set_all(b,0.);
      gsl_vector_set_all(x,0.);
      gsl_matrix_set_all(V,0.);
      gsl_vector_set_all(S,0.);
      gsl_vector_set_all(work_dummy,0.);
      tree_position_seed = global*num_neighbours;

      //Building up the linear system of equations

      for(int i = 0; i < num_neighbours; i++)
	{
	  //Getting node position and setting RBF to reference
	  x_node = coordinates[kD_tree[tree_position_seed+i]*3];
	  y_node = coordinates[kD_tree[tree_position_seed+i]*3+1];
	  z_node = coordinates[kD_tree[tree_position_seed+i]*3+2];
	  RBF->set_coordinate_offset(x_node, y_node, z_node);

	  //Setting main body of A matrix
	  for(int j = i+1; j < num_neighbours; j++)
	    {
	      //calculating the distance
	      index1 = kD_tree[tree_position_seed+i];
	      index2 = kD_tree[tree_position_seed+j];
	      x_1 = coordinates[index1*3];
	      x_1 -= coordinates[index2*3];
	      x_1 *= x_1;   
	      y_1 = coordinates[index1*3+1];
	      y_1 -= coordinates[index2*3+1];
	      y_1 *= y_1;
	      z_1 = coordinates[index1*3+2];
	      z_1 -= coordinates[index2*3+2];
	      z_1 *= z_1;
	      x_1 += (y_1+z_1);
	      //Evaluating radial basis function
	      //This square root is not super efficient
	      //Should be implemented in r^2 manner
	      value = (* RBF)(sqrt(x_1));
	      gsl_matrix_set(A,i,j,value);
	      gsl_matrix_set(A,j,i,value);
	    }

	  //Creating the missing entries in the coefficient matrix
	  //and the main part of the result vector
	  gsl_matrix_set(A,i,num_neighbours,1);
	  gsl_matrix_set(A,num_neighbours,i,1.);
	  gsl_matrix_set(A,i,num_neighbours+1,x_node);
	  gsl_matrix_set(A,num_neighbours+1,i,x_node);
	  gsl_matrix_set(A,i,num_neighbours+2,y_node);
	  gsl_matrix_set(A,num_neighbours+2,i,y_node);
	  gsl_matrix_set(A,i,num_neighbours+3,z_node);
	  gsl_matrix_set(A,num_neighbours+3,i,z_node);
	  gsl_matrix_set(A,i,num_neighbours+4,x_node*x_node);
	  gsl_matrix_set(A,num_neighbours+4,i,gsl_matrix_get(A,i,num_neighbours+4));
	  gsl_matrix_set(A,i,num_neighbours+5,y_node*y_node);
	  gsl_matrix_set(A,num_neighbours+5,i,gsl_matrix_get(A,i,num_neighbours+5));
	  gsl_matrix_set(A,i,num_neighbours+6,z_node*z_node);
	  gsl_matrix_set(A,num_neighbours+6,i,gsl_matrix_get(A,i,num_neighbours+6));
	  gsl_matrix_set(A,i,num_neighbours+7,x_node*y_node);
	  gsl_matrix_set(A,num_neighbours+7,i,gsl_matrix_get(A,i,num_neighbours+7));
	  gsl_matrix_set(A,i,num_neighbours+8,x_node*z_node);
	  gsl_matrix_set(A,num_neighbours+8,i,gsl_matrix_get(A,i,num_neighbours+8));
	  gsl_matrix_set(A,i,num_neighbours+9,y_node*z_node);
	  gsl_matrix_set(A,num_neighbours+9,i,gsl_matrix_get(A,i,num_neighbours+9));
 
      //Creating the data vector	  
	  if(selection == "x")
	    {
	      gsl_vector_set(b,i,RBF->Dx(x_eval,y_eval,z_eval));
	    }	  
	  else if(selection == "y")
	    {
	      gsl_vector_set(b,i,RBF->Dy(x_eval,y_eval,z_eval));
	    }
	  else if(selection == "z")
	    {
	      gsl_vector_set(b,i,RBF->Dz(x_eval,y_eval,z_eval));
	    }
	  else if(selection == "xx")
	    {
	      gsl_vector_set(b,i,RBF->Dxx(x_eval,y_eval,z_eval));
	    }	
	  else if(selection == "yy")
	    {
	      gsl_vector_set(b,i,RBF->Dyy(x_eval,y_eval,z_eval));
	    }
	  else if(selection == "zz")
	    {
	      gsl_vector_set(b,i,RBF->Dzz(x_eval,y_eval,z_eval));
	    }
	  else if(selection == "xy")
	    {
	      gsl_vector_set(b,i,RBF->Dxy(x_eval,y_eval,z_eval));
	    }
	  else if(selection == "xz")
	    {
	      gsl_vector_set(b,i,RBF->Dxz(x_eval,y_eval,z_eval));
	    }
	  else if(selection == "yz")
	    {
	      gsl_vector_set(b,i,RBF->Dyz(x_eval,y_eval,z_eval));
	    }
	  else if(selection == "Laplace")
	    {
	      gsl_vector_set(b,i,1./3.*(RBF->Dxx(x_eval,y_eval,z_eval)+RBF->Dyy(x_eval,y_eval,z_eval)+RBF->Dzz(x_eval,y_eval,z_eval)));
	    }
	}

      //Setting missing values in result vector
      if(selection == "x")
	{
	  gsl_vector_set(b,num_neighbours+1,1.);
	  gsl_vector_set(b,num_neighbours+4,2.*x_eval);
	  gsl_vector_set(b,num_neighbours+7,y_eval);
	  gsl_vector_set(b,num_neighbours+8,z_eval);
	}
      else if(selection == "y")
	{
	  gsl_vector_set(b,num_neighbours+2,1.);
	  gsl_vector_set(b,num_neighbours+5,2.*y_eval);
	  gsl_vector_set(b,num_neighbours+7,x_eval);
	  gsl_vector_set(b,num_neighbours+9,z_eval);
	}
      else if(selection == "z")
	{
	  gsl_vector_set(b,num_neighbours+3,1.);
	  gsl_vector_set(b,num_neighbours+6,2.*z_eval);
	  gsl_vector_set(b,num_neighbours+8,x_eval);
	  gsl_vector_set(b,num_neighbours+9,y_eval);
	}
      else if(selection == "xx")
	{
	  gsl_vector_set(b,num_neighbours+4,2.);
	}
      else if(selection == "yy")
	{
	  gsl_vector_set(b,num_neighbours+5,2.);
	}
      else if(selection == "zz")
	{
	  gsl_vector_set(b,num_neighbours+6,2.);
	}
      else if(selection == "xy")
	{
	  gsl_vector_set(b,num_neighbours+7,1.);
	}
      else if(selection == "xz")
	{
	  gsl_vector_set(b,num_neighbours+8,1.);
	}
      else if(selection == "yz")
	{
	  gsl_vector_set(b,num_neighbours+9,1.);
	}
      else if(selection == "Laplace")
	{
	  gsl_vector_set(b,num_neighbours+4,2.);
	  gsl_vector_set(b,num_neighbours+5,2.);
	  gsl_vector_set(b,num_neighbours+6,2.);
	}

      //Checking for the condition of the linear system via singular value
      //decomposition.
      gsl_linalg_SV_decomp(A, V, S, work_dummy);
      gsl_vector_minmax(S,&min,&max);
      condition.push_back(max/min);

      //Solving linear system using the SVD. 
      gsl_linalg_SV_solve(A, V, S, b, x);

      //Writing the output weights 
      for(int i = 0; i < num_neighbours; i++)
	{
	  (* weights)[tree_position_seed+i] = gsl_vector_get(x,i);
	}
    }
  //Cleanig up gsl work force
  gsl_matrix_free(A);
  gsl_vector_free(b);
  gsl_vector_free(x);
  gsl_matrix_free(V);
  gsl_vector_free(S);
  gsl_vector_free(work_dummy);


  return gsl_stats_mean(&condition[0], 1, num_nodes);
}


double mesh_free_3D::create_finite_differences_weights(string selection, vector<double> *weights, radial_basis_function_shape *RBF, vector<double> *adaptive_shape_parameter)
{

  double shape_save = RBF->show_epsilon();

  if(adaptive_shape_parameter->size() < num_nodes)
    {
      throw invalid_argument("MFREE_DIFF: Adaptive shape parameter vector too short.");
    }


  //Checking if the tree is up to date
  if(kD_update)
    {
      throw invalid_argument("UNSTRUC_GRID: Tree needs to be updated befor FD.");
    }

  //Determining the size of the problem
  if(kD_tree.size() % num_nodes != 0)
    {
      throw logic_error("UNSTRUC_GRID: The kD-tree size is invalid.");
    }
  int num_neighbours = kD_tree.size() / num_nodes;

  //Allocating work space for linear system to get finite differencing 
  //weights
  gsl_matrix *A = gsl_matrix_calloc(num_neighbours+10,num_neighbours+10);
  gsl_matrix *V = gsl_matrix_calloc(A->size1,A->size2);
  gsl_vector *S = gsl_vector_calloc(A->size1);
  gsl_vector *work_dummy = gsl_vector_calloc(A->size1);
  gsl_vector *b = gsl_vector_calloc(A->size1);
  gsl_vector *x = gsl_vector_calloc(A->size1);
  vector<double> condition;

  //Setting the stage for the output weights
  weights->resize(kD_tree.size());

  //Allocating all helper quantities in the process
  double x_eval, y_eval, z_eval, x_1, y_1, z_1,  value, x_node, y_node, z_node,  min, max;
  int tree_position_seed, index1, index2;


  //Looping through all grid nodes
  for(int global = 0; global < num_nodes; global++)
    {
      RBF->set_epsilon((*adaptive_shape_parameter)[global]);

      //Getting evaluation coordinate
      x_eval = coordinates[global*3];
      y_eval = coordinates[global*3+1];
      z_eval = coordinates[global*3+2];
      //Resetting linear system
      gsl_matrix_set_identity(A);
      gsl_vector_set_all(b,0.);
      gsl_vector_set_all(x,0.);
      gsl_matrix_set_all(V,0.);
      gsl_vector_set_all(S,0.);
      gsl_vector_set_all(work_dummy,0.);
      tree_position_seed = global*num_neighbours;

      //Building up the linear system of equations

      for(int i = 0; i < num_neighbours; i++)
	{
	  //Getting node position and setting RBF to reference
	  x_node = coordinates[kD_tree[tree_position_seed+i]*3];
	  y_node = coordinates[kD_tree[tree_position_seed+i]*3+1];
	  z_node = coordinates[kD_tree[tree_position_seed+i]*3+2];
	  RBF->set_coordinate_offset(x_node, y_node, z_node);

	  //Setting main body of A matrix
	  for(int j = i+1; j < num_neighbours; j++)
	    {
	      //calculating the distance
	      index1 = kD_tree[tree_position_seed+i];
	      index2 = kD_tree[tree_position_seed+j];
	      x_1 = coordinates[index1*3];
	      x_1 -= coordinates[index2*3];
	      x_1 *= x_1;   
	      y_1 = coordinates[index1*3+1];
	      y_1 -= coordinates[index2*3+1];
	      y_1 *= y_1;
	      z_1 = coordinates[index1*3+2];
	      z_1 -= coordinates[index2*3+2];
	      z_1 *= z_1;
	      x_1 += (y_1+z_1);
	      //Evaluating radial basis function
	      //This square root is not super efficient
	      //Should be implemented in r^2 manner
	      value = (* RBF)(sqrt(x_1));
	      gsl_matrix_set(A,i,j,value);
	      gsl_matrix_set(A,j,i,value);
	    }

	  //Creating the missing entries in the coefficient matrix
	  //and the main part of the result vector
	  gsl_matrix_set(A,i,num_neighbours,1);
	  gsl_matrix_set(A,num_neighbours,i,1.);
	  gsl_matrix_set(A,i,num_neighbours+1,x_node);
	  gsl_matrix_set(A,num_neighbours+1,i,x_node);
	  gsl_matrix_set(A,i,num_neighbours+2,y_node);
	  gsl_matrix_set(A,num_neighbours+2,i,y_node);
	  gsl_matrix_set(A,i,num_neighbours+3,z_node);
	  gsl_matrix_set(A,num_neighbours+3,i,z_node);
	  gsl_matrix_set(A,i,num_neighbours+4,x_node*x_node);
	  gsl_matrix_set(A,num_neighbours+4,i,gsl_matrix_get(A,i,num_neighbours+4));
	  gsl_matrix_set(A,i,num_neighbours+5,y_node*y_node);
	  gsl_matrix_set(A,num_neighbours+5,i,gsl_matrix_get(A,i,num_neighbours+5));
	  gsl_matrix_set(A,i,num_neighbours+6,z_node*z_node);
	  gsl_matrix_set(A,num_neighbours+6,i,gsl_matrix_get(A,i,num_neighbours+6));
	  gsl_matrix_set(A,i,num_neighbours+7,x_node*y_node);
	  gsl_matrix_set(A,num_neighbours+7,i,gsl_matrix_get(A,i,num_neighbours+7));
	  gsl_matrix_set(A,i,num_neighbours+8,x_node*z_node);
	  gsl_matrix_set(A,num_neighbours+8,i,gsl_matrix_get(A,i,num_neighbours+8));
	  gsl_matrix_set(A,i,num_neighbours+9,y_node*z_node);
	  gsl_matrix_set(A,num_neighbours+9,i,gsl_matrix_get(A,i,num_neighbours+9));
 
      //Creating the data vector	  
	  if(selection == "x")
	    {
	      gsl_vector_set(b,i,RBF->Dx(x_eval,y_eval,z_eval));
	    }	  
	  else if(selection == "y")
	    {
	      gsl_vector_set(b,i,RBF->Dy(x_eval,y_eval,z_eval));
	    }
	  else if(selection == "z")
	    {
	      gsl_vector_set(b,i,RBF->Dz(x_eval,y_eval,z_eval));
	    }
	  else if(selection == "xx")
	    {
	      gsl_vector_set(b,i,RBF->Dxx(x_eval,y_eval,z_eval));
	    }	
	  else if(selection == "yy")
	    {
	      gsl_vector_set(b,i,RBF->Dyy(x_eval,y_eval,z_eval));
	    }
	  else if(selection == "zz")
	    {
	      gsl_vector_set(b,i,RBF->Dzz(x_eval,y_eval,z_eval));
	    }
	  else if(selection == "xy")
	    {
	      gsl_vector_set(b,i,RBF->Dxy(x_eval,y_eval,z_eval));
	    }
	  else if(selection == "xz")
	    {
	      gsl_vector_set(b,i,RBF->Dxz(x_eval,y_eval,z_eval));
	    }
	  else if(selection == "yz")
	    {
	      gsl_vector_set(b,i,RBF->Dyz(x_eval,y_eval,z_eval));
	    }
	  else if(selection == "Laplace")
	    {
	      gsl_vector_set(b,i,1./3.*(RBF->Dxx(x_eval,y_eval,z_eval)+RBF->Dyy(x_eval,y_eval,z_eval)+RBF->Dzz(x_eval,y_eval,z_eval)));
	    }
	}

      //Setting missing values in result vector
      if(selection == "x")
	{
	  gsl_vector_set(b,num_neighbours+1,1.);
	  gsl_vector_set(b,num_neighbours+4,2.*x_eval);
	  gsl_vector_set(b,num_neighbours+7,y_eval);
	  gsl_vector_set(b,num_neighbours+8,z_eval);
	}
      else if(selection == "y")
	{
	  gsl_vector_set(b,num_neighbours+2,1.);
	  gsl_vector_set(b,num_neighbours+5,2.*y_eval);
	  gsl_vector_set(b,num_neighbours+7,x_eval);
	  gsl_vector_set(b,num_neighbours+9,z_eval);
	}
      else if(selection == "z")
	{
	  gsl_vector_set(b,num_neighbours+3,1.);
	  gsl_vector_set(b,num_neighbours+6,2.*z_eval);
	  gsl_vector_set(b,num_neighbours+8,x_eval);
	  gsl_vector_set(b,num_neighbours+9,y_eval);
	}
      else if(selection == "xx")
	{
	  gsl_vector_set(b,num_neighbours+4,2.);
	}
      else if(selection == "yy")
	{
	  gsl_vector_set(b,num_neighbours+5,2.);
	}
      else if(selection == "zz")
	{
	  gsl_vector_set(b,num_neighbours+6,2.);
	}
      else if(selection == "xy")
	{
	  gsl_vector_set(b,num_neighbours+7,1.);
	}
      else if(selection == "xz")
	{
	  gsl_vector_set(b,num_neighbours+8,1.);
	}
      else if(selection == "yz")
	{
	  gsl_vector_set(b,num_neighbours+9,1.);
	}
      else if(selection == "Laplace")
	{
	  gsl_vector_set(b,num_neighbours+4,2.);
	  gsl_vector_set(b,num_neighbours+5,2.);
	  gsl_vector_set(b,num_neighbours+6,2.);
	}

      //Checking for the condition of the linear system via singular value
      //decomposition.
      gsl_linalg_SV_decomp(A, V, S, work_dummy);
      gsl_vector_minmax(S,&min,&max);
      condition.push_back(max/min);

      //Solving linear system using the SVD. 
      gsl_linalg_SV_solve(A, V, S, b, x);

      //Writing the output weights 
      for(int i = 0; i < num_neighbours; i++)
	{
	  (* weights)[tree_position_seed+i] = gsl_vector_get(x,i);
	}
    }
  //Cleanig up gsl work force
  gsl_matrix_free(A);
  gsl_vector_free(b);
  gsl_vector_free(x);
  gsl_matrix_free(V);
  gsl_vector_free(S);
  gsl_vector_free(work_dummy);

  RBF->set_epsilon(shape_save);

  return gsl_stats_mean(&condition[0], 1, num_nodes);

}

double mesh_free_3D::differentiate(vector<double> *target_coordinates, vector<double> *in, string selection, radial_basis_function *RBF, vector<double> *out, int nn)
{

  if(target_coordinates->size() < dim)
    {
      throw invalid_argument("MFREE_DIFF: Target coordinate vector invalid.");
    }

  if(in->size() < num_nodes)
    {
      throw invalid_argument("MFREE_DIFF: Input vector invalid.");
    }

  int num_out_nodes = target_coordinates->size() / dim;
  vector<int> output_tree;
  vector<double> output_distances;
  output_tree.resize(nn*num_out_nodes);
  output_distances.resize(nn*num_out_nodes);

  //Creating a tree for the target vector.

  flann::Matrix<double> flann_base(&coordinates[0],num_nodes,dim);
  flann::Matrix<double> flann_search(&(*target_coordinates)[0],num_out_nodes,dim);
  flann::Matrix<int> flann_tree(&output_tree[0],num_out_nodes,nn);
  flann::Matrix<double> flann_distances(&distances[0],num_out_nodes,nn);

  //Creating flann index
  flann::Index<flann::L2<double> > index(flann_base, flann::KDTreeIndexParams(4));
  index.buildIndex();
 //Performing flann nearest neighbours search
  index.knnSearch(flann_search, flann_tree, flann_distances, nn, flann::SearchParams(128));


  //Allocating work space for linear system to get finite differencing 
  //weights
  gsl_matrix *A = gsl_matrix_calloc(nn+10,nn+10);
  gsl_matrix *V = gsl_matrix_calloc(A->size1,A->size2);
  gsl_vector *S = gsl_vector_calloc(A->size1);
  gsl_vector *work_dummy = gsl_vector_calloc(A->size1);
  gsl_vector *b = gsl_vector_calloc(A->size1);
  gsl_vector *x = gsl_vector_calloc(A->size1);
  vector<double> condition;

  //Setting the stage for the output weights
  vector<double> output_weights;
  output_weights.resize(output_tree.size());

  //Allocating all helper quantities in the process
  double x_eval, y_eval, z_eval, x_1, y_1, z_1,  value, x_node, y_node, z_node,  min, max;
  int tree_position_seed, index1, index2;


  //Looping through all grid nodes
  for(int global = 0; global < num_out_nodes; global++)
    {
      //Getting evaluation coordinate
      x_eval = (*target_coordinates)[global*3];
      y_eval = (*target_coordinates)[global*3+1];
      z_eval = (*target_coordinates)[global*3+2];
      //Resetting linear system
      gsl_matrix_set_identity(A);
      gsl_vector_set_all(b,0.);
      gsl_vector_set_all(x,0.);
      gsl_matrix_set_all(V,0.);
      gsl_vector_set_all(S,0.);
      gsl_vector_set_all(work_dummy,0.);
      tree_position_seed = global*nn;

      //Building up the linear system of equations

      for(int i = 0; i < nn; i++)
	{
	  //Getting node position and setting RBF to reference
	  x_node = coordinates[output_tree[tree_position_seed+i]*3];
	  y_node = coordinates[output_tree[tree_position_seed+i]*3+1];
	  z_node = coordinates[output_tree[tree_position_seed+i]*3+2];
	  RBF->set_coordinate_offset(x_node, y_node, z_node);

	  //Setting main body of A matrix
	  for(int j = i+1; j < nn; j++)
	    {
	      //calculating the distance
	      index1 = output_tree[tree_position_seed+i];
	      index2 = output_tree[tree_position_seed+j];
	      x_1 = coordinates[index1*3];
	      x_1 -= coordinates[index2*3];
	      x_1 *= x_1;   
	      y_1 = coordinates[index1*3+1];
	      y_1 -= coordinates[index2*3+1];
	      y_1 *= y_1;
	      z_1 = coordinates[index1*3+2];
	      z_1 -= coordinates[index2*3+2];
	      z_1 *= z_1;
	      x_1 += (y_1+z_1);
	      //Evaluating radial basis function
	      //This square root is not super efficient
	      //Should be implemented in r^2 manner
	      value = (* RBF)(sqrt(x_1));
	      gsl_matrix_set(A,i,j,value);
	      gsl_matrix_set(A,j,i,value);
	    }

	  //Creating the missing entries in the coefficient matrix
	  //and the main part of the result vector
	  gsl_matrix_set(A,i,nn,1);
	  gsl_matrix_set(A,nn,i,1.);
	  gsl_matrix_set(A,i,nn+1,x_node);
	  gsl_matrix_set(A,nn+1,i,x_node);
	  gsl_matrix_set(A,i,nn+2,y_node);
	  gsl_matrix_set(A,nn+2,i,y_node);
	  gsl_matrix_set(A,i,nn+3,z_node);
	  gsl_matrix_set(A,nn+3,i,z_node);
	  gsl_matrix_set(A,i,nn+4,x_node*x_node);
	  gsl_matrix_set(A,nn+4,i,gsl_matrix_get(A,i,nn+4));
	  gsl_matrix_set(A,i,nn+5,y_node*y_node);
	  gsl_matrix_set(A,nn+5,i,gsl_matrix_get(A,i,nn+5));
	  gsl_matrix_set(A,i,nn+6,z_node*z_node);
	  gsl_matrix_set(A,nn+6,i,gsl_matrix_get(A,i,nn+6));
	  gsl_matrix_set(A,i,nn+7,x_node*y_node);
	  gsl_matrix_set(A,nn+7,i,gsl_matrix_get(A,i,nn+7));
	  gsl_matrix_set(A,i,nn+8,x_node*z_node);
	  gsl_matrix_set(A,nn+8,i,gsl_matrix_get(A,i,nn+8));
	  gsl_matrix_set(A,i,nn+9,y_node*z_node);
	  gsl_matrix_set(A,nn+9,i,gsl_matrix_get(A,i,nn+9));
 
      //Creating the data vector	  
	  if(selection == "x")
	    {
	      gsl_vector_set(b,i,RBF->Dx(x_eval,y_eval,z_eval));
	    }	  
	  else if(selection == "y")
	    {
	      gsl_vector_set(b,i,RBF->Dy(x_eval,y_eval,z_eval));
	    }
	  else if(selection == "z")
	    {
	      gsl_vector_set(b,i,RBF->Dz(x_eval,y_eval,z_eval));
	    }
	  else if(selection == "xx")
	    {
	      gsl_vector_set(b,i,RBF->Dxx(x_eval,y_eval,z_eval));
	    }	
	  else if(selection == "yy")
	    {
	      gsl_vector_set(b,i,RBF->Dyy(x_eval,y_eval,z_eval));
	    }
	  else if(selection == "zz")
	    {
	      gsl_vector_set(b,i,RBF->Dzz(x_eval,y_eval,z_eval));
	    }
	  else if(selection == "xy")
	    {
	      gsl_vector_set(b,i,RBF->Dxy(x_eval,y_eval,z_eval));
	    }
	  else if(selection == "xz")
	    {
	      gsl_vector_set(b,i,RBF->Dxz(x_eval,y_eval,z_eval));
	    }
	  else if(selection == "yz")
	    {
	      gsl_vector_set(b,i,RBF->Dyz(x_eval,y_eval,z_eval));
	    }
	  else if(selection == "Laplace")
	    {
	      gsl_vector_set(b,i,1./3.*(RBF->Dxx(x_eval,y_eval,z_eval)+RBF->Dyy(x_eval,y_eval,z_eval)+RBF->Dzz(x_eval,y_eval,z_eval)));
	    }
	}

      //Setting missing values in result vector
      if(selection == "x")
	{
	  gsl_vector_set(b,nn+1,1.);
	  gsl_vector_set(b,nn+4,2.*x_eval);
	  gsl_vector_set(b,nn+7,y_eval);
	  gsl_vector_set(b,nn+8,z_eval);
	}
      else if(selection == "y")
	{
	  gsl_vector_set(b,nn+2,1.);
	  gsl_vector_set(b,nn+5,2.*y_eval);
	  gsl_vector_set(b,nn+7,x_eval);
	  gsl_vector_set(b,nn+9,z_eval);
	}
      else if(selection == "z")
	{
	  gsl_vector_set(b,nn+3,1.);
	  gsl_vector_set(b,nn+6,2.*z_eval);
	  gsl_vector_set(b,nn+8,x_eval);
	  gsl_vector_set(b,nn+9,y_eval);
	}
      else if(selection == "xx")
	{
	  gsl_vector_set(b,nn+4,2.);
	}
      else if(selection == "yy")
	{
	  gsl_vector_set(b,nn+5,2.);
	}
      else if(selection == "zz")
	{
	  gsl_vector_set(b,nn+6,2.);
	}
      else if(selection == "xy")
	{
	  gsl_vector_set(b,nn+7,1.);
	}
      else if(selection == "xz")
	{
	  gsl_vector_set(b,nn+8,1.);
	}
      else if(selection == "yz")
	{
	  gsl_vector_set(b,nn+9,1.);
	}
      else if(selection == "Laplace")
	{
	  gsl_vector_set(b,nn+4,2.);
	  gsl_vector_set(b,nn+5,2.);
	  gsl_vector_set(b,nn+6,2.);
	}

      //Checking for the condition of the linear system via singular value
      //decomposition.
      gsl_linalg_SV_decomp(A, V, S, work_dummy);
      gsl_vector_minmax(S,&min,&max);
      condition.push_back(max/min);

      //Solving linear system using the SVD. 
      gsl_linalg_SV_solve(A, V, S, b, x);

      //Writing the output weights 
      for(int i = 0; i < nn; i++)
	{
	  output_weights[tree_position_seed+i] = gsl_vector_get(x,i);
	}
    }
  //Cleanig up gsl work force
  gsl_matrix_free(A);
  gsl_vector_free(b);
  gsl_vector_free(x);
  gsl_matrix_free(V);
  gsl_vector_free(S);
  gsl_vector_free(work_dummy);


  double output = gsl_stats_mean(&condition[0], 1, num_out_nodes);

  out->resize(num_out_nodes);
  int seed, pos;

  //Performing the finite differencing

  for(int i = 0; i < num_out_nodes; i++)
    {
      value = 0.;
      seed = i*nn;
      for(int j = 0; j < nn; j++)
	{
	  pos = output_tree[seed+j];
	  value += (*in)[pos]*output_weights[seed+j];
	}
      (*out)[i] = value;
    }


 return output;




}
