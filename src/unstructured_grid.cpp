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

#include <mfree/unstructured_grid.h>

unstructured_grid::unstructured_grid()
{
  kD_update = true;
}


unstructured_grid::~unstructured_grid()
{
  coordinates.clear();
  kD_tree.clear();
  distances.clear();
}

int unstructured_grid::return_grid_size(bool dim_check)
{
  int value; 

  if(dim_check)
    {
      value = dim;
    }
  else
    {
      value = num_nodes;
    }
  return value;
} 


vector<double> unstructured_grid::operator () (int node)
{
  if(node >= 0 && node <= num_nodes)
    {
      vector<double> out;
      int root = node*dim;

      for(int i = 0; i < dim; i++)
	{
	  out.push_back(coordinates[root+i]);
	}
      return out;
    }
}

double unstructured_grid::operator () (int node, int component)
{
  if(node >= 0 && node <= num_nodes && component >= 0)
    {
      return coordinates[node*dim+component];
    }
}

void unstructured_grid::set(int node, vector<double> new_coordinates)
{
  if(node >= 0 && node <= num_nodes)
    {
      int root = node*dim;
      for(int i = 0; i < dim; i++)
	{
	  coordinates[root+i] = new_coordinates[i];
	}
      kD_update = true;
    }
}

void unstructured_grid::set(vector<double> *new_coordinates)
{

  if(new_coordinates->size() % dim != 0)
    {
      throw invalid_argument("UNSTRUC_GRID: New coordinates vector invalid.");
    }

  kD_update = true;
  coordinates.clear();

  vector<double>::iterator it;

  for(it = new_coordinates->begin(); it != new_coordinates->end(); it++)
    {
      coordinates.push_back(*it);
    }

  num_nodes = new_coordinates->size()/dim;
}

void unstructured_grid::set(int num_nodes_in)
{

  kD_update = true;
  num_nodes = num_nodes_in;
  coordinates.clear();
  coordinates.resize(dim*num_nodes);
}

void unstructured_grid::build_tree(int nearest_neighbours)
{

  if(nearest_neighbours < 1)
    {
      throw invalid_argument("UNSTRUC_GRID: invalid number of nearest neighbours.");
    }
  
  //Allocating kD tree related quantities

  kD_tree.clear();
  distances.clear();
  
  kD_tree.resize(nearest_neighbours*num_nodes);
  distances.resize(nearest_neighbours*num_nodes);
  
  //Assigning flann data structures
  
  flann::Matrix<double> flann_dataset(&coordinates[0],num_nodes,dim);
  flann::Matrix<int> flann_tree(&kD_tree[0],num_nodes,nearest_neighbours);
  flann::Matrix<double> flann_distances(&distances[0],num_nodes,nearest_neighbours);
  
  //Creating flann index
  flann::Index<flann::L2<double> > index(flann_dataset, flann::KDTreeIndexParams(4));
  index.buildIndex();
  
  
  //Performing flann nearest neighbours search
  index.knnSearch(flann_dataset, flann_tree, flann_distances, nearest_neighbours, flann::SearchParams(128));
  kD_update = false;
}

vector<int> unstructured_grid::neighbours(int node)
{

  vector<int> knn;
  //Make sure that routine only return if tree is okay.
  if(!kD_update)
    {
      //How many nearest neighours?
      int num_neighbours = kD_tree.size()/num_nodes;
      int index = node*num_neighbours;
      for(int i = 0; i < num_neighbours; i++)
	{
	  knn.push_back(kD_tree[index+i]);
	}
      return knn;
    }
} 

vector<int> unstructured_grid::neighbours()
{
  if(!kD_update)
    {
      return kD_tree;
    }
  else
    {
      throw invalid_argument("UNSTRUC_GRID: Tree needs to be updated."); 
    }
}

int unstructured_grid::neighbours_col(vector<int> *neighbours, vector<int> *length_counter)
{
  return findif_row_col_convert(num_nodes, &kD_tree, neighbours, length_counter);
}

double unstructured_grid::interpolate(vector<double> *output_grid, int stride, vector<double> *input_function, radial_basis_function *RBF, int knn, vector<double> *output_function)
{

  //Checking for validity of the input data

  if(input_function->size() != num_nodes)
    {
      throw invalid_argument("UNSTRUC_GRID: Input function for interpolate is invalid.");
    }

  if(output_grid->size() < dim)
    {
      throw invalid_argument("UNSTRUC_GRID: Input grid for interpolate is invalid.");
    }

  if(output_grid->size()%stride != 0 && output_grid->size()%stride < dim)
    {
      throw invalid_argument("UNSTRUC_GRID: Input grid for interpolate is invalid.");
    }

  if(stride < 1)
    {
      throw invalid_argument("UNSTRUC_GRID: Stride for input grid invalid.");
    }

  if(knn < 1)
    {
      throw invalid_argument("UNSTRUC_GRID: Unsufficient number of neighbours to interpolate.");
    }

  //Getting interpolant coordinates
  vector<double> interpolant_coordinates;

  for(int i = 0; i < output_grid->size(); i += stride)
    {
      for(int j = 0; j < dim; j++)
	{
	  interpolant_coordinates.push_back((*output_grid)[i+j]);
	}
    }
  int num_nodes_interpolant = interpolant_coordinates.size()/dim; 



  //Building the tree for the output grid
  vector<int> interpolant_tree;
  vector<double> interpolant_distances;
  interpolant_tree.resize(knn*num_nodes_interpolant);
  interpolant_distances.resize(knn*num_nodes_interpolant);

  flann::Matrix<double> flann_dataset(&coordinates[0],num_nodes,dim);
  flann::Matrix<double> flann_dataset_interpolant(&interpolant_coordinates[0],num_nodes_interpolant,dim);
  flann::Matrix<int> flann_tree(&interpolant_tree[0],num_nodes_interpolant,knn);
  flann::Matrix<double> flann_distances(&interpolant_distances[0],num_nodes_interpolant,knn);

  flann::Index<flann::L2<double> > index(flann_dataset, flann::KDTreeIndexParams(4));
  index.buildIndex();
  index.knnSearch(flann_dataset_interpolant, flann_tree, flann_distances, knn, flann::SearchParams(128));

  //For each individual output point, build the linear system and calculate interpolant

  //Allocating work space for linear system to get finite differencing 
  //weights
  gsl_matrix *A = gsl_matrix_calloc(knn,knn);
  gsl_matrix *V = gsl_matrix_calloc(A->size1,A->size2);
  gsl_vector *S = gsl_vector_calloc(A->size1);
  gsl_vector *work_dummy = gsl_vector_calloc(A->size1);
  gsl_vector *b = gsl_vector_calloc(A->size1);
  gsl_vector *x = gsl_vector_calloc(A->size1);
  vector<double> condition;

  //Setting the stage for the output coefficients
  vector<double> lambda;
  lambda.resize(interpolant_tree.size());

  //Allocating all helper quantities in the process
  double value, radius,min,max;
  int tree_position_seed, index1, index2;


  vector<double> node1;
  node1.resize(dim);

  //Looping through all outptu grid nodes
  for(int global = 0; global < num_nodes_interpolant; global++)
    {
      //Resetting linear system
      gsl_matrix_set_identity(A);
      gsl_vector_set_all(b,0.);
      gsl_vector_set_all(x,0.);
      gsl_matrix_set_all(V,0.);
      gsl_vector_set_all(S,0.);
      gsl_vector_set_all(work_dummy,0.);
      tree_position_seed = global*knn;

      //Looping through neighbour combinations and calculating distances
      for(int i = 0; i < knn; i++)
	{
	  for(int n = 0; n < dim; n++)
	    {
	      node1[n] = (coordinates[interpolant_tree[tree_position_seed+i]*dim+n]);
	    }
	  //Setting main body of A matrix
	  for(int j = i+1; j < knn; j++)
	    {
	      radius = 0;
	      for(int n = 0; n < dim; n++)
		{
		  radius += pow(node1[n] - coordinates[interpolant_tree[tree_position_seed+j]*dim+n],2.);
		}
	      value = (*RBF)(sqrt(radius));
	      gsl_matrix_set(A,i,j,value);
	      gsl_matrix_set(A,j,i,value);
	    }
	  //Setting the result vector
	  gsl_vector_set(b,i,(*input_function)[interpolant_tree[tree_position_seed+i]]);
	}

  
      //Checking for the condition of the linear system via singular value
      //decomposition.
      gsl_linalg_SV_decomp(A, V, S, work_dummy);
      gsl_vector_minmax(S,&min,&max);
      condition.push_back(max/min);

      //Solving linear system using the SVD. 
      gsl_linalg_SV_solve(A, V, S, b, x);

      //Writing the output weights 
      for(int i = 0; i < knn; i++)
	{
	  lambda[tree_position_seed+i] = gsl_vector_get(x,i);
	}
    }

  //Calculating the interpolant

  output_function->resize(num_nodes_interpolant);

  for(int i = 0; i < num_nodes_interpolant; i++)
    {
      tree_position_seed = i*knn;
      for(int n = 0; n < dim; n++)
	{
	  node1[n] = interpolant_coordinates[i*dim+n];
	}
      value = 0;
      for(int j = 0; j < knn; j++)
	{
	  radius = 0;
	  for(int n = 0; n < dim; n++)
	    {
	      radius += pow(node1[n] - coordinates[interpolant_tree[tree_position_seed+j]*dim+n],2.);
	    }
	  value += (*RBF)(sqrt(radius))*lambda[i*knn+j];
	}
      (*output_function)[i] = value;
    }

  return gsl_stats_mean(&condition[0], 1, num_nodes_interpolant);

}	      

double unstructured_grid::differentiate(vector<double> *in, string selection, radial_basis_function *RBF, vector<double> *out)
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

vector<double> unstructured_grid::create_finite_differences_weights_col(string selection, radial_basis_function *RBF, int max_length)
{
  vector<double> aux;
  create_finite_differences_weights(selection, &aux, RBF);
  return findif_row_col_convert(num_nodes, max_length, &kD_tree, &aux);
}

unstructured_grid_1D::unstructured_grid_1D(int num_nodes_in)
{

  dim = 1;

  if(num_nodes_in < 1)
    {
      throw invalid_argument("UNSTRUC_GRID: num nodes must be > 0");
    } 

  num_nodes = num_nodes_in;
  coordinates.resize(num_nodes);

  for(int i = 0; i < num_nodes; i++)
    {
      coordinates[i] = 0.0;
    }
}

unstructured_grid_1D::unstructured_grid_1D(int num_nodes_in, double *input_coordinates, int stride)
{

  dim = 1;

  if(num_nodes_in < 1)
    {
      throw invalid_argument("UNSTRUC_GRID: num nodes must be > 0");
    } 

  num_nodes = num_nodes_in;
  coordinates.resize(num_nodes);

  for(int i = 0; i < num_nodes; i++)
    {
      coordinates[i] = input_coordinates[i*stride];
    }
}

double unstructured_grid_1D::create_finite_differences_weights(string selection, vector<double> *weights, radial_basis_function *RBF)
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

unstructured_grid_2D::unstructured_grid_2D(int num_nodes_in)
{

  dim = 2;

  if(num_nodes_in < 1)
    {
      throw invalid_argument("UNSTRUC_GRID: num nodes must be > 0");
    } 

  num_nodes = num_nodes_in;
  coordinates.resize(num_nodes*2);

  for(int i = 0; i < num_nodes*2; i++)
    {
      coordinates[i] = 0.0;
    }
}

unstructured_grid_2D::unstructured_grid_2D(int num_nodes_in, double *input_coordinates, int stride)
{

  dim = 2;
  if(num_nodes_in < 1)
    {
      throw invalid_argument("UNSTRUC_GRID: num nodes must be > 0");
    } 

  num_nodes = num_nodes_in;
  coordinates.resize(num_nodes*2);

  for(int i = 0; i < num_nodes; i++)
    {
      coordinates[i*2] = input_coordinates[i*stride];
      coordinates[i*2+1] = input_coordinates[i*stride+1];
    }
}

double unstructured_grid_2D::create_finite_differences_weights(string selection, vector<double> *weights, radial_basis_function *RBF)
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


unstructured_grid_3D::unstructured_grid_3D(int num_nodes_in)
{

  dim = 3;

  if(num_nodes_in < 1)
    {
      throw invalid_argument("UNSTRUC_GRID: num nodes must be > 0");
    } 

  num_nodes = num_nodes_in;
  coordinates.resize(num_nodes*3);

  for(int i = 0; i < num_nodes*3; i++)
    {
      coordinates[i] = 0.0;
    }
}

unstructured_grid_3D::unstructured_grid_3D(int num_nodes_in, double *input_coordinates, int stride)
{

  dim = 3; 
  if(num_nodes_in < 1)
    {
      throw invalid_argument("UNSTRUC_GRID: num nodes must be > 0");
    } 

  num_nodes = num_nodes_in;
  coordinates.resize(num_nodes*2);

  for(int i = 0; i < num_nodes; i++)
    {
      coordinates[i*3] = input_coordinates[i*stride];
      coordinates[i*3+1] = input_coordinates[i*stride+1];
      coordinates[i*3+2] = input_coordinates[i*stride+2];
    }
}

double unstructured_grid_3D::create_finite_differences_weights(string selection, vector<double> *weights, radial_basis_function *RBF)
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


 
int findif_row_col_convert(int dim, vector<int> *knn_in, vector<double> *coefficients_in, vector<int> *knn_out, vector<double> *coefficients_out)
{

  //Some initial sanity checks

  if(knn_in->size() != coefficients_in->size())
    {
      throw invalid_argument("ROW_COL_CONV: Input vectors of unequal size.");
    }

  if(knn_in->size() % dim != 0)
    {
      throw invalid_argument("ROW_COL_CONV: Input vector length does not match grid size.");
    }

  int nn = knn_in->size() / dim;

  //Checking maximum length of col vector in matrix

  int max_length = 0;
  int current_length;

  for(int col = 0; col < dim; col++)
    {
      current_length = 0;
      for(int row = 0; row < dim; row++)
	{
	  for(int i = 0; i < nn; i++)
	    {
	      if((*knn_in)[row*nn+i] == col)
		{
		  current_length++;
		  break;
		}
	    }
	}
      if(current_length > max_length)
	{
	  max_length = current_length;
	}
    }

  knn_out->resize(dim*max_length);
  coefficients_out->resize(dim*max_length);


  for(int col = 0; col < dim; col++)
    {
      int index = 0;
      bool main_loop = 1;
      while(index < max_length) 
	{
	  if(main_loop)
	    {
	      for(int row = 0; row < dim; row++)
		{
		  for(int i = 0; i < nn; i++)
		    {
		      if((*knn_in)[row*nn+i] == col)
			{
			  (*knn_out)[col*max_length+index] = row;
			  (*coefficients_out)[col*max_length+index] = (*coefficients_in)[row*nn+i];
			  index++;
			}
		    }
		}
	      main_loop = 0;
	    }
	  (*knn_out)[col*max_length+index] = 0;
	  (*coefficients_out)[col*max_length+index] = 0.0;
	  index++;
	}
    }

  return max_length;
 }

int findif_row_col_convert(int dim, vector<int> *knn_in, vector<int> *knn_out, vector<int> *length_counter_out)
{

  //Some initial sanity checks

  if(knn_in->size() % dim != 0)
    {
      throw invalid_argument("ROW_COL_CONV: Input vector length does not match grid size.");
    }
  length_counter_out->resize(dim);

  int nn = knn_in->size() / dim;

  //Checking maximum length of col vector in matrix

  int max_length = 0;
  int current_length;

  for(int col = 0; col < dim; col++)
    {
      current_length = 0;
      for(int row = 0; row < dim; row++)
	{
	  for(int i = 0; i < nn; i++)
	    {
	      if((*knn_in)[row*nn+i] == col)
		{
		  current_length++;
		  break;
		}
	    }
	}
      (*length_counter_out)[col] = current_length;
      if(current_length > max_length)
	{
	  max_length = current_length;
	}
    }

  knn_out->resize(dim*max_length);


  for(int col = 0; col < dim; col++)
    {
      int index = 0;
      bool main_loop = 1;
      while(index < max_length) 
	{
	  if(main_loop)
	    {
	      for(int row = 0; row < dim; row++)
		{
		  for(int i = 0; i < nn; i++)
		    {
		      if((*knn_in)[row*nn+i] == col)
			{
			  (*knn_out)[col*max_length+index] = row;
			  index++;
			  break;
			}
		    }
		}
	      main_loop = 0;
	    }
	  (*knn_out)[col*max_length+index] = 0;
	  index++;
	}
    }

  return max_length;
 }

vector<double> findif_row_col_convert(int dim, int max_length, vector<int> *knn_in, vector<double> *coefficients_in)
{

  vector<double> coefficients_out;
  coefficients_out.resize(dim*max_length);

  //Some initial sanity checks

  if(knn_in->size() != coefficients_in->size())
    {
      throw invalid_argument("ROW_COL_CONV: Input vectors of unequal size.");
    }

  if(knn_in->size() % dim != 0)
    {
      throw invalid_argument("ROW_COL_CONV: Input vector length does not match grid size.");
    }

  int nn = knn_in->size() / dim;

  for(int col = 0; col < dim; col++)
    {
      int index = 0;
      bool main_loop = 1;
      while(index < max_length) 
	{
	  if(main_loop)
	    {
	      for(int row = 0; row < dim; row++)
		{
		  for(int i = 0; i < nn; i++)
		    {
		      if((*knn_in)[row*nn+i] == col)
			{
			  coefficients_out[col*max_length+index] = (*coefficients_in)[row*nn+i];
			  index++;
			  break;
			}
		    }
		}
	      main_loop = 0;
	    }
	  coefficients_out[col*max_length+index] = 0;
	  index++;
	}
    }

  return coefficients_out;
 }











