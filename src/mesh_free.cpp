/*** mesh_free.cpp
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

#include <mfree/mesh_free.h>

mesh_free::mesh_free(int dim_in, vector<double> *input, int stride) : dim(dim_in)
{

  if(dim == 0)
    {
      throw invalid_argument("MFREE: Dimensionality of 0 is not possible.");
    }

  kD_update = true;
  kD_tree.resize(0);
  distances.resize(0);

  if(stride == 0)
    {
      stride = dim;
    }

  coordinates.resize(0);

  if(input != NULL)
    {
      num_nodes = input->size() / dim;
      for(int i = 0; i < num_nodes; i++)
	{
	  for(int j = 0; j < dim; j++)
	    {
	      coordinates.push_back((*input)[i*dim+j]);
	    }
	}
    }
  else
    {
      num_nodes = 0;
      coordinates.resize(0);
    }
}

mesh_free::mesh_free(mesh_free &input) : dim(input.dim), num_nodes(input.num_nodes), coordinates(input.coordinates), kD_update(input.kD_update)
{

  if(kD_update)
    {
      kD_tree.resize(0);
      distances.resize(0);
    }
  else
    {
      kD_tree = input.kD_tree;
      distances = input.distances;
    }

}


mesh_free::~mesh_free()
{
  coordinates.clear();
  kD_tree.clear();
  distances.clear();
}

mesh_free mesh_free::operator + (mesh_free &input)
{
  if(dim != input.return_grid_size(1))
    {
      throw invalid_argument("MFREE: Cannot add two mesh-free objects of differing dimensionality.");
    }

  vector<double> out_coords;

  for(int i = 0; i < num_nodes; i++)
    {
      for(int j = 0; j < dim; j++)
	{
	  out_coords.push_back((*this)(i,j));
	}
    }

  for(int i = 0; i < input.return_grid_size(0); i++)
    {
      for(int j = 0; j < dim; j++)
	{
	  out_coords.push_back(input(i,j));
	}
    }

  mesh_free output(dim,&out_coords);

  return output;
}

void mesh_free::operator += (mesh_free &input)
{

  if(dim != input.return_grid_size(1))
    {
      throw invalid_argument("MFREE: Cannot add two mesh-free object of differing dimensionality.");
    }

  kD_update = true;

  for(int i = 0; i < input.return_grid_size(0); i++)
    {
      for(int j = 0; j < dim; j++)
	{
	  coordinates.push_back(input(i,j));
	}
      num_nodes++;
    }
}

mesh_free mesh_free::operator - (mesh_free &input)
{

  if(dim != input.return_grid_size(1))
    {
      throw invalid_argument("MFREE: Cannot subract two mesh-free objects of differing dimensionality.");
    }

  vector<double> coords_input;
  for(int j = 0; j < input.return_grid_size(0); j++)
    {
      for(int d = 0; d < dim; d++)
	{
	  coords_input.push_back(input(j,d));
	}
    }

  vector<int> tree;
  tree.resize(num_nodes);
  vector<double> distances;
  distances.resize(num_nodes);
  flann::Matrix<double> flann_search(&coordinates[0],num_nodes,dim);
  flann::Matrix<int> flann_tree(&tree[0],num_nodes,1);
  flann::Matrix<double> flann_distances(&distances[0],num_nodes,1);
  flann::Matrix<double> flann_dataset(&coords_input[0],input.return_grid_size(0),dim);

  flann::Index<flann::L2<double> > index(flann_dataset, flann::KDTreeIndexParams(4));
  index.buildIndex();
  index.knnSearch(flann_search, flann_tree, flann_distances, 1, flann::SearchParams(128));


  vector<double> coords_out;
  for(int i = 0; i < num_nodes; i++)
    {
      if(distances[i] != 0.0)
	{
	  for(int j = 0; j < dim; j++)
	    {
	      coords_out.push_back((*this)(i,j));
	    }
	}
    }
  mesh_free output(dim,&coords_out);

  return output;
}

void mesh_free::operator -= (mesh_free &input)
{

  if(dim != input.return_grid_size(1))
    {
      throw invalid_argument("MFREE: Cannot subtract two mesh-free object of differing dimensionality.");
    }

  vector<double> coords_input;
  for(int j = 0; j < input.return_grid_size(0); j++)
    {
      for(int d = 0; d < dim; d++)
	{
	  coords_input.push_back(input(j,d));
	}
    }

  vector<int> tree;
  tree.resize(num_nodes);
  vector<double> distances;
  distances.resize(num_nodes);
  flann::Matrix<double> flann_search(&coordinates[0],num_nodes,dim);
  flann::Matrix<int> flann_tree(&tree[0],num_nodes,1);
  flann::Matrix<double> flann_distances(&distances[0],num_nodes,1);
  flann::Matrix<double> flann_dataset(&coords_input[0],input.return_grid_size(0),dim);

  flann::Index<flann::L2<double> > index(flann_dataset, flann::KDTreeIndexParams(4));
  index.buildIndex();
  index.knnSearch(flann_search, flann_tree, flann_distances, 1, flann::SearchParams(128));


  vector<double> coords_out;
  for(int i = 0; i < num_nodes; i++)
    {
      if(distances[i] != 0.0)
	{
	  for(int j = 0; j < dim; j++)
	    {
	      coords_out.push_back((*this)(i,j));
	    }
	}
    }

  coordinates.clear();
  coordinates = coords_out;
  num_nodes = coordinates.size()/dim;
  kD_update = true;

}

void mesh_free::operator *= (double scale)
{
  kD_update = true;

  for(int i =0; i < coordinates.size(); i++)
    {
      coordinates[i] *= scale;
    }
}

void mesh_free::operator *= (vector<double> scale)
{

  if(scale.size() < dim)
    {
      throw invalid_argument("MFREE: Scale vector must be >= domain diemsion.");
    }
  for(int i = 0; i < num_nodes; i++)
    {
      for(int j = 0; j < dim; j++)
	{
	  coordinates[i*dim+j] *= scale[j];
	}
    }
}

void mesh_free::operator - (unsigned int n)
{

  for(int i = 0; i < n; i++)
    {
      for(int j = 0; j < dim; j++)
	{
	  if(coordinates.size() > 0)
	    {
	      coordinates.pop_back();
	    }
	}
    }
  num_nodes  = coordinates.size() / dim;

  kD_update = true;
}




int mesh_free::return_grid_size(bool dim_check)
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


vector<double> mesh_free::operator () (int node)
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

double mesh_free::operator () (int node, int component)
{
  if(node >= 0 && node <= num_nodes && component >= 0)
    {
      return coordinates[node*dim+component];
    }
}

void mesh_free::set(int node, vector<double> new_coordinates)
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

void mesh_free::set(vector<double> *new_coordinates)
{

  if(new_coordinates->size() % dim != 0)
    {
      throw invalid_argument("MFREE: New coordinates vector invalid.");
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

void mesh_free::set(int num_nodes_in)
{

  kD_update = true;
  num_nodes = num_nodes_in;
  coordinates.clear();
  coordinates.resize(dim*num_nodes);
}

void mesh_free::print_info(string statement)
{

  cout <<statement <<": \t dim:" <<dim <<"  nodes:" <<num_nodes <<"\t" <<flush;
  for(int i = 0; i < num_nodes; i++)
    {
      cout <<"{" <<flush; 
      for(int j = 0; j < dim; j++)
	{
	  if(j != dim-1)
	    {
	      cout <<(*this)(i,j) <<"," <<flush;
	    }
	  else
	    {
	      cout <<(*this)(i,j) <<flush;
	    }
	}
      if(i != num_nodes-1)
	{
	  cout <<"};" <<flush;  
	}
      else
	{
	  cout <<"}" <<flush;
	}
    }
  cout <<endl;
}

void mesh_free::write_ASCII(string filename, bool col_description)
{
  ofstream output(filename.c_str());

  if(col_description)
    {
      output <<"#" <<flush;
      for(int i = 1; i <=dim ; i++)
	{
	  if(i == dim)
	    {
	      output <<"x_" <<i <<endl;
	    }
	  else
	    {
	      output <<"x_" <<i <<" " <<flush;
	    }
	}
    }

  for(int i = 0; i < num_nodes; i++)
    { 
      for(int j = 0; j < dim; j++)
	{
	  if(j != dim-1)
	    {
	      output <<(*this)(i,j) <<" " <<flush;
	    }
	  else
	    {
	      output <<(*this)(i,j) <<endl;
	    }
	}
    }
  output.close();


}

void mesh_free::build_tree(int nearest_neighbours)
{

  if(nearest_neighbours < 1)
    {
      throw invalid_argument("MFREE: invalid number of nearest neighbours.");
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

vector<int> mesh_free::neighbours(int node, int nearest_neighbours)
{

  vector<int> knn;
  
  //Make sure that routine only return if tree is okay.
  if(kD_update)
    {
      build_tree(nearest_neighbours);
    }

  //How many nearest neighours?
  int num_neighbours = kD_tree.size()/num_nodes;
  int index = node*num_neighbours;
  for(int i = 0; i < num_neighbours; i++)
    {
      knn.push_back(kD_tree[index+i]);
    }
  return knn;
}
 

vector<int> mesh_free::neighbours(int nearest_neighbours)
{
  if(kD_update)
    {
      build_tree(nearest_neighbours);
    }
  return kD_tree;
}

vector<double> mesh_free::provide_distances(int nearest_neighbours)
{
  if(kD_update)
    {
      build_tree(nearest_neighbours);
    }
  return distances;
}

int mesh_free::neighbours_col(vector<int> *neighbours, vector<int> *length_counter, int nearest_neighbours)
{

  if(kD_update)
    {
      build_tree(nearest_neighbours);
    }

  return findif_row_col_convert(num_nodes, &kD_tree, neighbours, length_counter);
  
}

vector<int> mesh_free::embed(mesh_free *input, int knn)
{

  int ref_nodes = input->return_grid_size();

  vector<double> distances;
  distances.resize(ref_nodes*knn);

  vector<int> output;
  output.resize(ref_nodes*knn);


  if(input->return_grid_size(1) != dim)
    {
      throw invalid_argument("M_FREE: Dimensions must match for embed operation.");
    }

  vector<double> ref_coordinates;

  for(int i = 0; i < ref_nodes; i++)
    {
      for(int j = 0; j < dim; j++)
	{
	  ref_coordinates.push_back((*input)(i,j));
	}
    }

  
  flann::Matrix<double> flann_dataset(&coordinates[0],num_nodes,dim);
  flann::Matrix<int> flann_tree(&output[0],ref_nodes,knn);
  flann::Matrix<double> flann_distances(&distances[0],ref_nodes,knn);
  flann::Matrix<double> flann_coordinates(&ref_coordinates[0],ref_nodes,dim);
  flann::Index<flann::L2<double> > index(flann_dataset, flann::KDTreeIndexParams(4));
  index.buildIndex();
  
  
  //Performing flann nearest neighbours search
  index.knnSearch(flann_coordinates, flann_tree, flann_distances, knn, flann::SearchParams(128));

  return output;
}

vector<int> mesh_free::embed(vector<double> *input, int knn, int stride)
{

  if(stride == 0)
    {
      stride = dim;
    }

  if(input->size() < dim)
    {
      throw invalid_argument("M_FREE: Input coordinate vector too short for embed.");
    }

  int ref_nodes = floor((double) input->size() / (double) stride);
  if(ref_nodes == 0)
    {
      ref_nodes++;
    }


  vector<double> distances;
  distances.resize(ref_nodes*knn);


  vector<int> output;
  output.resize(ref_nodes*knn);


  vector<double> ref_coordinates;

  for(int i = 0; i < input->size(); i += stride)
    {
      for(int j = 0; j < dim; j++)
	{
	  ref_coordinates.push_back((*input)[i+j]);
	}
    }

  
  flann::Matrix<double> flann_dataset(&coordinates[0],num_nodes,dim);
  flann::Matrix<int> flann_tree(&output[0],ref_nodes,knn);
  flann::Matrix<double> flann_distances(&distances[0],ref_nodes,knn);
  flann::Matrix<double> flann_coordinates(&ref_coordinates[0],ref_nodes,dim);
  flann::Index<flann::L2<double> > index(flann_dataset, flann::KDTreeIndexParams(4));
  index.buildIndex();
  
  
  //Performing flann nearest neighbours search
  index.knnSearch(flann_coordinates, flann_tree, flann_distances, knn, flann::SearchParams(128));

  return output;
}



double mesh_free::interpolate(vector<double> *output_grid, vector<double> *input_function, vector<double> *output_function,  radial_basis_function *RBF, int knn, int stride)
{

  if(stride == 0)
    {
      stride = dim;
    }

  //Checking for validity of the input data

  if(input_function->size() != num_nodes)
    {
      throw invalid_argument("MFREE: Input function for interpolate is invalid.");
    }

  if(output_grid->size() < dim)
    {
      throw invalid_argument("MFREE: Input grid for interpolate is invalid.");
    }

  if(output_grid->size()%stride != 0 && output_grid->size()%stride < dim)
    {
      throw invalid_argument("MFREE: Input grid for interpolate is invalid.");
    }

  if(stride < 1)
    {
      throw invalid_argument("MFREE: Stride for input grid invalid.");
    }

  if(knn < 1)
    {
      throw invalid_argument("MFREE: Unsufficient number of neighbours to interpolate.");
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

  gsl_matrix_free(A);
  gsl_matrix_free(V);
  gsl_vector_free(S);
  gsl_vector_free(work_dummy);
  gsl_vector_free(b);
  gsl_vector_free(x);

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

double mesh_free::interpolate(vector<double> *output_grid, vector<double> *input_function, vector<double> *output_function,  radial_basis_function_shape *RBF, vector<double> *adaptive_shape_parameter, int knn, int stride)
{

  double shape_save = RBF->show_epsilon();

  if(stride == 0)
    {
      stride = dim;
    }

  //Checking for validity of the input data

  if(input_function->size() != num_nodes)
    {
      throw invalid_argument("MFREE: Input function for interpolate is invalid.");
    }

  if(output_grid->size() < dim)
    {
      throw invalid_argument("MFREE: Input grid for interpolate is invalid.");
    }

  if(output_grid->size()%stride != 0 && output_grid->size()%stride < dim)
    {
      throw invalid_argument("MFREE: Input grid for interpolate is invalid.");
    }

  if(stride < 1)
    {
      throw invalid_argument("MFREE: Stride for input grid invalid.");
    }

  if(knn < 1)
    {
      throw invalid_argument("MFREE: Unsufficient number of neighbours to interpolate.");
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

  if(adaptive_shape_parameter->size() < num_nodes_interpolant)
    {
      throw invalid_argument("MFREE: Unsufficient number of adaptive shape parameters.");
    }



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
      RBF->set_epsilon((*adaptive_shape_parameter)[global]);
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
  gsl_matrix_free(A);
  gsl_matrix_free(V);
  gsl_vector_free(S);
  gsl_vector_free(work_dummy);
  gsl_vector_free(b);
  gsl_vector_free(x);


  //Calculating the interpolant

  output_function->resize(num_nodes_interpolant);

  for(int i = 0; i < num_nodes_interpolant; i++)
    {
      RBF->set_epsilon((*adaptive_shape_parameter)[i]);
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
  RBF->set_epsilon(shape_save);

  return gsl_stats_mean(&condition[0], 1, num_nodes_interpolant);

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











