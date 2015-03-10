/*** grid_utils.cpp
This set of tools provides functionality for the SaWLens
workflow. This includes grid manipulations, averaging routines
and other useful utilities. 

Julian Merten
Universiy of Oxford
Feb 2015
julian.merten@physics.ox.ac.uk
http://www.julianmerten.net
***/

#include <mfree/grid_utils.h>

void grid_sampling(unstructured_grid *big_grid, unstructured_grid *small_grid, int rng_seed)
{

  int big_grid_size = big_grid->return_grid_size();
  int small_grid_size = small_grid->return_grid_size();

  //Checking for matching grid dimensions

  if(big_grid->return_grid_size(1) != small_grid->return_grid_size(1))
    {
      throw invalid_argument("GRD_CONV: Grid dimensions do not match.");
    }

  if(big_grid_size < small_grid_size)
    {
      throw invalid_argument("GRD_CONV: Second argument grid must be smaller.");
    }

  //Initialising the random number generator with the system time
  //Random numbers can then be drawm from r

  if(rng_seed == -1)
    {
      time_t current_time;
      ctime(&current_time);
      rng_seed = current_time;
    }

  const gsl_rng_type * T;
  gsl_rng * r;
  gsl_rng_env_setup();
  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_rng_set(r,rng_seed);

  vector<bool> check_sampling;

  for(int i = 0; i < big_grid_size; i++)
    {
      check_sampling.push_back(0);
    }
  int check_index = 0;
  int index;

  while(check_index < small_grid_size)
    {
      index = floor(gsl_rng_uniform(r)*big_grid_size);
      if(!check_sampling[index])
	{
	  vector<double> cpy_coordinates = (* big_grid)(index);
	  small_grid->set(check_index,cpy_coordinates);
	  check_sampling[index] = 1;
	  check_index++;
	}
    }
}

double grid_conversion(unstructured_grid *big_grid, double *big_function, int stride_big, unstructured_grid *small_grid, vector<double> *small_function, vector<double> *covariance, int knn, string selection)
{

  int big_grid_size = big_grid->return_grid_size();
  int small_grid_size = small_grid->return_grid_size();
  int big_grid_dim = big_grid->return_grid_size(1); 
  int small_grid_dim = small_grid->return_grid_size(1); 

  //Standard checks before 

  //Checking for matching grid dimensions

  if(big_grid_dim != small_grid_dim)
    {
      throw invalid_argument("GRD_CONV: Grid dimensions do not match.");
    }

  if(big_grid_size < small_grid_size)
    {
      throw invalid_argument("GRD_CONV: Second argument grid must be smaller.");
    }

  //Setting up necessary tree quantities
  vector<int> kD_tree;
  kD_tree.resize(small_grid_size*knn);
  vector<double> distances;
  distances.resize(small_grid_size*knn);
  flann::Matrix<int> flann_tree(&kD_tree[0],small_grid_size,knn);
  flann::Matrix<double> flann_distances(&distances[0],small_grid_size,knn);
 
  //Querying the grid coordinates

  vector<double> big_grid_coordinates;	     
  vector<double> small_grid_coordinates;

  for(int i = 0; i < big_grid_size; i++)
    {
      vector<double> dummy_coordinates;
      dummy_coordinates = (* big_grid)(i);
      for(int j = 0; j < dummy_coordinates.size(); j++)
	{
	  big_grid_coordinates.push_back(dummy_coordinates[j]);
	}
    }

  //Building tree index

  flann::Matrix<double> flann_dataset(&big_grid_coordinates[0],big_grid_size,big_grid_dim);
  flann::Index<flann::L2<double> > tree_index(flann_dataset, flann::KDTreeIndexParams(4));
  tree_index.buildIndex();
  
  for(int i = 0; i < small_grid_size; i++)
    {
      vector<double> dummy_coordinates;
      dummy_coordinates = (* small_grid)(i);
      for(int j = 0; j < dummy_coordinates.size(); j++)
	{
	  small_grid_coordinates.push_back(dummy_coordinates[j]);
	}
    }

  //Finding nearest neighbours
  
  flann::Matrix<double> reference_points(&small_grid_coordinates[0],small_grid_size,small_grid_dim);
  tree_index.knnSearch(reference_points, flann_tree, flann_distances, knn, flann::SearchParams(128));

  //Building up the statistics samples

  covariance->resize(small_grid_size*small_grid_size);
  small_function->resize(small_grid_size);
  vector<double> samples;

  for(int i = 0; i < small_grid_size; i++)
    {
      for(int j = 0; j < knn; j++)
	{
	  samples.push_back(big_function[kD_tree[i*knn+j]]);
	}
    }

  //Calculating means and covariances

  double max_distance = 0.;
  double current_distance;
  int counter;

  vector<double> sd;
  sd.resize(small_grid_size);

  for(int i = 0; i < small_grid_size; i++)
    {
      current_distance = distances[i*knn];
      if(current_distance > max_distance)
	{
	  max_distance = current_distance;
	}
      if(selection == "unweighted")
	{
	  (*small_function)[i] = gsl_stats_mean(&samples[i*knn],1,knn);
	  sd[i] = gsl_stats_sd_m(&samples[i*knn],1,knn,(*small_function)[i]);
	}
      else if(selection == "inv_dist_weighted")
	{
	  vector<double> weights;
	  for(int j = 0; j < knn; j++)
	    {
	      weights.push_back(1./(1.+distances[i*knn+j]));
	    }
	  (*small_function)[i] = gsl_stats_wmean(&weights[0],1,&samples[i*knn],1,knn);
	  sd[i] = gsl_stats_wsd_m(&weights[0],1,&samples[i*knn],1,knn,(*small_function)[i]);
	}


	  for(int j = i; j < small_grid_size; j++)
	    {
	      counter = 0;	
	      for(int l = 0; l < knn; l++)
		{	 
		  for(int k = 0; k < knn; k++)
		    {
		      if(kD_tree[i*knn+l] == kD_tree[j*knn+k])
			{
			  counter++;
			}
		    }
		}
	      /*
	      (* covariance)[i*small_grid_size+j] =0.04* (double) counter/ (double) knn;
	      if(i == j)
		{
		  (* covariance)[i*small_grid_size+j] *= 1.001;
		}
	      */
	      if(counter > 0)
		{
		  (* covariance)[i*small_grid_size+j] = (double) counter/ (double) (knn*knn);
		  //(* covariance)[i*small_grid_size+j] = gsl_stats_covariance(&samples[i*knn],1,&samples[j*knn],1,knn);
		}
	      //		  (* covariance)[i*small_grid_size+j] = gsl_stats_covariance(&samples[i*knn],1,&samples[j*knn],1,knn);
	    }
    }
  for(int i = 0; i < small_grid_size; i++)
    {
      for(int j = i; j < small_grid_size; j++)
	{
	  (* covariance)[i*small_grid_size+j] *= sd[i]*sd[j];
	  if((* covariance)[i*small_grid_size+j] > 0.95*(* covariance)[i*small_grid_size+i] && i!=j )
	    {
	      (* covariance)[i*small_grid_size+j] *= 0.95*(* covariance)[i*small_grid_size+j];
	    }

	  (* covariance)[j*small_grid_size+i] = (* covariance)[i*small_grid_size+j];
	}
    }
	
  return max_distance;
}

double grid_conversion(unstructured_grid *big_grid, double *big_function, int stride_big, unstructured_grid *small_grid, vector<double> *small_function)
{
  int big_grid_size = big_grid->return_grid_size();
  int small_grid_size = small_grid->return_grid_size();
  int big_grid_dim = big_grid->return_grid_size(1); 
  int small_grid_dim = small_grid->return_grid_size(1); 

  //Standard checks before 

  //Checking for matching grid dimensions

  if(big_grid_dim != small_grid_dim)
    {
      throw invalid_argument("GRD_CONV: Grid dimensions do not match.");
    }

  if(big_grid_size < small_grid_size)
    {
      throw invalid_argument("GRD_CONV: Second argument grid must be smaller.");
    }

  //Setting up necessary tree quantities
  vector<int> kD_tree;
  kD_tree.resize(small_grid_size);
  vector<double> distances;
  distances.resize(small_grid_size);
  flann::Matrix<int> flann_tree(&kD_tree[0],small_grid_size,1);
  flann::Matrix<double> flann_distances(&distances[0],small_grid_size,1);
 
  //Querying the grid coordinates

  vector<double> big_grid_coordinates;	     
  vector<double> small_grid_coordinates;

  for(int i = 0; i < big_grid_size; i++)
    {
      vector<double> dummy_coordinates;
      dummy_coordinates = (* big_grid)(i);
      for(int j = 0; j < dummy_coordinates.size(); j++)
	{
	  big_grid_coordinates.push_back(dummy_coordinates[j]);
	}
    }

  //Building tree index

  flann::Matrix<double> flann_dataset(&big_grid_coordinates[0],big_grid_size,big_grid_dim);
  flann::Index<flann::L2<double> > tree_index(flann_dataset, flann::KDTreeIndexParams(4));
  tree_index.buildIndex();
  
  for(int i = 0; i < small_grid_size; i++)
    {
      vector<double> dummy_coordinates;
      dummy_coordinates = (* small_grid)(i);
      for(int j = 0; j < dummy_coordinates.size(); j++)
	{
	  small_grid_coordinates.push_back(dummy_coordinates[j]);
	}
    }

  //Finding nearest neighbours
  
  flann::Matrix<double> reference_points(&small_grid_coordinates[0],small_grid_size,small_grid_dim);
  tree_index.knnSearch(reference_points, flann_tree, flann_distances, 1, flann::SearchParams(128));

  //Filling the output vector with the input vector at best-match positions

  double max_distance = 0.;
  double current_distance;

  small_function->resize(small_grid_size);

  for(int j = 0; j < small_grid_size; j++)
    {
      current_distance = distances[j];
      (*small_function)[j] = big_function[kD_tree[j]];
	
      if(max_distance < current_distance)
	{
	  max_distance = current_distance;
	}
    }
  return max_distance;
}

void grid_combination(vector<galaxy> *shear_field, vector<ccurve_estimator> *ccurve, vector<multiple_image_system> *msystems_in, vector<double> *shear1_covariance,vector<double> *shear2_covariance, vector<double> *coordinate_vector, vector<double> *shear1_out, vector<double>* shear2_out, vector<double> *ccurve_prefactor, vector<double> *shear1_covariance_out, vector<double> *shear2_covariance_out, vector<double> *redshifts)
{

  //Counting the number of nodes to the final grid

  int shear_values = shear_field->size();
  int ccurve_values = ccurve->size();
  int msystem_values = 0; 

  for(int i = 0; i < msystems_in->size(); i++)
    {
      msystem_values += (*msystems_in)[i]();
    }

  int num_nodes = shear_values + ccurve_values + msystem_values;
  coordinate_vector->resize(2*num_nodes);
  shear1_out->resize(num_nodes);
  shear2_out->resize(num_nodes);
  ccurve_prefactor->resize(num_nodes);
  redshifts->resize(num_nodes);

  for(int i = 0; i < shear_values; i++)
    {
      (*shear_field)[i].set("index", i);
      (*coordinate_vector)[2*i] = (*shear_field)[i].get("x");
      (*coordinate_vector)[2*i+1] = (*shear_field)[i].get("y");
      (*shear1_out)[i] = (*shear_field)[i].get("g1");
      (*shear2_out)[i] = (*shear_field)[i].get("g2");
      (*ccurve_prefactor)[i] = 0.;
      (*redshifts)[i] = (*shear_field)[i].get("z");
    }

  for(int i = 0; i < ccurve_values; i++)
    {
      (*ccurve)[i].set("index",shear_values+i);
      (*coordinate_vector)[2*(shear_values+i)] = (*ccurve)[i].get("x");
      (*coordinate_vector)[2*(shear_values+i)+1] = (*ccurve)[i].get("y");
      (*shear1_out)[shear_values+i] = 0.;
      (*shear2_out)[shear_values+i] = 0.;
      (*ccurve_prefactor)[shear_values+i] = 1.;
      (*redshifts)[i] = (*ccurve)[shear_values+i].get("z");
    }

  int grid_index = shear_values + ccurve_values - 1;

  for(int i = 0; i < msystems_in->size(); i++)
    {
      vector<int> index;
      for(int j = 0; j < (*msystems_in)[i](); j++)
	{
	  index.push_back(++grid_index);
	  (*coordinate_vector)[2*grid_index] = (*msystems_in)[i](j,0);
	  (*coordinate_vector)[2*grid_index+1] = (*msystems_in)[i](j,1);
	  (*shear1_out)[grid_index] = 0.;
	  (*shear2_out)[grid_index] = 0.;
	  (*ccurve_prefactor)[grid_index] = 0.;
	  (*redshifts)[i] = (*msystems_in)[grid_index].get_redshift();
	}
      (*msystems_in)[i].assign_indices(index);
    }


  //Now we also need to manipulate the covariance matrices, which is a little 
  //more complicated


  shear1_covariance_out->resize(num_nodes*num_nodes);
  shear2_covariance_out->resize(num_nodes*num_nodes);



  for(int i = 0; i < num_nodes; i++)
    {
      for(int j = i; j < num_nodes; j++)
	{
	  if(j < shear_values && i < shear_values)
	    {
	      (*shear1_covariance_out)[i*num_nodes+j] = (*shear1_covariance)[i*shear_values+j];
	      (*shear2_covariance_out)[i*num_nodes+j] = (*shear2_covariance)[i*shear_values+j];
	    }
	  else
	    {
	      (*shear1_covariance_out)[i*num_nodes+j] = 0.;
	      (*shear2_covariance_out)[i*num_nodes+j] = 0.;
	    }
	  (*shear1_covariance_out)[j*num_nodes+i] = (*shear1_covariance_out)[i*num_nodes+j]; 
	  (*shear2_covariance_out)[j*num_nodes+i] = (*shear2_covariance_out)[i*num_nodes+j];
	}
    }

 
}


coordinate_grid::coordinate_grid(int num_nodes)
{

  if(num_nodes < 1)
    {
      throw invalid_argument("C_GRID: Number of nodes must be larger than 0.");
    }

  coordinates.resize(2*num_nodes);

  for(int i = 0; i < 2*num_nodes; i++)
    {
      coordinates[i] = 0.;
    }
}

coordinate_grid::coordinate_grid(string type, int num_nodes, int refinement_level, int rng_seed)
{

  coordinates.clear();
  double PI = acos(-1.);

  if(rng_seed == -1)
    {
      time_t current_time;
      time(&current_time);
      rng_seed = current_time;
    }

  if(num_nodes < 1)
    {
      throw invalid_argument("C_GRID: Number of nodes must be larger than 0.");
    }


  if(type == "random")
    {

      const gsl_rng_type * T;
      gsl_rng * r;
      gsl_rng_env_setup();
      T = gsl_rng_default;
      r = gsl_rng_alloc (T);
      gsl_rng_set(r,rng_seed);


      vector<double> radii;
      vector<double> nodes_per_bin;
      
      radii.push_back(1.0);
      
      for(int i = 1; i <= refinement_level; i++)
	{
	  radii.push_back(pow(0.5,i));
	}
      
      double density = 0.;
      vector<double> areas;

      for(int i = 0; i <= refinement_level; i++)
	{
	  if(i != refinement_level)
	    {
	      areas.push_back(((double) i + 1.) * (PI*radii[i]*radii[i] - PI*radii[i+1]*radii[i+1])); 
	      density += areas[i];
	      
	    }
	  else
	    {
	      areas.push_back(((double) i + 1.) * PI*radii[i]*radii[i]);
	      density += areas[i];
	    }
	}
      density = num_nodes/density;

      double radius, phi, x, y;
      
      for(int i = 0; i <= refinement_level; i++)
	{
	  areas[i] *= density;
	  for(int j = 0; j < areas[i]; j++)
	    {
	      if(i == refinement_level)
		{
		  radius = sqrt(gsl_rng_uniform(r) * (radii[i]*radii[i]));
		}
	      else
		{
		  radius = sqrt(gsl_rng_uniform(r)*pow(radii[i],2.));
		}
	      phi = gsl_rng_uniform(r)*2.*PI;
	      x = radius*cos(phi);
	      y = radius*sin(phi);
	      coordinates.push_back(x);
	      coordinates.push_back(y);
	    }
	}
      gsl_rng_free (r);
    }

  else if(type == "regular")
    {

      //Finding out how many nodes we can have

      double nodes_per_refinement = 0.;

      for(int i = 0; i <= refinement_level; i++)
	{
	  if(i == refinement_level)
	    {
	      nodes_per_refinement += 1.;
	    }
	  else
	    {
	      nodes_per_refinement += 0.75;
	    }
	}

      nodes_per_refinement = num_nodes / nodes_per_refinement;

      int nodes = ceil(sqrt(nodes_per_refinement));

      double current_pixel_size = 2.0/nodes;
      nodes = floor(0.5/current_pixel_size);
      current_pixel_size = 0.5/nodes;

      for(int i = 0; i <= refinement_level; i++)
	{
	  vector<double> current_grid;
	  if(i == refinement_level)
	    {
	      current_grid = build_unit_grid(current_pixel_size,0);
	    }
	  else
	    {
	      current_grid = build_unit_grid(current_pixel_size,1);
	    }
	  for(int j = 0; j < current_grid.size(); j++)
	    {
	      current_grid[j] *= pow(0.5,i);
	      coordinates.push_back(current_grid[j]);
	    }
	  
	}
    }
}
		  
coordinate_grid::~coordinate_grid()
{

  coordinates.clear();

}

vector<double> coordinate_grid::operator() ()
{

  return coordinates;

}

void coordinate_grid::add_mask(double x1, double x2, double y1, double y2)
{

  if(x2 < x1 || y2 < y1)
    {
      throw invalid_argument("C_GRID: Mask boundaries are invalid.");
    }

  vector<double>::iterator it;
  for(it=coordinates.begin(); it < coordinates.end(); it +=2)
    {
      if(*it > x1 && *it < x2 && *(it+1) > y1 && *(it+1) < y2)
	{
	  coordinates.erase(it,it+2);
	  it = it-2;
	}
    }
}

void coordinate_grid::add_mask(double x1, double y1, double r)
{
  if(r < 0)
    {
      throw invalid_argument("C_GRID: Mask radius is invalid.");
    }
  vector<double>::iterator it;
  for(it=coordinates.begin(); it < coordinates.end(); it +=2)
    {
      if(pow(*it-x1,2.)+pow(*(it+1)-y1,2.) < r*r)
	{
	  coordinates.erase(it,it+2);
	  it = it-2;
	}
    }
}

void coordinate_grid::write(string filename)
{

  ofstream out(filename.c_str());

  for(int i = 0; i < coordinates.size(); i += 2)
    {
      out <<coordinates[i] <<"\t" <<coordinates[i+1] <<endl;
    }
}

void coordinate_grid::scale(double factor)
{

  for(int i = 0; i < coordinates.size(); i++)
    {
      coordinates[i] *= factor;
    }

}


vector<double> build_unit_grid(double pixel_size, bool spare_centre)
{
  vector<double> out;

  for(double x = -1.; x <= 1.+1e-12; x += pixel_size)
    {
      for(double y = -1.; y <= 1.+1e-12; y += pixel_size) 
	{
	  if(spare_centre)
	    {
	      if(abs(x) > 0.5 || abs(y) > 0.5)
		{
		  out.push_back(x);
		  out.push_back(y);
		}
	    }
	  else
	    {
	      out.push_back(x);
	      out.push_back(y);
	    }
	}
    }

  return out;
}
