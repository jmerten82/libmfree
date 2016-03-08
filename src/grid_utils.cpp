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

void grid_sampling(mesh_free *big_grid, mesh_free *small_grid, int rng_seed)
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

vector<double> optimise_grid_differentiation(mesh_free_differentiate *mesh_free_in,radial_basis_function_shape *rbf_in,  test_function *test_function_in, vector<string> differentials = vector<string>(), int refinements, int steps, double eps, string verbose_mode)
{
  bool verbose = 0;

  double eps_save = rbf_in->show_epsilon();

  if(verbose_mode != "")
    {
      verbose = 1;
    }

  ofstream out_log;

  if(verbose)
    {
      out_log.open(verbose_mode.c_str());
    }

  //Problem dimension
  int dim = mesh_free_in->return_grid_size();

  //Getting the distances from the mesh. This makes sure that tree is created. 

  vector<double> distances = mesh_free_in->provide_distances();
  int stride = distances.size() / dim;

  //Finding minimum nn distance with gsl vector views
  gsl_vector_view minimum_dists = gsl_vector_view_array_with_stride (&distances[1],stride,dim);
  double min_dist = gsl_vector_min(&minimum_dists.vector);  
 

  min_dist = sqrt(min_dist);

  double eps_start_save = eps;
  double eps_stop_save = 1./min_dist;
  double step_save = (eps_stop_save-eps_start_save)/steps;

  vector<double> base_function;
  vector<vector<double> > base_coords;
  for(int i = 0; i < dim; i++)
    {
      vector<double> coords;
      coords = (*mesh_free_in)(i);
      base_coords.push_back(coords);
      base_function.push_back((*test_function_in)(coords));
    }


  double final_eps;
  vector<double> output;

  for(int diffs = 0; diffs < differentials.size(); diffs++)
    {
      double eps_start = eps_start_save;
      double eps_stop = eps_stop_save;
      double step = step_save;
      double main_index = 0;
      ofstream out_log;
      if(verbose)
	{
	  string out_file = verbose_mode + "_" + differentials[diffs] + ".dat";
	  out_log.open(out_file.c_str());
	  out_log <<"#epsilon condition avg_error max_error" <<endl;
	}

      while(main_index <= refinements)
	{
	  vector<double> epsilons, avg_errors, max_errors, conditions;
	  for(double eps = eps_start; eps <= eps_stop; eps += step)
	    {
	      vector<double> difference_vector;
	      double condition;
	      rbf_in->set_epsilon(eps);
	      vector<double> current_diff_out;
	      condition = mesh_free_in->differentiate(&base_function,differentials[diffs],rbf_in, &current_diff_out);
	      for(int i = 0; i < dim; i++)
		{
		  double current_real = test_function_in->D(base_coords[i],differentials[diffs]);
		  difference_vector.push_back(abs((current_diff_out[i]-current_real)/current_real));
		} //End of all nodes loop

	      double avg_error = gsl_stats_mean(&difference_vector[0],1,difference_vector.size());
	      avg_errors.push_back(avg_error);
	      vector<double>::iterator max_iterator;
	      max_iterator = max_element(difference_vector.begin(),difference_vector.end());
	      max_errors.push_back(*max_iterator);
	      epsilons.push_back(eps);
	      conditions.push_back(condition);
	      if(verbose)
		{
		  out_log <<eps <<"\t" <<condition <<"\t" <<avg_error <<"\t" <<*max_iterator <<endl;
		}
	    } //End of eps loop

	  vector<double>::iterator best_eps_error;
	  best_eps_error = min_element(avg_errors.begin(),avg_errors.end());
	  int best_eps_index = distance(avg_errors.begin(),best_eps_error);
	  if(best_eps_index > 0)
	    {
	      eps_start = epsilons[best_eps_index-1];
	    }
	  else
	    {
	      eps_stop = epsilons[0];
	    }
	  if(best_eps_index < epsilons.size()-1)
	    {
	      eps_stop = epsilons[best_eps_index+1];
	    }
	  else
	    {
	      eps_stop = epsilons[epsilons.size()-1];
	    }
	  step = (eps_stop-eps_start)/steps; 
	  
	  final_eps = epsilons[best_eps_index];    
	  main_index++;
	} //End of all refinements loop 
      if(verbose)
	{
	  out_log.close();
	}
      output.push_back(final_eps);
    } //End of run over all derivatives

  rbf_in->set_epsilon(eps_save);
  return output;
}

vector<double> optimise_adaptive_grid_differentiation(mesh_free_differentiate *mesh_free_in,radial_basis_function_shape *rbf_in,  test_function *test_function_in, string differential, int knn, int refinements, int steps, double eps, string verbose_mode)
{

  bool verbose = false;

  if(verbose_mode != "")
    {
      verbose = true;
    }

  double eps_save = rbf_in->show_epsilon();

  vector<double> adaptive_shape_parameter;

  if(mesh_free_in->return_grid_size(1) != (*test_function_in)())
    {
      throw invalid_argument("SHAPE_OPT: Test function and mesh free not matching in dim.");
    }


  //Evaluating test function at all support nodes
  vector<vector<double> > support_coordinates;
  vector<double> support_function;
  for(int i = 0; i < mesh_free_in->return_grid_size(); i++)
    {
      vector<double> current_coordinates = (*mesh_free_in)(i);
      support_function.push_back((*test_function_in)(current_coordinates));
      support_coordinates.push_back(current_coordinates);
    }
  //Building the tree for the grid once to query some distances later.
  vector<int> interpolant_tree;
  vector<double> interpolant_distances;
  interpolant_tree.resize(knn);
  interpolant_distances.resize(knn);
  flann::Matrix<double> flann_dataset(&support_coordinates[0][0],mesh_free_in->return_grid_size(),mesh_free_in->return_grid_size(1));
  flann::Index<flann::L2<double> > index(flann_dataset, flann::KDTreeIndexParams(4));
  index.buildIndex();

  ofstream out_log;
  if(verbose)
    {
      out_log.open(verbose_mode.c_str());
      out_log <<"#pixel epsilon condition error" <<endl;
    }

  double current_target_value;
  for(int global = 0; global < mesh_free_in->return_grid_size(); global++)
    {
      current_target_value = test_function_in->D(support_coordinates[global],differential);

      //Finding maximum shape parameter. 
      vector<int> dummy_tree;
      dummy_tree.resize(1);
      vector<double> current_min_dist;
      current_min_dist.resize(1);
      flann::Matrix<double> flann_dataset_interpolant(&support_coordinates[global][0],1,mesh_free_in->return_grid_size(1));
      flann::Matrix<int> flann_tree(&dummy_tree[0],1,1);
      flann::Matrix<double> flann_distances(&current_min_dist[0],1,1);
      index.knnSearch(flann_dataset_interpolant, flann_tree, flann_distances,1, flann::SearchParams(128));

      //Doing the shape parameter loop. 
      double eps_start = eps;
      double eps_stop = 1./(sqrt(current_min_dist[0]+eps));
      double step = (eps_stop - eps_start) / steps;

      double main_index = 0;
      double final_eps;
      while(main_index <= refinements)
	{
	  vector<double> epsilons, errors, conditions;
	  for(double current_eps = eps_start; current_eps <= eps_stop; current_eps += step)
	    {
	      double condition,error;
	      vector<double> current_interpol_out;
	      rbf_in->set_epsilon(current_eps);
	      
	      condition = mesh_free_in->differentiate(&support_coordinates[global],&support_function,differential,rbf_in, &current_interpol_out, knn);
	      error = abs((current_interpol_out[0]-current_target_value)/current_target_value);
	      conditions.push_back(condition);
	      errors.push_back(error);
	      epsilons.push_back(current_eps);
	      if(verbose)
		{
		  out_log <<global <<"\t" <<current_eps <<"\t" <<condition <<"\t" <<error <<endl;
		}
	    } // end of eps loop


	  vector<double>::iterator best_eps_error;
	  best_eps_error = min_element(errors.begin(),errors.end());
	  int best_eps_index = distance(errors.begin(),best_eps_error);
	  if(best_eps_index > 0)
	    {
	      eps_start = epsilons[best_eps_index-1];
	    }
	  else
	    {
	      eps_stop = epsilons[0];
	    }
	  if(best_eps_index < epsilons.size()-1)
	    {
	      eps_stop = epsilons[best_eps_index+1];
	    }
	  else
	    {
	      eps_stop = epsilons[epsilons.size()-1];
	    }
	  step = (eps_stop-eps_start)/steps; 
	  
	  final_eps = epsilons[best_eps_index];    
	  main_index++;
	      
	} //end of refinement loop
      
      adaptive_shape_parameter.push_back(final_eps);

    } //End of all nodes loop

  if(verbose)
    {
      out_log.close();
    }

  rbf_in->set_epsilon(eps_save);

  return adaptive_shape_parameter;

}

double optimise_grid_interpolation(mesh_free *mesh_free_in, mesh_free *mesh_free_target, radial_basis_function_shape *rbf_in,  test_function *test_function_in, int knn, string test_function_switch, int refinements, int steps, double eps, string verbose_mode)
{

  bool verbose = 0;

  double eps_save = rbf_in->show_epsilon();

  if(verbose_mode != "")
    {
      verbose = 1;
    }

  ofstream out_log;

  //Problem dimension
  int dim = mesh_free_in->return_grid_size();
  int dim_target = mesh_free_target->return_grid_size();


  //Getting the distances from the mesh. This makes sure that tree is created. 

  vector<double> distances = mesh_free_in->provide_distances();
  int stride = distances.size() / dim;

  //Finding minimum nn distance with gsl vector views
  gsl_vector_view minimum_dists = gsl_vector_view_array_with_stride (&distances[1],stride,dim);
  double min_dist = gsl_vector_min(&minimum_dists.vector);  

  min_dist = sqrt(min_dist);

  double eps_start_save = eps;
  double eps_stop_save = 1./min_dist;
  double step_save = (eps_stop_save-eps_start_save)/steps;

  //Evaluating test functions
  vector<double> input_function;
  vector<double> target_function, target_coords;

  for(int i = 0; i < dim; i++)
    {
      vector<double> current_coords = (*mesh_free_in)(i);
      if(test_function_switch == "")
	{
	  input_function.push_back((*test_function_in)(current_coords));
	}
      else
	{
	  input_function.push_back(test_function_in->D(current_coords,test_function_switch));
	}
    }

  for(int i = 0; i < dim_target; i++)
    {
      vector<double> current_coords = (*mesh_free_target)(i);
      for(int j = 0; j < current_coords.size(); j++)
	{
	  target_coords.push_back(current_coords[j]);
	}
      if(test_function_switch == "")
	{
	  target_function.push_back((*test_function_in)(current_coords));
	}
      else
	{
	  target_function.push_back(test_function_in->D(current_coords,test_function_switch));
	}
    }


  double eps_start = eps_start_save;
  double eps_stop = eps_stop_save;
  double step = step_save;
  double main_index = 0;
  if(verbose)
    {
      out_log.open(verbose_mode.c_str());
      out_log <<"#epsilon condition avg_error max_error" <<endl;
    }

  double final_eps;

  while(main_index <= refinements)
    {
      vector<double> epsilons, avg_errors, max_errors, conditions;
      for(double eps = eps_start; eps <= eps_stop; eps += step)
	{
	  vector<double> difference_vector;
	  double condition;
	  rbf_in->set_epsilon(eps);
	  vector<double> current_interpol_out;
	  condition = mesh_free_in->interpolate(&target_coords,&input_function, &current_interpol_out, rbf_in, knn);
	  for(int i = 0; i < dim_target; i++)
	    {
	      difference_vector.push_back(abs((current_interpol_out[i]-target_function[i])/target_function[i]));
	    } //End of all nodes loop
	  
	  double avg_error = gsl_stats_mean(&difference_vector[0],1,difference_vector.size());
	  avg_errors.push_back(avg_error);
	  vector<double>::iterator max_iterator;
	  max_iterator = max_element(difference_vector.begin(),difference_vector.end());
	  max_errors.push_back(*max_iterator);
	  epsilons.push_back(eps);
	  conditions.push_back(condition);
	  if(verbose)
	    {
	      out_log <<eps <<"\t" <<condition <<"\t" <<avg_error <<"\t" <<*max_iterator <<endl;
	    }
	} //End of eps loop

      vector<double>::iterator best_eps_error;
      best_eps_error = min_element(avg_errors.begin(),avg_errors.end());
      int best_eps_index = distance(avg_errors.begin(),best_eps_error);
      if(best_eps_index > 0)
	{
	  eps_start = epsilons[best_eps_index-1];
	}
      else
	{
	  eps_stop = epsilons[0];
	}
      if(best_eps_index < epsilons.size()-1)
	{
	  eps_stop = epsilons[best_eps_index+1];
	}
      else
	{
	  eps_stop = epsilons[epsilons.size()-1];
	}
      step = (eps_stop-eps_start)/steps; 
      
      final_eps = epsilons[best_eps_index];    
      main_index++;
    } //End of all refinements loop 
  if(verbose)
    {
      out_log.close();
    }
  
  rbf_in->set_epsilon(eps_save);
  return final_eps;
  
}

vector<double> optimise_adaptive_grid_interpolation(mesh_free *mesh_free_in, mesh_free *mesh_free_target, radial_basis_function_shape *rbf_in,  test_function *test_function_in, int knn, string test_function_switch , int refinements, int steps, double eps, string verbose_mode)
{

  bool verbose = false;

  if(verbose_mode != "")
    {
      verbose = true;
    }

  vector<double> adaptive_shape_parameter;

  double eps_save = rbf_in->show_epsilon();

  if(mesh_free_in->return_grid_size(1) != (*test_function_in)())
    {
      throw invalid_argument("SHAPE_OPT: Test function and mesh free not matchgin in dim.");
    }

  //Evaluating test function at all support nodes
  vector<double> support_coordinates;
  vector<double> support_function;
  for(int i = 0; i < mesh_free_in->return_grid_size(); i++)
    {
      vector<double> current_coordinates = (*mesh_free_in)(i);
      if(test_function_switch == "")
	{
	  support_function.push_back((*test_function_in)(current_coordinates));
	}
      else
	{
	  support_function.push_back(test_function_in->D(current_coordinates,test_function_switch));
	}
      for(int j = 0; j < current_coordinates.size(); j++)
	{
	  support_coordinates.push_back(current_coordinates[j]);
	}
    }

  //Building the tree for the grid once to query some distances later.
  vector<int> interpolant_tree;
  vector<double> interpolant_distances;
  interpolant_tree.resize(knn);
  interpolant_distances.resize(knn);
  flann::Matrix<double> flann_dataset(&support_coordinates[0],mesh_free_in->return_grid_size(),mesh_free_in->return_grid_size(1));
  flann::Index<flann::L2<double> > index(flann_dataset, flann::KDTreeIndexParams(4));
  index.buildIndex();

  ofstream out_log;

  if(verbose)
    {
      out_log.open(verbose_mode.c_str());
      out_log <<"#pixel epsilon condition error" <<endl;
    }


  //Main loop over all target nodes

  vector<double> current_target_coordinates;
  for(int global = 0; global < mesh_free_target->return_grid_size(); global++)
    {
      double current_target_value;
      current_target_coordinates = (*mesh_free_target)(global);
      if(test_function_switch == "")
	{
	  current_target_value = (* test_function_in)(current_target_coordinates);
	}
      else
	{
	  current_target_value = test_function_in->D(current_target_coordinates,test_function_switch);
	}

      //Finding maximum shape parameter. 
      vector<int> dummy_tree;
      dummy_tree.resize(1);
      vector<double> current_min_dist;
      current_min_dist.resize(1);
      flann::Matrix<double> flann_dataset_interpolant(&current_target_coordinates[0],1,mesh_free_target->return_grid_size(1));
      flann::Matrix<int> flann_tree(&dummy_tree[0],1,1);
      flann::Matrix<double> flann_distances(&current_min_dist[0],1,1);
      index.knnSearch(flann_dataset_interpolant, flann_tree, flann_distances,1, flann::SearchParams(128));

      //Doing the shape parameter loop. 
      double eps_start = eps;
      double eps_stop = 1./(sqrt(current_min_dist[0]+eps));
      double step = (eps_stop - eps_start) / steps;

      double main_index = 0;
      double final_eps;
      while(main_index <= refinements)
	{
	  vector<double> epsilons, errors, conditions;
	  for(double current_eps = eps_start; current_eps <= eps_stop; current_eps += step)
	    {
	      double condition,error;
	      vector<double> current_interpol_out;
	      rbf_in->set_epsilon(current_eps);
	      
	      condition = mesh_free_in->interpolate(&current_target_coordinates,&support_function, &current_interpol_out, rbf_in, knn);		
	      error = abs((current_interpol_out[0]-current_target_value)/current_target_value);
	      conditions.push_back(condition);
	      errors.push_back(error);
	      epsilons.push_back(current_eps);
	      if(verbose)
		{
		  out_log <<global <<"\t" <<current_eps <<"\t" <<condition <<"\t" <<error <<endl;
		}
	    } // end of eps loop


	  vector<double>::iterator best_eps_error;
	  best_eps_error = min_element(errors.begin(),errors.end());
	  int best_eps_index = distance(errors.begin(),best_eps_error);
	  if(best_eps_index > 0)
	    {
	      eps_start = epsilons[best_eps_index-1];
	    }
	  else
	    {
	      eps_stop = epsilons[0];
	    }
	  if(best_eps_index < epsilons.size()-1)
	    {
	      eps_stop = epsilons[best_eps_index+1];
	    }
	  else
	    {
	      eps_stop = epsilons[epsilons.size()-1];
	    }
	  step = (eps_stop-eps_start)/steps; 
	  
	  final_eps = epsilons[best_eps_index];    
	  main_index++;
	      
	} //end of refinement loop
      
      adaptive_shape_parameter.push_back(final_eps);

    } //End of all nodes loop

  if(verbose)
    {
      out_log.close();
    }

  rbf_in->set_epsilon(eps_save);

  return adaptive_shape_parameter;

}
