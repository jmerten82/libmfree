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

vector<double> optimise_grid_differentiation(mesh_free_differentiate *mesh_free_in,radial_basis_function_shape *rbf_in,  test_function *test_function_in, vector<string> differentials = vector<string>(), int refinements, int steps, string verbose_mode)
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
 
  //Finding maximum grid distance with the help of the convex hull
  vector<Point_2> cgal_mesh;
  vector<Point_2> cgal_chull;

  for(int i = 0; i < dim; i++)
    {
      cgal_mesh.push_back(Point_2((*mesh_free_in)(i,0),(*mesh_free_in)(i,1)));
    }
  CGAL::convex_hull_2(cgal_mesh.begin(), cgal_mesh.end(), std::back_inserter(cgal_chull) );
  int dim2 = cgal_chull.size();
  double max_dist = 0.;
  for(int i = 0; i < dim2-1; i++)
    {
      for(int j = i; j < dim2; j++)
	{
	  double current_dist = pow(cgal_chull[i][0]-cgal_chull[j][0],2) + pow(cgal_chull[i][1]-cgal_chull[j][1],2);
	  if(current_dist > max_dist)
	    {
	      max_dist = current_dist;
	    }
	}
    } 

  min_dist = sqrt(min_dist);
  max_dist = sqrt(max_dist);

  double eps_start_save = 1./max_dist;
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
	  out_log <<"#Minimum distance on grid: " <<min_dist <<endl;
	  out_log <<"#Maximum distance on grid: " <<max_dist <<endl;
	  out_log <<"#epsilon condition avg_error max_error" <<endl;
	}

      while(main_index <= refinements)
	{
	  cout <<main_index <<endl;
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

double optimise_grid_interpolation(mesh_free *mesh_free_in, mesh_free *mesh_free_target, radial_basis_function_shape *rbf_in,  test_function *test_function_in, int knn, string test_function_switch, int refinements, int steps, string verbose_mode)
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
  int dim_target = mesh_free_target->return_grid_size();


  //Getting the distances from the mesh. This makes sure that tree is created. 

  vector<double> distances = mesh_free_in->provide_distances();
  int stride = distances.size() / dim;

  //Finding minimum nn distance with gsl vector views
  gsl_vector_view minimum_dists = gsl_vector_view_array_with_stride (&distances[1],stride,dim);
  double min_dist = gsl_vector_min(&minimum_dists.vector);  
 
  //Finding maximum grid distance with the help of the convex hull
  vector<Point_2> cgal_mesh;
  vector<Point_2> cgal_chull;

  for(int i = 0; i < dim; i++)
    {
      cgal_mesh.push_back(Point_2((*mesh_free_in)(i,0),(*mesh_free_in)(i,1)));
    }
  CGAL::convex_hull_2(cgal_mesh.begin(), cgal_mesh.end(), std::back_inserter(cgal_chull) );
  int dim2 = cgal_chull.size();
  double max_dist = 0.;
  for(int i = 0; i < dim2-1; i++)
    {
      for(int j = i; j < dim2; j++)
	{
	  double current_dist = pow(cgal_chull[i][0]-cgal_chull[j][0],2) + pow(cgal_chull[i][1]-cgal_chull[j][1],2);
	  if(current_dist > max_dist)
	    {
	      max_dist = current_dist;
	    }
	}
    } 

  min_dist = sqrt(min_dist);
  max_dist = sqrt(max_dist);

  double eps_start_save = 1./max_dist;
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
      string out_file = verbose_mode;
      out_log.open(out_file.c_str());
      out_log <<"#Minimum distance on grid: " <<min_dist <<endl;
      out_log <<"#Maximum distance on grid: " <<max_dist <<endl;
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
