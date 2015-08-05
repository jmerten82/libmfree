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
