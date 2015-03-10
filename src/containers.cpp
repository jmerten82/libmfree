/*** containers.cpp
     This collection of classes provides typical 
     data structures for the use in SaWLens.

Julian Merten
JPL/Caltech
August 2014
jmerten@caltech.edu
***/

#include <mfree/containers.h>

multiple_image_system::multiple_image_system(double redshift_in, double sigma_in)
{

  redshift = redshift_in;
  sigma = sigma_in;
  num_sys = 0;

}

multiple_image_system::~multiple_image_system()
{

  num_sys = 0;
  image_coordinates.clear();
  image_parities.clear();
  grid_indices.clear();

}

void multiple_image_system::add(vector<double> coordinates_in)
{

  int to_add = coordinates_in.size() / 2;
  if(num_sys + to_add > 5)
    {
      throw invalid_argument("MSYSTEM: You want to add too many systems");
    }

  for(int i = 0; i < to_add; i++)
    {
      image_coordinates.push_back(coordinates_in[i*2]);
      image_coordinates.push_back(coordinates_in[i*2+1]);
      num_sys++;
    }
}

void multiple_image_system::add(vector<double> coordinates_in, vector<bool> parities_in)
{

  int to_add = coordinates_in.size() / 2;
  if(num_sys + to_add > 5)
    {
      throw invalid_argument("MSYSTEM: You want to add too many systems");
    }

  if(parities_in.size() != to_add)
    {
      throw invalid_argument("MSYSTEM: Parities and coordinates lengths do not match.");
    }

  for(int i = 0; i < to_add; i++)
    {
      image_coordinates.push_back(coordinates_in[i*2]);
      image_coordinates.push_back(coordinates_in[i*2+1]);
      image_parities.push_back(parities_in[i]);
      num_sys++;
    }
}

void multiple_image_system::add(vector<double> coordinates_in, vector<int> indices)
{

  int to_add = coordinates_in.size() / 2;
  if(num_sys + to_add > 5)
    {
      throw invalid_argument("MSYSTEM: You want to add too many systems");
    }

  if(indices.size() != to_add)
    {
      throw invalid_argument("MSYSTEM: Index and coordinates lengths do not match.");
    }

  for(int i = 0; i < to_add; i++)
    {
      image_coordinates.push_back(coordinates_in[i*2]);
      image_coordinates.push_back(coordinates_in[i*2+1]);
      grid_indices.push_back(indices[i]);
      num_sys++;
    }

}

void multiple_image_system::add(vector<double> coordinates_in, vector<int> indices, vector<bool> parities_in)
{

  int to_add = coordinates_in.size() / 2;
  if(num_sys + to_add > 5)
    {
      throw invalid_argument("MSYSTEM: You want to add too many systems");
    }

  if(parities_in.size() != to_add || indices.size() != to_add)
    {
      throw invalid_argument("MSYSTEM: Parities/index and coordinates lengths do not match.");
    }

  for(int i = 0; i < to_add; i++)
    {
      image_coordinates.push_back(coordinates_in[i*2]);
      image_coordinates.push_back(coordinates_in[i*2+1]);
      image_parities.push_back(parities_in[i]);
      grid_indices.push_back(indices[i]);
      num_sys++;
    }
}

int multiple_image_system::operator() ()
{

  return num_sys;

}

int multiple_image_system::operator() (int i)
{

  if(i < 0 || i > num_sys)
    {
      throw invalid_argument("MSYSTEM: Invalid image selection.");
    }
  if(grid_indices.size() == num_sys)
    {
      return grid_indices[i];
    }
  else
    {
      return -1;
    }
}

double multiple_image_system::operator() (int i, int j)
{

  if(i < 0 || i > num_sys)
    {
      throw invalid_argument("MSYSTEM: Invalid image selection.");
    }

  if(j == 1)
    {
      return image_coordinates[i*2+1];
    }
  else
    {
      return image_coordinates[i*2];
    }
}

double multiple_image_system::get_redshift()
{
  return redshift;
}

double multiple_image_system::get_sigma()
{
  return sigma;
}

void multiple_image_system::set_parity(int i, bool parity)
{

 if(i < 0 || i > num_sys)
    {
      throw invalid_argument("MSYSTEM: Invalid image selection.");
    }

 image_parities[i] = parity;
}

bool multiple_image_system::get_parity(int i)
{

 if(i < 0 || i > num_sys)
    {
      throw invalid_argument("MSYSTEM: Invalid image selection.");
    }

 return image_parities[i];
}

void multiple_image_system::assign_indices(vector<int> index_in)
{

  if(index_in.size() < num_sys)
    {
      throw invalid_argument("MSYSTEM: Size of index vector is too short.");
    }

  if(grid_indices.size() != num_sys)
    {
      grid_indices.resize(num_sys);
    }

  for(int i = 0; i < num_sys; i++)
    {
      grid_indices[i] = index_in[i];
    }
}


galaxy::galaxy(double x_in, double y_in)
{

  x = x_in;
  y = y_in;
  z = 0.0;
  g1 = 0.0;
  g2 = 0.0; 
  F1 = 0.0;
  F2 = 0.0;
  G1 = 0.0;
  G2 = 0.0; 
  index = 0;
}

galaxy::galaxy(double x_in, double y_in, double g1_in, double g2_in)
{

  x = x_in;
  y = y_in;
  z = 0.0;
  g1 = g1_in;
  g2 = g2_in; 
  F1 = 0.0;
  F2 = 0.0;
  G1 = 0.0;
  G2 = 0.0; 
  index = 0;
}

galaxy::~galaxy()
{

}

void galaxy::set(string selection, double value)
{

  if(selection == "x")
    {
      x = value;
    }
  else if(selection == "y")
    {
      y = value;
    }
  else if(selection == "z")
    {
      z = value;
    }
  else if(selection == "g1")
    {
      g1 = value;
    }
  else if(selection == "g2")
    {
      g2 = value;
    }
  else if(selection == "F1")
    {
      F1 = value;
    }
  else if(selection == "F2")
    {
      F2 = value;
    }
  else if(selection == "G1")
    {
      G1 = value;
    }
  else if(selection == "G2")
    {
      G2 = value;
    }
  else if(selection == "index")
    {
      index = (int) value;
    }
  else
    {
      throw invalid_argument("GALAXY: Invalid selection for set.");
    }
}

double galaxy::get(string selection)
{

  if(selection == "x")
    {
      return x;
    }
  else if(selection == "y")
    {
      return y;
    }
  else if(selection == "z")
    {
      return z;
    }
  else if(selection == "g1")
    {
      return g1;
    }
  else if(selection == "g2")
    {
      return g2;
    }
  else if(selection == "F1")
    {
      return F1;
    }
  else if(selection == "F2")
    {
      return F2;
    }
  else if(selection == "G1")
    {
      return G1;
    }
  else if(selection == "G2")
    {
      return G2;
    }
  else
    {
      throw invalid_argument("GALAXY: Invalid selection for get.");
    }
}

int galaxy::operator()()
{
  return index;
}

ccurve_estimator::ccurve_estimator(double x_in, double y_in, double z_in)
{

  x = x_in;
  y = y_in;
  z = z_in;
  index = 0;

}

void ccurve_estimator::set(string selection, double value)
{
  if(selection == "x")
    {
      x = value;
    }
  else if(selection == "y")
    {
      y = value;
    }
  else if(selection == "z")
    {
      z = value;
    }
  else if(selection == "index")
    {
      index = (int) value;
    }
  else
    {
      throw invalid_argument("CCURVE_EST: Invalid selection for set.");
    }
}

double ccurve_estimator::get(string selection)
{
  if(selection == "x")
    {
      return x;
    }
  else if(selection == "y")
    {
      return y;
    }
  else if(selection == "z")
    {
      return z;
    }
  else
    {
      throw invalid_argument("CCURVE_EST: Invalid selection for get.");
    }
}

int ccurve_estimator::operator() ()
{
  return index;
}



  

 


