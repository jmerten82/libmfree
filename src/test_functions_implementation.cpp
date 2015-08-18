/*** test_functions_implementation.cpp
This is a collection of concrete implementations of test
functions for the optimisation of mesh-free domains. The general
base class for this is found in mfree/test_functions.h.
 
Julian Merten
Universiy of Oxford
Aug 2015
julian.merten@physics.ox.ac.uk
http://www.julianmerten.net
***/

#include <mfree/test_functions_implementation.h>

using namespace std;

bengts_function::bengts_function(vector<double> coordinates)
{

  dim = 2;

  if(coordinates.size() == 0)
    {
      coordinate_offset.push_back(0.);
      coordinate_offset.push_back(0.);
    }
  else if(coordinates.size() < dim)
    {
      throw invalid_argument("B_FUNCTION: Invalid offset vector.");
    }
  else
    {
      coordinate_offset.push_back(coordinates[0]);
      coordinate_offset.push_back(coordinates[1]);
    }

}

double bengts_function::operator() (vector<double> coordinates)
{

  if(coordinates.size() < dim)
    {
      throw invalid_argument("B_FUNCTION: Invalid coordinate query.");
    }

  double x = coordinates[0] - coordinate_offset[0];
  double y = coordinates[1] - coordinate_offset[1];

  return 1. + sin(4. * x) + cos(3. * x) + sin(2. * y);
}

double bengts_function::D(vector<double> coordinates, string selection)
{

  if(coordinates.size() < dim)
    {
      throw invalid_argument("B_FUNCTION: Invalid coordinate query.");
    }

  double x = coordinates[0] - coordinate_offset[0];
  double y = coordinates[1] - coordinate_offset[1];

  double value;

  if(selection == "x")
    {
      value = 4. * cos(4. * x) - 3. * sin(3. * x);
    }
  else if(selection == "y")
    {
      value = 2. * cos(2. * y);
    }
  else if(selection == "xx")
    {
      value = -16. * sin(4. * x) - 9. * cos(3. *x);
    }
  else if(selection == "yy")
    {
      value = -4. * sin(2. * y);
    }
  else if(selection == "xy")
    {
      value = 0.;
    }
  else
    {
      throw invalid_argument("B_FUNCTION: Invalid derivative selection.");
    }

  return value;

}
