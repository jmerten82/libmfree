/*** test_functions.cpp
This abstract class provides a test functions to 
optimize the parameters of a RBF enabled mesh-free domain. 
The classes here are base classes. Explicit test functions are 
defined elsewhere but inherit the functionality of the base classes. 
 
Julian Merten
Universiy of Oxford
Aug 2015
julian.merten@physics.ox.ac.uk
http://www.julianmerten.net
***/

#include <mfree/test_functions.h>

using namespace std;

int test_function::operator() ()
{
  return dim;
}

void test_function::reset_origin(vector<double> coordinates)
{

  if(coordinates.size() == 0)
    {
      coordinate_offset.clear();
      for(int i = 0; i < dim; i++)
	{
	  coordinate_offset.push_back(0.);
	}
    }

  else if(coordinates.size() < dim)
    {
      throw invalid_argument("T_FUNCTION: New origin vector invalid.");
    }

  else
    {
      coordinate_offset.clear();
      for(int i = 0; i < dim; i++)
	{
	  coordinate_offset.push_back(coordinates[i]);
	}
    }
}
