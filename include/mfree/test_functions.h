/*** test_functions.h
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

#ifndef    TEST_FUNCTIONS_H
#define    TEST_FUNCTIONS_H

#include <vector>
#include <stdexcept>

using namespace std;

class test_function
{

 protected:

  /*
    The dimensionality of the test function. 
  */

  int dim;

  /*
    A vector holding a possible coorindate offset in each dimensions. 
    By default this vector should be a series of 0s with length dim.
  */

  vector<double> coordinate_offset;

 public:

  /*
    The bracket operator evaluates the test function at the given coordinate.
    Since we don't have a standard implementation this virtual operator
    is abstract.
  */

  double virtual operator() (vector<double> coordinates) = 0; 

  /*
    This virtual abstract function evaluates any derivative of the function
    at a given coordinate. The availability of specific derivatives will
    depend on the concrete implementation.
  */

  double virtual D(vector<double> coordinates,string selection = "x") = 0;

  /*
    The overloaded empty bracket operator returns the dimensionality of the
    test function. 
  */

  int operator() ();

  /*
    This routine lets you reset the coordinate offset of the test function.
    If no argument is given, the coorindates will be set to the point 0.
  */

  void reset_origin(vector<double> coordinates = vector<double>());

};


#endif    /*TEST_FUNCTIONS_H*/
