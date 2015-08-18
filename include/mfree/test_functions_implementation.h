/*** test_functions_implementation.h
This is a collection of concrete implementations of test
functions for the optimisation of mesh-free domains. The general
base class for this is found in mfree/test_functions.h.
 
Julian Merten
Universiy of Oxford
Aug 2015
julian.merten@physics.ox.ac.uk
http://www.julianmerten.net
***/

#ifndef    TEST_FUNCTIONS_I_H
#define    TEST_FUNCTIONS_I_H

#include <mfree/test_functions.h>
#include <cmath>

/**
   This class implements the typical 2D test function as it is implemented
   in most Fornberg & Flyer et al. papers.
   f(x,y) = 1 + sin(4x) +cos(3x) + sin(2y)
**/


class bengts_function : public test_function
{

 public:

  /*
    The standard constructor which also allows to define an initial offset. 
  */

  bengts_function(vector<double> coordinates = vector<double>());

  /*
    The implementation of the functional form.
  */

  double operator() (vector<double> coordinates);

  /*
    The implementation of the derivatives. Implemented currently
    for this function are the following derivative selections
    x
    y
    xx
    yy
    xy
  */

  double D(vector<double> coordinates, string selection = "x");

};


#endif    /*TEST_FUNCTIONS_I_H*/
