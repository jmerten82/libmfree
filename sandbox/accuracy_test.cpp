/*** accuracy_test.cpp
     This sandbox routine tests in detail the accuracy of RBF FD.
     
     Julian Merten
     University of Oxford
     Jul 2017
     http://www.julianmerten.net
***/

#include <iostream>
#include <vector>
#include <mfree/radial_basis_function.h>
#include <mfree/rbf_implementation.h>
#include <mfree/rbf_polyharmonic_splines.h>
#include <mfree/test_functions.h>
#include <mfree/test_functions_implementation.h>
#include <mfree/mesh_free.h>
#include <mfree/mesh_free_differentiate.h>
#include <mfree/rwfits.h>
#include <mfree/grid_utils.h>

using namespace std;

int main()
{

  //Define RBF, polynomial degree, number of grid points and test function to be tested

  //Create random distribution of points

  //Evaluate test function and its derivatives on this grid

  //Find optimal shape parameters for the grid

  //Evaluate the derivate for a number of degrees in polynomial support



}
