/*** new_rbf_concept_test.cpp
     This is an implementation test for 
     the new, slimmer RBF implementation.
     Also, the PHS and their derivatives are 
     checked with these routines. 

     Julian Merten
     University of Oxford
     Dec 28 2016
     http://www.julianmerten.net
***/

#include <iostream>
#include <cmath>
#include <vector>
#include <sstream>
#include <mfree/radial_basis_function.h>
#include <mfree/rbf_implementation.h>
#include <mfree/rbf_polyharmonic_splines.h>
#include <mfree/rwfits.h>

using namespace std;

int main()
{

  vector<radial_basis_function*> rbfs;

  phs_first_order one;
  phs_second_order two;
  phs_third_order three;
  phs_fourth_order four;
  phs_fifth_order five;
  phs_sixth_order six;
  phs_seventh_order seven;
  phs_eighth_order eight;
  phs_nineth_order nine;
  phs_tenth_order ten;

  rbfs.push_back(&one);
  rbfs.push_back(&two);
  rbfs.push_back(&three);
  rbfs.push_back(&four);
  rbfs.push_back(&five);
  rbfs.push_back(&six);
  rbfs.push_back(&seven);
  rbfs.push_back(&eight);
  rbfs.push_back(&nine);
  rbfs.push_back(&ten);

  double step = 2./256.;
  string root = "./data/rbf_order";

  for(uint r = 0; r < rbfs.size(); r++)
    {
      vector<double> grid;
      double x,y;
      ostringstream order;
      order <<r;
      string name = root+order.str()+".fits";
      for(uint i = 0; i < 256; i++)
	{
	  y = -1. + i*step;
	  for(uint j = 0; j < 256; j++)
	    {
	      x = -1. + j*step;
	      grid.push_back(rbfs[r]->Dxx(x,y));
	    }
	}
      write_img_to_fits(name,&grid);
    }



  return 0;
}


