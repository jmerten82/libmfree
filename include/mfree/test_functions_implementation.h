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
    Laplace
    Neg_Laplace
  */

  double D(vector<double> coordinates, string selection = "x");

};

/**
   This class implements the 2D version of the NFW lensing potential. 
   It needs one parameter, which is the scale convergence. We follow
   here the defintion of the lensing notes by Massimo Meneghetti:
   \psi(r) = 4*\kappa_s*g(r), where
   g(r) = 0.5*ln^2(r/2) + h(r), where 
   h(r) = 2*arctan^2(sqrt(r-1/r+1)) for (x > 1)
   h(r) = -2*arctanh^2(sqrt(1-r/1+r)) for (x < 1)
   h(r) = 0 for(r = 1)
**/

class nfw_lensing_potential : public test_function
{

 protected:

  /*
    The one free paramater of the radially symmetric function, the
    scale convergence. 
  */
  double scale_convergence;

 public:

  /*
    Standard constructor. Needs the scale convergence and the coordinate
    offset. 
  */

  nfw_lensing_potential(double k_s_in = 1.0, vector<double> coordinates = vector<double>());

  /*
    This returns the potential at a specific coordinate.
  */

  double operator() (vector<double> coordinates);

  /*
    This returns the potential at a specific radius.
  */

  double operator() (double radius);

  /*
    The current implementation of the derivatives. Currently 
    available are:
    x
    y
    Half_Laplace (this one actually returns 1/2 the Laplace)
  */

  double D(vector<double> coordinates, string selection = "x");
}; 

/**

   This implements a 16 parameter function to describe the multiplicative
   bias in shear calibration as a function of signal-to-noise and
   Resoltuion (psf_size/size). Property of Henk Hoekstra. 
**/


class henks_function : public test_function
{

 protected:

  vector<double> params;

 public:

  /*
    The standars constructor must be called with an input parameter vector
    of at least 16 parameters. 
  */

  henks_function(vector<double> *input_parameters, vector<double> coordinates = vector<double>());

  /*
    Queries the function for a specific SNR-R coordinate pair.
  */

  double operator() (vector<double> coordinates);

  /*
    The current options for the derivative operator are either SNR or R. 
  */

  double D(vector<double> coordinates, string selection = "SNR");

};



#endif    /*TEST_FUNCTIONS_I_H*/
