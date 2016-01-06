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

nfw_lensing_potential::nfw_lensing_potential(double k_s_in, vector<double> coordinates)
{

  dim = 2;
  scale_convergence = k_s_in;
  coordinate_offset.push_back(0.);
  coordinate_offset.push_back(0.);

  if(coordinates.size() >= dim)
    {
      coordinate_offset[0] = coordinates[0];
      coordinate_offset[1] = coordinates[1];
    }
}

double nfw_lensing_potential::operator() (vector<double> coordinates)
{

  double radius;
  if(coordinates.size() >= dim)
    {
      radius = sqrt(pow(coordinates[0]-coordinate_offset[0],2)+pow(coordinates[1]-coordinate_offset[1],2));
    }

  else
    {
      throw invalid_argument("NFW_LENS_POTENTIAL: To few coordinate components provided.");
    }

  return (*this)(radius);
} 

double nfw_lensing_potential::operator()(double radius)
{

  if(radius < 1.e-3)
    {
      radius = 1.e-3;
    }
  double value = 0.5*pow(log(radius/2.),2);

  if(radius > 1.)
    {
      value +=  2.*pow(atan(sqrt((radius-1.)/(radius+1.))),2);
    }
  else if(radius < 1.)
    {
      value +=  -2.*pow(atanh(sqrt((1.-radius)/(radius+1.))),2);
    }

  return value*4.*scale_convergence;
}

double nfw_lensing_potential::D(vector<double> coordinates, string selection)
{

  if(coordinates.size() < dim)
    {
      throw invalid_argument("NFW_LENS_POTENTIAL_D: Too few coordinate components provided.");
    }

  double radius = sqrt(pow(coordinates[0]-coordinate_offset[0],2)+pow(coordinates[1]-coordinate_offset[1],2));

  double value = 4.*scale_convergence;
  double term = 1.;

  if(radius < 1.e-3)
    {
      radius = 1.e-3;
    }

  if(selection == "x" || selection == "y")
    {
      term = log(radius/2.);
      if(radius > 1.)
	{
	  term += 2./sqrt(radius*radius-1.)*atan(sqrt((radius-1.)/(radius+1.)));
	}
      
      else if(radius < 1.)
	{
	  term += 2./sqrt(1.-radius*radius)*atanh(sqrt((1.-radius)/(radius+1.)));
	}
      else
	{
	  term += 1.;
	}
      value *=term/pow(radius,2);
      
      if(selection == "x")
	{
	  if(abs(coordinates[0]-coordinate_offset[0]) < 1e-3 && coordinates[0]-coordinate_offset[0] < 0.)
	    {
	      value *= -1e-3;
	    }
	  else if(abs(coordinates[0]-coordinate_offset[0]) < 1e-3 && coordinates[0]-coordinate_offset[0] > 0.)
	    {
	      value *= 1e-3;
	    } 
	  else
	    {
	      value *= (coordinates[0]-coordinate_offset[0]);
	    }
	}
      else if(selection == "y")
	{
	  if(abs(coordinates[1]-coordinate_offset[1]) < 1e-3 && coordinates[1]-coordinate_offset[1] < 0.)
	    {
	      value *= -1e-3;
	    }
	  else if(abs(coordinates[1]-coordinate_offset[1]) < 1e-3 && coordinates[1]-coordinate_offset[1] > 0.)
	    {
	      value *= 1e-3;
	    } 
	  else
	    {
	      value *= (coordinates[1]-coordinate_offset[1]);
	    }
	}
    }

  else if(selection == "Laplace")
    {
      value *= 0.5;
      if(radius < 1.)
	{
	  value *= 1./(radius*radius-1.)*(1.-2./sqrt(1.-radius*radius)*atanh(sqrt((1.-radius)/(1.+radius))));
	}
      else if(radius > 1.)
	{
	  value *= 1./(radius*radius-1.)*(1.-2./sqrt(radius*radius-1.)*atan(sqrt((radius-1.)/(1.+radius))));
	}
      else
	{
	  value *= 1./3.; 
	}
    }

  else
    {
      throw invalid_argument("Invalid selection for test function D");
    }

  return value;
}



