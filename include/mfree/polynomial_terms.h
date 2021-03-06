/*** polynomial_terms.h
This is a set of helper functions that will make our life easier when it 
comes to dealing with polynomial support terms in the RBF formalism.

Julian Merten
Universiy of Oxford
Dec 2016
julian.merten@physics.ox.ac.uk
http://www.julianmerten.net
***/

#ifndef    POLYNOMIAL_TERMS_H
#define    POLYNOMIAL_TERMS_H

#include <vector>
#include <stdexcept>
#include <iostream>

using namespace std;

/*
  This is a special case for initial test implementations which implements polynomial support for a one-dimensional domain.
*/

vector<double> row_vector_from_polynomial_1D(double x, unsigned pdeg);

/*
  This is a special case for initial test implementations which implements polynomial support for a two-dimensional domain.
*/

vector<double> row_vector_from_polynomial_2D(double x, double y, unsigned pdeg);

/*
  This is a special case for initial test implementations which implements polynomial support for a three-dimensional domain.
*/

vector<double> row_vector_from_polynomial_3D(double x, double y, double z,  unsigned pdeg);

/*
  This 1D helper function delivers the polynomial support right hand side column
  vector depending on the derivative in question and the polynomial order. 
*/

vector<double> polynomial_support_rhs_column_vector_1D(string derivative = "x", unsigned pdeg = 0);

/*
  This 2D helper function delivers the polynomial support right hand side column
  vector depending on the derivative in question and the polynomial order. 
*/

vector<double> polynomial_support_rhs_column_vector_2D(string derivative = "x", unsigned pdeg = 0);

/*
  This 3D helper function delivers the polynomial support right hand side column
  vector depending on the derivative in question and the polynomial order. 
*/

vector<double> polynomial_support_rhs_column_vector_3D(string derivative = "x", unsigned pdeg = 0);

/*
  This is specifically needed for the calculation of an interpolant.
  Given the libmfree scheme and ordering of polynomials of a given order
  this function evaluates the polynomials at a given point and weighs them
  with the factor and returns the sum of all contributions. This
  routine is largely based on the 'row_vector_from_polynomial' routines.
  The optional stride relates to an offset in the gamma vector, which eventually
  desribes an ensemble of weights for many interpolant nodes. 
*/

double polynomial_support_evaluate_1D(double x_in, vector<double> *gammas, unsigned int stride = 0, unsigned int pdeg = 0);

/*
  Same as above but in 2D.
*/

double polynomial_support_evaluate_2D(double x_in, double y_in, vector<double> *gammas, unsigned int stride = 0, unsigned int pdeg = 0);

/*
  Same as above but in 3D.
*/

double polynomial_support_evaluate_3D(double x_in, double y_in, double z_in, vector<double> *gammas, unsigned int stride = 0, unsigned int pdeg = 0);



#endif    /*POLYNOMIAL_TERMS_H*/
