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

using namespace std;

/*
  This function creates a row vector for a polynomial term, given a set of 
  coordinates, also determining the dimensionality of the problem and the 
  desired rank of the polynomial support. It returns the full row vector.
*/


vector<double> row_vector_from_polynomial(vector<double> coordinates, uint pdeg = 2);


/*
  This is a special case for initial test implementations which implements polynomial support for a two-dimensional domain.
*/

vector<double> row_vector_from_polynomial_2D(vector<double> coordinates, uint pdeg = 0);

/*
  This helper function delivers the polynomial support right hand side column
  vector depending on the derivative in question and the polynomial order. 
*/

vector<double> polynomial_support_rhs_column_vector_2D(string derivative = "x", uint pdeg = 0);



#endif    /*POLYNOMIAL_TERMS_H*/
