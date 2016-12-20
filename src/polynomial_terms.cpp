/*** polynomial_terms.cpp
This is a set of helper functions that will make our life easier when it 
comes to dealing with polynomial support terms in the RBF formalism.

Julian Merten
Universiy of Oxford
Dec 2016
julian.merten@physics.ox.ac.uk
http://www.julianmerten.net
***/

#include <mfree/polynomial_terms.h>

vector<double> row_vector_from_polynomial(vector<double> coordinates, uint pdeg = 2)
{

  uint D = coordinates.size();
  vector<double> out;

  //special case, constant term
  out.push_back(1.);



  //main loop over all wanted p degrees
  for(uint current_p = 1; current_p <= pdeg; current_p++)
    {
      vector<int> current_combo;
      //looping through the available dimension slots
      for(uint d = 0; d < D; d++)
	{
	  for

	}//end dimension loop





    }// end p degree loop


  return out;


}
