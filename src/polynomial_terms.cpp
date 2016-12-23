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
//Work on this function is currently suspended until we implemented 2D tests.

vector<double> row_vector_from_polynomial(vector<double> coordinates, uint pdeg)
{

  uint D = coordinates.size();
  vector<double> out;

  //special case, constant term
  out.push_back(1.);



  //main loop over all wanted p degrees
  for(uint current_p = 1; current_p <= pdeg; current_p++)
    {
      //looping through the available dimension slots
      vector<uint> = current_combo;
      vector<uint> = current_combo;
      current_max = current_p;
      for(uint current = 0; uint < pdeg; current++)
	{
	  for(uint d = 0; d < D; d++)
	    {


	    }
	}

	}//end dimension loop





    }// end p degree loop


  return out;


}


vector<double> row_vector_from_polynomial_2D(vector<double> coordinates, uint pdeg)
{

  vector<double> row_vector;
  vector<double> x_values, y_values;

  x_values.push_back(1.);
  y_values.push_back(1.);

  for(uint i = 1; i <= pdeg; i++)
    {
      x_values.push_back(x_values[i]*coordinates[0]);
      y_values.push_back(y_values[i]*coordinates[1]);
    }

  row_vector.push_back(1.);

  for(uint i = 1; i <= pdeg; i++)
    {
      for(uint j = 1; j <= i; j++)
	{
	  row_vector.push_back(x_values[j+1]*y_values[j-i+1]);
	}
    }

  return row_vector;
}

vector<double> polynomial_support_rhs_column_vector_2D(string selection, uint pdeg)
{

  //Calculating length of vector
  uint length = 1;

  for(uint i = 1; i < pdeg; i++)
    {
      length += (i+1);
    }

  vector<double> out;

  for(uint i = 0; i < length; i++)
    {
      out.push_back(0.);
    }

  if(selection == "x")
    {
      out[1] = 1.;
    }

  else if(selection == "y")
    {
      out[2] = 1.;
    }

  else if(selection == "xx")
    {
      out[3] = 2.;	      
    }	
  else if(selection == "yy")
    {
      out[5] = 2.;
    }
  else if(selection == "xy")
    {
      out[4] = 1.;
    }
  else if(selection == "xxx")
    {
      out[6] = 6.;
    }
  else if(selection == "yyy")
    {
      out[9] = 6.;
    }
  else if(selection == "xxy")
    {
      out[7] = 2.;
    }
  else if(selection == "xyy")
    {
      out[8] = 2.;
    }
  else if(selection == "Laplace")
    {
      out[3] = 2.;
      out[5] = 2.;
    }
  else if(selection == "Neg_Laplace")
    {
      out[3] = 2.;
      out[5] = 2.; 
    }
  
  return out;

}

