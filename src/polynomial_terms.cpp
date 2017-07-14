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

vector<double> row_vector_from_polynomial_1D(double x, unsigned pdeg)
{

  vector<double> row_vector;
  row_vector.push_back(1.);

  for(unsigned i = 1; i <= pdeg; i++)
    {
      row_vector.push_back(row_vector[i-1]*x);
    }
  return row_vector;
}

vector<double> row_vector_from_polynomial_2D(double x, double y, unsigned pdeg)
{

  vector<double> row_vector;
  vector<double> x_values, y_values;

  x_values.push_back(1.);
  y_values.push_back(1.);
  for(int i = 1; i <= pdeg; i++)
    {
      x_values.push_back(x_values[i-1]*x);
      y_values.push_back(y_values[i-1]*y);
    }

  row_vector.push_back(1.);
  for(unsigned i = 0; i < pdeg; i++)
    {
      unsigned max_grade = i+1;
      for(unsigned j = 0; j <= max_grade; j++)
	{
	  row_vector.push_back(x_values[max_grade-j]*y_values[j]);
	}
    }

  return row_vector;
}

vector<double> row_vector_from_polynomial_3D(double x, double y, double z,  unsigned pdeg)
{

  vector<double> row_vector;
  vector<double> x_values, y_values, z_values;

  x_values.push_back(1.);
  y_values.push_back(1.);
  z_values.push_back(1.);

  for(unsigned i = 1; i <= pdeg; i++)
    {
      x_values.push_back(x_values[i-1]*x);
      y_values.push_back(y_values[i-1]*y);
      z_values.push_back(z_values[i-1]*z);
    }

  row_vector.push_back(1.);
  unsigned counter = 1;  

  for(int i = 0; i < pdeg; i++)
    {
      int max = i+1;
      for(int l = max; l >= 0; l--)
	{
	  int max_y = max-l;
	  for(int m = max_y; m >= 0; m--)
	    {
	      int max_z = max-l-m;
	      for(int n = max_z; n >= 0; n--)
		{
		  if(l+m+n == max)
		    {
		      row_vector.push_back(x_values[l]*y_values[m]*z_values[n]);
		    }
		}
	    }
	}

    }

  return row_vector;
}

vector<double> polynomial_support_rhs_column_vector_1D(string selection, unsigned pdeg)
{

  //Calculating length of vector
  unsigned length = (pdeg+1);

  vector<double> out;

  for(unsigned i = 0; i < length; i++)
    {
      out.push_back(0.);
    }

  if(selection == "x")
    {
      if(pdeg > 0)
	{
	  out[1] = 1.;
	}
    }
  else if(selection == "xx")
    {
      if(pdeg > 1)
	{
	  out[2] = 2.;
	}	      
    }	
  else if(selection == "xxx")
    {
      if(pdeg > 2)
	{
	  out[3] = 6.;
	}
    }
  else if(selection == "Laplace")
    {
      if(pdeg > 1)
	{
	  out[2] = 2.;
	}
    }
  else if(selection == "Neg_Laplace")
    {
      if(pdeg > 1)
	{
	  out[2] = 2.;
	}
    }
  else
    {
      throw invalid_argument("POLYTERMS: Invalid derivative selection");
    }
  
  return out;
}


vector<double> polynomial_support_rhs_column_vector_2D(string selection, unsigned pdeg)
{

  //Calculating length of vector
  unsigned length = (pdeg+1)*(pdeg+2)/2;

  vector<double> out;

  for(unsigned i = 0; i < length; i++)
    {
      out.push_back(0.);
    }

  if(selection == "x")
    {
      if(pdeg > 0)
	{
	  out[1] = 1.;
	}
    }

  else if(selection == "y")
    {
      if(pdeg > 0)
	{
	  out[2] = 1.;
	}
    }
  else if(selection == "xx")
    {
      if(pdeg > 1)
	{
	  out[3] = 2.;
	}	      
    }	
  else if(selection == "yy")
    {
      if(pdeg > 1)
	{
	  out[5] = 2.;
	}
    }
  else if(selection == "xy")
    {
      if(pdeg > 1)
	{
	  out[4] = 1.;
	}
    }
  else if(selection == "xxx")
    {
      if(pdeg > 2)
	{
	  out[6] = 6.;
	}
    }
  else if(selection == "yyy")
    {
      if(pdeg > 2)
	{
	  out[9] = 6.;
	}
    }
  else if(selection == "xxy")
    {
      if(pdeg > 2)
	{
	  out[7] = 2.;
	}
    }
  else if(selection == "xyy")
    {
      if(pdeg > 2)
	{
	  out[8] = 2.;
	}
    }
  else if(selection == "Laplace")
    {
      if(pdeg > 1)
	{
	  out[3] = 1.;
	  out[5] = 1.;
	}
    }
  else if(selection == "Neg_Laplace")
    {
      if(pdeg > 1)
	{
	  out[3] = 1.;
	  out[5] = -1.;
	} 
    }

  else
    {
      throw invalid_argument("POLYTERMS: Invalid derivative selection");
    }
  
  return out;

}

vector<double> polynomial_support_rhs_column_vector_3D(string selection, unsigned pdeg)
{

  //Calculating length of vector
  unsigned length = (pdeg+1)*(pdeg+2)*(pdeg+3)/6;


  vector<double> out;

  for(unsigned i = 0; i < length; i++)
    {
      out.push_back(0.);
    }

  if(selection == "x")
    {
      if(pdeg > 0)
	{
	  out[1] = 1.;
	}
    }

  else if(selection == "y")
    {
      if(pdeg > 0)
	{
	  out[2] = 1.;
	}
    }

  else if(selection == "z")
    {
      if(pdeg > 0)
	{
	  out[3] = 1.;
	}	      
    }	
  else if(selection == "xx")
    {
      if(pdeg > 1)
	{
	  out[4] = 2.;
	}
    }
  else if(selection == "xy")
    {
      if(pdeg > 1)
	{
	  out[5] = 1.;
	}
    }
  else if(selection == "xz")
    {
      if(pdeg > 1)
	{
	  out[6] = 1.;
	}
    }
  else if(selection == "yy")
    {
      if(pdeg > 1)
	{
	  out[7] = 2.;
	}
    }
  else if(selection == "yz")
    {
      if(pdeg > 1)
	{
	  out[8] = 1.;
	}
    }
  else if(selection == "zz")
    {
      if(pdeg > 1)
	{
	  out[9] = 2.;
	}
    }
  else if(selection == "xxx")
    {
      if(pdeg > 2)
	{
	  out[10] = 6.;
	}
    }
  else if(selection == "xxy")
    {
      if(pdeg > 2)
	{
	  out[11] = 2.;
	}
    }
  else if(selection == "xxz")
    {
      if(pdeg > 2)
	{
	  out[12] = 2.;
	}
    }
  else if(selection == "xyy")
    {
      if(pdeg > 2)
	{
	  out[13] = 2.;
	} 
    }
  else if(selection == "xyz")
    {
      if(pdeg > 2)
	{
	  out[14] = 1.;
	} 
    }
  else if(selection == "xzz")
    {
      if(pdeg > 2)
	{
	  out[15] = 2.;
	}
    }
  else if(selection == "yyy")
    {
      if(pdeg > 2)
	{
	  out[16] = 6.;
	} 
    }
  else if(selection == "yyz")
    {
      if(pdeg > 2)
	{
	  out[17] = 2.; 
	}
    }

  else if(selection == "yzz")
    {
      if(pdeg > 2)
	{
	  out[18] = 2.;
	}
    }
  else if(selection == "zzz")
    {
      if(pdeg > 2)
	{
	  out[19] = 6.;
	} 
    }

  else if(selection == "Laplace")
    {
      if(pdeg > 1)
	{
	  out[4] = 1.;
	  out[7] = 1.;
	}
    }
  else if(selection == "Neg_Laplace")
    {
      if(pdeg > 1)
	{
	  out[4] = 1.;
	  out[7] = -1.; 
	}
    }
  else
    {
      throw invalid_argument("POLYTERMS: Invalid derivative selection");
    }
  
  return out;

}

