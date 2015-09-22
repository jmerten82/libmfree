#include <iostream>
#include <gsl/gsl_matrix.h>
#include <mfree/mesh_free.h>
#include <mfree/rwfits.h>
#include <mfree/test_functions.h>
#include <mfree/test_functions_implementation.h>
#include <saw/findif_fast.h>
#include <saw/rw_fits.h>

using namespace std;

int main()
{

  double scale = 2.;
  int size = 200;

  string out1 = "./test_function1.fits";
  string out2 = "./test_function2.fits";

  gsl_vector *function1 = gsl_vector_calloc(size*size); 
  gsl_vector *function2 = gsl_vector_calloc(size*size); 
  gsl_vector *Dfunction1 = gsl_vector_calloc(size*size); 
  gsl_vector *Dfunction2 = gsl_vector_calloc(size*size); 
  gsl_vector *DDfunction1 = gsl_vector_calloc(size*size); 
  gsl_vector *DDfunction2 = gsl_vector_calloc(size*size); 

  gsl_vector *NDfunction1 = gsl_vector_calloc(size*size);
  gsl_vector *NDDfunction1 = gsl_vector_calloc(size*size);

  gsl_vector *NDfunction2 = gsl_vector_calloc(size*size);
  gsl_vector *NDDfunction2 = gsl_vector_calloc(size*size);

  double step = scale / size;
  double x,y;

  findif_grid_fast grid(size,size);
  grid.set_scale(scale);


  bengts_function test1;
  vector<double> new_coords;
  new_coords.push_back(0.25);
  new_coords.push_back(-0.25);
  nfw_lensing_potential test2(5.0,new_coords);
  vector<double> coordinates;
  coordinates.resize(2);

  int index = 0;
  y =-1.0;
  for(int i = 0; i < 200; i++)
    {
      coordinates[1] = y;
      x = -1.0;
      for(int j = 0; j < 200; j++)
	{
	  coordinates[0] = x;
	  gsl_vector_set(function1,index,test1(coordinates));
	  gsl_vector_set(function2,index,test2(coordinates));
	  gsl_vector_set(Dfunction1,index,test1.D(coordinates,"x"));
	  gsl_vector_set(Dfunction2,index,test2.D(coordinates,"x"));
	  gsl_vector_set(DDfunction1,index,test1.D(coordinates,"xx"));
	  gsl_vector_set(DDfunction2,index,test2.D(coordinates,"Laplace"));
	  x += step;
	  index++;
	}
      y += step;
    }
  grid.derivative_parser(function1,NDfunction1,"#%1.0%first%x%add%#");
  grid.derivative_parser(function1,NDDfunction1,"#%1.0%second%x%add%#");

  grid.derivative_parser(function2,NDfunction2,"#%1.0%first%x%add%#");
  grid.derivative_parser(function2,NDDfunction2,"#%0.5%second%x%add%#%0.5%second%y%add%#");

  write_pimg(out1,size,size,function1 );
  write_imge(out1,"Dx",size,size,Dfunction1);
  write_imge(out1,"NDx",size,size,NDfunction1);
  write_imge(out1,"Dxx",size,size,DDfunction1);
  write_imge(out1,"NDxx",size,size,NDDfunction1);

  write_pimg(out2,size,size,function2);
  write_imge(out2,"Dx",size,size,Dfunction2);
  write_imge(out2,"NDx",size,size,NDfunction2);
  write_imge(out2,"Laplace",size,size,DDfunction2);
  write_imge(out2,"NLaplace",size,size,NDDfunction2);




  return 0;
}
