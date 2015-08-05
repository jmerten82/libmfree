/*** test_rbf.cpp
Tests the basic functionality of the current RBF implementation.
 
Julian Merten
Universiy of Oxford
Aug 2015
julian.merten@physics.ox.ac.uk
http://www.julianmerten.net
***/

#include <iostream>
#include <fstream>
#include <mfree/radial_basis_function.h>
#include <mfree/rbf_implementation.h>
#include <mfree/rwfits.h>

using namespace std;

int main()
{

  //Creating a simple Gaussian RBF

  //Testing standard constructor

  gaussian_rbf rbf1;
  coordinate c1;
  c1 = rbf1.show_coordinate_offset();
  
  cout <<"Gauss1  " <<"x: " <<c1.x <<"\t" <<"x: " <<c1.y <<"\t" <<"z: " <<c1.z <<"\t" <<"eps: " <<rbf1.show_epsilon() <<endl;

  coordinate c2;
  c2.x = 1.0;
  c2.y = 2.0; 
  c2.z = 3.0;

  gaussian_rbf rbf2(c2,42.0);
  c2 = rbf2.show_coordinate_offset();

  cout <<"Gauss2  "<<"x: " <<c2.x <<"\t" <<"x: " <<c2.y <<"\t" <<"z: " <<c2.z <<"\t" <<"eps: " <<rbf2.show_epsilon() <<endl;

  gaussian_rbf rbf3 = rbf2;
  coordinate c3 = rbf3.show_coordinate_offset();
  cout <<"Gauss3  "<<"x: " <<c3.x <<"\t" <<"x: " <<c3.y <<"\t" <<"z: " <<c3.z <<"\t" <<"eps: " <<rbf3.show_epsilon() <<endl;

  rbf3 *= 2.0;
  rbf3 -= 5.0;
  rbf3 /= 10000.0;
  rbf3.set_coordinate_offset(50.,-50.);
  c3 = rbf3.show_coordinate_offset();
  cout <<"Gauss4  "<<"x: " <<c3.x <<"\t" <<"x: " <<c3.y <<"\t" <<"z: " <<c3.z <<"\t" <<"eps: " <<rbf3.show_epsilon() <<endl;

  cubic_spline_rbf spline1;
  coordinate s1;
  s1 = spline1.show_coordinate_offset();
  cout <<"Spline1  "<<"x: " <<s1.x <<"\t" <<"x: " <<s1.y <<"\t" <<"z: " <<s1.z <<endl;

  cubic_spline_rbf spline2(1.5);
  coordinate s2;
  s2 = spline2.show_coordinate_offset();
  cout <<"Spline2  "<<"x: " <<s2.x <<"\t" <<"x: " <<s2.y <<"\t" <<"z: " <<s2.z <<endl;

  coordinate s3;
  s3.x = -50.0;
  s3.y = 50.0;
  s3.z = 0.;

  cubic_spline_rbf spline3(s3);
  s1 = spline3.show_coordinate_offset();
  cout <<"Spline3  "<<"x: " <<s1.x <<"\t" <<"x: " <<s1.y <<"\t" <<"z: " <<s1.z <<endl;

  cout <<"Plotting 1D Gauss4 to gauss1d.dat..." <<flush;
  ofstream output("./gauss1d.dat");
  for(double x = -500.0; x < 500.0; x+=1.)
    {
      output <<x <<" " <<rbf3(x) <<" " <<rbf3.Dx(x) <<" " <<rbf3.Dy(x) <<" " <<rbf3.Dxx(x) <<endl;
    }
  cout <<"done." <<endl;

  output.close();

  cout <<"Plotting 1D Spline3 to spline1d.dat..." <<flush;
  ofstream output2("./spline1d.dat");
  for(double x = -500.0; x < 500.0; x+=1.)
    {
      output2 <<x <<" " <<spline3(x) <<" " <<spline3.Dx(x) <<" " <<spline3.Dy(x) <<" " <<spline3.Dxx(x) <<endl;
    }
  cout <<"done." <<endl;

  output2.close();

  cout <<"Ploting 2D Gauss4 and Spline3 to gauss2d.fits and spline2d.fits..." <<flush;

  vector<double> gauss, gauss_dx, gauss_dxy;
  vector<double> spline, spline_dx, spline_dxy;
  for(int i = 0; i < 1000; i++)
    {
      for(int j = 0; j < 1000; j++)
	{
	  double x = (double) j - 500.;
	  double y = (double) i - 500.;
	  gauss.push_back(rbf3(x,y));
	  gauss_dx.push_back(rbf3.Dx(x,y));
	  gauss_dxy.push_back(rbf3.Dxy(x,y));
	  spline.push_back(spline3(x,y));
	  spline_dx.push_back(spline3.Dx(x,y));
	  spline_dxy.push_back(spline3.Dxy(x,y));
	}
    }

  write_img_to_fits("./gauss2d.fits",&gauss);
  write_img_to_fits("./gauss2d.fits",&gauss_dx,"Dx");
  write_img_to_fits("./gauss2d.fits",&gauss_dxy,"Dxy");

  write_img_to_fits("./spline2d.fits",&spline);
  write_img_to_fits("./spline2d.fits",&spline_dx,"Dx");
  write_img_to_fits("./spline2d.fits",&spline_dxy,"Dxy");

  cout <<"done." <<endl;

  return 0;

}
