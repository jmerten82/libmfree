/*** test_rbf.cpp
Tests the basic functionality of the current mesh-free domain implementation.
 
Julian Merten
Universiy of Oxford
Aug 2015
julian.merten@physics.ox.ac.uk
http://www.julianmerten.net
***/

#include <iostream>
#include <fstream>
#include <mfree/mesh_free.h>
#include <mfree/mesh_free_differentiate.h>
#include <mfree/grid_utils.h>

using namespace std;

int main()
{


  mesh_free mesh1;
  mesh1.print_info("Grid1");


  vector<double> test_coords;
  test_coords.push_back(1.0);
  test_coords.push_back(1.0);
  test_coords.push_back(0.0);
  test_coords.push_back(0.0);
  test_coords.push_back(-1.0);
  test_coords.push_back(-1.0);
  test_coords.push_back(-42.0);

  mesh_free mesh2(2,&test_coords);
  mesh2.print_info("Grid2");

  test_coords.pop_back();
  test_coords.pop_back();
  test_coords.pop_back();
  
  mesh_free mesh3(2,&test_coords);
  mesh3.print_info("Grid3");

  
  mesh_free mesh4;
  mesh4 = mesh2 - mesh3;
  mesh4.print_info("Grid4");

  mesh2 *= 30.;
  mesh2.print_info("Grid2 * 30.0");

  mesh2 += mesh3;
  mesh2.print_info("Grid2* 30.0 + Grid3");

  mesh2 -= mesh4;
  mesh2.print_info("Grid2* 30.0 + Grid3 - Grid4");

  mesh2 -= mesh3;
  mesh2.print_info("Grid2* 30.0 + Grid3 - Grid4 - Grid3");

  mesh2.build_tree();

  coordinate_grid nomesh("regular", 100);

  vector<double> coordinates = nomesh();

  mesh_free_2D my_domain(&coordinates);
  my_domain *= 100.;

  my_domain.print_info();
  my_domain.build_tree();

  mesh_free_2D another_domain;
  another_domain = my_domain;
  another_domain *= 0.5;
  another_domain.print_info();

  my_domain -= another_domain;
  my_domain.print_info();









  
  return 0;
}
