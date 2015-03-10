#include <iostream>
#include <vector>
#include <mfree/rwfits.h>

using namespace std;

int main()
{

  vector<int> test_vector;

  for(int i = 0; i < 2500; i++)
    {
      test_vector.push_back(i);
    }

  write_img_to_fits("./test_fits.fits", &test_vector);
  
  vector<float> test_float;

  for(int i = 0; i < 20000; i++)
    {
      test_float.push_back((float) i);
    }

  write_img_to_fits("./test_fits.fits",&test_float,"float",100,200);

  vector<double> test_double;

  for(int i = 0; i < 40000; i++)
    {
      test_double.push_back((double) i);
    }

  write_img_to_fits("./test_fits.fits", &test_double, "double");


  vector<double> in_test;

  read_img_from_fits("./test_fits.fits",&in_test);

  write_img_to_fits("./rw_fits.fits",&in_test);

  read_img_from_fits("./test_fits.fits",&in_test,"double");

  write_img_to_fits("./rw_fits.fits",&in_test,"double");

  double cat_size= 34.5;
  int cat_number = 45;

  write_header_to_fits("./rw_fits.fits","KATZE", cat_size);
  write_header_to_fits("./rw_fits.fits","KATZEN",cat_number,"double","Wieviele");

  vector<double> test;

  for(int i = 0; i < 6; i++)
    {
      test.push_back(1.);
    }

  add_WCS_to_fits("./rw_fits.fits", test);
  add_WCS_to_fits("./rw_fits.fits", test, "linear", "double");

  double cat_size_again;
  int cat_number_again;

  read_header_from_fits("./rw_fits.fits", "KATZE", &cat_size_again);
  read_header_from_fits("./rw_fits.fits", "KATZEN", &cat_number_again, "double");

  cout <<cat_size_again <<"  " <<cat_number_again <<endl;


  return 0;

}
