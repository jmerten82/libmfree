/*** rwfits.cpp
These small utility routines let the user read/write vectors
from the C++ stdlib into FITS files and reverse. 
Additional routines for reading and writing headers are also
included, together with WCS functioanlity..   
 
Julian Merten
Universiy of Oxford
Feb 2015
julian.merten@physics.ox.ac.uk
http://www.julianmerten.net
***/

/**
   Most of the rwfits routines are templates so they 
   would be defined directly in the header.
**/

#include <mfree/rwfits.h>

using namespace std;

void add_WCS_to_fits(string fits_filename, vector<double> WCS, string format, string extension_name)
{

  if(!(WCS.size() == 6 || WCS.size() == 8))
    {
      throw invalid_argument("RWFITS: WCS input of invalid size.");
    }

  vector<double> system;

  if(WCS.size() == 8)
    {
      system = WCS;
    }
  else
    {
      for(int i = 0; i < 5; i++)
	{
	  system.push_back(WCS[i]);
	}
      system.push_back(0.0);
      system.push_back(0.0);
      system.push_back(WCS[5]);
    }

  if(extension_name == "")
    {
      write_header_to_fits(fits_filename,"CRPIX1", system[0]);
      write_header_to_fits(fits_filename,"CRPIX2", system[1]);
      write_header_to_fits(fits_filename,"CRVAL1", system[2]);
      write_header_to_fits(fits_filename,"CRVAL2", system[3]);
      string temp1, temp2;
      if(format == "J2000")
	{
	  temp1 = "RA---TAN ";
	  temp2 = "DEC--TAN";
	}
      else
	{
	  temp1 = " ";
	  temp2 = " ";
	}
      write_header_to_fits(fits_filename,  "CTYPE1",temp1);
      write_header_to_fits(fits_filename,  "CTYPE2",temp2);
      write_header_to_fits(fits_filename,  "CD1_1", system[4]);
      write_header_to_fits(fits_filename,  "CD1_2", system[5]);
      write_header_to_fits(fits_filename,  "CD2_1", system[6]);
      write_header_to_fits(fits_filename,  "CD2_2", system[7]);
    }
  else
    {
      write_header_to_fits(fits_filename,  "CRPIX1", system[0],extension_name);
      write_header_to_fits(fits_filename,  "CRPIX2", system[1],extension_name);
      write_header_to_fits(fits_filename,  "CRVAL1", system[2],extension_name);
      write_header_to_fits(fits_filename,  "CRVAL2", system[3],extension_name);
      string temp1, temp2;
      if(format == "J2000")
	{
	  temp1 = "RA---TAN ";
	  temp2 = "DEC--TAN";
	}
      else
	{
	  temp1 = " ";
	  temp2 = " ";
	}
      write_header_to_fits(fits_filename, "CTYPE1",temp1,extension_name);
      write_header_to_fits(fits_filename, "CTYPE2",temp2,extension_name);
      write_header_to_fits(fits_filename, "CD1_1", system[4],extension_name);
      write_header_to_fits(fits_filename, "CD1_2", system[5],extension_name);
      write_header_to_fits(fits_filename, "CD2_1", system[6],extension_name);
      write_header_to_fits(fits_filename, "CD2_2", system[7],extension_name);
    }


}
