/*** fits.cpp
These small utility routines let the user read/write vectors
from the C++ stdlib into FITS files and reverse. 
Additional routines for reading and writing headers are also
included.   
 
Julian Merten
JPL/Caltech
Dec 2014
jmerten@caltech.edu
***/

#include <mfree/fits.h>

template <class T> void write_img_to_fits(string fits_filename, vector<T> *input, int x_dim = 0, int y_dim = 0, string extension_name = "")
{

  //Creating FITS pointer and image dimensions
  fitsfile *fptr;
  int naxis = 2;
  if(y_dim = 0)
    {
      y_dim = floor(sqrt(input->size()));
      x_dim = y_dim; 
    }
  long naxes[2]  = {x_dim, y_dim};

  //Geting the type of the input vector
  int image_type = read_img_type(&input);

  //Setting fitsfile status
  int status = 0;

  //Trying to open FITS file, or try to create it. Throw error if this fails.
  string filename = "!"+fits_filename;

  if(fits_open_file(&fptr, fits_filename.c_str(), READWRITE, &status))
    {
      if(fits_create_file(&fptr, filename.c_str(), &status))
	{
	  throw runtime_error("MFREE FITS: Cannot open or create FITS file.");
	} 
    }

  //Creating a new image in the FITS file
  if(fits_create_img(fptr, image_type, naxis, naxes, &status))
    {
      throw runtime_error("MFREE FITS: Could not create image.");
    }

  //Writing pixel data from input vector
  long fpixel[2] = {1,1};
  if(fits_write_pix(fptr, image_type, fpixel, x_dim*y_dim, &input[0], &status))
    {
       throw runtime_error("MFREE FITS: Could not write pixel data.");
    }

  //Setting extension name if necessary
  if(extension_name != "")
    {
      if(fits_write_key(fptr,TSTRING,"EXTNAME",extension_name.c_str(),"",&status))
	{
	  throw runtime_error("MFREE FITS: Could no create extension name.");
	}
    }
  //Writing creator and system time information into HDU
  char *time;
  int time_status;
  string time_format;
  fits_get_system_time(time, &time_status, &status);
  if(time_status)
    {
      time_format = "LOCAL";
    }
  else
    {
      time_format = "UTC";
    }
  if(fits_write_key(fptr,TSTRING,"CREATOR","libmfree","",&status) && fits_write_key(fptr,TSTRING,"DATE",time,time_format.c_str(),&status))
    {
      throw runtime_error("MFREE FITS: Could not write aux headers.");
    }

}

template <class T> vector<T> read_img_from_fits(string fits_filename, string extension_name = "" )
{

}

template <class T> void write_header_to_fits(string fits_filename, string keyword, T input, string description= "", string extension_name = "")
{

}

template <class T> T read_header_from_fits(string fits_filename, string keyword, string extension_name = "")
{

}
