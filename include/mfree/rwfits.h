/*** rwfits.h
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

#ifndef    RWFITS_H
#define    RWFITS_H

#include <vector>
#include <ctime>
#include <cmath>
#include <stdexcept>
#include <mfree/mesh_free.h>
#include <mfree/mesh_free_differentiate.h>
#include <CCfits/CCfits>
#include <flann/flann.hpp>

using namespace std;
using namespace CCfits;

/*
  Writes a vector as the image of a FITS file. If no  x and y dimensions
  are given, the image is expected to have an aspect ratio of 1.
  If no extenion is given, the primary HDU is written. 
*/

template <class T> void write_img_to_fits(string fits_filename, vector<T> *input, string extension_name = "", int x_dim = 0, int y_dim = 0);

/*
  Reads a vector from the image of a FITS file.
  If no extenion is given, the primary HDU is read. 
*/

template <class T> void read_img_from_fits(string fits_filename, string extension_name = "" );

/*
  This routine reads a column from a FITS binary table. You can specify the
  extension index, if it is not set you can specify the extensions name.
  If both are not set, the extension with index 1 is read. See
  CCFits documentation for conventions. The routine returns the size of
  the output vector. The column to read is either identified by name 
  or by its index. If none is specified, the first one is read.
*/

template <class T> int read_column_from_fits_table(string fits_filename, vector<T> *output, string column_name = "", int column_index = 1, string extension_name = "", int extension_index = 1);

/*
  This routine reads a number of FITS table columns into a vector of vectors.
  The indices of the columns are defined by the content of the index vector.
  If no index vector is passed, all columns are read. 
*/

template <class T> vector<int> read_columns_from_fits_table(string fits_filename, vector<vector<T> > *output, vector<int> index_vector = vector<int>(), string extension_name = "", int extension_index = 1);


/*
  This routine writes a column to a FITS binary table. You can specify the
  extension index, if it is not set you can specify the extensions name.
  If both are not set, the extension with index 1 is written to. See
  CCFits documentation for conventions. 
  You cam also specify the column name. 
*/

template <class T> void write_column_to_fits_table(string fits_filename, vector<T> *input, string column_name = "", string extension_name = "");

/*
  This routine writes all vectors in the umbrella vector as separate columns
  into the FITS file table. Extension name cane be specified and a vector
  with names can be provided. 
*/

template <class T> void write_columns_to_fits_table(string fits_filename, vector<vector<T> > *input, vector<string> column_names = vector<string>(), string extension_name = "");

/*
  Writes a header with keyword and description into the HDU of a FITS file.
  If no extenion is given, the primary HDU is written. 
*/

template <class T> void write_header_to_fits(string fits_filename, string keyword, T input, string extension_name = "", string description= "");

/*
  Read a header  identified by keyword  
  from the HDU of a FITS file and writes it into output. 
  If no extenion is given, the primary HDU is read. 
*/

template <class T> void read_header_from_fits(string fits_filename, string keyword,  T *output, string extension_name = "");

/*
  This routine writes a WCS system into either the primary image 
  or an image extension. If the extension name is left blank, it is
  assumed that the WCS is for the primary image. The format of the
  input vector is x/y coordinate of image centre (pixels), x/y of
  coordinate centre and the four values of the translation matrix.
  If only two values are give, it is expected that the diagonal elements
  are zero. The format of the WCS can be either "J2000" or "linear".
*/

void add_WCS_to_fits(string fits_filename, vector<double> WCS, string format = "J2000", string extension_name = "");   

/* 
   Necessary datatype routines to define the datatype of a FITS
   table or image. This is necessary since the FITS libraries
   need this type hard-wired. The default type is going to be
   double.
*/

template<typename T> inline static int read_img_type(vector<T> *input)
{
  return DOUBLE_IMG;
}

template<typename T> inline static int read_header_type(T *input)
{
  return TDOUBLE;
}

template<typename T> inline static ValueType read_table_type(vector<T> *input)
{
  return Tdouble;
}


/*
  This routine takes a 2D mesh-free domain and some function and places
  it as a Voronoi diagram into a FITS file. The original coordinates
  are placed into a linear WCS. The dimension of the oversampled 
  output has to be given and will be applied to the larger of the
  two side lengths. If an extension is given, a FITS extension
  is written instead of the primary image. 
*/

template <class T> void voronoi_to_fits(mesh_free_2D *grid, vector<T> *function, string filename, string extension = "", int dim = 512); 

//Explicit definitions of the template routines

template <class T> void write_img_to_fits(string fits_filename, vector<T> *input, string extension_name = "", int x_dim = 0, int y_dim = 0)
{

  //Creating image dimensions

  std::auto_ptr<FITS> img(0);
  if(y_dim == 0)
    {
      y_dim = floor(sqrt(input->size()));
      x_dim = y_dim; 
    }
  long naxes[2] = {x_dim,y_dim};
  vector<long> extAx(2,0);
  extAx.front() = x_dim;
  extAx.back() = y_dim;
  long naxis = 2;

  //Geting the type of the input vector
  int image_type = read_img_type(input);

  //Setting fitsfile status
  int status = 0;

  valarray<T> temp;
  temp.resize(input->size());
  copy(input->begin(),input->end(),&temp[0]);

  //Trying to open FITS file, or try to create it.
  if(extension_name == "")
    {
      img.reset(new FITS("!"+fits_filename, image_type, naxis, naxes));
      img->pHDU().write(1,x_dim*y_dim, temp);
      img->pHDU().addKey("Creator","libmfree","");
      img->pHDU().writeDate();
    }
  
  else
    {
      img.reset(new FITS(fits_filename,Write));
      ExtHDU* imageExt = img->addImage(extension_name, image_type, extAx);
      imageExt->write(1, x_dim*y_dim, temp);
      imageExt->addKey("Creator","libmfree","");
      imageExt->writeDate();
    } 
}

template <class T> void read_img_from_fits(string fits_filename, vector<T> *output,  string extension_name = "" )
{

  //Create FITS pointer and create temporary valarray
  std::auto_ptr<FITS> pInfile(new FITS(fits_filename, Read, true));
  valarray<T> temp;

  //Opening either pHDU or extension and read image content

  if(extension_name == "")
    {
      PHDU& img = pInfile->pHDU();
      img.read(temp);
    }
  else
    {
      ExtHDU& img = pInfile->extension(extension_name);
      img.read(temp);
    }

  output->resize(temp.size());
  copy(&temp[0],&temp[temp.size()],output->begin());

}

template <class T> void write_header_to_fits(string fits_filename, string keyword, T input, string extension_name = "", string description= "")
{
  std::auto_ptr<FITS> img(0);

  if(extension_name == "")
    {
      img.reset(new FITS(fits_filename,Write));
      img->pHDU().addKey(keyword,input,description);
    }
  else
    {
      img.reset(new FITS(fits_filename,Write,extension_name));
      img->currentExtension().addKey(keyword,input,description);
    }
}

template <class T> void read_header_from_fits(string fits_filename, string keyword,  T *output, string extension_name = "")
{
  std::auto_ptr<FITS> pInfile(new FITS(fits_filename, Read, true));

  if(extension_name == "")
    {
      PHDU& img = pInfile->pHDU();
      img.readKey(keyword,*output);
    }

  else
    {
      ExtHDU& img = pInfile->extension(extension_name);
      img.readKey(keyword,*output);
    }

}

template <class T> int read_column_from_fits_table(string fits_filename, vector<T> *output, string column_name, int column_index, string extension_name, int extension_index)
{

  //Create FITS pointer and create temporary valarray
  std::auto_ptr<FITS> pInfile(new FITS(fits_filename, Read, true));

  //We need this for conversion reasons. Not the most efficient. 

  valarray<T> temp;

  //Opening the right table


  if(extension_name == "")
    {
      ExtHDU& table =  pInfile->extension(extension_index);
      if(column_name == "")
	{
	  Column& column = table.column(column_index);
	  column.read(temp, 0, column.rows());
	}
      else
	{
	  Column& column = table.column(column_name);
	  column.read(temp, 0, column.rows());
	}      
    }
  else
    {
      ExtHDU& table = pInfile->extension(extension_name);
      if(column_name == "")
	{
	  Column& column = table.column(column_index);
	  column.read(temp, 0, column.rows());
	}
      else
	{
	  Column& column = table.column(column_name);
	  column.read(temp, 0, column.rows());
	}
      
    }
  output->clear();
  output->resize(temp.size());
  
  copy(&temp[0],&temp[temp.size()],output->begin());
  
  return temp.size();
  
}

template <class T> vector<int> read_columns_from_fits_table(string fits_filename, vector<vector<T> > *output, vector<int> index_vector, string extension_name, int extension_index)
{
  //Create FITS pointer and create temporary valarray
  std::auto_ptr<FITS> pInfile(new FITS(fits_filename, Read, true));

  output->clear();

  //Opening the right table

  if(extension_name == "")
    {
      ExtHDU& table =  pInfile->extension(extension_index);
      int num_cols;
      if(index_vector.size() == 0)
	{
	  num_cols = table.numCols();
	}
      else
	{
	  num_cols = index_vector.size();
	}      
      for(int i = 0; i < num_cols; i++)
	{
	  //We need this for conversion reasons. Not the most efficient. 
	  valarray<T> temp;
	  vector<T> temp2;
	  int index;
	  if(index_vector.size() == 0)
	    {
	      index = i+1;
	    }
	  else
	    {
	      index = index_vector[i];
	    }
	  Column& column = table.column(index);
	  column.read(temp, 0, column.rows());
	  temp2.resize(temp.size());
	  copy(&temp[0],&temp[temp.size()],temp2.begin());
	  output->push_back(temp2);
	}   
    }
  else
    {
      ExtHDU& table =  pInfile->extension(extension_index);
      int num_cols;
      if(index_vector.size() == 0)
	{
	  num_cols = table.numCols();
	}
      else
	{
	  num_cols = index_vector.size();
	}
      
      for(int i = 0; i < num_cols; i++)
	{
	  //We need this for conversion reasons. Not the most efficient. 
	  valarray<T> temp;
	  vector<T> temp2;
	  int index;
	  if(index_vector.size() == 0)
	    {
	      index = i+1;
	    }
	  else
	    {
	      index = index_vector[i];
	    }
	  Column& column = table.column(index);
	  column.read(temp, 0, column.rows());
	  temp2.resize(temp.size());
	  copy(&temp[0],&temp[temp.size()],temp2.begin());
	  output->push_back(temp2);
	}
    }
 
  vector<int> sizes;
  for(int i = 0; i < output->size(); i++)
    {
      sizes.push_back((*output)[i].size());
    }
  return sizes; 
}

template <class T> void write_column_to_fits_table(string fits_filename, vector<T> *input, string column_name, string extension_name)
{

  std::auto_ptr<FITS> pFits(0);
  pFits.reset( new FITS(fits_filename,Write) );

  valarray<T> temp;
  temp.resize(input->size());
  for(int i = 0; i < input->size(); i++)
    {
      temp[i] = (*input)[i];
    }  
  Table* newTable = pFits->addTable(extension_name,input->size());
  newTable->addColumn(read_table_type(input),column_name,1);  
  newTable->column(column_name).write(temp,1);
}

template <class T> void write_columns_to_fits_table(string fits_filename, vector<vector<T> > *input, vector<string> column_names, string extension_name)
{

  std::auto_ptr<FITS> pFits(0);
  pFits.reset( new FITS(fits_filename,Write) );

  Table* newTable = pFits->addTable(extension_name,(*input)[0].size());

  for(int i = 0; i < input->size(); i++)
    {
      string current_name;
      if(column_names.size() >= input->size())
	{
	  current_name = column_names[i];
	}
      else
	{
	  current_name = "";
	}
      valarray<T> temp;
      temp.resize((*input)[i].size());
      for(int j = 0; j < temp.size(); j++)
	{
	  temp[j] = (*input)[i][j];
	} 
      newTable->addColumn(read_table_type(&(*input)[i]),current_name,1);  
      newTable->column(current_name).write(temp,1);
    }
}





/*
  Template specialisations. 
*/

template<> inline int read_img_type<float> (vector<float> *input)
{
  return FLOAT_IMG;
}

template<> inline int read_header_type<float> (float *input)
{
  return TFLOAT;
}

template<> inline ValueType read_table_type<float> (vector<float> *input)
{
    return Tfloat;
}

template<> inline int read_img_type<unsigned long> (vector<unsigned long> *input)
{
  return ULONG_IMG;
}

template<> inline int read_header_type<unsigned long> (unsigned long *input)
{
  return TULONG;
}

template<> inline ValueType read_table_type<unsigned long> (vector<unsigned long> *input)
{
    return Tulong;
}

template<> inline int read_img_type<long> (vector<long> *input)
{
  return LONG_IMG;
}

template<> inline int read_header_type<long> (long *input)
{
  return TLONG;
}

template<> inline ValueType read_table_type<long> (vector<long> *input)
{
    return Tlong;
}

template<> inline int read_img_type<unsigned int> (vector<unsigned int> *input)
{
  return USHORT_IMG;
}

template<> inline int read_header_type<unsigned short> (unsigned short *input)
{
  return TUSHORT;
}

template<> inline ValueType read_table_type<unsigned short> (vector<unsigned short> *input)
{
    return Tushort;
}

template<> inline int read_img_type<int> (vector<int> *input)
{
  return SHORT_IMG;
}

template<> inline int read_header_type<int> (int *input)
{
  return TINT;
}

template<> inline ValueType read_table_type<int> (vector<int> *input)
{
    return Tint;
}

template<> inline int read_img_type<unsigned char> (vector<unsigned char> *input)
{
  return BYTE_IMG;
}

template<> inline int read_header_type<unsigned char> (unsigned char *input)
{
  return TBYTE;
}

template<> inline ValueType read_table_type<unsigned char> (vector<unsigned char> *input)
{
    return Tbyte;
}

template<> inline int read_img_type<char> (vector<char> *input)
{
  return BYTE_IMG;
}

template<> inline int read_header_type<char> (char *input)
{
  return TBYTE;
}

template<> inline ValueType read_table_type<char> (vector<char> *input)
{
    return Tbyte;
}

template<> inline int read_img_type<bool> (vector<bool> *input)
{
  return BYTE_IMG;
}

template<> inline int read_header_type<bool> (bool *input)
{
  return TBIT;
}
template<> inline ValueType read_table_type<bool> (vector<bool> *input)
{
    return Tbit;
}



template <class T> void voronoi_to_fits(mesh_free_2D *grid, vector<T> *function, string filename, string extension = "", int dim = 512)
{
  int grid_dim = grid->return_grid_size(); 
  //Initial sanity checks
  
  if(grid_dim > function->size())
    {
      throw invalid_argument("MFREE VORO: Input function invalid.");
    }
  
  //Checking for the grid size
  
  gsl_vector *x = gsl_vector_calloc(grid_dim);
  gsl_vector *y = gsl_vector_calloc(grid_dim);
  
  for(int i = 0; i < grid_dim; i++)
    {
      gsl_vector_set(x,i,(*grid)(i,0)); 
      gsl_vector_set(y,i,(*grid)(i,1));
    }
  
  double x_max, x_min, y_max, y_min, x_length, y_length, ratio;
  gsl_vector_minmax(x,&x_min,&x_max);
  gsl_vector_minmax(y,&y_min,&y_max); 
  x_length = x_max - x_min;
  y_length = y_max - y_min;
  ratio = x_length/y_length;
  
  int x_dim, y_dim;
  if(ratio > 1.0)
    {
      x_dim = dim;
      y_dim = floor((double) dim * ratio +0.5);
    }
  else
    {
      y_dim = dim;
      x_dim = floor((double) dim * ratio +0.5);
    }
  double pixel_size = x_length / (double) x_dim;


  //Building the tree for the grid

  vector<double> coordinates;
  for(int i = 0; i < grid_dim; i++)
    {
      coordinates.push_back((*grid)(i,0));
      coordinates.push_back((*grid)(i,1));
    }

  vector<double> query;
  double x_c,y_c;
  
  vector<int> kD_tree;
  vector<double> distances;

  kD_tree.resize(x_dim*y_dim);
  distances.resize(x_dim*y_dim);

  y_c = y_min;
  for(int i = 0; i < y_dim; i++)
    {
      double x_c = x_min;
      for(int j = 0; j < x_dim; j++)
	{
	  query.push_back(x_c);
	  query.push_back(y_c);
	  x_c += pixel_size;
	}
      y_c += pixel_size;
    }

  flann::Matrix<double> flann_dataset(&coordinates[0],grid_dim,2);
  flann::Matrix<double> flann_query(&query[0],x_dim*y_dim,2);
  flann::Matrix<int> flann_tree(&kD_tree[0],x_dim*y_dim,1);
  flann::Matrix<double> flann_distances(&distances[0],x_dim*y_dim,1);
  flann::Index<flann::L2<double> > index(flann_dataset, flann::KDTreeIndexParams(4));
  index.buildIndex();
  index.knnSearch(flann_query, flann_tree, flann_distances, 1, flann::SearchParams(128));

  vector<double> out;
  for(int i = 0; i < x_dim*y_dim; i++)
    {
      out.push_back((*function)[kD_tree[i]]);
    }

  vector<double> WCS;
  WCS.push_back(1.0);
  WCS.push_back(1.0);
  WCS.push_back(x_min);
  WCS.push_back(y_min);
  WCS.push_back(pixel_size);
  WCS.push_back(0.0);
  WCS.push_back(0.0);
  WCS.push_back(pixel_size);

  if(extension == "")
    {
      write_img_to_fits(filename, &out);
      add_WCS_to_fits(filename,WCS,"linear");
    }
  else
    {
      write_img_to_fits(filename, &out,extension);
      add_WCS_to_fits(filename,WCS,"linear",extension);
    }
}



#endif    /*RWFITS_H*/
