/*** rwhdf5.h
     These simple routines are meant as helpers for the 
     mfree classes for HDF5 access. It can write mesh free
     structures to HDF5 and read them from the same files.

     Julian Merten
     University of Oxford
     Feb 2015
     julian.merten@physics.ox.ac.uk
     http://www.julianmerten.net
***/

#include <mfree/rwhdf5.h>

void write_mfree_to_hdf5(mesh_free *input, string filename)
{

  //Creating HDF5 file
  H5File file(filename, H5F_ACC_TRUNC);

  //Getting information about the grid
  hsize_t dims[2];

  dims[1] = input->return_grid_size();
  dims[0] = input->return_grid_size(1);

  //Creating HDF5 dataspace for the grid coordinates
  DataSpace dataspace(2,dims);

  //And linking it to a dataset
  DataSet dataset = file.createDataSet("Coordinates", H5T_NATIVE_DOUBLE, dataspace);

  //Reading out the nodes
  vector<double> temp;

  for(int i = 0; i < dims[1]; i++)
    {
      for(int j = 0; j < dims[0]; j++)
	{ 
	  temp.push_back((* input)(i,j));
	}
    }

  //Writing coordinates into dataset
  dataset.write(&input[0], H5T_NATIVE_DOUBLE);

  //Finally adding some attributes to the dataset
  hsize_t dim[1] = {2};  
  DataSpace attr_dataspace1 = DataSpace (1, dim);
  DataSpace attr_dataspace2 = DataSpace (1, dim);
  Attribute attribute1 = dataset.createAttribute("DIM", H5T_NATIVE_INT,attr_dataspace1);
  Attribute attribute2 = dataset.createAttribute("#Nodes", H5T_NATIVE_INT,attr_dataspace2);
  attribute1.write(H5T_NATIVE_INT, &dims[0]);
  attribute2.write(H5T_NATIVE_INT, &dims[1]);

  //No need to close dataspaces and sets in a C++ implementation.

}


void read_mfree_from_hdf5(string filename, mesh_free *output)
{
  //WORK TO BE DONE

}
