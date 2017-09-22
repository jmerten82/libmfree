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

  hid_t file_id, dataset_id, attribute_id, attribute_type, dataspace_id;
  herr_t status;

  hsize_t dims[2];

  dims[0] = input->return_grid_size();
  dims[1] = input->return_grid_size(1);

  file_id = H5Fcreate(filename.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
  attribute_type = H5Tcopy(H5T_NATIVE_INT);
  dataspace_id = H5Screate(H5S_SCALAR);

  attribute_id = H5Acreate2 (file_id, "Dim",attribute_type, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id,attribute_type,&dims[1]);
  status = H5Aclose(attribute_id);
  status = H5Sclose(dataspace_id);
  dataspace_id = H5Screate(H5S_SCALAR);
  attribute_id = H5Acreate2 (file_id, "Number of nodes",attribute_type, dataspace_id, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Awrite(attribute_id,attribute_type,&dims[0]);
  status = H5Aclose(attribute_id);
  status = H5Sclose(dataspace_id);

  vector<double> temp;

  for(int i = 0; i < dims[0]; i++)
    {
      for(int j = 0; j < dims[1]; j++)
	{ 
	  temp.push_back((* input)(i,j));
	}
    }
  dataspace_id = H5Screate_simple(2, dims, NULL);
  dataset_id = H5Dcreate2(file_id, "coordinates", H5T_NATIVE_DOUBLE, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,&temp[0]);
  status = H5Dclose(dataset_id);
  status = H5Sclose(dataspace_id);
  status = H5Fclose(file_id);
}


void read_mfree_from_hdf5(string filename, mesh_free *output)
{
  //WORK TO BE DONE

}
