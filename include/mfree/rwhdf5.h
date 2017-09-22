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

#ifndef    RWHDF5_H
#define    RWHDF5_H

#include <vector>
#include <stdexcept>
#include <cmath>
#include <hdf5.h>
#include <mfree/mesh_free.h>

/*
  This writes all necessary data of a mesh free structure
  into an HDF5 file. 
*/

void write_mfree_to_hdf5(mesh_free *input, string filename);

/*
  Reads a meshfree HDF5 file and returns a new unstructured grid;
*/

void  read_mfree_from_hdf5(string filename, mesh_free *output);





#endif    /*RWHDF5_H*/
