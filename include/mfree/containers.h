/*** containers.h
     This collection of classes provides typical 
     data structures for the use in SaWLens.

Julian Merten
JPL/Caltech
May 2014
jmerten@caltech.edu
***/

#ifndef    CONTAINERS_H
#define    CONTAINERS_H

#include <vector>
#include <stdexcept>

using namespace std;

/**
     This class provides a simple framework to save, 
     manipulate and access information about
     multiple image system.
**/

class multiple_image_system
{

 protected:

  /*
    The source redshift of the system.
  */

  double redshift;

  /*
    The positional error on the images of the system.
  */

  double sigma;

  /*
    The number of images in the system.
  */

  int num_sys;

  /*
    The coordinates of all the images in the system. The size 
    of this object is num_sys * 2, since both the x and the y coordinate
    are saved consecutively.
  */

  vector<double> image_coordinates;

  /*
    The parity of each of the systems. This is a vector of length num_sys.
    It returns true if a system is even and false if it is odd.
  */

  vector<bool> image_parities;

  /*
    Holds potentially assigned grid indices for each image. These need to 
    be set manually.
  */

  vector<int> grid_indices;

 public:

  /*
    Standard constructor. Just needs the redshift and a typical 
    positional error of the system.
  */

  multiple_image_system(double redshift, double sigma);

  /*
    Standard destructor. Does not much.
  */

  ~multiple_image_system();

  /*
    Adds one or several images to the system. If you want to add
    only one system, the length of the input vector should be two 
    and contains the two coordinates of the image. If you want to add more
    the length should be four, six...etc. 
  */

  void add(vector<double> coordinates);
  /*
    This version of the add function lets you also set the parity of the 
    newly added system. Length of the first vector is 2*number_of_added_sys. 
    The length of the parity vector is numbe_of_added_sys.
  */

  void add(vector<double> coordinates, vector<bool> parities);

  /*
    This version adds system and their respective grid indices.
  */

  void add(vector<double> coordinates, vector<int> indices);

  /*
    This version adds images, indices and parities.
  */

  void add(vector<double> coordinates, vector<int> indices, vector<bool> parities);

  /*
    The bracket empty bracket operator returns the number of multiple images
    in the system.
  */

  int operator() ();

  /*
    The bracket operator with one argument returns the grid index of the 
    i-th image in the system. If it is not assigned it returns -1.
  */

  int operator() (int i);

  /*
    The bracket operator with two arguments returns the j-th coordinate 
    component of the i-th image in the system.
  */

  double operator() (int i, int j);

  /*
    Returns the redshift of the system.
  */

  double get_redshift();

  /*
    Returns the positional error on image positions.
  */

  double get_sigma();

  /*
    Sets the parity of the i-th image.
  */

  void set_parity(int i, bool parity); 

  /*
    Returns the parity of the i-th image.
  */

  bool get_parity(int i); 

  /*
    Assigns grid indices to each image in the system. 
  */

  void assign_indices(vector<int> index_in);

};

/**
   This class describes a galaxy in the SaWLens framework. It holds
   its position and the typical measurements connected to it.
**/

class galaxy
{

 protected:

  /*
    The x position of the galaxy.
  */

  double x;

  /*
    The y position of the galaxy.
  */

  double y;

  /*
    The redshift of the galaxy.
  */

  double z;

  /*
    The first component of the shear1.
  */

  double g1;

  /*
    The second component of the shear.
  */

  double g2;

  /*
    The first compoment of the F flexion.
  */

  double F1;

  /*
    The second component of the F flexion.
  */

  double F2;

  /*
    The first compoment of the G flexion.
  */

  double G1;

  /*
    The second component of the G flexion.
  */

  double G2;

  /*
    This is a index value, which allows each galaxy to carry an integer 
    identifier.
  */

  int index;

 public:

  /*
    Standard constructor, initialising the galaxy with a position.
  */

  galaxy(double x, double y);

  /*
    Another constructor that lets you also initialise the shear and redshift
    information.
  */

  galaxy(double x, double y, double g1, double g2);

  /*
    Standard destructor.
  */

  ~galaxy();

  /*
    Lets you set certain properties of the galaxy. 
    Selections are
    x
    y
    z
    g1
    g2
    F1
    F2
    G1
    G2
    index
  */

  void set(string selection, double value);

/*
    Returns certain properties of the galaxy. 
    Selections are
    x
    y
    z
    g1
    g2
    F1
    F2
    G1
    G2
  */

  double get(string selection);

  /*
    The bracket operator returns the index of the galaxy.
  */

  int operator() ();
};

/**
   Finally this class describes a critical line estimator.
**/

class ccurve_estimator
{

 protected:

  /*
    The x position of the estimator.
  */

  double x;

  /*
    The y position of the estimator.
  */

  double y;

  /*
    The redshift of the estimator.
  */

  double z;

  /*
    A general index of the estimator.
  */

  double index;

  public:

  /*
    `The standard constructor of the class. Taking the two positions
    and the redshift of the estimator.
  */

  ccurve_estimator(double x, double y, double z);


  /*
    Sets a certain property of the estimator.
    Selections are:
    x
    y
    z
    index
  */

  void set(string selection, double value);

  /*
    Sets a certain property of the estimator.
    Selections are:
    x
    y
    z
    index
  */

  double get(string selection);

  /*
    The bracket operator return the index of the estimator.
  */

  int operator() ();

};



#endif    /*CONTAINERS_H*/
