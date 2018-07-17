/*** cuda_manager.cpp
     This tests the central CUDA manager class, which 
     manages all devices and their memory for libmfree.

Julian Merten
INAF OA Bologna
Jul 2018
julian.merten@oabo.inaf.it
http://www.julianmerten.net
***/

#include <iostream>
#include <omp.h>
#include <mfree/cuda/cuda_manager.h>

using namespace std;

int main()
{

  //Constructing the manager
  cuda_manager man1(2.0,7.5);

  if(man1.check())
    {
      cout <<"libmfree can run on your system." <<endl;
    }
  else
    {
      cout <<"libfmree cannot run on your system." <<endl;
    }

  man1.report(true);

  //Test all other methods of the class. 


  return 0;
}
