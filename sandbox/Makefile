# LIBDIR and INCDIR: Directory where libmfree is going to be installed
# BINDIR: Directory where eventual more complex drivers will be installed
INSTALLLIBDIR	= ${HOME}/lib
INSTALLINCDIR	= ${HOME}/include
INSTALLBINDIR	= ${HOME}/bin
CUDADIR		= /usr/local/cuda

# INCS and LIBS: specify here where to find the external libraries 
INCS		= -I${INSTALLINCDIR}
CUDAINCS	= -I$(CUDADIR)/include -I$(CUDADIR)/samples/common/inc
LIBS 		= -L${INSTALLLIBDIR} -L/usr/lib/atlas-base/ -L/usr/lib/x86_64-linux-gnu/hdf5/serial/
CUDALIBS	= -L$(CUDADIR)/lib64 -L../

# Internal stuff. Fell free to modify if you know what you are doing
LINKS			= -lmfree -lgsl -lgslcblas -lcfitsio -lCCfits -lhdf5 -lhdf5_hl -lhdf5_cpp
CUDALINKS		= -lcudamfree -lcuda -lcudart -lcusolver -lcublas
MFREESRCDIR		= ./drv
CUDASRCDIR		= ./cuda/drv
MFREEINC 		= ../include
MFREEBINDIR		= ./bin
CUDABINDIR		= ./cuda/bin
CC 			= g++
NVCC			= $(CUDADIR)/bin/nvcc
CCFLAGS			= -fPIC -fopenmp
NVCCFLAGS		= -Xcompiler '-fPIC -fopenmp'	
GSLINC 			= /usr/local/include
CCFITSINC		= /usr/local/include
HDF5INC			= /usr/local/include
ADDINC 			= -I$(MFREEINC) -I$(GSLINC) -I$(CFITSIOINC) -I$(CCFITSINC) -I$(HDF5INC) $(INCS) $(CUDAINCS)
SRC 			= $(wildcard $(MFREESRCDIR)/*.cpp)
CUDASRC 		= $(wildcard ${CUDASRCDIR}/*.cu)
BIN			= $(SRC:$(MFREESRCDIR)/%.cpp=$(MFREEBINDIR)/%)	
CUDABIN 		= $(CUDASRC:$(CUDASRCDIR)/%.cu=$(CUDABINDIR)/%)

.base: 
	if ! [ -e $(MFREEBINDIR) ]; then mkdir $(MFREEBINDIR);  fi; touch $(MFREEBINDIR);
	if ! [ -e $(CUDABINDIR) ]; then mkdir $(CUDABINDIR);  fi; touch $(CUDABINDIR);

.PHONY: clean

$(MFREEBINDIR)/%: $(MFREESRCDIR)/%.cpp .base
	$(CC) $(CCFLAGS) -o $@  $< $(ADDINC) ${LINKS}

$(CUDABINDIR)/%: $(CUDASRCDIR)/%.cu .base
	$(NVCC) $(NVCCFLAGS) -o $@  $< $(DIRTYHACK) $(MFREECUDALIB) $(ADDINC) ${CUDALIBS} $(CUDALINKS) ${LINKS} $(LIBS)

sandbox: ${BIN}

cuda: ${CUDABIN}

default: sandbox
