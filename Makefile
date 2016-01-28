LIBNAMESTAT 	= libmfree.a
LIBNAMEDYN 	= libmfree.so
MFREESRC	= ./src
MFREEOBS	= ./obs
MFREEINC 	= ./include/	
DRIVER 		= 
CC 		= g++
CCFLAGS		= -fPIC -frounding-math
GSLINC 		= /usr/local/include
CCFITSINC	= /usr/local/include
HDF5INC		= /usr/local/include
ADDINC 		= -I$(MFREEINC) -I$(GSLINC) -I$(CFITSIOINC) -I$(CCFITSINC) -I$(HDF5INC)
SRC 		= $(wildcard $(MFREESRC)/*.cpp)
OBJ		= $(SRC:$(MFREESRC)/%.cpp=$(MFREEOBS)/%.o)


.PHONY: clean distclean doc	


$(MFREEOBS)/%.o: $(MFREESRC)/%.cpp 
	$(CC) $(CCFLAGS) -c -o $@  $< $(ADDINC)

$(LIBNAMESTAT): $(OBJ)
	ar rv $(LIBNAMESTAT) $(OBJ)

$(LIBNAMEDYN): $(OBJ)	
	$(CC) $(CCFLAGS) -shared -o $(LIBNAMEDYN) $(OBJ) 	



default: $(LIBNAMESTAT)

all:	 $(LIBNAMESTAT)
install: $(LIBNAMESTAT)
shared: $(LIBNAMEDYN)

radial_basis_function.o:
rbf_implementation.o: radial_basis_function.o
mesh_free.o: radial_basis_function.o 
mesh_free_differentiate.o: mesh_free.o
grid_utils.o: mesh_free.o
rwfits.o:
rwhdf5.o:


