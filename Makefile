LIBNAMESTAT 	= libmfree.a
LIBNAMEDYN 	= libmfree.so
MFREESRC	= ./src
MFREEOBS	= ./obs
MFREEINC 	= ./include/	
DRIVER 		= 
CC 		= g++
CCFLAGS		= -fPIC
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
unstructured_grid.o: radial_basis_function.o
grid_utils.o: unstructured_grid.o
containers.o:
rwfits.o:
rwhdf5.o:


