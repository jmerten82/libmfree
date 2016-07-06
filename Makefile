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


.base: 
	if ! [ -e $(MFREEOBS) ]; then mkdir $(MFREEOBS);  fi; touch $(MFREEOBS)

.PHONY: clean


$(MFREEOBS)/%.o: $(MFREESRC)/%.cpp .base
	$(CC) $(CCFLAGS) -c -o $@  $< $(ADDINC)

$(LIBNAMESTAT): $(OBJ)
	ar rv $(LIBNAMESTAT) $(OBJ)

$(LIBNAMEDYN): $(OBJ)	
	$(CC) $(CCFLAGS) -shared -o $(LIBNAMEDYN) $(OBJ) 	



default: $(LIBNAMESTAT)

all:	 $(LIBNAMESTAT)
install: $(LIBNAMESTAT)
shared: $(LIBNAMEDYN)
clean:	.base
	rm -rf $(MFREEOBS);
	rm -f $(LIBNAMESTAT);
	rm -f $(LIBNAMEDYN)

radial_basis_function.o:
rbf_implementation.o: radial_basis_function.o
mesh_free.o: radial_basis_function.o 
mesh_free_differentiate.o: mesh_free.o
grid_utils.o: mesh_free.o
rwfits.o:
rwhdf5.o:


