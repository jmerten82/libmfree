CUDAROOT	= 	/usr/local/cuda
MFREEROOT	= 	$(HOME)/codes/dev/libmfree
MFREECUDAROOT	=	$(MFREEROOT)/src/cuda
CC		=	/usr/bin/g++
NVCC		=	$(CUDAROOT)/bin/nvcc
NVCCFLAG	=	-Xcompiler '-fPIC -fopenmp' 
CUDALIB		=	$(CUDAROOT)/lib64
CUDAINC 	=	$(CUDAROOT)/include
MFREEINC 	=	$(MFREEROOT)/include
ADDINC		=	-I$(MFREEINC) -I$(CUDAINC)
MFREECUDALINK	=	$(MFREECUDAROOT)/cuda_libmfree_link.o
MFREECUDALIB	=	$(MFREEROOT)/libcudamfree.so

SRC		= 	$(wildcard $(MFREECUDAROOT)/*.cu)
OBJ		= 	$(SRC:$(MFREECUDAROOT)/%.cu=$(MFREECUDAROOT)/%.o)

default: $(SRC)
	$(NVCC) $(NVCCFLAG) -dc $(SRC) $(ADDINC)  
	$(NVCC) $(NVCCFLAG) -dlink -o $(MFREECUDALINK) $(OBJ) 
	$(CC) -shared -o $(MFREECUDALIB) $(OBJ) $(MFREECUDALINK) -L$(CUDALIB) -lcudart
