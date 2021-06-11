CUDA_HOME  ?= /Soft/cuda/11.2.1

NVCC        = $(CUDA_HOME)/bin/nvcc
NVCC_FLAGS  = -O3 -Wno-deprecated-gpu-targets -I$(CUDA_HOME)/include -arch=compute_35 --ptxas-options=-v -I$(CUDA_HOME)/sdk/CUDALibraries/common/inc -g
LD_FLAGS    = -lcudart -Xlinker -rpath,$(CUDA_HOME)/lib64 -I$(CUDA_HOME)/sdk/CUDALibraries/common/lib
EXE	    = main.exe
OBJ	    = main.o

CC 			= gcc
CC_FLAGS 	= -lm -g

default: all

all: $(EXE)

main.o: main.cu
	$(NVCC) -c -o $@ $+ $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $+ -o $@ $(LD_FLAGS)

clean:
	rm -f *.o $(EXE)
