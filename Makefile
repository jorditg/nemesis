CC=g++
CFLAGS=-g --std=c++11 -Wall
LIBFLAGS=-lOpenCL

HEADERS=nn.hpp OpenCLKernels.hpp common.hpp 
SOURCES=main.cpp nn.cpp OpenCLKernels.cpp common.cpp 
EXECUTABLE=nn-opencl

all: $(EXECUTABLE)

nn-opencl: $(HEADERS) $(SOURCES) Makefile
		$(CC) $(CFLAGS) $(SOURCES) $(LIBFLAGS) -o$(EXECUTABLE)

clean:
	rm -f *.o *~ $(EXECUTABLE)

