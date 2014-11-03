CC=g++
CFLAGS=-g --std=c++11 -Wall
LIBFLAGS=-lOpenCL

HEADERS=nn.hpp OpenCLMatrixMultiplication.hpp common.hpp OpenCLErrorReduce.hpp
SOURCES=main.cpp nn.cpp OpenCLMatrixMultiplication.cpp OpenCLErrorReduce.cpp
EXECUTABLE=nn-opencl

all: $(EXECUTABLE)

nn-opencl: $(HEADERS) $(SOURCES) Makefile
		$(CC) $(CFLAGS) $(SOURCES) $(LIBFLAGS) -o$(EXECUTABLE)

clean:
	rm -f *.o *~ $(EXECUTABLE)

