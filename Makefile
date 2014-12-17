CC=g++
CFLAGS=-g --std=c++11 -Wall
LIBFLAGS=-lOpenCL -pthread

HEADERS=nn.hpp OpenCLKernels.hpp common.hpp mg.hpp mnist.hpp
SOURCES=main.cpp nn.cpp OpenCLKernels.cpp common.cpp mg.cpp mnist.cpp
EXECUTABLE=nn-opencl

all: $(EXECUTABLE)

nn-opencl: $(HEADERS) $(SOURCES) Makefile
		$(CC) $(CFLAGS) $(SOURCES) $(LIBFLAGS) -o$(EXECUTABLE)

clean:
	rm -f *.o *~ $(EXECUTABLE)

