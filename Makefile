CC=g++
CFLAGS=-g --std=c++11 -Wall
LIBFLAGS=-lOpenCL

HEADERS=basic.hpp nn.hpp oclobject.hpp
SOURCES=basic.cpp main.cpp nn.cpp oclobject.cpp
EXECUTABLE=nn-opencl

all: $(EXECUTABLE)

nn-opencl: $(HEADERS) $(SOURCES) Makefile
		$(CC) $(CFLAGS) $(SOURCES) $(LIBFLAGS) -o$(EXECUTABLE)

clean:
	rm -f *.o *~ $(EXECUTABLE)

