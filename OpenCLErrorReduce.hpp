/* 
 * File:   MatrixMultiplication.hpp
 * Author: jdelatorre
 *
 * Created on 23 de octubre de 2014, 10:25
 */

#ifndef ERRORREDUCE_HPP
#define	ERRORREDUCE_HPP

#define __CL_ENABLE_EXCEPTIONS  // enable use of exceptions of the OpenCL API

#include <string>
#include <fstream>

#include "CL/cl.hpp"

#include "common.hpp"


class OpenCLErrorReduce {
public:
    inline OpenCLErrorReduce(
                            const cl::Context & c, 
                            const std::vector<cl::Device> & d,
                            const cl::CommandQueue & q,
                            matrix const & Y,
                            matrix const & T) 
                            : context(c), devices(d), queue(q), y(Y), t(T) {
        opencl_initialize();
    };
    virtual ~OpenCLErrorReduce();
    cl_float run();
private:
    const cl::Context & context;
    const std::vector<cl::Device> & devices;
    const cl::CommandQueue & queue;
    
    size_t global_size[1];
    size_t local_size[1];
    
    matrix const & y;
    matrix const & t;
    
    matrix error;
    
    cl::Program *program;    
    cl::Kernel *kernel;
    bool lds;
    
    inline void readfile(const std::string &filepath, std::string &buffer) {
        std::ifstream fin(filepath.c_str());
        getline(fin, buffer, char(-1));
        fin.close();
    };
    
    void opencl_initialize();
    
    cl_float run_CE_Kernel_local();

};

#endif	/* MATRIXMULTIPLICATION_HPP */

