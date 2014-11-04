/*
 * File:   MatrixMultiplication.hpp
 * Author: jdelatorre
 *
 * Created on 23 de octubre de 2014, 10:25
 */

#define __CL_ENABLE_EXCEPTIONS  // enable use of exceptions of the OpenCL API


#ifndef MATRIXMULTIPLICATION_HPP
#define MATRIXMULTIPLICATION_HPP

#include <string>
#include <fstream>

#include "CL/cl.hpp"

#include "common.hpp"


class OpenCLMatrixMultiplication {
 public:
    inline OpenCLMatrixMultiplication(
                            const cl::Context & c,
                            const std::vector<cl::Device> &d,
                            const int d_id,
                            const cl::CommandQueue & q)
                            : context(c), devices(d), device_id(d_id), queue(q) {
        opencl_select_kernel();
    };

    virtual ~OpenCLMatrixMultiplication();
    void run(matrix const &A,
             matrix const &B,
             matrix const &result);
  
 private:
    const cl::Context & context;
    const std::vector<cl::Device> & devices;
    const int device_id;
    const cl::CommandQueue & queue;
    
    cl::Program *program;
    cl::Kernel *kernel;
    bool lds;
    
    inline void readfile(const std::string &filepath, std::string &buffer) {
        std::ifstream fin(filepath.c_str());
        getline(fin, buffer, char(-1));
        fin.close();
    };
    
    void opencl_select_kernel();
    
    void run_mmmKernel_local(matrix const &A,
                             matrix const &B,
                             matrix const &result);
    void run_mmmKernel(matrix const &A,
                       matrix const &B,
                       matrix const &result);
};

#endif  /* MATRIXMULTIPLICATION_HPP */

