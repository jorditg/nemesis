/*
 * File:   OpenCLKernels.hpp
 * Author: jdelatorre
 *
 * Created on 23 de octubre de 2014, 10:25
 */

#define __CL_ENABLE_EXCEPTIONS  // enable use of exceptions of the OpenCL API


#ifndef OPENCLKERNELS_HPP
#define OPENCLKERNELS_HPP

#include <string>
#include <fstream>

#include "CL/cl.hpp"

#include "common.hpp"


class OpenCLKernels {
 public:
    inline OpenCLKernels(
            const cl::Context & c,
            const std::vector<cl::Device> &d,
            const int d_id,
            const cl::CommandQueue & q)
            : context(c), devices(d), device_id(d_id), queue(q) {
        opencl_init();
    };

    virtual ~OpenCLKernels();
    
    void runMatrixMultiplicationSigmoid(
            matrix_cl_float const &A,              
            matrix_cl_float const &B,
            matrix_cl_float const &C,
            bool setBias = true, 
            bool calcSigmoid = true);
    
    void runElementWiseSubstract(
            matrix_cl_float const &t,              
            matrix_cl_float const &y,
            matrix_cl_float &e);    
    
    cl_float runCrossEntropy(
            matrix_cl_float const &t, 
            matrix_cl_float const &y, 
            matrix_cl_float &error);
    
    void runTranspose(
            matrix_cl_float const &a,
            matrix_cl_float &transpose);
    
    void runElementWiseMultiplicationBySigmoidDerivativeKernel(
            matrix_cl_float const &deltas,              
            matrix_cl_float const &activations);
 private:
    const std::string sourceFile = "NN_Kernels.cl";
    
    const cl::Context & context;
    const std::vector<cl::Device> & devices;
    const int device_id;
    const cl::CommandQueue & queue;
    
    cl::Program *program;
    
    // kernels
    
    cl::Kernel *matrixMultiplicationSigmoidKernel;
    const std::string matrixMultiplicationSigmoidKernel_name = 
                      "matrixMultiplicationSigmoidKernelLocal";
    
    cl::Kernel *elementWiseSubstractKernel;
    const std::string elementWiseSubstractKernel_name =
                      "elementWiseSubstractKernel";
    
    cl::Kernel *crossEntropyKernelLocal;
    const std::string crossEntropyKernelLocal_name =
                      "crossEntropyKernelLocal";
    
    cl::Kernel *transposeKernelLocal;
    const std::string transposeKernelLocal_name =
                      "transpose";
    
    cl::Kernel *elementWiseMultiplicationBySigmoidDerivativeKernel;
    const std::string elementWiseMultiplicationBySigmoidDerivativeKernel_name =
                      "elementWiseMultiplicationBySigmoidDerivativeKernel";
    
    bool lds;
    
    inline void readfile(const std::string &filepath, std::string &buffer) {
        std::ifstream fin(filepath.c_str());
        getline(fin, buffer, char(-1));
        fin.close();
    };
    
    void opencl_init();
    
};

#endif  /* MATRIXMULTIPLICATION_HPP */

