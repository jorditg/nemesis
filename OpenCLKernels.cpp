/* 
 * File:   OpenCLKernels.cpp
 * Author: jdelatorre
 * 
 * Created on 23 de octubre de 2014, 10:25
 */

#include <string>
#include <iostream>

#include <algorithm>    // std::min

#include "OpenCLKernels.hpp"
#include "common.hpp"

OpenCLKernels::~OpenCLKernels() {    
    delete crossEntropyKernelLocal;
    delete elementWiseSubstractKernel;
    delete matrixMultiplicationSigmoidKernel;
    delete program;
}

void OpenCLKernels::opencl_init() {
    // create a CL program using kernel source
    std::string sourceString;
    readfile(sourceFile, sourceString);
    
    cl::Program::Sources sources;
    sources.push_back(std::make_pair(sourceString.c_str(), 0));
    // don't need to specify length as we used a null terminated string

    // create the OpenCL program
    program = new cl::Program(context, sources);
    
    try {
        program->build(devices);
    } catch (const cl::Error &e) {
        // get compilation log in case of failure
     std::cout << "Build Status: "
        << program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(devices[device_id])
        << std::endl;
     std::cout << "Build Options:\t"
        << program->getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(devices[device_id])
        << std::endl;
     std::cout << "Build Log:\t "
        << program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[device_id])
        << std::endl;
    }
    
    lds = true;
    try {
      matrixMultiplicationSigmoidKernel = 
            new cl::Kernel(*program, 
                           matrixMultiplicationSigmoidKernel_name);
      elementWiseSubstractKernel =
            new cl::Kernel(*program,
                           elementWiseSubstractKernel_name);
      crossEntropyKernelLocal =
            new cl::Kernel(*program,
                           crossEntropyKernelLocal_name);
      
    } catch (cl::Error &e) {
        std::cout << e.err() << e.what() << std::endl;
    }
}

void OpenCLKernels::
     runMatrixMultiplicationSigmoid(matrix_cl_float const &A,              
                                    matrix_cl_float const &B,
                                    matrix_cl_float const &C,
                                    bool setBias) {
    // It's correct, cols and rows are in this order
    const size_t global_size[2] = {size_t(C.cols/4),
                                   size_t(C.rows/4)};
    
    // Proposed block size = 8
    size_t blockSize_r = 8, blockSize_c = 8;

    blockSize_c = std::min(global_size[0], blockSize_c/4);
    blockSize_r = std::min(global_size[1], blockSize_r/4);
    
    // if global size < block size the reduce block size to global size
    
    
    // float4 elements in kernel
    const size_t local_size[2] = { blockSize_c, blockSize_r };

    // -----------------------------------------------------------------------
    // Setting kernel arguments
    // -----------------------------------------------------------------------    
    matrixMultiplicationSigmoidKernel->setArg(0, *(A.data.deviceData));
    matrixMultiplicationSigmoidKernel->setArg(1, *(B.data.deviceData));
    matrixMultiplicationSigmoidKernel->setArg(2, *(C.data.deviceData));
    matrixMultiplicationSigmoidKernel->setArg(3, A.cols);
    matrixMultiplicationSigmoidKernel->setArg(4, A.offset);
    matrixMultiplicationSigmoidKernel->setArg(5, B.offset);
    matrixMultiplicationSigmoidKernel->setArg(6, C.offset);
    matrixMultiplicationSigmoidKernel->setArg(7, cl::__local((blockSize_c*4)*(blockSize_r*4)*sizeof(cl_float)));
    matrixMultiplicationSigmoidKernel->setArg(8, setBias?1:0);
    matrixMultiplicationSigmoidKernel->setArg(9, 1);    // calculate sigmoid after matrix multiplication

    // -----------------------------------------------------------------------
    // Define ndrange iteration space: global and local sizes based on
    // parameters obtained from user

    // Refer to the sample documentation for clarification about
    // how work is devided among work-groups and work-items.
    // -----------------------------------------------------------------------
  
    std::cout << "Launching for device\n"
              << " (global size: " << global_size[0]
              << ", "  << global_size[1] << ")\n"
              << " (local size: " << local_size[0]
              << ", "  << local_size[1] << ")\n";
    
    const cl::NDRange offset = cl::NullRange;
    const cl::NDRange global(global_size[0], global_size[1]);
    const cl::NDRange local(local_size[0], local_size[1]);
    queue.enqueueNDRangeKernel(*matrixMultiplicationSigmoidKernel, offset, global, local);
    queue.finish();

    std::cout << "Matmult finished\n";
}

// NOT TESTED YET
void OpenCLKernels::runElementWiseSubstract(
            matrix_cl_float const &tm,              
            matrix_cl_float const &ym,
            matrix_cl_float const &em) {

    em.cols = tm.cols;
    em.rows = tm.rows;
    
    const size_t blockSize = 512;  // float4's
    const size_t data_size_float4_global = ym.rows*ym.cols/4;
    
    int global_size[1] = {data_size_float4_global};
    int local_size[1] = {std::min(blockSize, global_size[0])};
    
    assert(global_size[0] % local_size[0] == 0);
    
    std::cout << "Launching for device\n"
              << " (global size: " << global_size[0] << ")\n"
              << " ( local size: " << local_size[0] << ")\n";

    elementWiseSubstractKernel->setArg(0, *(tm.data.deviceData));
    elementWiseSubstractKernel->setArg(1, *(ym.data.deviceData));
    elementWiseSubstractKernel->setArg(2, *(em.data.deviceData));
    
    const cl::NDRange offset = cl::NullRange;
    const cl::NDRange global(global_size[0]);
    const cl::NDRange local(local_size[0]);
    queue.enqueueNDRangeKernel(*elementWiseSubstractKernel, offset, global, local);
    queue.finish();
}

// NOT TESTED YET
cl_float OpenCLKernels::runCrossEntropy(matrix_cl_float const &t, 
                                        matrix_cl_float const &y, 
                                        matrix_cl_float &error) {

    const size_t blockSize = 4096;  // float4's
    const size_t data_size_float4_global = y.rows*y.cols/4;
    
    int global_size[1] = {data_size_float4_global / 2};
    int local_size[1] = {std::min(blockSize, global_size[0])};

    assert(global_size[0] % local_size[0] == 0);
    
    const size_t error_size = 4 * global_size[0]/local_size[0];
    
    error.rows = 1;
    error.cols = error_size;
    
    // -----------------------------------------------------------------------
    // Setting kernel arguments
    // -----------------------------------------------------------------------    
    crossEntropyKernelLocal->setArg(0, t.data.deviceData);
    crossEntropyKernelLocal->setArg(1, y.data.deviceData);
    crossEntropyKernelLocal->setArg(2, error.data.deviceData);
    crossEntropyKernelLocal->setArg(4, cl::__local(local_size[0] * 4 * sizeof(cl_float)));

    // -----------------------------------------------------------------------
    // Define ndrange iteration space: global and local sizes based on
    // parameters obtained from user

    // Refer to the sample documentation for clarification about
    // how work is devided among work-groups and work-items.
    // -----------------------------------------------------------------------
  
    std::cout << "Launching for device\n"
            << " (global size: " << global_size[0] << ")\n"
            << " (local size: " << local_size[0] << ")\n";
    
    const cl::NDRange offset = cl::NullRange;
    const cl::NDRange global(global_size[0]);
    const cl::NDRange local(local_size[0]);
    queue.enqueueNDRangeKernel(*crossEntropyKernelLocal, offset, global, local);
    queue.finish();

    std::cout << "CE kernel finished\n";
    
    error.data.readFromDevice(queue);
    
    std::vector<cl_float> & e = error.data.hostData;
    cl_float ce = 0.0;
    for (size_t i = 0; i < error_size; i++) {
        ce += e[i];
    }
    
    return ce;
}
