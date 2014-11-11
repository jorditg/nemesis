/* 
 * File:   MatrixMultiplication.cpp
 * Author: jdelatorre
 * 
 * Created on 23 de octubre de 2014, 10:25
 */

#include <string>
#include <iostream>

#include <algorithm>    // std::min

#include "OpenCLMatrixMultiplication.hpp"

OpenCLMatrixMultiplication::~OpenCLMatrixMultiplication() {
    delete kernel;
    delete program;
}

void OpenCLMatrixMultiplication::opencl_select_kernel() {
    // create a CL program using kernel source
    std::string sourceFile = "MatrixMultiplication_Kernels.cl";
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
    
//    if (devices[device_id].getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() > 1024) {
        lds = true;
        try {
          kernel = new cl::Kernel(*program, "mmmKernel_local");
        } catch (cl::Error &e) {
            std::cout << e.err() << e.what() << std::endl;
        }
//    } else {
//        lds = false;
//        kernel = new cl::Kernel(*program, "mmmKernel");
//    }
}

void OpenCLMatrixMultiplication::
     run_mmmKernel_local(matrix_cl_float const &A,
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
    kernel->setArg(0, *(A.data.deviceData));
    kernel->setArg(1, *(B.data.deviceData));
    kernel->setArg(2, *(C.data.deviceData));
    kernel->setArg(3, A.cols);
    kernel->setArg(4, A.offset);
    kernel->setArg(5, B.offset);
    kernel->setArg(6, C.offset);
    kernel->setArg(7, cl::__local((blockSize_c*4)*(blockSize_r*4)*sizeof(cl_float)));
    kernel->setArg(8, setBias?1:0);

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
    queue.enqueueNDRangeKernel(*kernel, offset, global, local);
    queue.finish();

    std::cout << "Matmult finished\n";
}

//void OpenCLMatrixMultiplication::
//     run_mmmKernel(matrix_cl_float const &A,
//                   matrix_cl_float const &B,
//                   matrix_cl_float const &C) {
//    const size_t blockSize_r = 8, blockSize_c = 8;
//    
//    const size_t global_size[2] = {size_t(C.cols/4),
//                                   size_t(C.rows/4)};
//    // float4 elements in kernel
//
//    const size_t local_size[2] = {blockSize_c, blockSize_r};
//
//    // -----------------------------------------------------------------------
//    // Setting kernel arguments
//    // -----------------------------------------------------------------------
//    kernel->setArg(0, *(A.data.deviceData));
//    kernel->setArg(1, *(B.data.deviceData));
//    kernel->setArg(2, *(C.data.deviceData));
//    kernel->setArg(3, A.cols);
//    kernel->setArg(4, B.cols);
//    kernel->setArg(5, A.offset);
//    kernel->setArg(6, B.offset);
//    kernel->setArg(7, C.offset);
//    
//    // -----------------------------------------------------------------------
//    // Define ndrange iteration space: global and local sizes based on
//    // parameters obtained from user.
//
//    // Refer to the sample documentation for clarification about
//    // how work is devided among work-groups and work-items.
//    // -----------------------------------------------------------------------
//      
//    std::cout << "Launching for device\n"
//              << " (global size: " << global_size[0]
//              << ", "  << global_size[1] << ")\n"
//              << " (local size: " << local_size[0]
//              << ", "  << local_size[1] << ")\n";
//    
//    const cl::NDRange offset = cl::NullRange;
//    const cl::NDRange global(global_size[0], global_size[1]);
//    const cl::NDRange local(local_size[0], local_size[1]);
//    queue.enqueueNDRangeKernel(*kernel, offset, global, local);
//    queue.finish();
//
//    std::cout << "Matmult finished\n";
//}

void OpenCLMatrixMultiplication::run(matrix_cl_float const &A,              
                                     matrix_cl_float const &B,
                                     matrix_cl_float const &C,
                                     bool setBias) {
//    if (lds) {
        run_mmmKernel_local(A, B, C, setBias);
//    } else {
//        run_mmmKernel(A, B, C);
//    }
}

