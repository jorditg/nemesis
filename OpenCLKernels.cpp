/* 
 * File:   OpenCLKernels.cpp
 * Author: jdelatorre
 * 
 * Created on 23 de octubre de 2014, 10:25
 */

#include <string>
#include <vector>
#include <iostream>
#include <boost/math/common_factor.hpp>

#include "NN_Kernels.h"
#include "OpenCLKernels.hpp"
#include "common.hpp"

OpenCLKernels::~OpenCLKernels() {
    delete elementWiseMultiplicationBySigmoidDerivativeKernel;
//    delete transposeKernelLocal;
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
    } catch(const cl::Error &e) {
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
                           matrixMultiplicationSigmoidKernel_name.c_str());
      elementWiseSubstractKernel =
            new cl::Kernel(*program,
                           elementWiseSubstractKernel_name.c_str());
      crossEntropyKernelLocal =
            new cl::Kernel(*program,
                           crossEntropyKernelLocal_name.c_str());

//      transposeKernelLocal =
//            new cl::Kernel(*program,
//                           transposeKernelLocal_name.c_str());
      
      elementWiseMultiplicationBySigmoidDerivativeKernel =
            new cl::Kernel(*program,
                           elementWiseMultiplicationBySigmoidDerivativeKernel_name.c_str());
      
    } catch(const cl::Error &e) {
        std::cout << e.err() << e.what() << std::endl;
    }
}

void OpenCLKernels::
     runMatrixMultiplicationSigmoid(matrix_cl_float const &A,
                                    matrix_cl_float const &B,
                                    matrix_cl_float const &C,
                                    bool setBias,
                                    bool calcSigmoid,
                                    bool sumToC,
                                    cl_float multTheSum) {
    // It's correct, cols and rows are in this order
    const size_t global_size[2] = {size_t(C.cols/4),
                                   size_t(C.rows/4)};
    
    assert(C.rows == A.rows && C.cols == B.cols && A.cols == B.rows);
    
    // Proposed block size = 8
    size_t blockSize_r = 8, blockSize_c = 8;

    // ??¿¿esto creo que no está bien!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    blockSize_c = boost::math::gcd(global_size[0], blockSize_c);
    blockSize_r = boost::math::gcd(global_size[1], blockSize_r);
    
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
    matrixMultiplicationSigmoidKernel->setArg(4, A.offset/4);
    matrixMultiplicationSigmoidKernel->setArg(5, B.offset/4);
    matrixMultiplicationSigmoidKernel->setArg(6, C.offset/4);
    matrixMultiplicationSigmoidKernel->setArg(7,
          cl::Local((blockSize_c*4)*(blockSize_r*4)*sizeof(cl_float)));
    matrixMultiplicationSigmoidKernel->setArg(8, setBias?1:0);
    matrixMultiplicationSigmoidKernel->setArg(9,
          calcSigmoid?1:0);    // calculate sigmoid after matrix multiplication
    matrixMultiplicationSigmoidKernel->setArg(10,
          A.colMajorOrdered?1:0);    // A in column-major order
    matrixMultiplicationSigmoidKernel->setArg(11,
          B.colMajorOrdered?1:0);    // B in column-major order
    matrixMultiplicationSigmoidKernel->setArg(12,
          sumToC?1:0);    // Result should be sumed to previous value of C or only assigned
    matrixMultiplicationSigmoidKernel->setArg(13,
          multTheSum); // If sumToC== true value that multiplies the result previous to sum
    
    // -----------------------------------------------------------------------
    // Define ndrange iteration space: global and local sizes based on
    // parameters obtained from user

    // Refer to the sample documentation for clarification about
    // how work is devided among work-groups and work-items.
    // -----------------------------------------------------------------------
  
//    std::cout << "Launching for device\n"
//              << " (global size: " << global_size[0]
//              << ", "  << global_size[1] << ")\n"
//              << " (local size: " << local_size[0]
//              << ", "  << local_size[1] << ")\n";
    
    const cl::NDRange offset = cl::NullRange;
    const cl::NDRange global(global_size[0], global_size[1]);
    const cl::NDRange local(local_size[0], local_size[1]);
    queue.enqueueNDRangeKernel(*matrixMultiplicationSigmoidKernel,
                               offset,
                               global,
                               local);
    queue.finish();

//    std::cout << "Matmult finished\n";
}


void OpenCLKernels::runElementWiseSubstract(
            matrix_cl_float const &tm,
            matrix_cl_float const &ym,
            matrix_cl_float &em) {

    assert(tm.cols == ym.cols && tm.rows == ym.rows &&
           tm.cols == em.cols && tm.rows == em.rows);
    
    const size_t blockSize = 512;  // float4's
    const size_t data_size_float4_global = ym.rows*ym.cols/4;
    
    size_t global_size[1] = {data_size_float4_global};
    size_t local_size[1] = {boost::math::gcd(blockSize, global_size[0])};
    
    assert(global_size[0] % local_size[0] == 0);
    
//    std::cout << "Launching for device\n"
//              << " (global size: " << global_size[0] << ")\n"
//              << " ( local size: " << local_size[0] << ")\n";

    elementWiseSubstractKernel->setArg(0, *(tm.data.deviceData));
    elementWiseSubstractKernel->setArg(1, *(ym.data.deviceData));
    elementWiseSubstractKernel->setArg(2, *(em.data.deviceData));
    elementWiseSubstractKernel->setArg(3, tm.offset/4);
    elementWiseSubstractKernel->setArg(4, ym.offset/4);
    elementWiseSubstractKernel->setArg(5, em.offset/4);
    
    const cl::NDRange offset = cl::NullRange;
    const cl::NDRange global(global_size[0]);
    const cl::NDRange local(local_size[0]);
    queue.enqueueNDRangeKernel(*elementWiseSubstractKernel,
                               offset,
                               global,
                               local);
    queue.finish();
}

// NOT TESTED YET
void OpenCLKernels::runElementWiseMultiplicationBySigmoidDerivativeKernel(
            matrix_cl_float const &deltas,
            matrix_cl_float const &activations) {

    assert(deltas.cols == activations.cols
           && deltas.rows == activations.rows);
    
    const size_t blockSize = 512;  // float4's
    const size_t data_size_float4_global = deltas.rows*deltas.cols/4;
    
    size_t global_size[1] = {data_size_float4_global};
    size_t local_size[1] = {boost::math::gcd(blockSize, global_size[0])};
    
    assert(global_size[0] % local_size[0] == 0);
    
//    std::cout << "Launching for device\n"
//              << " (global size: " << global_size[0] << ")\n"
//              << " ( local size: " << local_size[0] << ")\n";

    elementWiseMultiplicationBySigmoidDerivativeKernel->
        setArg(0, *(deltas.data.deviceData));
    elementWiseMultiplicationBySigmoidDerivativeKernel->
        setArg(1, *(activations.data.deviceData));
    elementWiseMultiplicationBySigmoidDerivativeKernel->
        setArg(2, deltas.offset/4);
    elementWiseMultiplicationBySigmoidDerivativeKernel->
        setArg(3, activations.offset/4);
    
    const cl::NDRange offset = cl::NullRange;
    const cl::NDRange global(global_size[0]);
    const cl::NDRange local(local_size[0]);
    queue.enqueueNDRangeKernel(
        *elementWiseMultiplicationBySigmoidDerivativeKernel,
        offset,
        global,
        local);
    queue.finish();
}

cl_float OpenCLKernels::runCrossEntropy(matrix_cl_float const &t,
                                        matrix_cl_float const &y,
                                        matrix_cl_float &error) {
    const size_t blockSize = 2048;  // float4's
    const size_t data_size_float4_global = y.rows*y.cols/4;

    size_t global_size[1] = {data_size_float4_global / 2};
    size_t local_size[1] = {boost::math::gcd(blockSize, global_size[0])};
    
    assert(data_size_float4_global * 4 <= error.data.hostData.size());    
    assert(global_size[0] % local_size[0] == 0);
    
    // -----------------------------------------------------------------------
    // Setting kernel arguments
    // -----------------------------------------------------------------------
    crossEntropyKernelLocal->setArg(0, *(t.data.deviceData));
    crossEntropyKernelLocal->setArg(1, *(y.data.deviceData));
    crossEntropyKernelLocal->setArg(2, *(error.data.deviceData));
    crossEntropyKernelLocal->setArg(3,
                           cl::Local(local_size[0] * 4 * sizeof(cl_float)));
    crossEntropyKernelLocal->setArg(4, y.offset/4);

    // -----------------------------------------------------------------------
    // Define ndrange iteration space: global and local sizes based on
    // parameters obtained from user

    // Refer to the sample documentation for clarification about
    // how work is devided among work-groups and work-items.
    // -----------------------------------------------------------------------
  
//    std::cout << "Launching for device\n"
//            << " (global size: " << global_size[0] << ")\n"
//            << " (local size: " << local_size[0] << ")\n";
    
    const cl::NDRange offset = cl::NullRange;
    const cl::NDRange global(global_size[0]);
    const cl::NDRange local(local_size[0]);
    queue.enqueueNDRangeKernel(*crossEntropyKernelLocal, offset, global, local);
    queue.finish();

    //std::cout << "CE kernel finished\n";
    
    error.data.readFromDevice(queue);

    const size_t error_size = 4 * global_size[0]/local_size[0];
    std::vector<cl_float> & e = error.data.hostData;
    cl_float ce = 0.0;
    for (size_t i = 0; i < error_size; i++) {
        ce += e[i];
    }
    
    return -ce/(y.rows*y.cols);
}


//// NOT TESTED
//void OpenCLKernels::runTranspose(matrix_cl_float const &a,
//                                 matrix_cl_float &transpose) {
//
//    // -----------------------------------------------------------------------
//    // Setting kernel arguments
//    // -----------------------------------------------------------------------
//    transposeKernelLocal->setArg(0, transpose.data.deviceData);
//    transposeKernelLocal->setArg(1, a.data.deviceData);
//    transposeKernelLocal->setArg(2, a.cols);
//    transposeKernelLocal->setArg(3, a.rows);
//    transposeKernelLocal->setArg(4, cl::Local((TRANSPOSE_BLOCK_DIM+1)
//                                                * TRANSPOSE_BLOCK_DIM
//                                                * sizeof(cl_float)));
//    transposeKernelLocal->setArg(5, transpose.offset);
//    transposeKernelLocal->setArg(6, a.offset);
//
//    size_t global_size[2] = {size_t(a.cols), size_t(a.rows)};
//    size_t local_size[2] = {TRANSPOSE_BLOCK_DIM, TRANSPOSE_BLOCK_DIM };
//    
//    const cl::NDRange offset = cl::NullRange;
//    const cl::NDRange global(global_size[0], global_size[1]);
//    const cl::NDRange local(local_size[0], local_size[1]);
//    queue.enqueueNDRangeKernel(*transposeKernelLocal, offset, global, local);
//    queue.finish();
//}
