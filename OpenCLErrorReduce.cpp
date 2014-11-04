/* 
 * File:   MatrixMultiplication.cpp
 * Author: jdelatorre
 * 
 * Created on 23 de octubre de 2014, 10:25
 */
#include <cassert>

#include <string>
#include <algorithm>
#include <vector>
#include <iostream>

#include "OpenCLErrorReduce.hpp"


OpenCLErrorReduce::~OpenCLErrorReduce() {
    delete kernel;
    delete program;
}

void OpenCLErrorReduce::opencl_initialize() {
    assert(t.rows == y.rows && t.cols == y.cols);
    
    const size_t blockSize = 4096;  // float4's
    const size_t data_size_float4_global = y.rows*y.cols/4;
    
    global_size[0] = {data_size_float4_global / 2};
    local_size[0] = std::min(blockSize, global_size[0]);

    assert(global_size[0] % local_size[0] == 0);
    
    const size_t error_size = 4 * global_size[0]/local_size[0];
    
    error.hostData.resize(error_size);
    error.createBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR);

    
    // create a CL program using kernel source
    std::string sourceFile = "Reduction_kernels.cl";
    std::string sourceString;
    readfile(sourceFile, sourceString);
    
    cl::Program::Sources sources;
    sources.push_back(std::make_pair(sourceString.c_str(), 0));
    // don't need to specify length as we used a null terminated string
    
    // create the OpenCL program
    program = new cl::Program(context, sources);
    
    try {
        program->build(devices);
    } catch(const cl::Error &e) {  // get compilation log in case of failure
        std::cerr << e.what() << "returned err: " << e.err() << std::endl <<
        program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << std::endl;
    }
    
    kernel = new cl::Kernel(*program, "cross_entropy");
    
}

cl_float OpenCLErrorReduce::run_CE_Kernel_local() {

    // -----------------------------------------------------------------------
    // Setting kernel arguments
    // -----------------------------------------------------------------------    
    kernel->setArg(0, y.deviceData);
    kernel->setArg(1, t.deviceData);
    kernel->setArg(2, error.deviceData);
    kernel->setArg(4, cl::__local(local_size[0] * 4 * sizeof(cl_float)));

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
    queue.enqueueNDRangeKernel(*kernel, offset, global, local);
    queue.finish();

    std::cout << "CE kernel finished\n";
    
    error.readFromDevice(queue);
    
    std::vector<cl_float> & e = error.hostData;
    cl_float ce = 0.0;
    for (size_t i = 0; i < e.size(); i++) {
        ce += e[i];
    }
    
    return ce;
}

cl_float OpenCLErrorReduce::run() {
    return run_CE_Kernel_local();
}
