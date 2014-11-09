// Copyright 2014 <Jordi de la Torre>

#ifndef NN_HPP_
#define NN_HPP__

#define __CL_ENABLE_EXCEPTIONS  // enable use of exceptions of the OpenCL API
#include <CL/cl.hpp>

#include <vector>
#include <string>

#include "common.hpp"
#include "OpenCLMatrixMultiplication.hpp"
//#include "OpenCLErrorReduce.hpp"

class nn {
    cl_int numberOfElements;
    cl_int numberOfTrainingData;
    cl_int numberOfLayers;

    std::vector<cl_int> elementsPerLayer;
    
    std::vector<cl_float> activations_host;
    std::vector<cl_float> weights_host;
    std::vector<cl_float> deltas_host;
    std::vector<cl_float> t_host;
    
    host_device_memory_map<cl_float> activations; // inputs and calculated activations
    host_device_memory_map<cl_float> weights;  // all the weights of the NN
    host_device_memory_map<cl_float> deltas;   // delta errors (Backprop)
    host_device_memory_map<cl_float> t;        // real output value

    cl::Context *context;   // unique OpenCL context
    std::vector<cl::Device> devices;
    cl::CommandQueue *queue;   // unique OpenCL command queue;

    OpenCLMatrixMultiplication *matmult;
    //OpenCLErrorReduce *ce;

    void opencl_device_memory_allocation();
    void opencl_cleanup();

    // FeedForward calculation matrices
    matrix_cl_float & FF_get_1st_matrix_for_product(cl_int order);
    matrix_cl_float & FF_get_2nd_matrix_for_product(cl_int order);
    matrix_cl_float & FF_get_result_matrix_for_product(cl_int order);

    public:
    
    explicit nn(const std::string &filename);
    ~nn();

    void populate_random_weights(cl_float min, cl_float max);
    void populate_fixed_weights();
    
    void FF();
};

#endif
