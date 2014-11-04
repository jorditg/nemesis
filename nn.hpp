// Copyright 2014 <Jordi de la Torre>

#ifndef NN_HPP_
#define NN_HPP__

#define __CL_ENABLE_EXCEPTIONS  // enable use of exceptions of the OpenCL API
#include <CL/cl.hpp>

#include <vector>
#include <string>

#include "common.hpp"
#include "OpenCLMatrixMultiplication.hpp"

class nn {

    cl_int numberOfTrainingData;
    cl_int numberOfLayers;

    std::vector<cl_int> elementsPerLayer;

    matrix<cl_float> inputs;   // data input
    matrix<cl_float> weights;  // all the weights of the NN
    matrix<cl_float> t;        // real output value
    matrix<cl_float> output1;  // buffer for calculations 1        
    matrix<cl_float> output2;  // buffer for calculations 2

    cl::Context *context;   // unique OpenCL context
    std::vector<cl::Device> devices;
    cl::CommandQueue *queue;   // unique OpenCL command queue;

    OpenCLMatrixMultiplication *matmult;   
    OpenCLErrorReduce *ce;

    void opencl_device_memory_allocation();
    void opencl_cleanup();

    // FeedForward calculation matrices
    inline cl_uint get_number_of_product_matrices() { return numberOfLayers-1; }
    matrix & FF_get_1st_matrix_for_product(cl_uint order);
    matrix & FF_get_2nd_matrix_for_product(cl_uint order);
    matrix & FF_get_result_matrix_for_product(cl_uint order);

    inline matrix const & output(cl_uint order) {
        return (order%2)?output2:output1;
    };
    
    public:
    explicit nn(const std::string &filename);
    ~nn();

    void populate_random_weights(cl_float min, cl_float max);
    void populate_fixed_weights();
    
    std::vector<cl_float> & FF();
};

#endif
