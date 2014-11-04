// Copyright 2014 <Jordi de la Torre>

#ifndef NN_HPP_
#define NN_HPP__

#define __CL_ENABLE_EXCEPTIONS  // enable use of exceptions of the OpenCL API
#include <CL/cl.hpp>

#include <vector>
#include <string>

#include "common.hpp"
#include "OpenCLMatrixMultiplication.hpp"
#include "OpenCLErrorReduce.hpp"

class nn {

    cl_int numberOfTrainingData;
    cl_int numberOfLayers;

    std::vector<cl_int> elementsPerLayer;

    matrix_cl_float inputs;   // data input
    matrix_cl_float weights;  // all the weights of the NN
    matrix_cl_float t;        // real output value
    matrix_cl_float output1;  // buffer for calculations 1
    matrix_cl_float output2;  // buffer for calculations 2

    matrix_cl_float &y = output1;  // can point to output1 or output2 depending
                                   // on which the last buffer call is
  

    cl::Context *context;   // unique OpenCL context
    std::vector<cl::Device> devices;
    cl::CommandQueue *queue;   // unique OpenCL command queue;

    OpenCLMatrixMultiplication *matmult;
    OpenCLErrorReduce *ce;

    void opencl_device_memory_allocation();
    void opencl_cleanup();

    // FeedForward calculation matrices
    inline cl_int get_number_of_product_matrices()
                        { return numberOfLayers-1; }
    matrix_cl_float & FF_get_1st_matrix_for_product(cl_int order);
    matrix_cl_float & FF_get_2nd_matrix_for_product(cl_int order);
    matrix_cl_float & FF_get_result_matrix_for_product(cl_int order);

    inline matrix_cl_float & output(cl_int order) {
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
