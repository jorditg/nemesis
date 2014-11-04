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

    std::vector<cl_float> inputs;           // data input
    std::vector<cl_float> weights;          // neural network weights
    std::vector<cl_float> t;                // real output value
    std::vector<cl_float> output1;
    std::vector<cl_float> output2;

    cl::Context *context;   // unique OpenCL context
    std::vector<cl::Device> devices;
    cl::CommandQueue *queue;   // unique OpenCL command queue;

    cl::Buffer *inputsBuffer;
    cl::Buffer *weightsBuffer;
    cl::Buffer *tBuffer;
    cl::Buffer *outputBuffer1;
    cl::Buffer *outputBuffer2;

    OpenCLMatrixMultiplication *matmult;
    
    OpenCLErrorReduce *ce;

    void load_weights(const std::string & filename,
                      std::vector<cl_float> &weights);

    void load_csv_data(const std::string & filename,
                     std::vector<cl_float> & input,
                     std::vector<cl_float> & output,
                     cl_int &rows, cl_int &layers,
                     std::vector<cl_int> &elements);

    void opencl_initialize();
    void opencl_device_memory_allocation();
    void opencl_cleanup();

    // FeedForward calculation matrices
    inline cl_uint get_number_of_product_matrices() { return numberOfLayers-1; }
    matrix_type FF_get_1st_matrix_for_product(cl_uint order);
    matrix_type FF_get_2nd_matrix_for_product(cl_uint order);
    matrix_type FF_get_result_matrix_for_product(cl_uint order);

    inline cl::Buffer & outputBuffer(cl_uint order) {
        return (order%2)?*outputBuffer2:*outputBuffer1;
    };

    inline std::vector<cl_float> & output(cl_uint order) {
        return (order%2)?output2:output1;
    };
    
     // memory transfer functions from device to host
    void device2HostWeightsTransfer();
    void device2HostOutput1Transfer();
    void device2HostOutput2Transfer();
    
    void device2HostTransfer(const cl::Buffer & buffer, size_t size);

    public:
    explicit nn(const std::string &filename);
    ~nn();

    void populate_random_weights(cl_float min, cl_float max);
    void populate_fixed_weights();
    
    std::vector<cl_float> & FF();
};

#endif
