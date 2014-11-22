// Copyright 2014 <Jordi de la Torre>

#ifndef NN_HPP_
#define NN_HPP__

#define __CL_ENABLE_EXCEPTIONS  // enable use of exceptions of the OpenCL API
#include <CL/cl.hpp>

#include <vector>
#include <string>

#include "common.hpp"
#include "OpenCLKernels.hpp"
//#include "OpenCLErrorReduce.hpp"

class nn {
    const size_t CROSS_ENTROPY_ERROR_SIZE = 32768;
    
    cl_int numberOfNeurons;
    cl_int numberOfWeights;
    cl_int numberOfTrainingData;
    cl_int numberOfLayers;

    std::vector<cl_int> elementsPerLayer;
    
    std::vector<cl_float> activations_host;
    std::vector<cl_float> weights_host;
//    std::vector<cl_float> weights_transposed_host;
    std::vector<cl_float> deltas_host;
    std::vector<cl_float> t_host;
    std::vector<cl_float> cross_entropy_error_host;
    
    host_device_memory_map<cl_float> activations;  // inputs and calculated activations
    host_device_memory_map<cl_float> weights;  // all the weights of the NN
//    host_device_memory_map<cl_float> weights_transposed;  // all the weights of the NN
    host_device_memory_map<cl_float> deltas;   // delta errors (Backprop)
    host_device_memory_map<cl_float> t;        // real output value
    host_device_memory_map<cl_float> cross_entropy_error;        // real output value

    cl::Context *context;   // unique OpenCL context
    std::vector<cl::Device> devices;
    cl::CommandQueue *queue;   // unique OpenCL command queue;

    OpenCLKernels *openclKernels;

    cl_float learningRate = 0.3f;
    
    
    inline cl_int get_weights_matrix_offset(cl_int layer) {
        cl_int offset = 0;
        for(int i = 0; i < layer; i++)
            offset += elementsPerLayer[i]*elementsPerLayer[i+1];
        return offset;
    }
    
    inline cl_int get_activations_matrix_offset(cl_int layer) {
        cl_int offset = 0;
        for(int i = 0; i < layer; i++) 
            offset += elementsPerLayer[i];        
        offset *= numberOfTrainingData;        
        return offset;
    }
    
    inline cl_int get_deltas_matrix_offset(cl_int layer) {
        assert(layer > 0);
        cl_int offset = 0;
        for(int i = 1; i < layer; i++) 
            offset += elementsPerLayer[i];        
        offset *= numberOfTrainingData;        
        return offset;        
    }
    
//    void transposeWeights();
    
 public:
    
    explicit nn(const std::string &filename);
    ~nn();

    void populate_random_weights(cl_float min, cl_float max);
    void populate_fixed_weights();
    
    void test_matrix_multiplication(const int nr_rows_A, 
                                    const int nr_cols_A,
                                    const int nr_rows_B, 
                                    const int nr_cols_B);

    
    void FF();
    cl_float cross_entropy();
    void BP();
    
};

#endif
