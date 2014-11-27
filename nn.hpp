
#ifndef NN_HPP_
#define NN_HPP__

#define __CL_ENABLE_EXCEPTIONS  // enable use of exceptions of the OpenCL API

#include <CL/cl.hpp>

#include <vector>
#include <string>

#include "common.hpp"
#include "OpenCLKernels.hpp"

class nn {
    const cl_uint CROSS_ENTROPY_ERROR_SIZE = 32768;
    
    cl_uint numberOfNeurons;
    cl_uint numberOfWeights;
    cl_uint numberOfTrainingData;
    cl_uint numberOfLayers;

    cl_float learningRate = 0.3f;  // Typìcal value 0.3
    cl_float momentum = 0.9f;      // Typical value 0.9
    size_t maxEpochs = 10000;      // Typical value 5000000
    cl_float minError = 0.01f;     // Typical value 0.01

    size_t printEpochs = 100;      // Typical value 1000
    
    std::vector<cl_uint> elementsPerLayer;
    
    std::vector<cl_float> activations_host;
    std::vector<cl_float> weights_host;
    std::vector<cl_float> increment_weights_host;
//    std::vector<cl_float> weights_transposed_host;
    std::vector<cl_float> deltas_host;
    std::vector<cl_float> t_host;
    std::vector<cl_float> cross_entropy_error_host;

    std::vector<cl_uint> activations_offsets;
    std::vector<cl_uint> weights_offsets;
    std::vector<cl_uint> deltas_offsets;

    host_device_memory_map<cl_float> activations;
    // inputs and calculated activations
    host_device_memory_map<cl_float> weights;  // all the weights of the NN
    host_device_memory_map<cl_float> increment_weights;  // all the weights of the NN
//    host_device_memory_map<cl_float> weights_transposed;  // all the weights of the NN
    host_device_memory_map<cl_float> deltas;   // delta errors (Backprop)
    host_device_memory_map<cl_float> t;        // real output value
    host_device_memory_map<cl_float> cross_entropy_error;  // real output value

    cl::Context *context;   // unique OpenCL context
    std::vector<cl::Device> devices;
    cl::CommandQueue *queue;   // unique OpenCL command queue;

    OpenCLKernels *openclKernels;
    
 public:
    
    explicit nn(const std::string &filename);
    ~nn();

    void populate_sparse_weights();
    void populate_random_weights(cl_float min, cl_float max);
    void populate_fixed_weights();
    
    void test_matrix_multiplication(const cl_uint nr_rows_A,
                                    const cl_uint nr_cols_A,
                                    const cl_uint nr_rows_B,
                                    const cl_uint nr_cols_B);

    
    inline void load_weights(std::string filename) {
        load_csv_vector(filename, weights_host);
    }

    inline void save_weights(std::string filename) {
        weights.readFromDevice(*queue);
        save_csv_vector(filename, weights_host);
    }
    
    void FF();  // Feed forward calculation
    cl_float cross_entropy();   // cross entropy error calculation
    void BP();  // Backpropagation calculation
    
    void train();
    
};

#endif
