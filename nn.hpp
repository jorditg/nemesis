
#ifndef NN_HPP_
#define NN_HPP__

#define __CL_ENABLE_EXCEPTIONS  // enable use of exceptions of the OpenCL API

#include <CL/cl.hpp>

#include <vector>
#include <string>

#include "common.hpp"
#include "mg.hpp"
#include "OpenCLKernels.hpp"

class nn {
    const cl_uint CROSS_ENTROPY_ERROR_SIZE = 32768;
    
    cl_uint numberOfNeurons;    
    cl_uint numberOfWeights;    
    cl_uint numberOfTrainingData;
    cl_uint numberOfTestData;
    cl_uint numberOfLayers;

    cl_float learningRate = 0.27f;  // Typìcal value 0.3
    cl_float momentum = 0.8f;      // Typical value 0.9
    size_t maxEpochs = 100000;      // Typical value 5000000
    cl_float minError = 0.001f;     // Typical value 0.01
    //cl_uint minibatchSize = 256;     
    
    size_t printEpochs = 100;      // Typical value 1000
    
    std::vector<cl_uint> elementsPerLayer;
    
    // activations of all the neurons for all the training data for one epoch
    std::vector<cl_float> activations_host;
    // activations of all the neurons for all the test data for one epoch
    std::vector<cl_float> activations_test_host;
    // weights of all neurons
    std::vector<cl_float> weights_host;
    // last weight increment calculated from back propagation
    std::vector<cl_float> increment_weights_host;
    // deltas of all activation layers
    std::vector<cl_float> deltas_host;
    // output values of the training data
    std::vector<cl_float> t_host;
    // output values of the test data
    std::vector<cl_float> t_test_host;
    // vector required for the host side calculation of the cross entropy
    // after first reduce in device
    std::vector<cl_float> cross_entropy_error_host;
    
    // offsets required for finding activation values over the vector
    std::vector<cl_uint> activations_offsets;
    std::vector<cl_uint> activations_test_offsets;
    std::vector<cl_uint> weights_offsets;
    std::vector<cl_uint> deltas_offsets;
    
    //std::vector<cl_uint> minibatch_idx_host;
    
    // classes for mapping the host memory with the device memory
    host_device_memory_map<cl_float> activations;
    host_device_memory_map<cl_float> activations_test;
    // inputs and calculated activations
    host_device_memory_map<cl_float> weights;  // all the weights of the NN
    host_device_memory_map<cl_float> increment_weights;  // all the weights of the NN
    host_device_memory_map<cl_float> deltas;   // delta errors (Backprop)
    host_device_memory_map<cl_float> t;        // real output value
    host_device_memory_map<cl_float> t_test;        // real output value
    host_device_memory_map<cl_float> cross_entropy_error;  // real output value

    //host_device_memory_map<cl_uint> minibatch_idx;
    
    cl::Context *context;   // unique OpenCL context
    std::vector<cl::Device> devices;
    cl::CommandQueue *queue;   // unique OpenCL command queue;

    OpenCLKernels *openclKernels;
        
 public:
    
    explicit nn(const std::string &nn_file,
                const std::string &train_file,
                const std::string &test_file);
    ~nn();

    void populate_sparse_weights();
    void populate_random_weights(const cl_float min, const cl_float max);
    void populate_fixed_weights(const cl_float val);
    
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
    
    void FF();  // Feed forward (all sigmoid) calculation of training data
    void FF_test(); // Feed forward (all sigmoid) calculation of test data
    void print_classification_results_test();
    void print_classification_results_train();
    
    cl_float cross_entropy();   // CE training calculation   
    cl_float cross_entropy_test();  // CE test calculation
    
    void BP();  // Backpropagation calculation (all sigmoid))
    
    void train();   // Training for all sigmoid
    
    // Classification neural network (all sigmoid except last layer -> softmax)
    
};

#endif
