
#ifndef NN_HPP_
#define NN_HPP__

#define __CL_ENABLE_EXCEPTIONS  // enable use of exceptions of the OpenCL API

#include <CL/cl.hpp>

#include <vector>
#include <string>
#include <math.h> 

#include "common.hpp"
#include "mg.hpp"
#include "OpenCLKernels.hpp"

class nn {
    const cl_uint CROSS_ENTROPY_ERROR_SIZE = 1048576;
    
    cl_uint numberOfNeurons;    
    cl_uint numberOfWeights;    
    cl_uint numberOfTrainingData;
    cl_uint numberOfTestData;
    cl_uint numberOfLayers;    

    bool NAG = true;    // true uses Nesterov-accelerated gradient. 
                        // false uses Classical Momentum
    
    cl_uint epoch = 0;  // epoch of training
    cl_float ce = 0.0;  
    cl_float ce_test = 0.0;
    
    cl_uint minibatchSize = 256;
    cl_float learningRate = 0.3f;  // Typìcal value 0.3
    cl_float momentum = 0.9f;      // Typical value 0.9
    size_t maxEpochs = 10000;      // Typical value 5000000
    cl_float minError = 0.001f;     // Typical value 0.01
    cl_float lambda = 1.0f;     // L2 regularization parameter
    
    size_t printEpochs = 250;      // Typical value 1000
    
    std::vector<cl_uint> elementsPerLayer;
    
    // Whole training data set
    std::vector<cl_float> training_data;
    std::vector<cl_float> training_data_output;
    
    // activations of all the neurons for all the training data for one epoch
    std::vector<cl_float> activations_host;
    // activations of all the neurons for all the test data for one epoch
    std::vector<cl_float> activations_test_host;
    // bias
    std::vector<cl_float> bias_host;
    // weights of all neurons
    std::vector<cl_float> weights_host;
    // last weight increment calculated from back propagation
    std::vector<cl_float> increment_weights_host;
    // last bias increment calculated from back propagation
    std::vector<cl_float> increment_bias_host;    
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
    std::vector<cl_uint> bias_offsets;
    std::vector<cl_uint> deltas_offsets;
      
    // classes for mapping the host memory with the device memory
    host_device_memory_map<cl_float> activations;
    host_device_memory_map<cl_float> activations_test;
    host_device_memory_map<cl_float> bias;
    // inputs and calculated activations
    host_device_memory_map<cl_float> weights;  // all the weights of the NN
    host_device_memory_map<cl_float> increment_weights;  // all the inc weights of the NN
    host_device_memory_map<cl_float> increment_bias;  // all the inc bias of the NN
    host_device_memory_map<cl_float> deltas;   // delta errors (Backprop)
    host_device_memory_map<cl_float> t;        // real output value
    host_device_memory_map<cl_float> t_test;        // real output value
    host_device_memory_map<cl_float> cross_entropy_error;  // real output value

    //host_device_memory_map<cl_uint> minibatch_idx;
    
    cl::Context *context;   // unique OpenCL context
    std::vector<cl::Device> devices;
    cl::CommandQueue *queue;   // unique OpenCL command queue;

    OpenCLKernels *openclKernels;
    
    /*
     * Momentum update rule extracted from "On the importance of initialization and momentum in deep learning",
     * Hinton et al. 2013.
     * According to this paper:
     * momentum_max is chosen between 0.999, 0.995, 0.99, 0.9 and 0
     * learning rate is chosen between 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001
     */
    inline void update_momentum_rule_Hinton2013(cl_uint t) {
        const cl_float momentum_max = 0.9;   // Values used: 0.999, 0.995, 0.99, 0.9, 0
        const cl_float new_momentum = 1.0f - std::pow( 2.0f, -1.0f - std::log2(t / 250.0f + 1.0f));
        momentum = std::min(momentum_max, new_momentum);                            
    }

    void print_results_data_header();
    void print_results_data(cl_float ce1, 
                            cl_float ce2, 
                            cl_float ce, 
                            cl_float ce1_test, 
                            cl_float ce2_test, 
                            cl_float ce_test);  
    
    cl_float percentage_classification_results_test();
    cl_float percentage_classification_results_train();
    
    void NAG_preupdate();
    void NAG_postupdate(); 
public:
    
    explicit nn(const std::string &nn_file,
                const std::string &train_file,
                const std::string &test_file);
    ~nn();

    void populate_sparse_weights(cl_float stddev = 0.1f);
    void populate_random_weights(const cl_float min, const cl_float max);
    void populate_fixed_weights(const cl_float val);
    
    void test_matrix_multiplication(const cl_uint nr_rows_A,
                                    const cl_uint nr_cols_A,
                                    const cl_uint nr_rows_B,
                                    const cl_uint nr_cols_B);

    
    inline void load_float_vector(std::string filename, std::vector<cl_float> & v) {
        load_csv_vector(filename, v);
    }

    inline void save_float_vector(std::string filename, std::vector<cl_float> & v) {
        save_csv_vector(filename, v);
    }
    
    void FF();  // Feed forward (all sigmoid) calculation of training data
    void FF_test(); // Feed forward (all sigmoid) calculation of test data

    
    cl_float cross_entropy();   // CE training calculation   
    cl_float cross_entropy_test();  // CE test calculation    
    cl_float L2_regularization();
    
    void BP();  // Backpropagation calculation (all sigmoid))
    
    void train();   // Training for all sigmoid
    
    // Classification neural network (all sigmoid except last layer -> softmax)
    
};

#endif
