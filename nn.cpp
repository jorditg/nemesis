// Copyright 2014 <Jordi de la Torre>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include <cassert>
#include <vector>
#include <algorithm>
#include <string>
#include <iomanip>
#include <future>
#include <iostream>

#include "nn.hpp"
#include "OpenCLKernels.hpp"
// #include "common.hpp"
#include "mnist.hpp"
#include "dng.hpp"

nn::nn() : activations(activations_host),
          activations_test(activations_test_host),
          bias(bias_host),
          weights(weights_host),
          increment_weights(increment_weights_host),
          // increment_bias(increment_bias_host),
          deltas(deltas_host),
          t(t_host),
          t_test(t_test_host),
          buffer_error(buffer_error_host) {
    
    opencl_init();
}

nn::~nn() {
    delete openclKernels;
    delete queue;
    delete context;
}

void nn::opencl_init() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);  // get available OpenCL platforms
    // get OpenCL devices for first platform
    platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    // create a context for these devices
    context = new cl::Context(devices);
    // Create queue of first device
    queue = new cl::CommandQueue(*context, devices[0]);
    // instantitate kernels
    openclKernels = new OpenCLKernels(*context, devices, 0, *queue);
}

void nn::load_MNIST_train_and_test_DATA(
                    const std::string &train_file,
                    const std::string &train_labels_file,
                    const std::string &test_file,
                    const std::string &test_labels_file) {
    // load input data into host memory
    size_t r, c;  // local variables not used
    
    read_mnist_images_file(train_file,
                           training_data,
                           r,
                           c);
    numberOfTrainingData = static_cast<cl_uint>(r);
    
    read_mnist_labels_file(train_labels_file,
                           training_data_output,
                           r,
                           c);
    
    read_mnist_images_file(test_file,
                           activations_test.hostData,
                           r,
                           c);
    numberOfTestData = static_cast<cl_uint>(r);
    
    read_mnist_labels_file(test_labels_file,
                           t_test.hostData,
                           r,
                           c);
    
    trainDataLoaded = true;
    testDataLoaded = true;
}

void nn::calculate_offsets() {
    // calculate offsets of every layer inside the vectors
    activations_offsets.resize(numberOfLayers);
    activations_test_offsets.resize(numberOfLayers);
    deltas_offsets.resize(numberOfLayers);
    weights_offsets.resize(numberOfLayers - 1);
    bias_offsets.resize(numberOfLayers - 1);
    activations_offsets[0] = 0;
    activations_test_offsets[0] = 0;
    weights_offsets[0] = 0;
    bias_offsets[0] = 0;
    deltas_offsets[0] = 0;   // never used in the algorithm
    deltas_offsets[1] = 0;
    for (cl_uint i = 1; i < numberOfLayers; i++) {
      activations_offsets[i] = activations_offsets[i-1] +
                               minibatchSize*elementsPerLayer[i-1];
      activations_test_offsets[i] = activations_test_offsets[i-1] +
                               numberOfTestData*elementsPerLayer[i-1];
      weights_offsets[i] = weights_offsets[i-1] +
                           elementsPerLayer[i-1]*elementsPerLayer[i];
      bias_offsets[i] = bias_offsets[i-1] + elementsPerLayer[i];
      deltas_offsets[i] = activations_offsets[i] - activations_offsets[1];
    }
}

void nn::allocate_NN_memory_on_host() {
    // host memory allocation for neural network
    numberOfWeights = 0;
    numberOfNeurons = 0;
    for ( cl_uint i = 0; i < numberOfLayers-1; i++ ) {
        numberOfWeights += elementsPerLayer[i]*elementsPerLayer[i+1];
        numberOfNeurons += elementsPerLayer[i];
    }
    numberOfNeurons += elementsPerLayer[numberOfLayers-1];
    
    bias.hostData.resize(numberOfNeurons - elementsPerLayer[0]);
    weights.hostData.resize(numberOfWeights);
    increment_weights.hostData.resize(numberOfWeights);
    // increment_bias.hostData.resize(numberOfNeurons - elementsPerLayer[0]);
    // there are no deltas in input layer
    deltas.hostData.resize((numberOfNeurons
                            -elementsPerLayer[0])*minibatchSize);
    buffer_error.hostData.resize(BUFFER_ERROR_SIZE);
}

// Call it always after allocate_NN_memory_on_host()
void nn::allocate_DATA_memory_on_host() {
    activations.hostData.resize(numberOfNeurons * minibatchSize);
    t.hostData.resize(elementsPerLayer[numberOfLayers-1] * minibatchSize);
    activations_test.hostData.resize(numberOfNeurons * numberOfTestData);
    t_test.hostData.resize(elementsPerLayer[numberOfLayers-1] * numberOfTestData);
}

void nn::allocate_memory_on_device() {
    // device memory allocation
    // Create OpenCL buffers for the matrices based on allocated memory regions
    // Create buffers with CL_MEM_USE_HOST_PTR to minimize copying and
    // model situation when matrices are hosted by some native library that
    // uses OpenCL to accelerate calculations
    activations.createBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    activations_test.createBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    bias.createBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    weights.createBuffer(*context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR);
    increment_weights.createBuffer(*context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR);
    // increment_bias.createBuffer(*context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR);
    deltas.createBuffer(*context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR);
    t.createBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    t_test.createBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    buffer_error.createBuffer(*context,
                                     CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR);
    
}

void nn::load_data_to_device() {
    activations.writeToDevice(*queue);
    activations_test.writeToDevice(*queue);
    bias.writeToDevice(*queue);
    weights.writeToDevice(*queue);
    increment_weights.writeToDevice(*queue);
    t.writeToDevice(*queue);
    t_test.writeToDevice(*queue);
}

/**

 * Sparse random initialization (Martens, 2010)
 * Stddev bibliography values: 0.1, 0.001
 */

void nn::populate_normal_random_weights(cl_float mean, cl_float stddev) {
    
  // suppose all the elements are 0 initialized only
  // sets the differents from 0
  boost::mt19937 rng;
  boost::normal_distribution<> nd(mean, stddev);
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);

  for (std::vector<cl_float>::iterator it = weights.hostData.begin();
       it != weights.hostData.end(); ++it)
    *it = var_nor();
  
}


/**

 * Sparse random initialization (Martens, 2010)
 * Stddev bibliography values: 0.1, 0.001
 */

void nn::populate_normal_sparse_weights(const cl_float mean, 
                                        const cl_float stddev, 
                                        const cl_uint initElementsPerLayer) {
    
  // suppose all the elements are 0 initialized only
  // sets the differents from 0
  boost::mt19937 rng;
  boost::normal_distribution<> nd(mean, stddev);
  boost::variate_generator<boost::mt19937&,
                           boost::normal_distribution<> > var_nor(rng, nd);

  const cl_uint init_elements = initElementsPerLayer;

  for (std::vector<cl_float>::iterator it = weights.hostData.begin();
       it != weights.hostData.end(); ++it)
    *it = 0.0f;

  for (cl_uint i = 1; i < numberOfLayers; i++) {
      boost::random::uniform_real_distribution<>
                   dist(0, elementsPerLayer[i-1]-1);
      for (cl_uint to_idx = 0; to_idx < elementsPerLayer[i]; to_idx++) {
          for (cl_uint k = 0; k < init_elements; k++) {
              const cl_uint from_idx = dist(rng);
              const cl_uint idx = weights_offsets[i-1] +
                                  elementsPerLayer[i] * from_idx +
                                  to_idx;
              weights.hostData[idx] = var_nor();
          }
      }
  }
}

void nn::populate_random_weights(const cl_float min, const cl_float max) {
  boost::random::mt19937 gen;
  boost::random::uniform_real_distribution<> dist(min, max);

  for (std::vector<cl_float>::iterator it = weights.hostData.begin();
       it != weights.hostData.end(); ++it)
    *it = dist(gen);
}

void nn::populate_fixed_weights(const cl_float val) {
  
  for (std::vector<cl_float>::iterator it = weights.hostData.begin() ;
       it != weights.hostData.end(); ++it)
    *it = val;
}

void nn::FF(host_device_memory_map<cl_float> &act,
            std::vector<cl_uint> &off,
            cl_uint rows) {
    const cl_uint N = numberOfLayers - 1;
    
    matrix_cl_float A(act);
    matrix_cl_float B(weights);
    matrix_cl_float C(act);
    matrix_cl_float bias_val(bias);  // offset set to 0
    bool calcSigmoid = true;
    for ( cl_uint i = 0; i < N; i++ ) {
        A.set(rows, elementsPerLayer[i], off[i]);
        B.set(elementsPerLayer[i], elementsPerLayer[i+1], weights_offsets[i]);
        C.set(rows, elementsPerLayer[i+1], off[i+1]);
        bias_val.offset = bias_offsets[i];
        
        if (i == N-1) {
            calcSigmoid = false;
        }
        
        openclKernels->
                  runMatrixMultiplicationSigmoid(A, B, C, &bias_val, calcSigmoid);
        if (i == N-1) {
            openclKernels->runSoftMax(C);
        }
    }
}

cl_float nn::percentage_classification_results(
            host_device_memory_map<cl_float> &act,
            std::vector<cl_uint> &act_off,
            host_device_memory_map<cl_float> &out,
            cl_uint rows) {

    // out.readFromDevice(*queue); // (doesn't change)

    act.readFromDevice(*queue);
    // SE PUEDE ACOTAR PARA NO TANTAS TRANSFERENCIAS SOLO BAJAR OUTPUTS

    const cl_uint N = elementsPerLayer[numberOfLayers-1];

    const cl_uint off = act_off[numberOfLayers-1];

    std::vector<cl_float> &v = out.hostData;
    std::vector<cl_float> &w = act.hostData;

    cl_uint good = 0;
    cl_uint bad = 0;
    for (cl_uint i = 0; i < rows; i++) {
        cl_float max1 = 0.0f;
        cl_uint pos1 = 0;
        for (cl_uint j = 0; j < N; j++) {
            const cl_float val = v[i*N+j];
            if (val > max1) {
                pos1 = j;
                max1 = val;
            }
        }
        cl_float max2 = 0.0f;
        cl_uint pos2 = 0;
        for (cl_uint j = 0; j < N; j++) {
            const cl_float val = w[off+i*N+j];
            if (val > max2) {
                pos2 = j;
                max2 = val;
            }
        }
        if (pos1 == pos2) {
            good++;
        } else {
            bad++;
        }
    }
    
    return cl_float(good)/cl_float(good+bad)*100;
}

void nn::BP() {
    
    matrix_cl_float tm(t);
    matrix_cl_float act(activations);
    matrix_cl_float wei(weights);
    // matrix_cl_float bias_inc(increment_bias);
    matrix_cl_float del(deltas);
    matrix_cl_float del_r(deltas);

    // first of all calculate the deltas of the last layer
    // delta {output_layer} = (y - t)
    const cl_uint last = numberOfLayers - 1;
    tm.set(minibatchSize, elementsPerLayer[last], 0);
    act.set(tm.rows, tm.cols, activations_offsets[last]);
    del_r.set(minibatchSize,
              elementsPerLayer[last],
              deltas_offsets[last]);

    openclKernels->runElementWiseSubstract(act, tm, del_r);
    
    
    // next calculate deltas for next layers
    // delta {previous layer} = delta {next_layer} * weights * activation_function_derivative
    // In case of sigmoid and softmax:
    // activation_function_derivative = activation * ( 1 - activation ) 
    for (cl_int i = numberOfLayers - 2; i > 0; i--) {
        del.set(minibatchSize,
                elementsPerLayer[i+1],
                deltas_offsets[i+1]);
        del_r.set(minibatchSize,
                  elementsPerLayer[i],
                  deltas_offsets[i]);
        // wei transposed
        wei.set(elementsPerLayer[i+1],
                elementsPerLayer[i],
                weights_offsets[i],
                true);
        openclKernels->
            runMatrixMultiplicationSigmoid(del, wei, del_r);
        
        act.set(minibatchSize,
                elementsPerLayer[i],
                activations_offsets[i]);
        openclKernels->
            runElementWiseMultiplicationBySigmoidDerivativeKernel(del_r, act);
    }
}

void nn::WA() {
    matrix_cl_float act(activations);
    matrix_cl_float wei(weights);
    matrix_cl_float bias_val(bias);
    matrix_cl_float wei_inc(increment_weights);
    // matrix_cl_float bias_inc(increment_bias);
    matrix_cl_float del(deltas);
    
    // Weight actualization
    for (cl_int i = numberOfLayers - 2; i >= 0; i--) {
        // act transposed
        act.set(elementsPerLayer[i], minibatchSize,
                activations_offsets[i], true);
        del.set(minibatchSize, elementsPerLayer[i+1],
                deltas_offsets[i+1]);
        wei.set(elementsPerLayer[i], elementsPerLayer[i+1],
                weights_offsets[i]);
        wei_inc.set(elementsPerLayer[i], elementsPerLayer[i+1],
                    weights_offsets[i]);
        bias_val.set(1, elementsPerLayer[i+1], bias_offsets[i]);

        const bool sum = true;
        const cl_float learningRateOverMinibatchSize =
                            learningRate/cl_float(minibatchSize);
        openclKernels->runMatrixMultiplicationSigmoid(
                            act,
                            del,
                            wei_inc,
                            nullptr,
                            false,
                            sum,
                            momentum,
                            -learningRateOverMinibatchSize);
        
        openclKernels->runRowSum(del, bias_val, 1.0f,
                                 -learningRateOverMinibatchSize);
                
    }

    const size_t wei_sz = wei.data.hostData.size();
    wei.set(1, wei_sz, 0);
    wei_inc.set(1, wei_sz, 0);
    if (enableL2Regularization)  // if L2-regularization
        openclKernels->runElementWiseSum(wei_inc, wei, wei_inc,
                     1.0f, - learningRate*lambda/numberOfTrainingData);
    openclKernels->runElementWiseSum(wei, wei_inc, wei);
}


void nn::NAG_preupdate() {
    matrix_cl_float wei(weights);
    matrix_cl_float wei_inc(increment_weights);
    const size_t wei_size = weights.hostData.size();
    wei.set(1, wei_size, 0);
    wei_inc.set(1, wei_size, 0);
     
    openclKernels->runElementWiseSum(
                            wei,
                            wei_inc,
                            wei,
                            1.0f,
                            momentum);
}

void nn::NAG_postupdate() {
    matrix_cl_float wei(weights);
    matrix_cl_float wei_inc(increment_weights);
    const size_t wei_size = weights.hostData.size();
    wei.set(1, wei_size, 0);
    wei_inc.set(1, wei_size, 0);
     
    openclKernels->runElementWiseSum(
                            wei,
                            wei_inc,
                            wei,
                            1.0f,
                            -momentum);
}

void nn::print_results_data_header_with_L2_regularization() {
    std::cout << "\tTRAIN\t\t\t\t\t\t\t\tTEST" << std::endl;
    std::cout << "Epoch\tCE1\t\tCE2\t\tCE\t\t%Train\t\tCE1\t\tCE2\t\tCE\t\t%Test\n";
}

void nn::print_results_data_with_L2_regularization(
                            cl_float ce1,
                            cl_float ce2,
                            cl_float ce,
                            cl_float ce1_test,
                            cl_float ce2_test,
                            cl_float ce_test) {
    cl_uint ctrain = percentage_classification_results_train();
    cl_uint ctest = percentage_classification_results_test();
    
    std::cout << std::fixed << std::setprecision(6)
              << epoch << "\t"
              << ce1 << "\t"
              << ce2 << "\t"
              << ce << "\t"
              << ctrain << "%\t"
              << ce1_test << "\t"
              << ce2_test << "\t"
              << ce_test << "\t"
              << ctest << "%\n";
}

void nn::print_results_data_header() {
    std::cout << "\tTRAIN\t\t\tTEST\n";
    std::cout << "Epoch\tCE\t\t%Train\t\tCE\t\t%Test\n";
}

void nn::print_results_data(
                            cl_float ce,
                            cl_float ce_test) {
    const cl_float training_percentage = percentage_classification_results_train();
    const cl_float test_percentage = percentage_classification_results_test();
    
    // if(test_percentage > 70) learningRate = 0.01;
    
    std::cout << std::fixed << std::setprecision(6)
              << epoch << "\t"
              << ce << "\t"
              << training_percentage << "%\t"
              << ce_test << "\t"
              << test_percentage << "%\n";
}

void nn::print_data() {
    const cl_float ce_noreg = CE_train();

    FF_test();
    const cl_float ce_test_noreg = CE_test();

    ce = ce_noreg;
    ce_test = ce_test_noreg;

    if (enableL2Regularization) {
        cl_float sqr_weights = L2_regularization();
        const cl_float reg = 0.5f*sqr_weights*lambda;
        const cl_float ce_reg = reg/cl_float(numberOfTrainingData);
        const cl_float ce_test_reg = reg/cl_float(numberOfTestData);
        ce += ce_reg;
        ce_test += ce_test_reg;
        print_results_data_with_L2_regularization(ce_noreg, ce_reg, ce, ce_test_noreg, ce_test_reg, ce_test);
    } else {
        print_results_data(ce, ce_test);
    }       
}

void nn::train() {
    
    if (trainRunning) return;
    // only one training at time allowed
    
    trainRunning = true;
    
    minibatch_generator mg(numberOfTrainingData,
                           minibatchSize,
                           training_data,
                           activations.hostData,
                           elementsPerLayer[0],
                           training_data_output,
                           t.hostData,
                           elementsPerLayer[numberOfLayers-1]
                           );
    const size_t minibatch_size_bytes = minibatchSize*elementsPerLayer[0]*
                                        sizeof(cl_float);
    const size_t minibatch_size_output_bytes =
                     minibatchSize*
                     elementsPerLayer[numberOfLayers-1]*sizeof(cl_float);
        
    auto fut = std::async(&minibatch_generator::load_generated_minibatch, &mg);
    
#if DROPOUT
      dng dropout(elementsPerLayer,
                  weights_host,
                  weights_offsets,
                  bias_host,
                  bias_offsets,
                  increment_weights_host);
#endif
    
    if (enableL2Regularization)
        print_results_data_header_with_L2_regularization();
    else
        print_results_data_header();
    for (epoch = 0; epoch < maxEpochs; epoch++) {
        
        if (stopTraining) {
            stopTraining = false;
            break;
        }

        if (enableMomentumRule) {
            update_momentum_rule_Hinton2013(epoch);
        }
        
#if DROPOUT
          // dropout and load to OpenCL device
          dropout.dropout_neurons();
          weights.writeToDevice(*queue);
          increment_weights.writeToDevice(*queue);
          bias.writeToDevice(*queue);
#endif
        // wait for minibatch thread to finish
        fut.get();
        // load to device the thread calculated minibatch
        activations.writeToDevice(*queue, minibatch_size_bytes);
        t.writeToDevice(*queue, minibatch_size_output_bytes);
        // launch next minibatch calculation
        fut = std::async(&minibatch_generator::load_generated_minibatch, &mg);
        
        if (enableNAG) NAG_preupdate();
        FF_train();        
        BP();
        if (enableNAG) NAG_postupdate();
        WA();

#if DROPOUT
        // update weights and bias into dropout class controller    
        weights.readFromDevice(*queue);
        increment_weights.readFromDevice(*queue);
        bias.readFromDevice(*queue);
        dropout.update_from_last_dropout();
#endif
        
        if (epoch % printEpochs == 0) {
#if DROPOUT
            // if dropout we have to load all the weights and multiply them
            // by 0.5 in order to make the correct inference
            dropout.transfer_all_weights_to_nn();
            weights.writeToDevice(*queue);
            bias.writeToDevice(*queue);
            matrix_cl_float W(weights);
            W.set(weights.hostData.size(), 1);            
            openclKernels->runMatrixScalarMultiplication(W, 0.5f);
            matrix_cl_float B(bias);
            B.set(bias.hostData.size(), 1);
            openclKernels->runMatrixScalarMultiplication(B, 0.5f);
#endif
            print_data();
            if (ce < minError) break;
        }        
                
    }
      
    trainRunning = false;
}

cl_float nn::CE(
        host_device_memory_map<cl_float> &activ,
        std::vector<cl_uint> &off,
        host_device_memory_map<cl_float> &out,
        cl_uint rows) {
    matrix_cl_float tm(out);
    matrix_cl_float act(activ);
    matrix_cl_float ce(buffer_error);

    const cl_uint elemLastLayer = elementsPerLayer[numberOfLayers-1];
    
    tm.set(rows, elemLastLayer, 0);
    act.set(rows, elemLastLayer, off[numberOfLayers-1]);
    
    // act.data.readFromDevice(*queue);
    // tm.data.readFromDevice(*queue);
    // print(act, "y");
    // print(tm, "t");

    return openclKernels->runCrossEntropy(tm, act, ce);
}

cl_float nn::L2_regularization() {
    matrix_cl_float w(weights);
    matrix_cl_float ce(buffer_error);
    
    w.set(1, weights.hostData.size(), 0);
    
    return openclKernels->runL2Regularization(w, ce);
}


/**
 * Format of the file:
 *  DATA DESCRIPTION
 *  Number of layers
 *  Elements layer 0 (input)
 *  Elements layer 1 
 *  ...  
 *  Weights and Biases present in file (0: false, 1:true)
 *  bias: This and the rest present only if last = true
 *  ...
 *  row-aligned weights (layer0xlayer1 ...)
 * 
 */         

void nn::save_NN(const std::string filename) {
    std::ofstream saveFile(filename, std::ios::out | std::ios::binary);
    
    saveFile.write(reinterpret_cast<const char*>(&numberOfLayers),
                   sizeof(numberOfLayers));
    saveFile.write(reinterpret_cast<const char*>(&elementsPerLayer[0]),
                   numberOfLayers*sizeof(elementsPerLayer[0]));
    const bool weightsPresent = true;
    saveFile.write(reinterpret_cast<const char*>(&weightsPresent),
                   sizeof(weightsPresent));
    bias.readFromDevice(*queue);
    saveFile.write(reinterpret_cast<const char*>(&bias.hostData[0]),
                   bias.hostData.size()*sizeof(cl_float));
    weights.readFromDevice(*queue);
    saveFile.write(reinterpret_cast<const char*>(&weights.hostData[0]),
                   weights.hostData.size()*sizeof(cl_float));
}

void nn::load_NN(const std::string filename) {
    std::ifstream loadFile(filename, std::ios::in | std::ios::binary);
    // PENDING
    loadFile.read(reinterpret_cast<char*>(&numberOfLayers),
                  sizeof(numberOfLayers));
    elementsPerLayer.resize(numberOfLayers);
    loadFile.read(reinterpret_cast<char*>(&elementsPerLayer[0]),
                  numberOfLayers*sizeof(elementsPerLayer[0]));
    
    allocate_NN_memory_on_host();
    
    bool weightsPresent;
    loadFile.read(reinterpret_cast<char*>(&weightsPresent),
                  sizeof(weightsPresent));
    
    if (weightsPresent) {
        loadFile.read(reinterpret_cast<char*>(&bias.hostData[0]),
                      bias.hostData.size()*sizeof(cl_float));
        weights.readFromDevice(*queue);
        loadFile.read(reinterpret_cast<char*>(&weights.hostData[0]),
                      weights.hostData.size()*sizeof(cl_float));
    }
    
    neuralNetworkDefined = true;
}

//void nn::test_matrix_multiplication(const cl_uint nr_rows_A,
//                                    const cl_uint nr_cols_A,
//                                    const cl_uint nr_rows_B,
//                                    const cl_uint nr_cols_B) {
//  boost::random::mt19937 gen;
//  boost::random::uniform_real_distribution<> dist(-5.0f, 5.0f);
//  
//    matrix_cl_float A(deltas);
//    matrix_cl_float B(activations);
//    matrix_cl_float C(weights);
//    
//    assert(nr_rows_A % 4 == 0 &&
//           nr_rows_B % 4 == 0 &&
//           nr_cols_A % 4 == 0 &&
//           nr_cols_B % 4 == 0 &&
//           nr_cols_A == nr_rows_B);
//    
//    // Test of not transposed matrices ( CHECKED. IT WORKS!)
//    
//    const cl_uint nr_rows_C = nr_rows_A;
//    const cl_uint nr_cols_C = nr_cols_B;
//    
//    for (cl_uint i = 0; i < nr_rows_A; i++) {
//        for (cl_uint j = 0; j < nr_cols_A; j++) {
//            //A.data.hostData[j + nr_cols_A*i] = cl_float(j+1);
//            A.data.hostData[j + nr_cols_A*i] = dist(gen);
//        }
//    }
//    
//    for (cl_uint i = 0; i < nr_rows_B; i++) {
//        for (cl_uint j = 0; j < nr_cols_B; j++) {
//            //B.data.hostData[j + nr_cols_B*i] = 1.0f/cl_float(i+1);
//            B.data.hostData[j + nr_cols_B*i] = dist(gen);
//        }
//    }
//    
//    A.set(nr_rows_A, nr_cols_A, 0);
//    B.set(nr_rows_B, nr_cols_B, 0);
//    C.set(nr_rows_C, nr_cols_C, 0);
//    
//    A.data.writeToDevice(*queue);
//    B.data.writeToDevice(*queue);
//    
//    openclKernels->
//            runMatrixMultiplicationSigmoid(A, B, C);
//    C.data.readFromDevice(*queue);
//
//    print(A, "A");
//    print(B, "B");
//    print(C, "Not transposed matrices");
//
//    // Test A transposed
//    
//    for (cl_uint i = 0; i < nr_cols_A; i++) {
//        for (cl_uint j = 0; j < nr_rows_A; j++) {
//            //A.data.hostData[j + nr_rows_A*i] = cl_float(i+1);
//            A.data.hostData[j + nr_rows_A*i] = dist(gen);
//        }
//    }
//    
//    for (cl_uint i = 0; i < nr_rows_B; i++) {
//        for (cl_uint j = 0; j < nr_cols_B; j++) {
//            //B.data.hostData[j + nr_cols_B*i] = 1.0f/cl_float(i+1);
//            B.data.hostData[j + nr_cols_B*i] = dist(gen);
//        }
//    }
//    
//    A.set(nr_rows_A, nr_cols_A, 0, true);
//    B.set(nr_rows_B, nr_cols_B, 0);
//    C.set(nr_rows_C, nr_cols_C, 0);
//
//    A.data.writeToDevice(*queue);
//    B.data.writeToDevice(*queue);
//    
//    openclKernels->
//            runMatrixMultiplicationSigmoid(A, B, C);
//    C.data.readFromDevice(*queue);
//
//    print(A, "A", true);
//    print(B, "B");
//    print(C, "Result with A transposed");
//
//    // Test B transposed
//    
//    for (cl_uint i = 0; i < nr_rows_A; i++) {
//        for (cl_uint j = 0; j < nr_cols_A; j++) {
//            //A.data.hostData[j + nr_cols_A*i] = cl_float(j+1);
//            A.data.hostData[j + nr_cols_A*i] = dist(gen);
//        }
//    }
//    
//    for (cl_uint i = 0; i < nr_cols_B; i++) {
//        for (cl_uint j = 0; j < nr_rows_B; j++) {
//            //B.data.hostData[j + nr_rows_B*i] = 1.0f/cl_float(j+1);
//            B.data.hostData[j + nr_rows_B*i] = dist(gen);
//        }
//    }
//    
//    A.set(nr_rows_A, nr_cols_A, 0);
//    B.set(nr_rows_B, nr_cols_B, 0, true);
//    C.set(nr_rows_C, nr_cols_C, 0);
//
//    A.data.writeToDevice(*queue);
//    B.data.writeToDevice(*queue);
//    
//    openclKernels->
//            runMatrixMultiplicationSigmoid(A, B, C);
//    C.data.readFromDevice(*queue);
//
//    print(A, "A");
//    print(B, "B", true);
//    print(C, "Result with B transposed");
//
//    // Test A and B transposed
//    
//    for (cl_uint i = 0; i < nr_cols_A; i++) {
//        for (cl_uint j = 0; j < nr_rows_A; j++) {
//            //A.data.hostData[j + nr_rows_A*i] = cl_float(i+1);
//            A.data.hostData[j + nr_rows_A*i] = dist(gen);
//        }
//    }
//    
//    for (cl_uint i = 0; i < nr_cols_B; i++) {
//        for (cl_uint j = 0; j < nr_rows_B; j++) {
//            //B.data.hostData[j + nr_rows_B*i] = 1.0f/cl_float(j+1);
//            B.data.hostData[j + nr_rows_B*i] = dist(gen);
//        }
//    }
//    
//    A.set(nr_rows_A, nr_cols_A, 0, true);
//    B.set(nr_rows_B, nr_cols_B, 0, true);
//    C.set(nr_rows_C, nr_cols_C, 0);
//
//    A.data.writeToDevice(*queue);
//    B.data.writeToDevice(*queue);
//    
//    openclKernels->
//            runMatrixMultiplicationSigmoid(A, B, C);
//    C.data.readFromDevice(*queue);
//
//    print(A, "A", true);
//    print(B, "B", true);
//    print(C, "Result with A and B transposed");
//    
//    exit(0);
//    
//}

//void nn::test_dropout() {
//    elementsPerLayer.resize(4);
//    elementsPerLayer[0] = 24;
//    elementsPerLayer[1] = 16;
//    elementsPerLayer[2] = 12;
//    elementsPerLayer[3] = 4;
//    
//    const size_t weights_sz = 24*16+16*12+12*4;
//    
//    weights_host.resize(weights_sz);
//    
//    for(size_t i=0;i<weights_sz;i++) {
//        weights_host[i] = (i+1)%10;
//    }
//    
//    weights_offsets.resize(3);
//    weights_offsets[0] = 0;
//    weights_offsets[1] = 24*16;
//    weights_offsets[2] = 24*16 + 16*12;
//    
//    bias_host.resize(4+12+16);
//    bias_offsets.resize(3);    
//
//    for(size_t i=0;i<bias_host.size();i++) {
//        bias_host[i] = (i+1)%10;
//    }
//    
//    bias_offsets[0] = 0;
//    bias_offsets[1] = 16;
//    bias_offsets[2] = 16+12;
//
//    dng d(elementsPerLayer, weights_host, weights_offsets, bias_host, bias_offsets);
//    
//    print_vector(weights_host, elementsPerLayer[0], elementsPerLayer[1], weights_offsets[0]);
//    std::cout << "\n";
//    print_vector(weights_host, elementsPerLayer[1], elementsPerLayer[2], weights_offsets[1]);
//    std::cout << "\n";
//    print_vector(weights_host, elementsPerLayer[2], elementsPerLayer[3], weights_offsets[2]);
//    std::cout << "\n";
//    print_vector(bias_host, elementsPerLayer[1], 1, bias_offsets[0]);
//    std::cout << "\n";
//    print_vector(bias_host, elementsPerLayer[2], 1, bias_offsets[1]);
//    std::cout << "\n";
//    print_vector(bias_host, elementsPerLayer[3], 1, bias_offsets[2]);
//    std::cout << "\n";
//    
//    d.weigths_dropout();
//    
//    print_vector(weights_host, elementsPerLayer[0], elementsPerLayer[1], weights_offsets[0]);
//    std::cout << "\n";
//    print_vector(weights_host, elementsPerLayer[1], elementsPerLayer[2], weights_offsets[1]);
//    std::cout << "\n";
//    print_vector(weights_host, elementsPerLayer[2], elementsPerLayer[3], weights_offsets[2]);
//    std::cout << "\n";
//    print_vector(bias_host, elementsPerLayer[1], 1, bias_offsets[0]);
//    std::cout << "\n";
//    print_vector(bias_host, elementsPerLayer[2], 1, bias_offsets[1]);
//    std::cout << "\n";
//    print_vector(bias_host, elementsPerLayer[3], 1, bias_offsets[2]);
//    std::cout << "\n";
//        
//    d.weights_update_from_last_dropout();
//    
//    d.transfer_all_weights_to_nn();
//    
//    print_vector(weights_host, elementsPerLayer[0], elementsPerLayer[1], weights_offsets[0]);
//    std::cout << "\n";
//    print_vector(weights_host, elementsPerLayer[1], elementsPerLayer[2], weights_offsets[1]);
//    std::cout << "\n";
//    print_vector(weights_host, elementsPerLayer[2], elementsPerLayer[3], weights_offsets[2]);
//    std::cout << "\n";
//    print_vector(bias_host, elementsPerLayer[1], 1, bias_offsets[0]);
//    std::cout << "\n";
//    print_vector(bias_host, elementsPerLayer[2], 1, bias_offsets[1]);
//    std::cout << "\n";
//    print_vector(bias_host, elementsPerLayer[3], 1, bias_offsets[2]);
//    std::cout << "\n";
//        
//}
//
