// Copyright 2014 <Jordi de la Torre>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <cassert>
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>

#include "nn.hpp"
#include "OpenCLKernels.hpp"
#include "common.hpp"

void nn::populate_random_weights(cl_float min, cl_float max) {
  boost::random::mt19937 gen;
  boost::random::uniform_real_distribution<> dist(min, max);

  for (std::vector<cl_float>::iterator it = weights.hostData.begin();
       it != weights.hostData.end(); ++it)
    *it = dist(gen);
}

void nn::populate_fixed_weights() {
  
  for (std::vector<cl_float>::iterator it = weights.hostData.begin() ;
       it != weights.hostData.end(); ++it)
    *it = 0.00005;
}

nn::nn(const std::string &filename)
              : activations(activations_host),
                weights(weights_host),
                weights_transposed(weights_transposed_host),
                deltas(deltas_host),
                t(t_host) {
    
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);  // get available OpenCL platforms
    
    // get OpenCL devices for first platform
    platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
        
    // create a context for these devices
    context = new cl::Context(devices);
   
    // Create queue of first device
    queue = new cl::CommandQueue(*context, devices[0]);

    // load input data into host memory
    load_csv_data(filename,
                  activations.hostData,
                  t.hostData,
                  numberOfTrainingData,
                  numberOfLayers,
                  elementsPerLayer);
       
    // host memory allocation for neural network
    numberOfWeights = 0;
    numberOfNeurons = 0;
    for ( int i = 0; i < numberOfLayers-1; i++ ) {
        numberOfWeights += elementsPerLayer[i]*elementsPerLayer[i+1];
        numberOfNeurons += elementsPerLayer[i];
    }
    numberOfNeurons += elementsPerLayer[numberOfLayers-1];
    
    activations.hostData.resize(numberOfNeurons*numberOfTrainingData);
    weights.hostData.resize(numberOfWeights);
    weights_transposed.hostData.resize(numberOfWeights);
    // there are no deltas in input layer
    deltas.hostData.resize((numberOfNeurons
                            -elementsPerLayer[0])*numberOfTrainingData);

    load_csv_vector("weights.txt", weights.hostData);
    
    // device memory allocation
    // Create OpenCL buffers for the matrices based on allocated memory regions
    // Create buffers with CL_MEM_USE_HOST_PTR to minimize copying and
    // model situation when matrices are hosted by some native library that
    // uses OpenCL to accelerate calculations.
    
    // Create buffers and copy host contents
    activations.createBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    weights.createBuffer(*context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR);
    weights_transposed.createBuffer(*context,
                                    CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR);
    deltas.createBuffer(*context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR);
    t.createBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

    activations.writeToDevice(*queue);
    weights.writeToDevice(*queue);
    t.writeToDevice(*queue);
    
    // instantitate kernels
    openclKernels = new OpenCLKernels(*context, devices, 0, *queue);
    // ce = new OpenCLErrorReduce(*context, devices, *queue, y, t);
};

nn::~nn() {
  //    delete ce;
    delete openclKernels;
    delete queue;
    delete context;
}

void nn::transposeWeights() {
    // done in host
    weights.readFromDevice(*queue);
    for (int l = 0; l < numberOfLayers - 1; l++) {
        for (int i = 0; i < elementsPerLayer[l] ; i++) {
            for (int j = 0; j < elementsPerLayer[l+1]; j++) {
                weights_transposed.hostData.assign(
                    j*elementsPerLayer[l] + i,
                    weights.hostData[i*elementsPerLayer[l+1] + j]);
            }
        }
    }
    weights_transposed.writeToDevice(*queue);
}

void nn::FF() {
    const cl_int N = numberOfLayers - 1;
    
    matrix_cl_float A(activations);
    matrix_cl_float B(weights);
    matrix_cl_float C(activations);
    
    A.offset = 0;
    A.rows = numberOfTrainingData;
    B.offset = 0;
    C.offset = elementsPerLayer[0]*numberOfTrainingData;
    for ( cl_int i = 0; i < N; i++ ) {
        A.cols = elementsPerLayer[i];
        B.rows = A.cols;
        B.cols = elementsPerLayer[i+1];
        C.rows = A.rows;
        C.cols = B.cols;
        openclKernels->runMatrixMultiplicationSigmoid(A, B, C, (i != (N - 1)));
        // C.data.readFromDevice(*queue);
        // print_vector(C.data.hostData, C.rows, C.cols, C.offset);
        if (i < N-1) {
            A.offset = C.offset;
            C.offset += elementsPerLayer[i+1]*numberOfTrainingData;
            B.offset += elementsPerLayer[i]*elementsPerLayer[i+1];
        }
    }
 
    C.data.readFromDevice(*queue);  // copy calculatied activations back to host
    
    // devolvemos referencia a vector output en host que contiene
    // los resultados finales

    print_vector(C.data.hostData, C.rows, C.cols, C.offset);
}

void nn::BP() {
    const cl_int N = numberOfLayers - 1;
    
    matrix_cl_float tm(t);
    matrix_cl_float act(activations);
    matrix_cl_float wei(weights);
    matrix_cl_float del(deltas);
    matrix_cl_float del_r(deltas);
    matrix_cl_float wei_t(weights_transposed);
    
    // first of all calculate the deltas of the last layer
    tm.set(numberOfTrainingData, elementsPerLayer[N], 0);
    const int offset_act = (numberOfNeurons - elementsPerLayer[N])
                           *numberOfTrainingData;
    act.set(tm.rows, tm.cols, offset_act);

    const int offset_del = (numberOfNeurons - elementsPerLayer[0]
                            - elementsPerLayer[N])*numberOfTrainingData;
    del.set(tm.rows, tm.cols, offset_del);

    openclKernels->runElementWiseSubstract(tm, act, del);

    del.data.readFromDevice(*queue);
    print(tm, "t");
    print(act, "y");
    print(del, "t-y");

    transposeWeights();
    wei_t.set(0, 0, wei_t.data.hostData.size());  // rows/cols will be set later
    for (int i = N - 1; i > 1; i--) {
        del_r.set(del.rows,
                  elementsPerLayer[i],
                  del.offset - elementsPerLayer[i]*numberOfTrainingData);
        wei_t.set(elementsPerLayer[i],
                  elementsPerLayer[i-1],
                  wei_t.offset - elementsPerLayer[i]*elementsPerLayer[i-1]);
        act.set(del_r.rows,
                del_r.cols,
                act.offset - elementsPerLayer[i]*numberOfTrainingData);
        print(wei_t, "Wt");
        print(act, "delta");
        openclKernels->
            runMatrixMultiplicationSigmoid(del, wei_t, del_r, false, false);

        del_r.data.readFromDevice(*queue);
        print(del_r, "delta*Wt");

        openclKernels->
            runElementWiseMultiplicationBySigmoidDerivativeKernel(del_r, act);

        del_r.data.readFromDevice(*queue);
        print(del_r, "delta_r*Wt.*s*(s-1)");

        del.set(del_r.rows, del_r.cols, del_r.offset);
    }
}
