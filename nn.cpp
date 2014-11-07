// Copyright 2014 <Jordi de la Torre>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <cassert>
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>

#include "nn.hpp"
#include "OpenCLMatrixMultiplication.hpp"
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

nn::nn(const std::string &filename) : activations(activations_host),
                                      weights(weights_host), 
                                      deltas(deltas_host),
                                      t(t_host),
                                      y(activations_host) {
    
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
    numberOfElements = 0;
    for ( int i = 0; i < numberOfLayers-1; i++ ) {
        numberOfElements += (elementsPerLayer[i])*elementsPerLayer[i+1];
    }
    
    activations.hostData.resize(numberOfElements);
    weights.hostData.resize(numberOfElements);
    deltas.hostData.resize(numberOfElements);

    load_csv_vector("weights.txt", weights.hostData);
    
    // device memory allocation
    // Create OpenCL buffers for the matrices based on allocated memory regions
    // Create buffers with CL_MEM_USE_HOST_PTR to minimize copying and
    // model situation when matrices are hosted by some native library that
    // uses OpenCL to accelerate calculations.
    
    // Create buffers and copy host contents
    activations.createBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    weights.createBuffer(*context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR);
    deltas.createBuffer(*context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR);
    t.createBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);

    activations.writeToDevice(*queue);
    weights.writeToDevice(*queue);
    t.writeToDevice(*queue);
    
    // instantitate kernels
    matmult = new OpenCLMatrixMultiplication(*context, devices, 0, *queue);
    // ce = new OpenCLErrorReduce(*context, devices, *queue, y, t);
};

nn::~nn() {
  //    delete ce;
    delete matmult;
    delete queue;
    delete context;
}

/*
 * The next functions encode the required order of execution for feedforwarding
 * the neural network that is encoded in the next table:
 * 
 * ORDER ->     0          1          2           3         ...
 * MATRIX
 *   |
 *  \/
 * 1st      inputs      output1     output2     output1     ...
 * 2nd      weights0    weights2    weights3    weights4    ...
 * result   output1     output2     output1     output2     ...
 * 
 */


void nn::FF() {
    const cl_int N = numberOfLayers - 1;
    
    matrix<cl_float> & A(activations);
    matrix<cl_float> & B(weights);
    matrix<cl_float> & C(activations);
    
    A.offset = 0;
    A.rows = numberOfTrainingData;
    B.offset = 0;
    C.offset = elementsPerLayer[0]*elementsPerLayer[1];
    for ( cl_int i = 0; i < N; i++ ) {
        A.cols = elementsPerLayer[i];
        B.rows = A.cols;
        B.cols = elementsPerLayer[i+1];
        C.rows = A.cols;
        C.cols = B.cols;
        matmult->run(A, B, C);
        //C.readFromDevice(*queue);
        //continue;
        
        A.offset = C.offset;
        C.offset += elementsPerLayer[i+1]*numberOfTrainingData;        
        B.offset += elementsPerLayer[i]*elementsPerLayer[i+1];
    }


 
    C.data.readFromDevice(*queue);  // copy calculatied activations back to host
    
    // devolvemos referencia a vector output en host que contiene
    // los resultados finales
    
    print_vector(C.data.hostData, C.rows, C.cols, C.offset);
}
