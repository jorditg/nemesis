// Copyright 2014 <Jordi de la Torre>


#include <cassert>

#include <boost/tokenizer.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include <vector>
#include <algorithm>
#include <string>

#include <iostream>     // cout, endl
#include <fstream>      // fstream


#include "nn.hpp"
#include "OpenCLMatrixMultiplication.hpp"
#include "common.hpp"

void nn::populate_random_weights(cl_float min, cl_float max) {
  boost::random::mt19937 gen;
  boost::random::uniform_real_distribution<> dist(min, max);
  
  for (std::vector<cl_float>::iterator it = weights.begin() ;
       it != weights.end(); ++it)
    *it = dist(gen);
}

void nn::populate_fixed_weights() {
  
  for (std::vector<cl_float>::iterator it = weights.begin() ;
       it != weights.end(); ++it)
    *it = 0.00005;
}

nn::nn(const std::string &filename) {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);  // get available OpenCL platforms
    
    // get OpenCL devices for first platform
    platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
        
    // create a context for these devices
    context = new cl::Context(devices);
   
    // Create queue of first device
    queue = new cl::CommandQueue(*context, devices[0]);    
    
    // load input data into host memory
    load_csv_data(filename, inputs.hostData, t.hostData, numberOfTrainingData,
                  numberOfLayers, elementsPerLayer);
    
    // initialize rows and cols 
    inputs.rows = numberOfTrainingData;
    inputs.cols = elementsPerLayer[0];
    t.rows = inputs.rows;
    t.cols = elementsPerLayer[elementsPerLayer.size()-1];
    
    // host memory allocation for neural network weights
    cl_uint numberOfWeights = 0;
    for ( cl_int i = 0; i < numberOfLayers-1; i++ )
        numberOfWeights += elementsPerLayer[i]*elementsPerLayer[i+1];
    weights.hostData.resize(numberOfWeights);

    load_csv_vector("weights.txt", weights.hostData);
    
    // outputs buffer
    cl_uint maxLayerNeurons = *std::max_element(std::begin(elementsPerLayer)+1,
                                               std::end(elementsPerLayer));
    output1.hostData.resize(maxLayerNeurons*numberOfTrainingData);
    output2.hostData.resize(maxLayerNeurons*numberOfTrainingData);

    // device memory allocation
    // Create OpenCL buffers for the matrices based on allocated memory regions
    // Create buffers with CL_MEM_USE_HOST_PTR to minimize copying and
    // model situation when matrices are hosted by some native library that
    // uses OpenCL to accelerate calculations.
    
    // Create buffers and copy host contents    
    inputs.createBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    weights.createBuffer(*context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR);
    t.createBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);      
    output1.createBuffer(*context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR);
    output2.createBuffer(*context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR);

    // instantitate kernels
    matmult = new OpenCLMatrixMultiplication(*context, devices, 0, *queue);
    ce = new OpenCLErrorReduce(*context, devices, *queue, y, t);
};

nn::~nn() {
    delete ce;
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


matrix const & nn::FF_get_1st_matrix_for_product(cl_uint order) {
    if (order == 0) {
        return inputs;
    } else {
        if (order % 2) {
            return outputs1.set(numberOfTrainingData, elementsPerLayer[order]);
        } else {
            return outputs2.set(numberOfTrainingData, elementsPerLayer[order]);
        }
    }
}

matrix_type const & nn::FF_get_2nd_matrix_for_product(cl_uint order) {

    // calculate distance from origin of the weight matrix
    cl_uint offset = 0;
    for (cl_uint i = 0; i < order; i++)
        offset += elementsPerLayer[i]*elementsPerLayer[i+1];

    return weights.set(elementsPerLayer[order], 
                       elementsPerLayer[order+1], 
                       offset);
}


matrix_type nn::FF_get_result_matrix_for_product(cl_uint order) {
    matrix & result = output(order);
    return result.set(numberOfTrainingData, 
                      elementsPerLayer[order+1]);
}

std::vector<cl_float> & nn::FF() {
    const cl_uint N = get_number_of_product_matrices();
    
    for ( cl_uint i = 0; i < N; i++ ) {
        matrix & A = FF_get_1st_matrix_for_product(i);
        matrix & B = FF_get_2nd_matrix_for_product(i);
        matrix & C = FF_get_result_matrix_for_product(i);
        matmult->run(A, B, C);
    }


    matrix &result = output(N-1);
    
    // transferimos los datos finales calculados de device a host
    result.readFromDevice(*queue);
    
    // devolvemos referencia a vector output en host que contiene
    // los resultados finales
    
    print_vector(result.hostData, 8);

    return result;
}
