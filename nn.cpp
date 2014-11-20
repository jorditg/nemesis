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
                t(t_host),
                cross_entropy_error(cross_entropy_error_host) {
    
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
    cross_entropy_error.hostData.resize(CROSS_ENTROPY_ERROR_SIZE);

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
    cross_entropy_error.createBuffer(*context,
                                     CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR);
    
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
    int offset = 0;
    for (int l = 0; l < numberOfLayers - 1; l++) {
        for (int i = 0; i < elementsPerLayer[l] ; i++) {
            for (int j = 0; j < elementsPerLayer[l+1]; j++) {
                weights_transposed.hostData[offset + i +
                                            j*elementsPerLayer[l]] =
                    weights.hostData[offset + j +
                                     i*elementsPerLayer[l+1]];
            }
        }
        offset += elementsPerLayer[l]*elementsPerLayer[l+1];
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

void nn::test_matrix_multiplication() {
    matrix_cl_float A(deltas);
    matrix_cl_float B(activations);
    matrix_cl_float C(weights);
    
    // Test of not transposed matrices ( CHECKED. IT WORKS!)
    
    const int nr_rows_A = 12;
    const int nr_cols_A = 16;
    
    const int nr_rows_B = 16;
    const int nr_cols_B = 8;
    
    const int nr_rows_C = nr_rows_A;
    const int nr_cols_C = nr_cols_B;
//    
//    for (int i = 0; i < nr_rows_A; i++) {
//        for (int j = 0; j < nr_cols_A; j++) {
//            A.data.hostData[j + nr_cols_A*i] = cl_float(j+1);
//        }
//    }
//    
//    for (int i = 0; i < nr_rows_B; i++) {
//        for (int j = 0; j < nr_cols_B; j++) {
//            B.data.hostData[j + nr_cols_B*i] = 1.0f/cl_float(i+1);
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
//            runMatrixMultiplicationSigmoid(A, B, C,
//                                           false, false,
//                                           false, false);
//    C.data.readFromDevice(*queue);
//
//    print(A, "A");
//    print(B, "B");
//    print(C, "Not transposed matrices");

    // Test A transposed

    for (int i = 0; i < nr_cols_A; i++) {
        for (int j = 0; j < nr_rows_A; j++) {
            A.data.hostData[j + nr_rows_A*i] = cl_float(i+1);
        }
    }
    
    for (int i = 0; i < nr_rows_B; i++) {
        for (int j = 0; j < nr_cols_B; j++) {
            B.data.hostData[j + nr_cols_B*i] = 1.0f/cl_float(i+1);
        }
    }
    
    A.set(nr_rows_A, nr_cols_A, 0);
    B.set(nr_rows_B, nr_cols_B, 0);
    C.set(nr_rows_C, nr_cols_C, 0);

    A.data.writeToDevice(*queue);
    B.data.writeToDevice(*queue);
    
    openclKernels->
            runMatrixMultiplicationSigmoid(A, B, C,
                                           false, false,
                                           true, false);
    C.data.readFromDevice(*queue);

    print(A, "A", true);
    print(B, "B");
    print(C, "Result with A transposed");
  
    exit(0);
    
}

void nn::BP() {
    const cl_int N = numberOfLayers - 1;
    
    matrix_cl_float tm(t);
    matrix_cl_float act(activations);
    matrix_cl_float wei(weights);
    matrix_cl_float del(deltas);
    matrix_cl_float del_r(deltas);
    matrix_cl_float wei_t(weights_transposed);
    
    transposeWeights();
    // Testing transpose
    // wei.set(elementsPerLayer[0], elementsPerLayer[1], 0);
    // print(wei, "W");
    // wei_t.set(elementsPerLayer[1], elementsPerLayer[0], 0);
    // print(wei_t, "WT");


    // first of all calculate the deltas of the last layer
    tm.set(numberOfTrainingData, elementsPerLayer[N], 0);
    const int offset_act = (numberOfNeurons - elementsPerLayer[N])
                           *numberOfTrainingData;
    act.set(tm.rows, tm.cols, offset_act);

    const int offset_del_r = (numberOfNeurons - elementsPerLayer[0]
                            - elementsPerLayer[N])*numberOfTrainingData;
    del_r.set(numberOfTrainingData, elementsPerLayer[N], offset_del_r);

    openclKernels->runElementWiseSubstract(tm, act, del_r);

    // del_r.data.readFromDevice(*queue);
    // Testing t-y
    // print(tm, "t");
    // print(act, "y");
    // print(del_r, "t-y");


    wei_t.offset = wei_t.data.hostData.size();

    for (int i = N - 1; i > 1; i--) {
        del.set(del_r.rows, del_r.cols, del_r.offset);
        del_r.set(numberOfTrainingData, elementsPerLayer[i],
                  del_r.offset - numberOfTrainingData*elementsPerLayer[i]);
        wei_t.set(elementsPerLayer[i+1], elementsPerLayer[i],
                  wei_t.offset - elementsPerLayer[i]*elementsPerLayer[i+1]);
        openclKernels->
            runMatrixMultiplicationSigmoid(del, wei_t, del_r, false, false);

        wei_t.data.readFromDevice(*queue);
        del.data.readFromDevice(*queue);
        del_r.data.readFromDevice(*queue);
        print(wei_t, "Wt");
        print(del, "delta");
        print(del_r, "delta*Wt");
        
        act.set(numberOfTrainingData, elementsPerLayer[i],
                act.offset - elementsPerLayer[i]*numberOfTrainingData);
        openclKernels->
            runElementWiseMultiplicationBySigmoidDerivativeKernel(del_r, act);

        print(act, "act");
        del_r.data.readFromDevice(*queue);
        print(del_r, "delta_r*Wt.*s*(s-1)");
    }
    
    // Weight actualization
    act.offset = (numberOfNeurons - elementsPerLayer[N])
                     *numberOfTrainingData;
    del.offset = (numberOfNeurons - elementsPerLayer[0])
                     *numberOfTrainingData;
    wei.offset = wei.data.hostData.size();
    for (int i = N - 1; i > 1; i--) {
        act.set(elementsPerLayer[i], numberOfTrainingData,
                act.offset - elementsPerLayer[i]*numberOfTrainingData);
        del.set(numberOfTrainingData, elementsPerLayer[i+1],
                del.offset - elementsPerLayer[i+1]*numberOfTrainingData);
        wei.set(elementsPerLayer[i], elementsPerLayer[i+1],
                  wei.offset - elementsPerLayer[i]*elementsPerLayer[i+1]);

        wei.data.readFromDevice(*queue);
        print(wei, "Wei");

        const bool AColMajor = true;
        const bool SumToWeights = true;
        const cl_float weightIncrementMultiplier = 1.0f;
        openclKernels->
            runMatrixMultiplicationSigmoid(act, del, wei, false, false,
                                           AColMajor, false, SumToWeights,
                                           weightIncrementMultiplier);

        act.data.readFromDevice(*queue);
        del.data.readFromDevice(*queue);
        wei.data.readFromDevice(*queue);
        print(act, "Act");
        print(del, "delta");
        print(wei, "Wei = Wei + 1.0*ActT*delta");
    }
}

cl_float nn::cross_entropy() {
    matrix_cl_float tm(t);
    matrix_cl_float act(activations);
    matrix_cl_float ce(cross_entropy_error);

    tm.set(numberOfTrainingData, elementsPerLayer[numberOfLayers-1], 0);

    const int offset_act = (numberOfNeurons -
                            elementsPerLayer[numberOfLayers-1])
                           *numberOfTrainingData;
    act.set(tm.rows, tm.cols, offset_act);

    return openclKernels->runCrossEntropy(tm, act, ce);
}
