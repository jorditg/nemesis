// Copyright 2014 <Jordi de la Torre>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include <cassert>
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>

#include "nn.hpp"
#include "OpenCLKernels.hpp"
#include "common.hpp"

/**

 * Sparse random initialization (Martens, 2010)

 */

void nn::populate_sparse_weights() {
  boost::mt19937 rng;
  boost::normal_distribution<> nd(0.0f, 1.0f);
  boost::variate_generator<boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);

  const cl_uint init_elements = 15;

  for (std::vector<cl_float>::iterator it = weights.hostData.begin();
       it != weights.hostData.end(); ++it)
    *it = 0.0;

  for(cl_uint i = 1; i < numberOfLayers; i++) {
      boost::random::uniform_real_distribution<> dist(0, elementsPerLayer[i-1]);
      for(cl_uint to_idx = 0; to_idx < elementsPerLayer[i]; to_idx++) {         
          for(cl_uint k = 0; k < init_elements; k++) {
              cl_uint from_idx = dist(rng);
              weights.hostData[weights_offsets[i-1] +
                               elementsPerLayer[i] * from_idx +
                               to_idx] = var_nor();
          }
          // set biases to 0
          weights.hostData[weights_offsets[i-1] + to_idx] = 0.0;
      }
  }
}

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
                increment_weights(increment_weights_host),
                //  weights_transposed(weights_transposed_host),
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
    for ( cl_uint i = 0; i < numberOfLayers-1; i++ ) {
        numberOfWeights += elementsPerLayer[i]*elementsPerLayer[i+1];
        numberOfNeurons += elementsPerLayer[i];
    }
    numberOfNeurons += elementsPerLayer[numberOfLayers-1];
    
    activations.hostData.resize(numberOfNeurons*numberOfTrainingData);
    weights.hostData.resize(numberOfWeights);
    increment_weights.hostData.resize(numberOfWeights);
    //weights_transposed.hostData.resize(numberOfWeights);
    // there are no deltas in input layer
    deltas.hostData.resize((numberOfNeurons
                            -elementsPerLayer[0])*numberOfTrainingData);
    cross_entropy_error.hostData.resize(CROSS_ENTROPY_ERROR_SIZE);

    // calculate offsets of every layer inside the vectors
    activations_offsets.resize(numberOfLayers);
    deltas_offsets.resize(numberOfLayers);
    weights_offsets.resize(numberOfLayers-1);
    activations_offsets[0] = 0;
    weights_offsets[0] = 0;
    deltas_offsets[0] = 0;   // never used in the algorithm
    deltas_offsets[1] = 0;
    for (cl_uint i = 1; i < numberOfLayers; i++) {
      activations_offsets[i] = activations_offsets[i-1] +
                               numberOfTrainingData*elementsPerLayer[i-1];
      weights_offsets[i] = weights_offsets[i-1] +
                           elementsPerLayer[i-1]*elementsPerLayer[i];
      deltas_offsets[i] = activations_offsets[i] - activations_offsets[1];
    }
    
    //load_csv_vector("weights.txt", weights.hostData);
    populate_random_weights(0.00001, 0.00009);


    // device memory allocation
    // Create OpenCL buffers for the matrices based on allocated memory regions
    // Create buffers with CL_MEM_USE_HOST_PTR to minimize copying and
    // model situation when matrices are hosted by some native library that
    // uses OpenCL to accelerate calculations.
    
    // Create buffers and copy host contents
    activations.createBuffer(*context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR);
    weights.createBuffer(*context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR);
    increment_weights.createBuffer(*context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR);
//    weights_transposed.createBuffer(*context,
//                                    CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR);
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

//void nn::transposeWeights() {
//    // done in host
//    weights.readFromDevice(*queue);
//    int offset = 0;
//    for (int l = 0; l < numberOfLayers - 1; l++) {
//        for (int i = 0; i < elementsPerLayer[l] ; i++) {
//            for (int j = 0; j < elementsPerLayer[l+1]; j++) {
//                weights_transposed.hostData[offset + i +
//                                            j*elementsPerLayer[l]] =
//                    weights.hostData[offset + j +
//                                     i*elementsPerLayer[l+1]];
//            }
//        }
//        offset += elementsPerLayer[l]*elementsPerLayer[l+1];
//    }
//    weights_transposed.writeToDevice(*queue);
//}

void nn::FF() {
    const cl_uint N = numberOfLayers - 1;
    
    matrix_cl_float A(activations);
    matrix_cl_float B(weights);
    matrix_cl_float C(activations);
    
    A.offset = 0;
    A.rows = numberOfTrainingData;
    B.offset = 0;
    C.offset = elementsPerLayer[0]*numberOfTrainingData;
    for ( cl_uint i = 0; i < N; i++ ) {
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
 
    //C.data.readFromDevice(*queue);  // copy calculatied activations back to host
    //print_vector(C.data.hostData, C.rows, C.cols, C.offset);
}

void nn::BP() {
    matrix_cl_float tm(t);
    matrix_cl_float act(activations);
    matrix_cl_float wei(weights);
    matrix_cl_float wei_inc(increment_weights);
    matrix_cl_float del(deltas);
    matrix_cl_float del_r(deltas);

    // first of all calculate the deltas of the last layer
    const cl_uint last = numberOfLayers - 1;
    tm.set(numberOfTrainingData, elementsPerLayer[last], 0);
    act.set(tm.rows, tm.cols, activations_offsets[last]);
    del_r.set(numberOfTrainingData,
              elementsPerLayer[last],
              deltas_offsets[last]);

    openclKernels->runElementWiseSubstract(tm, act, del_r);
    
    for (cl_int i = numberOfLayers - 2; i > 0; i--) {
        del.set(numberOfTrainingData,
                elementsPerLayer[i+1],
                deltas_offsets[i+1]);
        del_r.set(numberOfTrainingData,
                  elementsPerLayer[i],
                  deltas_offsets[i]);
        // wei transposed
        wei.set(elementsPerLayer[i+1],
                elementsPerLayer[i],
                weights_offsets[i],
                true);
        openclKernels->
            runMatrixMultiplicationSigmoid(del, wei, del_r, false, false);

        //wei.data.readFromDevice(*queue);
        //del.data.readFromDevice(*queue);
        //del_r.data.readFromDevice(*queue);
        //print(wei, "W", true);
        //print(del, "delta");
        //print(del_r, "delta*Wt");
        
        act.set(numberOfTrainingData,
                elementsPerLayer[i],
                activations_offsets[i]);
        openclKernels->
            runElementWiseMultiplicationBySigmoidDerivativeKernel(del_r, act);

        //print(act, "act");
        //del_r.data.readFromDevice(*queue);
        //print(del_r, "delta_r*Wt.*s*(s-1)");
    }
    
    // Weight actualization
    for (cl_int i = numberOfLayers - 2; i >= 0; i--) {
        // act transposed
        act.set(elementsPerLayer[i], numberOfTrainingData,
                activations_offsets[i], true);
        del.set(numberOfTrainingData, elementsPerLayer[i+1],
                deltas_offsets[i+1]);
        wei.set(elementsPerLayer[i], elementsPerLayer[i+1],
                weights_offsets[i]);
        wei_inc.set(elementsPerLayer[i], elementsPerLayer[i+1],
                    weights_offsets[i]);
        

        //wei.data.readFromDevice(*queue);
        //print(wei, "Wei");

        const bool sum = true;
        const cl_float learningRateOverTrainingData =
                       learningRate/cl_float(numberOfTrainingData);
        
        openclKernels->runMatrixMultiplicationSigmoid(
                            act,
                            del,
                            wei_inc,
                            false,
                            false,
                            sum,
                            momentum,
                            learningRateOverTrainingData);
        
        openclKernels->runElementWiseSum(wei, wei_inc, wei);

        //act.data.readFromDevice(*queue);
        //del.data.readFromDevice(*queue);
        //wei.data.readFromDevice(*queue);
        //print(act, "Act", true);
        //print(del, "delta");
        //print(wei, "Wei = Wei + 1.0*ActT*delta");
    }
}

void nn::train() {
    
    for (size_t epoch = 0; epoch < maxEpochs; epoch++) {
        FF();
        cl_float ce = cross_entropy();
        if (epoch % printEpochs == 0) {
            std::cout << "Epoch: " << epoch << "   CE: " << ce << std::endl;
        }
        if (ce < minError) {
            std::cout << "Epoch: " << epoch << "   CE: " << ce << std::endl;
            break;
        }
        BP();
    }
}

cl_float nn::cross_entropy() {
    matrix_cl_float tm(t);
    matrix_cl_float act(activations);
    matrix_cl_float ce(cross_entropy_error);

    tm.set(numberOfTrainingData, elementsPerLayer[numberOfLayers-1], 0);

    const cl_uint offset_act = (numberOfNeurons -
                                elementsPerLayer[numberOfLayers-1])
                               *numberOfTrainingData;
    act.set(tm.rows, tm.cols, offset_act);

    return openclKernels->runCrossEntropy(tm, act, ce);
}

void nn::test_matrix_multiplication(const cl_uint nr_rows_A,
                                    const cl_uint nr_cols_A,
                                    const cl_uint nr_rows_B,
                                    const cl_uint nr_cols_B) {
    matrix_cl_float A(deltas);
    matrix_cl_float B(activations);
    matrix_cl_float C(weights);
    
    assert(nr_rows_A % 4 == 0 &&
           nr_rows_B % 4 == 0 &&
           nr_cols_A % 4 == 0 &&
           nr_cols_B % 4 == 0 &&
           nr_cols_A == nr_rows_B);
    
    // Test of not transposed matrices ( CHECKED. IT WORKS!)
    
    const cl_uint nr_rows_C = nr_rows_A;
    const cl_uint nr_cols_C = nr_cols_B;
    
    for (cl_uint i = 0; i < nr_rows_A; i++) {
        for (cl_uint j = 0; j < nr_cols_A; j++) {
            A.data.hostData[j + nr_cols_A*i] = cl_float(j+1);
        }
    }
    
    for (cl_uint i = 0; i < nr_rows_B; i++) {
        for (cl_uint j = 0; j < nr_cols_B; j++) {
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
                                           false, false);
    C.data.readFromDevice(*queue);

    print(A, "A");
    print(B, "B");
    print(C, "Not transposed matrices");

    // Test A transposed
    
    for (cl_uint i = 0; i < nr_cols_A; i++) {
        for (cl_uint j = 0; j < nr_rows_A; j++) {
            A.data.hostData[j + nr_rows_A*i] = cl_float(i+1);
        }
    }
    
    for (cl_uint i = 0; i < nr_rows_B; i++) {
        for (cl_uint j = 0; j < nr_cols_B; j++) {
            B.data.hostData[j + nr_cols_B*i] = 1.0f/cl_float(i+1);
        }
    }
    
    A.set(nr_rows_A, nr_cols_A, 0, true);
    B.set(nr_rows_B, nr_cols_B, 0);
    C.set(nr_rows_C, nr_cols_C, 0);

    A.data.writeToDevice(*queue);
    B.data.writeToDevice(*queue);
    
    openclKernels->
            runMatrixMultiplicationSigmoid(A, B, C, false, false);
    C.data.readFromDevice(*queue);

    print(A, "A", true);
    print(B, "B");
    print(C, "Result with A transposed");

    // Test B transposed
    
    for (cl_uint i = 0; i < nr_rows_A; i++) {
        for (cl_uint j = 0; j < nr_cols_A; j++) {
            A.data.hostData[j + nr_cols_A*i] = cl_float(j+1);
        }
    }
    
    for (cl_uint i = 0; i < nr_cols_B; i++) {
        for (cl_uint j = 0; j < nr_rows_B; j++) {
            B.data.hostData[j + nr_rows_B*i] = 1.0f/cl_float(j+1);
        }
    }
    
    A.set(nr_rows_A, nr_cols_A, 0);
    B.set(nr_rows_B, nr_cols_B, 0, true);
    C.set(nr_rows_C, nr_cols_C, 0);

    A.data.writeToDevice(*queue);
    B.data.writeToDevice(*queue);
    
    openclKernels->
            runMatrixMultiplicationSigmoid(A, B, C, false, false);
    C.data.readFromDevice(*queue);

    print(A, "A");
    print(B, "B", true);
    print(C, "Result with B transposed");

    // Test A and B transposed
    
    for (cl_uint i = 0; i < nr_cols_A; i++) {
        for (cl_uint j = 0; j < nr_rows_A; j++) {
            A.data.hostData[j + nr_rows_A*i] = cl_float(i+1);
        }
    }
    
    for (cl_uint i = 0; i < nr_cols_B; i++) {
        for (cl_uint j = 0; j < nr_rows_B; j++) {
            B.data.hostData[j + nr_rows_B*i] = 1.0f/cl_float(j+1);
        }
    }
    
    A.set(nr_rows_A, nr_cols_A, 0, true);
    B.set(nr_rows_B, nr_cols_B, 0, true);
    C.set(nr_rows_C, nr_cols_C, 0);

    A.data.writeToDevice(*queue);
    B.data.writeToDevice(*queue);
    
    openclKernels->
            runMatrixMultiplicationSigmoid(A, B, C, false, false);
    C.data.readFromDevice(*queue);

    print(A, "A", true);
    print(B, "B", true);
    print(C, "Result with A and B transposed");
    
    exit(0);
    
}
