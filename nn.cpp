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

void nn::load_weights(const std::string & filename,
                       std::vector<cl_float> &weights) {
    std::ifstream in(filename.c_str());
    if (!in.is_open()) {
        std::cout << "File already opened. Exiting\n";
        exit(1);
    }

    std::string line;

    typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;
    std::vector< std::string > vec;
    std::vector<cl_float>::iterator wit = weights.begin();
    
    while (getline(in, line)) {
        Tokenizer tok(line);
        vec.assign(tok.begin(), tok.end());
        // vector now contains strings from one row, output to cout here
        // std::copy(vec.begin(), vec.end(),
        //           std::ostream_iterator<std::string>(std::cout, "|"));
        // std::cout << "\n----------------------" << std::endl;

        // check that there is not incomplete data

        
        for (std::vector<std::string>::iterator it = vec.begin();
             it != vec.end(); ++it) {
          *wit = std::stof(*it);
          wit++;
        }
    }
    
    assert(wit == weights.end());
}


void nn::load_csv_data(const std::string & filename,
                       std::vector<cl_float> & input,
                       std::vector<cl_float> & output,
                       cl_int &rows, cl_int &layers,
                       std::vector<cl_int> &elements) {
    std::ifstream in(filename.c_str());
    if (!in.is_open()) {
        std::cout << "File already opened. Exiting\n";
        exit(1);
    }

    std::string line;

    getline(in, line);               // read number of data lines
    rows = std::stoi(line);
        
    typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;
    std::vector< std::string > vec;

    getline(in, line);              // read elements per layer

    Tokenizer tok(line);
    vec.assign(tok.begin(), tok.end());

    layers = 0;
    for (std::vector<std::string>::iterator it = vec.begin() ;
         it != vec.end(); ++it) {
        elements.push_back(std::stoi(*it));
        layers++;
    }

    
    const cl_int cols = elements[0] + elements[elements.size()-1];
    // cols to read = number of inputs + number of outputs

    cl_int n = 0;
    while (getline(in, line)) {
        Tokenizer tok(line);
        vec.assign(tok.begin(), tok.end());
        // vector now contains strings from one row, output to cout here
        // std::copy(vec.begin(), vec.end(),
        //           std::ostream_iterator<std::string>(std::cout, "|"));
        // std::cout << "\n----------------------" << std::endl;

        // check that there is not incomplete data
        assert(vec.size() == size_t(cols));

        cl_int i = 0;
        for (std::vector<std::string>::iterator it = vec.begin();
             it != vec.end(); ++it) {
            if (i < elements[0]) input.push_back(std::stof(*it));
            else
              output.push_back(std::stof(*it));
            i++;
        }
        n++;
        if (n == rows) break;
    }
    
    assert((input.size() / size_t(elements[0])) == size_t(rows));
}

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
    
    opencl_initialize();
    
    // load input data into host memory
    load_csv_data(filename, inputs, t, numberOfTrainingData,
                  numberOfLayers, elementsPerLayer);
    
    // host memory allocation for neural network weights
    cl_uint numberOfWeights = 0;
    for ( cl_int i = 0; i < numberOfLayers-1; i++ )
        numberOfWeights += elementsPerLayer[i]*elementsPerLayer[i+1];
    weights.resize(numberOfWeights);

    load_weights("weights.txt", weights);

    // outputs buffer
    cl_uint maxLayerNeurons = *std::max_element(std::begin(elementsPerLayer)+1,
                                               std::end(elementsPerLayer));
    output1.resize(maxLayerNeurons*numberOfTrainingData);
    output2.resize(maxLayerNeurons*numberOfTrainingData);

    // weight initialization
    //    ???  PENDING ¿¿¿

    // device memory allocation
    opencl_device_memory_allocation();

    // instantitate kernels
    matmult = new OpenCLMatrixMultiplication(*context, devices, 0, *queue);
    ce = new OpenCLErrorReduce(*context, devices, *queue, y, t);
};

nn::~nn() {
    opencl_cleanup();
}

/**
 * Initializes queue, device and context variables
 */
void nn::opencl_initialize() {
 
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);  // get available OpenCL platforms
    
    // get OpenCL devices for first platform
    platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
        
    // create a context for these devices
    context = new cl::Context(devices);
   
    // Create queue of first device
    queue = new cl::CommandQueue(*context, devices[0]);
}

void nn::opencl_device_memory_allocation() {
    
    // Create OpenCL buffers for the matrices based on allocated memory regions
    // Create buffers with CL_MEM_USE_HOST_PTR to minimize copying and
    // model situation when matrices are hosted by some native library that
    // uses OpenCL to accelerate calculations.
    
    // Create buffers and copy host contents
    
    inputsBuffer = new cl::Buffer(*context,
                                  CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                                  inputs.size()*sizeof(cl_float),
                                  &inputs[0]);

    weightsBuffer = new cl::Buffer(*context,
                                   CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                   weights.size()*sizeof(cl_float),
                                   &weights[0]);
    
    tBuffer = new cl::Buffer(*context,
                             CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                             t.size()*sizeof(cl_float),
                             &t[0]);
            
    outputBuffer1 = new cl::Buffer(*context,
                                   CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                   output1.size()*sizeof(cl_float),
                                   &output1[0]);

    outputBuffer2 = new cl::Buffer(*context,
                                   CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                   output2.size(),
                                   &output2[0]);
}

void nn::opencl_cleanup() {
 
    delete inputsBuffer;
    delete weightsBuffer;
    delete tBuffer;
    delete outputBuffer1;
    delete outputBuffer2;
    
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


matrix_type nn::FF_get_1st_matrix_for_product(cl_uint order) {
    if (order == 0) {
        return matrix_type(*inputsBuffer,
                           numberOfTrainingData,           
                           elementsPerLayer[0]
                           );
    } else {
        if (order % 2) {
            return matrix_type(*outputBuffer1,                               
                               numberOfTrainingData,
                               elementsPerLayer[order]);
        } else {
            return matrix_type(*outputBuffer2,
                               numberOfTrainingData,
                               elementsPerLayer[order]);
        }
    }
}

matrix_type nn::FF_get_2nd_matrix_for_product(cl_uint order) {

    // calculate distance from origin of the weight matrix
    cl_uint offset = 0;
    for (cl_uint i = 0; i < order; i++)
        offset += elementsPerLayer[i]*elementsPerLayer[i+1];
    //offset++;

    return matrix_type(*weightsBuffer,
                       elementsPerLayer[order],
                       elementsPerLayer[order+1],
                       offset);

}


matrix_type nn::FF_get_result_matrix_for_product(cl_uint order) {
    return matrix_type(outputBuffer(order),
                       numberOfTrainingData,
                       elementsPerLayer[order+1]);
}


void nn::device2HostWeightsTransfer() {
    const size_t size = weights.size()*sizeof(cl_float);

    queue->enqueueMapBuffer(*weightsBuffer,
                            CL_TRUE,    // blocking map
                            CL_MAP_READ,
                            0,
                            size);

    // Finish here is only required for correct time measurment
    // on the next iteration
    // It does not affect correctness of calculations because
    // you use the in-order OpenCL queue here.
    queue->finish();
}


void nn::device2HostOutput1Transfer() {
    const size_t size = output1.size()*sizeof(cl_float);

    queue->enqueueMapBuffer(*outputBuffer1,
                            CL_TRUE,  // blocking map
                            CL_MAP_READ,
                            0,
                            size);

    // Finish here is only required for correct time measurment
    // on the next iteration
    // It does not affect correctness of calculations because
    // you use the in-order OpenCL queue here.

    queue->finish();
}

void nn::device2HostOutput2Transfer() {
    const size_t size = output2.size()*sizeof(cl_float);

    queue->enqueueMapBuffer(*outputBuffer2,
                            CL_TRUE,  // blocking map
                            CL_MAP_READ,
                            0,
                            size);

    // Finish here is only required for correct time measurment
    // on the next iteration
    // It does not affect correctness of calculations because
    // you use the in-order OpenCL queue here.
    queue->finish();
}

void nn::device2HostTransfer(const cl::Buffer & buffer, size_t size) {
    queue->enqueueMapBuffer(buffer,
                            CL_TRUE,  // blocking map
                            CL_MAP_READ,
                            0,
                            size);

    // Finish here is only required for correct time measurment
    // on the next iteration
    // It does not affect correctness of calculations because
    // you use the in-order OpenCL queue here.
    queue->finish();
}

std::vector<cl_float> & nn::FF() {
    const cl_uint N = get_number_of_product_matrices();
    
    for ( cl_uint i = 0; i < N; i++ ) {
        matrix_type A = FF_get_1st_matrix_for_product(i);
        matrix_type B = FF_get_2nd_matrix_for_product(i);
        matrix_type C = FF_get_result_matrix_for_product(i);
        matmult->run(A, B, C);
    }


    std::vector<cl_float> &result = output(N-1);
    // transferimos los datos finales calculados de device a host
    device2HostTransfer(outputBuffer(N-1), result.size());
    // devolvemos referencia a vector output en host que contiene
    // los resultados finales
    
    print_vector(result, 8);

    return result;
}
