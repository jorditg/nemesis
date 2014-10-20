// Copyright 2014 <Jordi de la Torre>

#include <cassert>

#include <vector>
#include <algorithm>
#include <string>
#include <iostream>     // cout, endl
#include <fstream>      // fstream
#include <boost/tokenizer.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

#include "nn.hpp"


void nn::load_csv_data(const std::string & filename,
                       std::vector<cl_float> & input,
                       std::vector<cl_float> & output,
                       cl_uint &rows, cl_uint &layers,
                       std::vector<cl_uint> &elements) {
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

    
    const cl_uint cols = elements[0] + elements[elements.size()-1];
    // cols to read = number of inputs + number of outputs

    size_t n = 0;
    while (getline(in, line)) {
        Tokenizer tok(line);
        vec.assign(tok.begin(), tok.end());
        // vector now contains strings from one row, output to cout here
        // std::copy(vec.begin(), vec.end(),
        //           std::ostream_iterator<std::string>(std::cout, "|"));
        // std::cout << "\n----------------------" << std::endl;
        assert(vec.size() == cols);  // check that there is not incomplete data

        unsigned i = 0;
        for (std::vector<std::string>::iterator it = vec.begin();
             it != vec.end(); ++it) {
            if (i < elements[0]) input.push_back(std::stof(*it));
            else
              output.push_back(std::stof(*it));
            i++;
        }
        n++;
        if(n==rows) break;
    }
    
    assert((input.size() / elements[0]) == rows);
}

void nn::populate_random_weights(cl_float min, cl_float max) {
  boost::random::mt19937 gen;
  boost::random::uniform_real_distribution<> dist(min, max);
  
  for (std::vector<cl_float>::iterator it = weights.begin() ;
       it != weights.end(); ++it)
    *it = dist(gen);
}

nn::nn(const std::string &filename) {
    
    oclobjects = new OpenCLBasic("Intel", "all", "0");
    
    // OpenCL matrix multiplication compilation (local version)
const string program_file_name = "MatrixMultiplication_Kernels.cl";
    const string program_text = "";
    const string kernel_name_local = "mmmKernel_local";
    const string build_options = "";
    matmult_local = new OpenCLProgramOneKernel(*oclobjects,
                                         program_file_name,
                                         program_text,
                                         kernel_name_local,
                                         build_options);
    
    // OpenCL matrix multiplication compilation (non-local version)
    const string kernel_name = "mmmKernel";
    matmult = new OpenCLProgramOneKernel(*oclobjects,
                                         program_file_name,
                                         program_text,
                                         kernel_name,
                                         build_options);
    
    // load input data into host memory
    load_csv_data(filename, inputs, t, numberOfTrainingData,
                  numberOfLayers, elementsPerLayer);
    
    // host memory allocation for neural network weights
    cl_uint numberOfWeights = 0;
    for ( cl_uint i = 0; i < numberOfLayers-1; i++ )
        numberOfWeights += elementsPerLayer[i]*elementsPerLayer[i+1];
    weights.resize(numberOfWeights);
    
    // outputs buffer
    cl_uint maxLayerNeurons = *std::max_element(std::begin(elementsPerLayer)+1,
                                               std::end(elementsPerLayer));
    output1.resize(maxLayerNeurons*numberOfTrainingData);
    output2.resize(maxLayerNeurons*numberOfTrainingData);

    // weight initialization
    //    ???  PENDING ¿¿¿

    // device memory allocation
    device_memory_allocation();
};

nn::~nn() {
    // cl_int err;

    // err = clEnqueueUnmapMemObject(oclobjects->queue,
    //                               inputsBuffer.device,
    //                               &inputs[0], 0, 0, 0);
    // SAMPLE_CHECK_ERRORS(err);
    // err = clEnqueueUnmapMemObject(oclobjects->queue,
    //                               weightsBuffer.device,
    //                               &weights[0], 0, 0, 0);
    // SAMPLE_CHECK_ERRORS(err);
    // err = clEnqueueUnmapMemObject(oclobjects->queue,
    //                               tBuffer.device,
    //                               &t[0], 0, 0, 0);
    // SAMPLE_CHECK_ERRORS(err);
    // err = clEnqueueUnmapMemObject(oclobjects->queue,
    //                               outputBuffer1.device,
    //                               &output1[0], 0, 0, 0);
    // SAMPLE_CHECK_ERRORS(err);
    // err = clEnqueueUnmapMemObject(oclobjects->queue,
    //                               outputBuffer2.device,
    //                               &output2[0], 0, 0, 0);
    // SAMPLE_CHECK_ERRORS(err);
        
    delete oclobjects;
    delete matmult;
}

void nn::device_memory_allocation() {
    cl_int err = 0;  // OpenCL error code
    
    // Create OpenCL buffers for the matrices based on allocated memory regions
    // Create buffers with CL_MEM_USE_HOST_PTR to minimize copying and
    // model situation when matrices are hosted by some native library that
    // uses OpenCL to accelerate calculations.
    
    inputsBuffer.device = clCreateBuffer(
        oclobjects->context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        inputs.size()*sizeof(cl_float),
        &inputs[0],
        &err);
    SAMPLE_CHECK_ERRORS(err);

    weightsBuffer.device = clCreateBuffer(
        oclobjects->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        weights.size()*sizeof(cl_float),
        &weights[0],
        &err);
    SAMPLE_CHECK_ERRORS(err);

    tBuffer.device = clCreateBuffer(
        oclobjects->context,
        CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
        t.size()*sizeof(cl_float),
        &t[0],
        &err);
    SAMPLE_CHECK_ERRORS(err);
            
    outputBuffer1.device = clCreateBuffer(
        oclobjects->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        output1.size()*sizeof(cl_float),
        &output1[0],
        &err);
    SAMPLE_CHECK_ERRORS(err);

    outputBuffer2.device = clCreateBuffer(
        oclobjects->context,
        CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
        output2.size(),
        &output2[0],
        &err);
    SAMPLE_CHECK_ERRORS(err);
        
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
    matrix_type ret;
    if (order == 0) {
        ret.data = inputsBuffer.device;
        ret.width = elementsPerLayer[0];
        ret.heigth = numberOfTrainingData;
    } else {
        if (order % 2) {
            ret.data = outputBuffer1.device;
            ret.width = elementsPerLayer[order];
            ret.heigth = numberOfTrainingData;
        } else {
            ret.data = outputBuffer2.device;
            ret.width = elementsPerLayer[order];
            ret.heigth = numberOfTrainingData;
        }
    }
    return ret;
}

matrix_type nn::FF_get_2nd_matrix_for_product(cl_uint order) {
    matrix_type ret;

    // calculate distance from origin of the weight matrix
    cl_uint offset = 0;
    for (cl_uint i = 0; i < order; i++)
        offset += elementsPerLayer[i]*elementsPerLayer[i+1];
    offset++;

    // ??¿¿ PENDING ??¿¿
    // offset to be IMPLEMENTED

    ret.data = weightsBuffer.device;  // + offset;
    ret.width = elementsPerLayer[order+1];
    ret.heigth = elementsPerLayer[order];

    return ret;
}

matrix_type nn::FF_get_result_matrix_for_product(cl_uint order) {
    matrix_type ret;

    ret.width = elementsPerLayer[order+1];
    ret.heigth = numberOfTrainingData;

    if (order % 2) {
        ret.data = outputBuffer2.device;
    } else {
        ret.data = outputBuffer1.device;
    }
    return ret;
}


void nn::execute_mat_mult_mmmKernel_local(matrix_type const &A,
                                          matrix_type const &B,
                                          matrix_type const &result) {
    cl_int err;
    
    const size_t blockSize = 8;
    
    const size_t global_size[2] = {size_t(result.width/4), size_t(result.heigth/4)};
    // float4 elements in kernel

    const size_t local_size[2] = {blockSize, blockSize };

    // -----------------------------------------------------------------------
    // Setting kernel arguments
    // -----------------------------------------------------------------------

    err = clSetKernelArg(matmult->kernel, 0, sizeof(cl_mem), &A.data);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(matmult->kernel, 1, sizeof(cl_mem), &B.data);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(matmult->kernel, 2, sizeof(cl_mem), &result.data);
    SAMPLE_CHECK_ERRORS(err);

    // PENDING !!!!!!!!!!!!!!!!!!
    //    err = clSetKernelArg(matmult->kernel, 3, sizeof(cl_uint), &A.width);
    //SAMPLE_CHECK_ERRORS(err);
    //err = clSetKernelArg(matmult->kernel, 4, sizeof(cl_uint), &B.width);
    //SAMPLE_CHECK_ERRORS(err);

    // -----------------------------------------------------------------------
    // Define ndrange iteration space: global and local sizes based on
    // parameters obtained from user

    // Refer to the sample documentation for clarification about
    // how work is devided among work-groups and work-items.
    // -----------------------------------------------------------------------
  
    std::cout << "Launching for device\n" 
            << " (global size: " << global_size[0] << ", "  << global_size[1] << ")\n"
            << " (local size: " << local_size[0] << ", "  << local_size[1] << ")\n";
    

    err = clEnqueueNDRangeKernel(oclobjects->queue,
                                 matmult_local->kernel, 2, 0,
                                 global_size, local_size, 0, 0, 0);
    SAMPLE_CHECK_ERRORS(err);

    err = clFinish(oclobjects->queue);
    SAMPLE_CHECK_ERRORS(err);

    std::cout << "Matmult finished\n";
}

void nn::execute_mat_mult_mmmKernel(matrix_type const &A,
                                    matrix_type const &B,
                                    matrix_type const &result) {
    cl_int err;
    
    const size_t blockSize = 8;
    
    const size_t global_size[2] = {size_t(result.width/4), size_t(result.heigth/4)};
    // float4 elements in kernel

    const size_t local_size[2] = {blockSize, blockSize};

    // -----------------------------------------------------------------------
    // Setting kernel arguments
    // -----------------------------------------------------------------------

    err = clSetKernelArg(matmult->kernel, 0, sizeof(cl_mem), &A.data);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(matmult->kernel, 1, sizeof(cl_mem), &B.data);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(matmult->kernel, 2, sizeof(cl_mem), &result.data);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(matmult->kernel, 3, sizeof(cl_int), &A.width);
    SAMPLE_CHECK_ERRORS(err);
    err = clSetKernelArg(matmult->kernel, 4, sizeof(cl_int), &B.width);
    SAMPLE_CHECK_ERRORS(err);

    // -----------------------------------------------------------------------
    // Define ndrange iteration space: global and local sizes based on
    // parameters obtained from user.

    // Refer to the sample documentation for clarification about
    // how work is devided among work-groups and work-items.
    // -----------------------------------------------------------------------
  
    std::cout << "Launching for device\n" 
            << " (global size: " << global_size[0] << ", "  << global_size[1] << ")\n"
            << " (local size: " << local_size[0] << ", "  << local_size[1] << ")\n";

    err = clEnqueueNDRangeKernel(oclobjects->queue,
                                 matmult->kernel, 2, 0,
                                 global_size, local_size, 0, NULL, NULL);
    SAMPLE_CHECK_ERRORS(err);

    err = clFinish(oclobjects->queue);
    SAMPLE_CHECK_ERRORS(err);
    
    std::cout << "Matmult finished\n";
}

void nn::device2HostWeightsTransfer() {
    cl_int err;
    
    const size_t size = weights.size()*sizeof(cl_float);

    clEnqueueMapBuffer(oclobjects->queue,
                       weightsBuffer.device,
                       CL_TRUE,    // blocking map
                       CL_MAP_READ, 0, size, 0, 0, 0, &err);
    SAMPLE_CHECK_ERRORS(err);

    // Finish here is only required for correct time measurment
    // on the next iteration
    // It does not affect correctness of calculations because
    // you use the in-order OpenCL queue here.
    err = clFinish(oclobjects->queue);
    SAMPLE_CHECK_ERRORS(err);
}


void nn::device2HostOutput1Transfer() {
    cl_int err;

    const size_t size = output1.size()*sizeof(cl_float);

    clEnqueueMapBuffer(oclobjects->queue,
                       outputBuffer1.device, CL_TRUE,  // blocking map
                       CL_MAP_READ, 0, size, 0, 0, 0, &err);
    SAMPLE_CHECK_ERRORS(err);

    // Finish here is only required for correct time measurment
    // on the next iteration
    // It does not affect correctness of calculations because
    // you use the in-order OpenCL queue here.

    err = clFinish(oclobjects->queue);
    SAMPLE_CHECK_ERRORS(err);
}

void nn::device2HostOutput2Transfer() {
    cl_int err;
    
    const size_t size = output2.size()*sizeof(cl_float);

    clEnqueueMapBuffer(oclobjects->queue,
                       outputBuffer2.device, CL_TRUE,  // blocking map
                       CL_MAP_READ, 0, size, 0, 0, 0, &err);
    SAMPLE_CHECK_ERRORS(err);

    // Finish here is only required for correct time measurment
    // on the next iteration
    // It does not affect correctness of calculations because
    // you use the in-order OpenCL queue here.
    err = clFinish(oclobjects->queue);
    SAMPLE_CHECK_ERRORS(err);
}

void nn::FF() {
    for ( cl_uint i = 0; i < get_number_of_product_matrices(); i++ ) {
        matrix_type A = FF_get_1st_matrix_for_product(i);
        matrix_type B = FF_get_2nd_matrix_for_product(i);
        matrix_type C = FF_get_result_matrix_for_product(i);
        execute_mat_mult_mmmKernel(A, B, C);
    }
}
