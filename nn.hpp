// Copyright 2014 <Jordi de la Torre>

#ifndef NN_HPP_
#define NN_HPP__


#include <CL/cl.h>

#include <vector>
#include <string>

#include "oclobject.hpp"

struct matrix_type {
  cl_mem data;
  cl_int heigth;
  cl_int width;
};

class nn {

  cl_uint numberOfTrainingData;
  cl_uint numberOfLayers;
    
  std::vector<cl_uint> elementsPerLayer;
  
  std::vector<cl_float> inputs;           // data input
  std::vector<cl_float> weights;          // neural network weights
  std::vector<cl_float> t;                // real output value
  std::vector<cl_float> output1;
  std::vector<cl_float> output2;
    
  OpenCLBasic *oclobjects;
  
  OpenCLDeviceMemory<cl_float> inputsBuffer;
  OpenCLDeviceMemory<cl_float> weightsBuffer;
  OpenCLDeviceMemory<cl_float> tBuffer;
  OpenCLDeviceMemory<cl_float> outputBuffer1;
  OpenCLDeviceMemory<cl_float> outputBuffer2;
    
  OpenCLProgramOneKernel *matmult;    // matrix multiplication
  OpenCLProgramOneKernel *matmult_local;    // matrix multiplication local

  OpenCLProgramOneKernel *ce;         // cross entropy
    
  void load_csv_data(const std::string & filename,
                     std::vector<cl_float> & input,
                     std::vector<cl_float> & output,
                     cl_uint &rows, cl_uint &layers,
                     std::vector<cl_uint> &elements);

  void device_memory_allocation();
        
  // FeedForward calculation matrices
  inline cl_uint get_number_of_product_matrices() { return numberOfLayers-1; }
  matrix_type FF_get_1st_matrix_for_product(cl_uint order);
  matrix_type FF_get_2nd_matrix_for_product(cl_uint order);
  matrix_type FF_get_result_matrix_for_product(cl_uint order);
  
  // matrix multiplication available kernels
  void execute_mat_mult_mmmKernel_local(matrix_type const &A,
                                        matrix_type const &B,
                                        matrix_type const &result);
  void execute_mat_mult_mmmKernel(matrix_type const &A,
                                  matrix_type const &B,
                                  matrix_type const &result);
    
  // memory transfer functions from device to host
  void device2HostWeightsTransfer();
  void device2HostOutput1Transfer();
  void device2HostOutput2Transfer();
    
 public:
  explicit nn(const std::string &filename);
  ~nn();
    
  void populate_random_weights(cl_float min, cl_float max);
  void FF();
};

#endif
