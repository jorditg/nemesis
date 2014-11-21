/* 
 * File:   common.hpp
 * Author: jdelatorre
 *
 * Created on 23 de octubre de 2014, 12:36
 */

#ifndef COMMON_HPP
#define COMMON_HPP

#include <CL/cl.hpp>
#include <boost/tokenizer.hpp>
#include <boost/format.hpp>
#include <cassert>
#include <vector>
#include <string>

#include <fstream>
#include <iostream>

template<typename T>
struct host_device_memory_map {
  std::vector<T> & hostData;
  cl::Buffer * deviceData;
  
  explicit inline host_device_memory_map(std::vector<T> & v) : hostData(v) {}
  
  inline host_device_memory_map(const host_device_memory_map<T> & orig) :
                                hostData(orig.hostData), 
                                deviceData(orig.deviceData) {}

  inline void createBuffer(const cl::Context & context,
                           const cl_mem_flags flags) {
    deviceData = new cl::Buffer(context,
                                flags,
                                hostData.size()*sizeof(T),
                                &hostData[0]);
  }
  
  inline void readFromDevice(const cl::CommandQueue & queue) {
      queue.enqueueReadBuffer(*deviceData,
                              CL_TRUE,
                              0,
                              hostData.size()*sizeof(cl_float),
                              &hostData[0]);
      queue.finish();
  }

  inline void writeToDevice(const cl::CommandQueue & queue) {
      queue.enqueueWriteBuffer(*deviceData,
                               CL_TRUE, 
                               0,
                               hostData.size()*sizeof(cl_float),
                               &hostData[0]);
      queue.finish();
  }
  
  inline ~host_device_memory_map() {
      if (deviceData) delete deviceData;
  }
};

template <typename T>
struct opencl_matrix {
    host_device_memory_map<T> & data;
    cl_int rows;
    cl_int cols;
    cl_int offset;
    bool colMajorOrdered = false;   // default is row major
    
    explicit inline opencl_matrix(host_device_memory_map<T> & d) :
                         data(d), rows(0), cols(0), offset(0) {}
    
    inline opencl_matrix(const opencl_matrix<T> & orig) :
                         data(orig.data), rows(orig.rows),
                         cols(orig.cols), offset(orig.offset) {}
    
    inline opencl_matrix const & set(cl_int r, cl_int c, cl_int o = 0, bool matrixInColMajorOrder = false) {
        rows = r;
        cols = c;
        offset = o;
        colMajorOrdered = matrixInColMajorOrder;
        return *this;
    }
};

typedef opencl_matrix<cl_float> matrix_cl_float;

void load_csv_data(const std::string & filename,
                   std::vector<cl_float> & input,
                   std::vector<cl_float> & output,
                   cl_int &rows,
                   cl_int &layers,
                   std::vector<cl_int> &elements);

void load_csv_vector(const std::string & filename,
                     std::vector<cl_float> &weights);

void print_vector(const std::vector<cl_float> & v,
                  cl_int rows,
                  cl_int cols,
                  cl_int offset);

void print(const matrix_cl_float &m,
           const std::string header = "",
           const bool rows2cols = false);

#endif  /* COMMON_HPP */

