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
  
  inline host_device_memory_map(std::vector<T> & v) : hostData(v) {}

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

  inline void writeToDevice(cl::CommandQueue & queue) {
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

typedef matrix<cl_float> matrix_cl_float;

template<typename T>
struct matrix {
    host_device_memory_map<T> const &data;
    cl_uint rows;
    cl_uint cols;
    cl_uint offset;
    
    matrix(host_device_memory_map const &d) 
            : data(d), rows(0), cols(0), offset(0) {}
    
    inline matrix const & set (cl_uint r, cl_uint c, cl_uint o = 0) {
        rows = r;
        cols = c;
        offset = o;
        return *this;
    }
};

void print_vector(const std::vector<cl_float> &v, 
                  int rows, 
                  int cols, 
                  int offset = 0);

void load_csv_data(const std::string & filename,
                   std::vector<cl_float> & input,
                   std::vector<cl_float> & output,
                   cl_int &rows,
                   std::vector<cl_int> &elements);

void load_csv_vector(const std::string & filename,
                     std::vector<cl_float> &weights);

#endif  /* COMMON_HPP */

