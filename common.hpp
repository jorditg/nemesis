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
struct matrix {
  std::vector<T> hostData;
  cl::Buffer * deviceData;
  cl_int rows;
  cl_int cols;
  cl_int offset;    // offset where data begins
  
  inline matrix() : rows(0), cols(0), offset(0) {}
  
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
  
  inline matrix & set(cl_int r, cl_int c, cl_int o = 0) {
      rows = r;
      cols = c;
      offset = o;

      return *this;
  }

  inline ~matrix() {
      if (deviceData) delete deviceData;
  }
};

typedef matrix<cl_float> matrix_cl_float;

void print_vector(const std::vector<cl_float> &v, int rows, int cols);

void load_csv_data(const std::string & filename,
                   std::vector<cl_float> & input,
                   std::vector<cl_float> & output,
                   cl_int &rows,
                   cl_int &layers,
                   std::vector<cl_int> &elements);

void load_csv_vector(const std::string & filename,
                     std::vector<cl_float> &weights);

#endif  /* COMMON_HPP */

