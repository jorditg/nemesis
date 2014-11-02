/* 
 * File:   common.hpp
 * Author: jdelatorre
 *
 * Created on 23 de octubre de 2014, 12:36
 */

#ifndef COMMON_HPP
#define COMMON_HPP


#include <CL/cl.hpp>
#include <boost/format.hpp>
#include <vector>
#include <iostream>



struct matrix_type {
  cl::Buffer const & data;
  cl_int rows;
  cl_int cols;
  cl_int offset;    // offset where data begins
  
  inline matrix_type(cl::Buffer const &d, cl_int h, cl_int w, cl_int o = 0) :
      data(d), rows(h), cols(w), offset(o) {}
};

inline void print_vector(const std::vector<cl_float> &v, int cols) {
  for (size_t i = 0; i < v.size(); i++) {
    std::cout << boost::format("%5.5f") % v[i] << " ";
    if (!((i+1) % cols)) std::cout << std::endl;
  }
}

#endif  /* COMMON_HPP */

