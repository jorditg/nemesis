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
  cl_int height;
  cl_int width;
  
  inline matrix_type(cl::Buffer const &d, cl_int h, cl_int w) :
      data(d), height(h), width(w) {}
};

inline void print_vector(const std::vector<cl_float> &v, int cols) {
  for (size_t i = 0; i < v.size(); i++) {
    std::cout << boost::format("%3.3f") % v[i] << " ";
    if (!((i+1) % cols)) std::cout << std::endl;
  }
}

#endif  /* COMMON_HPP */

