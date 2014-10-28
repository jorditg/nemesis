/* 
 * File:   common.hpp
 * Author: jdelatorre
 *
 * Created on 23 de octubre de 2014, 12:36
 */

#ifndef COMMON_HPP
#define	COMMON_HPP

#include <CL/cl.hpp>

struct matrix_type {
  cl::Buffer & data;
  cl_int height;
  cl_int width;
  
  inline matrix_type(cl::Buffer &d, cl_int h, cl_int w) : data(d), height(h), width(w) {};
};

#endif	/* COMMON_HPP */

