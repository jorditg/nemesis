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


template<typename T>
struct matrix {
  std::vector<T> hostData;  
  cl::Buffer * deviceData;
  cl_int rows;
  cl_int cols;
  cl_int offset;    // offset where data begins
  
  inline matrix() : rows(0), cols(0), offset(0) {}
  
  inline void createBuffer(const cl::Context & context, const cl_mem_flags flags) {
    deviceData = new cl::Buffer(context, 
                                flags, 
                                hostData->size()*sizeof(T), 
                                &(*hostData[0]));
  }
  
  inline void readFromDevice(cl::CommandQueue & queue) {
      queue.enqueueMapBuffer(*deviceData, 
                             CL_TRUE, 
                             CL_MAP_READ, 
                             0, 
                             hostData->size());
      queue.finish();
  }

  inline void writeToDevice(cl::CommandQueue & queue) {
      queue.enqueueMapBuffer(*deviceData, 
                             CL_TRUE, 
                             CL_MAP_WRITE, 
                             0, 
                             hostData->size());
      queue.finish();
  }
  
  inline matrix const & set(cl_int r, cl_int c, cl_int o = 0) {
      rows = r; cols = c; offset = o;
      return *this;
  }

  inline ~matrix() {
      if (deviceData) delete deviceData;
  }
};

inline void print_vector(const std::vector<cl_float> &v, int cols) {
  for (size_t i = 0; i < v.size(); i++) {
    std::cout << boost::format("%5.5f") % v[i] << " ";
    if (!((i+1) % cols)) std::cout << std::endl;
  }
}

void load_csv_data(const std::string & filename,
                   std::vector<cl_float> & input,
                   std::vector<cl_float> & output,
                   unsigned &rows, 
                   unsigned &layers,
                   std::vector<unsigned> &elements) {
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

void load_csv_vector(const std::string & filename, 
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


#endif  /* COMMON_HPP */

