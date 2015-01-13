/* 
 * File:   dng.h
 * Author: jordi
 *
 * Created on 29 de diciembre de 2014, 17:38
 */

#ifndef DNG_HPP
#define DNG_HPP

#include <random>
#include <vector>
#include "CL/cl.hpp"

/** Dropout Network Generator:
 */
class dng {
 public:
    dng(std::vector<cl_uint> &el,
         std::vector<cl_float> &w,
         std::vector<cl_uint> &w_off,
         std::vector<cl_float> &b,
         std::vector<cl_uint> &b_off);
    
    void weigths_dropout();  // get a weight vector with probability of 0.5
                             // to get a neuron dropped
    void weights_update_from_last_dropout();  // get all weights ready for inference (halfed)
    
    void transfer_all_weights_to_nn();
    
 private:
    std::vector<cl_uint> &elementsPerLayerActualEpoch;
    std::vector<cl_float> &weightsActualEpoch;
    std::vector<cl_uint> &weightsOffsetsActualEpoch;
    std::vector<cl_float> &biasActualEpoch;
    std::vector<cl_uint> &biasOffsetsActualEpoch;
    std::vector<std::vector<cl_uint> > indexes;

    std::vector<cl_uint> elementsPerLayer;
    std::vector<cl_float> weightsAll;
    std::vector<cl_uint> weightsOffsets;
    std::vector<cl_float> biasAll;
    std::vector<cl_uint> biasOffsets;
        
    std::mt19937_64 gen;
};

#endif  /* DNG_H */
