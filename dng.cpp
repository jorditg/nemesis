/* 
 * File:   dng.cpp
 * Author: jordi
 * 
 * Created on 29 de diciembre de 2014, 17:38
 */

#include "dng.hpp"

#include <random>
#include <cstdint>
#include <algorithm>

dng::dng(std::vector<cl_uint> &el,
         std::vector<cl_float> &w, 
         std::vector<cl_uint> &w_off,
         std::vector<cl_float> &b, 
         std::vector<cl_uint> &b_off) 
        : elementsPerLayerActualEpoch(el),
          weightsActualEpoch(w), 
          weightsOffsetsActualEpoch(w_off),
          biasActualEpoch(b),
          biasOffsetsActualEpoch(b_off),
          elementsPerLayer(el),
          weightsAll(w),
          weightsOffsets(w_off),
          biasAll(b),
          biasOffsets(b_off) {
    std::random_device rd;
    gen.seed(rd());
    indexes.resize(elementsPerLayer.size());  // excluding output (no dropout in output)
    for(cl_uint i = 0; i < elementsPerLayer.size(); i++) {
        indexes.reserve(elementsPerLayer[i]);
    }    
}

void dng::weigths_dropout() {
    std::uint64_t rnd = gen();
    std::uint8_t used = 0;    
    
    // Input layer without dropout. Insert in last indexes all the neurons
    elementsPerLayerActualEpoch[0] = elementsPerLayer[0];        
    weightsOffsetsActualEpoch[0] = 0;
    indexes[0].clear();
    for(cl_uint e = 0; e < elementsPerLayer[0]; e++) {
        indexes[0].push_back(e);
    }
    // Other layers random choosing
    const cl_uint N = elementsPerLayer.size();
    biasOffsetsActualEpoch[0] = 0;
    for (cl_uint l = 1; l < N; l++) {
        indexes[l].clear();
        if (l < (N - 1)) {
            // hidden layers (random choosing)
            for (cl_uint e = 0; e < elementsPerLayer[l]; e++) {
                if(rnd & 1) {
                    indexes[l].push_back(e);
                }
                rnd >>= 1;
                used++;
                if(used == 64) {
                    used = 0;
                    rnd = gen();
                }
            }
        } else {
            // output layer (all elements)
            indexes[N-1].clear();
            for(cl_uint e = 0; e < elementsPerLayer[N-1]; e++) {
                indexes[N-1].push_back(e);
            }
        }
        
        
        // size must be the next multiple of 16 (OpenCL kernel limitations)
        cl_uint sz;
        if(indexes[l].size() % 16)
            sz= (cl_uint(indexes[l].size())/16 + 1)*16;
        else
            sz = indexes[l].size();
        
        elementsPerLayerActualEpoch[l] = sz;
        for(cl_uint i = 0; i < elementsPerLayerActualEpoch[l-1]; i++) {
            const cl_uint ciae = weightsOffsetsActualEpoch[l-1] + 
                                 elementsPerLayerActualEpoch[l]*i;
            const cl_uint ci = (i >= indexes[l-1].size())?
                               0:weightsOffsets[l-1] + elementsPerLayer[l]*indexes[l-1][i];
            for(cl_uint j = 0; j < elementsPerLayerActualEpoch[l]; j++) {
                weightsActualEpoch[ciae + j] = 
                   (i >= indexes[l-1].size() || 
                    j >= indexes[l].size())?0.0f:weightsAll[ci + indexes[l][j]];
            }
        }
    
        for(cl_uint j = 0; j < sz; j++) {
            biasActualEpoch[biasOffsetsActualEpoch[l-1] + j] = 
               (j >= indexes[l].size())?0.0f:biasAll[biasOffsets[l-1] + indexes[l][j]];
        }
        
        if(l < N-1) {
            weightsOffsetsActualEpoch[l] = weightsOffsetsActualEpoch[l-1] + 
                                       elementsPerLayerActualEpoch[l-1]*
                                       elementsPerLayerActualEpoch[l];
            biasOffsetsActualEpoch[l] = biasOffsetsActualEpoch[l-1] + 
                                        elementsPerLayerActualEpoch[l];
        }
    }  
}

void dng::weights_update_from_last_dropout() {
    for (cl_uint l = 1; l < elementsPerLayer.size(); l++) {
        for(cl_uint i = 0; i < indexes[l-1].size(); i++) {
            const cl_uint ciae = weightsOffsetsActualEpoch[l-1] + 
                                 elementsPerLayerActualEpoch[l]*i;
            const cl_uint ci = weightsOffsets[l-1] + 
                               elementsPerLayer[l]*indexes[l-1][i];
            for(cl_uint j = 0; j < indexes[l].size(); j++) {
                weightsAll[ci + indexes[l][j]] = 
                                weightsActualEpoch[ciae + j];                        
            }
        }
        for(cl_uint j = 0; j < indexes[l].size(); j++) {
            biasAll[biasOffsets[l-1] + indexes[l][j]] = 
                    biasActualEpoch[biasOffsetsActualEpoch[l-1] + j]; 
        }    
    }
}

void dng::transfer_all_weights_to_nn() {
    weightsActualEpoch.resize(weightsAll.size());
    std::copy(weightsAll.begin(), weightsAll.end(), weightsActualEpoch.begin());
    std::copy(elementsPerLayer.begin(), elementsPerLayer.end(), elementsPerLayerActualEpoch.begin());
    std::copy(weightsOffsets.begin(), weightsOffsets.end(), weightsOffsetsActualEpoch.begin());

    biasActualEpoch.resize(biasAll.size());
    std::copy(biasAll.begin(), biasAll.end(), biasActualEpoch.begin());
    std::copy(biasOffsets.begin(), biasOffsets.end(), biasOffsetsActualEpoch.begin());
}
