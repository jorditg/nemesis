/* 
 * File:   dng.cpp
 * Author: jordi
 * 
 * Created on 29 de diciembre de 2014, 17:38
 */

#include <algorithm>
#include <vector>
#include <iostream> // can be removed after testing
#include <iomanip>      // std::setprecision

#include "dng.hpp"

dng::dng(std::vector<cl_uint> &el,
         std::vector<cl_float> &w,
         std::vector<cl_uint> &w_off,
         std::vector<cl_float> &b,
         std::vector<cl_uint> &b_off,
         std::vector<cl_float> &inc_w)
        : elementsPerLayerActualEpoch(el),
          weightsActualEpoch(w),
          weightsOffsetsActualEpoch(w_off),
          biasActualEpoch(b),
          biasOffsetsActualEpoch(b_off),
          incrementWeightsActualEpoch(inc_w),
          elementsPerLayer(el),
          weightsAll(w),
          weightsOffsets(w_off),
          biasAll(b),
          biasOffsets(b_off),
          incrementWeightsAll(inc_w) {
    // reserve enough memory to select all the neurons
    // except output layer where we don't do dropout
    indexes.resize(elementsPerLayer.size());
    for (cl_uint i = 0; i < indexes.size(); i++) {
        indexes.reserve(elementsPerLayer[i]);
    }
}

void dng::calculate_offsets_actual_epoch() {
    const cl_uint numberOfLayers = elementsPerLayer.size();    
    // calculate offsets of every layer inside the vectors
    weightsOffsetsActualEpoch.resize(numberOfLayers - 1);
    biasOffsetsActualEpoch.resize(numberOfLayers - 1);
    weightsOffsetsActualEpoch[0] = 0;
    biasOffsetsActualEpoch[0] = 0;
    for (cl_uint i = 1; i < numberOfLayers; i++) {
      weightsOffsetsActualEpoch[i] = weightsOffsetsActualEpoch[i-1] +
                           elementsPerLayerActualEpoch[i-1]*elementsPerLayerActualEpoch[i];
      biasOffsetsActualEpoch[i] = biasOffsetsActualEpoch[i-1] + elementsPerLayerActualEpoch[i];
    }
}

void dng::dropout_neurons() {
    const cl_uint layers = elementsPerLayer.size();
    
    // Input layer without dropout. Insert in last indexes all the neurons
    elementsPerLayerActualEpoch[0] = elementsPerLayer[0];    
    indexes[0].clear();
    for (cl_uint e = 0; e < elementsPerLayer[0]; e++)
        indexes[0].push_back(e);
        
    // output layer without dropout. Insert in last indexes all the neurons
    elementsPerLayerActualEpoch[layers-1] = elementsPerLayer[layers-1];
    indexes[layers-1].clear();
    for (cl_uint e = 0; e < elementsPerLayer[layers-1]; e++)
        indexes[layers-1].push_back(e);

    // hidden layers random choosing    
    for (cl_uint l = 1; l < layers - 1; l++) {
        indexes[l].clear();
        // hidden layers (random choosing)
        for (cl_uint e = 0; e < elementsPerLayer[l]; e++) {
            if (rndbool.next())
                indexes[l].push_back(e);
        }
        // size must be the next multiple of 8 (OpenCL kernel limitations)
        cl_uint sz = (indexes[l].size() % 16)?
                     (cl_uint(indexes[l].size())/16 + 1)*16 :
                     indexes[l].size();
        
        elementsPerLayerActualEpoch[l] = sz;
    }    
    
    calculate_offsets_actual_epoch();
    const cl_uint weights_sz = weightsOffsetsActualEpoch[layers - 2] +
                               elementsPerLayerActualEpoch[layers - 2] * 
                               elementsPerLayerActualEpoch[layers - 1];
    weightsActualEpoch.resize(weights_sz);
    //std::fill(weightsActualEpoch.begin(), weightsActualEpoch.end(), 0);
    
    const cl_uint bias_sz = biasOffsetsActualEpoch[layers-2] + 
                            elementsPerLayerActualEpoch[layers - 1];
    biasActualEpoch.resize(bias_sz);
    //std::fill(biasActualEpoch.begin(), biasActualEpoch.end(), 0);
    
    for(cl_uint l = 1; l < layers; l++) {
        // copy selected weights
        for (cl_uint i = 0; i < indexes[l-1].size(); i++) {            
            for (cl_uint j = 0; j < indexes[l].size(); j++) {
                const cl_uint idx_fr = get_idx(weightsOffsets[l-1],
                                               indexes[l-1][i], indexes[l][j],
                                               elementsPerLayer[l]);

                const cl_uint idx_to = get_idx(weightsOffsetsActualEpoch[l-1],
                                               i, j,
                                               elementsPerLayerActualEpoch[l]);
                        
                weightsActualEpoch[idx_to] = weightsAll[idx_fr];
                incrementWeightsActualEpoch[idx_to] = incrementWeightsAll[idx_fr];
            }
        }
        // copy selected biases
        for (cl_uint j = 0; j < indexes[l].size(); j++) {
            biasActualEpoch[biasOffsetsActualEpoch[l-1] + j] =
                biasAll[biasOffsets[l-1] + indexes[l][j]];
        }
    }    
}

void dng::update_from_last_dropout() {
    const cl_uint layers = elementsPerLayer.size();
    
    for(cl_uint l = 1; l < layers; l++) {
        // copy selected weights
        for (cl_uint i = 0; i < indexes[l-1].size(); i++) {            
            for (cl_uint j = 0; j < indexes[l].size(); j++) {
                const cl_uint idx_fr = get_idx(weightsOffsets[l-1],
                                               indexes[l-1][i], indexes[l][j],
                                               elementsPerLayer[l]);

                const cl_uint idx_to = get_idx(weightsOffsetsActualEpoch[l-1],
                                               i, j,
                                               elementsPerLayerActualEpoch[l]);
                        
                weightsAll[idx_fr] = weightsActualEpoch[idx_to];
                incrementWeightsAll[idx_fr] = incrementWeightsActualEpoch[idx_to];
            }
        }
        // copy selected biases
        for (cl_uint j = 0; j < indexes[l].size(); j++) {
            biasAll[biasOffsets[l-1] + indexes[l][j]] = 
                    biasActualEpoch[biasOffsetsActualEpoch[l-1] + j];
        }
    }    
}

void dng::transfer_all_weights_to_nn() {
    weightsActualEpoch.resize(weightsAll.size());
    std::copy(weightsAll.begin(), weightsAll.end(), weightsActualEpoch.begin());
    std::copy(elementsPerLayer.begin(),
              elementsPerLayer.end(),
              elementsPerLayerActualEpoch.begin());
    std::copy(weightsOffsets.begin(),
              weightsOffsets.end(),
              weightsOffsetsActualEpoch.begin());

    biasActualEpoch.resize(biasAll.size());
    std::copy(biasAll.begin(), biasAll.end(), biasActualEpoch.begin());
    std::copy(biasOffsets.begin(),
              biasOffsets.end(),
              biasOffsetsActualEpoch.begin());
}


//void test_dng() {  
//    auto print = [] (const std::vector<cl_float> &v,
//                     cl_uint rows,
//                     cl_uint cols,
//                     cl_uint offset) {
//        cl_uint lines = 0;
//        cl_uint end = rows*cols + offset;
//        for (size_t i = offset; i < end; i++) {
//          std::cout << v[i] << " ";
//          if (!((i+1 - offset) % cols)) {
//              std::cout << std::endl;
//              lines++;
//          }
//          if (lines == rows ) break;
//        }
//    };
//
//    std::vector<cl_uint> elementsPerLayer = {128, 64, 32, 16, 8};
//    std::vector<cl_float> weights;
//    std::vector<cl_uint> weightsOffsets = {0, 128*64, 128*64 + 64*32, 128*64 + 64*32 + 32*16};
//    std::vector<cl_float> bias;
//    std::vector<cl_uint> biasOffsets = {0, 64, 64 + 32, 64 + 32 + 16};    
//        
//    weights.resize(128*64 + 64*32 + 32*16 + 16*8);
//    bias.resize(64 + 32 + 16 + 8);
//
//    auto fill_weights = [&] (int o, int r, int c) {
//        for (int i = 0; i < r; i++) {
//            for (int j = 0; j < c; j++) {
//                weights[o + i*c + j] = cl_float((i+1)*(j+1)%10);
//            }
//        }        
//    };    
//    fill_weights(weightsOffsets[0], 128, 64);
//    fill_weights(weightsOffsets[1], 64, 32);
//    fill_weights(weightsOffsets[2], 32, 16);
//    fill_weights(weightsOffsets[3], 16, 8);
//    
//    auto fill_biases = [&] (int o, int n) {
//        for(int i = 0; i < n; i++) {
//            bias[o + i] = cl_float(i);
//        }            
//    };
//    
//    fill_biases(biasOffsets[0], 64);
//    fill_biases(biasOffsets[1], 32);
//    fill_biases(biasOffsets[2], 16);
//    fill_biases(biasOffsets[3], 8);
//    
//    dng d(elementsPerLayer, weights, weightsOffsets, bias, biasOffsets);
//    
//    std::cout << "Begin:\n";
//    
//    for(cl_uint i = 0; i < elementsPerLayer.size()-1; i++) {
//        std::cout << "Weights" << i << "\n";
//        print(weights, elementsPerLayer[i], elementsPerLayer[i+1], weightsOffsets[i]);
//        std::cout << "\n";
//    }
//    for(cl_uint i = 0; i < elementsPerLayer.size()-1; i++) {
//        std::cout << "Bias" << i << "\n";
//        print(bias, 1, elementsPerLayer[i+1], biasOffsets[i]);
//        std::cout << "\n";
//    }
//    
//    d.dropout_neurons();
//    
//    std::cout << "After dropout:\n";
//    
//    for(cl_uint i = 0; i < elementsPerLayer.size()-1; i++) {
//        std::cout << "Weights" << i << "\n";
//        print(weights, elementsPerLayer[i], elementsPerLayer[i+1], weightsOffsets[i]);
//        std::cout << "\n";
//    }
//    for(cl_uint i = 0; i < elementsPerLayer.size()-1; i++) {
//        std::cout << "Bias" << i << "\n";
//        print(bias, 1, elementsPerLayer[i+1], biasOffsets[i]);
//        std::cout << "\n";
//    }
//
//    d.update_from_last_dropout();
//    d.transfer_all_weights_to_nn();
//
//    std::cout << "End:\n";
//    for(cl_uint i = 0; i < elementsPerLayer.size()-1; i++) {
//        std::cout << "Weights" << i << "\n";
//        print(weights, elementsPerLayer[i], elementsPerLayer[i+1], weightsOffsets[i]);
//        std::cout << "\n";
//    }
//    for(cl_uint i = 0; i < elementsPerLayer.size()-1; i++) {
//        std::cout << "Bias" << i << "\n";
//        print(bias, 1, elementsPerLayer[i+1], biasOffsets[i]);
//        std::cout << "\n";
//    }    
//}
