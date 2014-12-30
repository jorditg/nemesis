/* 
 * File:   dng.cpp
 * Author: jordi
 * 
 * Created on 29 de diciembre de 2014, 17:38
 */

#include "dng.hpp"

#include <random>
#include <cstdint>

dng::dng(std::vector<cl_uint> &el,
         std::vector<cl_float> &w, 
         std::vector<cl_uint> &w_off) 
        : elementsPerLayerActualEpoch(el),
          weightsActualEpoch(w), 
          weightsOffsetsActualEpoch(w_off),
          elementsPerLayer(el),
          weightsAll(w),
          weightsOffsets(w_off) {
    std::random_device rd;
    gen.seed(rd());
    indexes.resize(elementsPerLayer.size());  // excluding output (no dropout in output)
    for(cl_uint i = 0; i < elementsPerLayer.size(); i++) {
        indexes.reserve(elementsPerLayer[i]);
    }    
}

// A pesar de ser bueno porque te da las neuronas con una probabilidad de 0,5 
// no es bueno para nuestro algoritmo porque requerimos matrices múltiplos de
// 8 y es raro que se de e3sa coincidencia. Utilizaremos por tanto uno que 
// reporte medida fija igual a la mitad exacta de las neuronas de cada capa
// de esta forma será más fácil controlar que cumplen con el requisito de 
// multiplos de 8

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
    for (cl_uint l = 1; l < N; l++) {
        indexes[l].clear();
        if (l < (N - 1)) {
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
            // output layer
            indexes[N-1].clear();
            for(cl_uint e = 0; e < elementsPerLayer[N-1]; e++) {
                indexes[N-1].push_back(e);
            }
        }
        elementsPerLayerActualEpoch[l] = indexes[l].size();        
        for(cl_uint i = 0; i < indexes[l-1].size(); i++) {
            const cl_uint ciae = weightsOffsetsActualEpoch[l-1] + 
                                 elementsPerLayerActualEpoch[l]*i;
            const cl_uint ci = weightsOffsets[l-1] + 
                               indexes[l-1][i]*elementsPerLayer[l];
            for(cl_uint j = 0; j < indexes[l].size(); j++) {
                weightsActualEpoch[ciae + j] =
                        weightsAll[ci + indexes[l][j]];
            }
        }
        if(l < N-1) {
            weightsOffsetsActualEpoch[l] = weightsOffsetsActualEpoch[l-1] + 
                                       elementsPerLayerActualEpoch[l-1]*
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
                               indexes[l-1][i]*elementsPerLayer[l];
            for(cl_uint j = 0; j < indexes[l].size(); j++) {
                weightsAll[ci + indexes[l][j]] = 
                                weightsActualEpoch[ciae + j];                        
            }
        }
    
    }
}
