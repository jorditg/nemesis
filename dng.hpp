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
#include <cstdint>

//#include "CL/cl.hpp"

typedef unsigned cl_uint;   // remove after testing
typedef float cl_float;     // remove after testing
typedef int cl_int;         // remove after testing
void test_dng();

class rnd_bool {
    std::mt19937_64 gen;
    std::uint64_t rnd;
    std::uint8_t bits_used;
 public:    
    inline rnd_bool() {
        gen.seed(std::random_device()());
        rnd = gen();
        bits_used = 0;
    }
    
    //inline void print_rnd() { std::cout << rnd << std::endl; }
	
    inline bool next() {
        const bool ret = (rnd & 1);
        bits_used++;
        rnd >>= 1;
        if(bits_used == 64) {
            rnd = gen();
            bits_used = 0;
        }
	return ret;
    }
};


/** Dropout Network Generator:
 */
class dng {
 public:
    dng(std::vector<cl_uint> &el,
         std::vector<cl_float> &w,
         std::vector<cl_uint> &w_off,
         std::vector<cl_float> &b,
         std::vector<cl_uint> &b_off,
         std::vector<cl_float> &inc_w);
    
    void dropout_neurons();  
    
    void update_from_last_dropout();  
    void transfer_all_weights_to_nn();
    
 private:
    std::vector<cl_uint> &elementsPerLayerActualEpoch;
    std::vector<cl_float> &weightsActualEpoch;
    std::vector<cl_uint> &weightsOffsetsActualEpoch;
    std::vector<cl_float> &biasActualEpoch;
    std::vector<cl_uint> &biasOffsetsActualEpoch;
    std::vector<cl_float> &incrementWeightsActualEpoch;
    
    std::vector<std::vector<cl_uint> > indexes;

    std::vector<cl_uint> elementsPerLayer;
    std::vector<cl_float> weightsAll;
    std::vector<cl_uint> weightsOffsets;
    std::vector<cl_float> biasAll;
    std::vector<cl_uint> biasOffsets;
    std::vector<cl_float> incrementWeightsAll;
    
    rnd_bool rndbool;
    
    inline cl_uint get_idx(cl_uint offset, cl_uint row, cl_uint col, cl_uint nr_cols) {
        return offset + row * nr_cols + col;
    };
    void calculate_offsets_actual_epoch();
};

#endif  /* DNG_H */
