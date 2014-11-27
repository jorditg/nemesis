/*
 * File:   main.cpp
 * Author: jordi
 *
 * Created on 9 de octubre de 2014, 21:24
 */

#include <cstdlib>
#include <string>
#include "nn.hpp"

/*
 *
 */
int main(int argc, char** argv) {
    const std::string data_file = "data.txt";
    const std::string weights_file = "weights.txt";
    
    //nn1.test_matrix_multiplication(1024, 512, 512, 64);
    
    nn nn1(data_file);
    //nn1.load_weights(weights_file);
    nn1.populate_sparse_weights();
    nn1.train();
    nn1.save_weights("output_weights.txt");
    
    return 0;
}

