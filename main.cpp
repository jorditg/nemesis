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
    const std::string nn_file = "nn.txt";
    
    const std::string train_file = "train-images.idx3-ubyte";
    const std::string train_labels_file = "train-labels.idx1-ubyte";
    const std::string test_file = "t10k-images.idx3-ubyte";
    const std::string test_labels_file = "t10k-labels.idx1-ubyte";      
    
    //nn1.test_matrix_multiplication(1024, 512, 512, 64);
    
    nn nn1(nn_file, train_file, train_labels_file, test_file, test_labels_file);
    //nn1.load_weights(weights_file);
    
    //nn1.test_matrix_multiplication(128, 64, 64, 80);
    nn1.populate_normal_random_weights();
    nn1.train();
    
    return 0;
}

