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
    const std::string filename = "data.txt";

    nn nn1(filename);
    // nn1.FF();
    // cl_float ce = nn1.cross_entropy();
    // std::cout << "CE: " << ce << std::endl;
    nn1.test_matrix_multiplication(32, 24, 24, 8);
    //nn1.BP();
    
    return 0;
}

