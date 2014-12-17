/* 
 * File:   mnist.hpp
 * Author: jordi
 *
 * Created on 17 de diciembre de 2014, 22:00
 */

#ifndef MNIST_HPP
#define	MNIST_HPP

#include <string>
#include <vector>

void read_mnist_images_file(const std::string filename, 
                            std::vector<float> &v, 
                            size_t &r, 
                            size_t &c);

void read_mnist_labels_file(const std::string filename, 
                            std::vector<float> &v, 
                            size_t &r, 
                            size_t &c);

#endif	/* MNIST_HPP */

