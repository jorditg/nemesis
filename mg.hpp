/* 
 * File:   minibatch_generator.hpp
 * Author: jdelatorre
 *
 * Created on 26 de noviembre de 2014, 10:56
 */

#ifndef MG_HPP
#define	MG_HPP

#include <vector>
#include <boost/random/mersenne_twister.hpp>

class minibatch_generator {
    boost::random::mt19937 gen;
    unsigned sourceSize;
    unsigned destSize;
    std::vector<bool> selected;
    std::vector<unsigned> index;

public:
    minibatch_generator(unsigned source, unsigned dest);
    std::vector<unsigned> & generate(std::vector<unsigned> & dest_vector);
};

#endif	/* MINIBATCH_GENERATOR_HPP */

