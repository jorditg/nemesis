#include "mg.hpp"

#include <vector>
#include <boost/random/uniform_int_distribution.hpp>


minibatch_generator::minibatch_generator(unsigned source, unsigned dest) : sourceSize(source), destSize(dest) {
    selected.resize(sourceSize);
    index.resize(sourceSize);
}
std::vector<unsigned> & minibatch_generator::generate(std::vector<unsigned> & dest_vector) {
    boost::random::uniform_int_distribution<> dist(0, sourceSize - 1);

    for(unsigned i = 0; i < sourceSize; i++) {
        index[i] = i;
        selected[i] = false;
    }
    dest_vector.resize(destSize);
    for(unsigned i = 0; i < destSize; i++) {
        unsigned sel = dist(gen);
        while(selected[sel % sourceSize]) sel++;
        sel = sel % sourceSize;
        selected[sel] = true;
        index[i] = sel;            
    }

    return dest_vector;
}

