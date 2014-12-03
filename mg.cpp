#include "mg.hpp"

#include <vector>
#include <boost/random/uniform_int_distribution.hpp>


minibatch_generator::minibatch_generator(cl_uint sourceSz, 
                                         cl_uint destSz, 
                                         std::vector<cl_uint> &destVec) : 
                                         sourceSize(sourceSz), 
                                         destSize(destSz), 
                                         dest_vector(destVec) {
    dest_vector = destVec;
    selected.resize(sourceSize);
    index.resize(sourceSize);
}

std::vector<cl_uint> & minibatch_generator::generate() {
    boost::random::uniform_int_distribution<> dist(0, sourceSize - 1);

    for(cl_uint i = 0; i < sourceSize; i++) {
        index[i] = i;
        selected[i] = false;
    }
    dest_vector.resize(destSize);
    for(cl_uint i = 0; i < destSize; i++) {
        cl_uint sel = dist(gen);
        while(selected[sel % sourceSize]) sel++;
        sel = sel % sourceSize;
        selected[sel] = true;
        dest_vector[i] = sel;            
    }

    return dest_vector;
}

