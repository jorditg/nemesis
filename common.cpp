#include "common.hpp"

#include <string>
#include <vector>

typedef boost::tokenizer< boost::escaped_list_separator<char> > Tokenizer;

void load_csv_data(const std::string & filename,
                   std::vector<cl_float> & input,
                   std::vector<cl_float> & output,
                   cl_int &rows,
                   cl_int &layers,
                   std::vector<cl_int> &elements) {
    
    std::ifstream in(filename.c_str());
    if (!in.is_open()) {
        std::cout << "File already opened. Exiting\n";
        exit(1);
    }

    std::string line;

    getline(in, line);               // read number of data lines
    rows = std::stoi(line);
        
    std::vector< std::string > vec;

    getline(in, line);              // read elements per layer

    Tokenizer tok(line);
    vec.assign(tok.begin(), tok.end());

    layers = 0;
    for (std::vector<std::string>::iterator it = vec.begin() ;
         it != vec.end(); ++it) {
        elements.push_back(std::stoi(*it) + 1 /* bias element */);
        layers++;
    }
    elements[elements.size()-1]--;  // output elements doesn't have bias

    const cl_int cols = elements[0] + elements[elements.size()-1];
    // cols to read = number of inputs + 1 (bias) + number of outputs

    cl_int n = 0;
    while (getline(in, line)) {
        line = "1," + line;     // Insert the bias element not present in file
        Tokenizer tok(line);
        vec.assign(tok.begin(), tok.end());
        // vector now contains strings from one row, output to cout here
        // std::copy(vec.begin(), vec.end(),
        //           std::ostream_iterator<std::string>(std::cout, "|"));
        // std::cout << "\n----------------------" << std::endl;

        // check that there is not incomplete data
        assert(vec.size() == size_t(cols));

        cl_int i = 0;
        for (std::vector<std::string>::iterator it = vec.begin();
             it != vec.end(); ++it) {
            if (i < elements[0]) input.push_back(std::stof(*it));
            else
              output.push_back(std::stof(*it));
            i++;
        }
        n++;
        if (n == rows) break;
    }
    
    assert((input.size() / size_t(elements[0])) == size_t(rows));
}

void load_csv_vector(const std::string & filename,
                     std::vector<cl_float> &weights) {
    
    std::ifstream in(filename.c_str());
    if (!in.is_open()) {
        std::cout << "File already opened. Exiting\n";
        exit(1);
    }

    std::string line;

    std::vector< std::string > vec;
    std::vector<cl_float>::iterator wit = weights.begin();
    
    while (getline(in, line)) {
        Tokenizer tok(line);
        vec.assign(tok.begin(), tok.end());
        // vector now contains strings from one row, output to cout here
        // std::copy(vec.begin(), vec.end(),
        //           std::ostream_iterator<std::string>(std::cout, "|"));
        // std::cout << "\n----------------------" << std::endl;

        // check that there is not incomplete data
        for (std::vector<std::string>::iterator it = vec.begin();
             it != vec.end(); ++it) {
          *wit = std::stof(*it);
          wit++;
        }
    }
    
    assert(wit == weights.end());
}


void print_vector(const std::vector<cl_float> &v,
                  cl_int rows,
                  cl_int cols,
                  cl_int offset = 0) {
  int lines = 0;
  for (size_t i = offset; i < v.size(); i++) {
    std::cout << boost::format("%5.5f") % v[i] << " ";
    if (!((i+1 - offset) % cols)) {
        std::cout << std::endl;
        lines++;
    }
    if (lines == rows ) break;
  }
}

void print(const matrix_cl_float &m,
           const std::string header,
           const bool rows2cols) {
  if (!header.empty())
    std::cout << header << std::endl;

  if (!rows2cols)
    print_vector(m.data.hostData, m.rows, m.cols, m.offset);
  else
    print_vector(m.data.hostData, m.cols, m.rows, m.offset);
}
