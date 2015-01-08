/* 
 * File:   CLI.cpp
 * Author: jdelatorre
 * 
 * Created on 23 de diciembre de 2014, 14:53
 */

#include <string>
#include <sstream>
#include <iostream>
#include <string>
#include <stdexcept>      // std::invalid_argument

#include "CL/cl.hpp"

#include "cli.hpp"
#include "nn.hpp"


cli::~cli() {
    if(train_thread != nullptr) delete train_thread;
}

void cli::set(std::istringstream & is, const std::string & cmd) {    
    
    if(neural_network.isTraining()) {
        std::cout << "Error: NN training.\n";
        std::cout << "       Use <pause> or <stop> before using <set> command" << "\n";
        return;
    }
    
    bool error = false;
    
    std::string token;
    is >> token;
    
    if (token == "lr" || token == "momentum") {    
        std::string val;
        is >> val;
        cl_float v;
        try {
            v = std::stof(val);
        } catch (const std::invalid_argument& ia) {
            error = true;
        }
        if (val < 0.0f || val > 1.0f) error = true;
        
        if(!error) {
            if(token == "lr") {
                neural_network.setLR(val);
            } else {
                neural_network.setM(val);
            }
        } else {
            std::cout << "Error: Not valid value. Should be between 0.0 and 1.0\n"
        }
    } else if (token == "nag") {    // set NAG
        TODO_msg(cmd);
    } else if (token == "rule") {   // set a new rule
        TODO_msg(cmd);
    } else if (token == "nn") { // set a NN architecture
        TODO_msg(cmd);
    } else {
        unknown_command_msg(cmd);
    }
}

void cli::load (std::istringstream & is, const std::string & cmd) {
    std::string what, filepath;
    
    is >> what;
    is >> filepath; // should be "/file/path"
    if (filepath[0] == '\"' && filepath[filepath.size()-1] == '\"') {
        filepath = filepath.substr(1, filepath.size()-2);
    }
    
    if(!filepath.empty()) {
        if (what == "trainingset") {
            TODO_msg(cmd);
            return;
        } else if (what == "testset") {
            TODO_msg(cmd);
            return;
        } else if (what == "nn") {
            TODO_msg(cmd);
            return;
        }
    }
    // if no return before => command error
    unknown_command_msg(cmd);
}

void cli::save (std::istringstream & is, const std::string & cmd) {
    std::string what, filepath;
    
    is >> what;
    is >> filepath; // should be "/file/path"
    if (filepath[0] == '\"' && filepath[filepath.size()-1] == '\"') {
        filepath = filepath.substr(1, filepath.size()-2);
    }
    
    if(!filepath.empty()) {
        if (what == "nn") {
            TODO_msg(cmd);
            return;
        } 
    }
    // if no return before => command error
    unknown_command_msg(cmd);    
}

void cli::train(std::istringstream & is, const std::string & cmd) {
    std::string what;
    
    is >> what;
    
    if (what == "run") {
        train_thread = new std::thread t(nn::train, &neural_network);
    } else if (what == "pause") {
        TODO_msg(cmd);
    } else if (what == "stop") {
        TODO_msg(cmd);        
    } else {
        unknown_command_msg(cmd);
    }
}

void cli::loop() {
    
    std::string token, cmd;    
    
    do {
        if (!getline(std::cin, cmd)) // Block here waiting for input
            cmd = "quit";
        
        std::istringstream is(cmd);

        token.clear();  // getline() could return empty or blank line
        is >> std::skipws >> token;

        if (token == "quit") {
            break;
        } else if (token == "set") {
            set(is, cmd);
        } else if (token == "load") {
            load(is, cmd);
        } else if (token == "save") {
            save(is, cmd);
        } else if (token == "train") {
            train(is, cmd);
        } else {
            unknown_command_msg(cmd);
        }
    } while (token != "quit");

}
