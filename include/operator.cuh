// These are your operators
// Operators are C++ controllers that corresponds to computations in the model 
// They handle transfer of weights to gpu and also calling the kernel on input
// As you see function definitions are not binding, but giving you some suggestions on how to structure your code, you can proceed with your own structure as well.
// kernels are independent of operators, this make less coupling and more modularity
#pragma once

#include "config.h"
#include "prelude.h"

class AbstractOperator {
  public:
    ~AbstractOperator() {
        throw runtime_error("Destructor not implemented, beware of "
                            "weights!, fallback to derived class");
    }

    AbstractOperator() {
        throw runtime_error(
            "Constructor not implemented, fallback to derived class");
    }

    bool to_gpu() const {
        throw runtime_error(
            "to_gpu() not implemented, fallback to derived class");
    }

    bool call() {
        throw runtime_error(
            "call() not implemented, fallback to derived class");
    }

    bool is_in_gpu() {
        throw runtime_error(
            "is_in_gpu() not implemented, fallback to derived class");
    }

  private:
    bool in_gpu = false;
};

