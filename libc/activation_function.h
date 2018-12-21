#ifndef _ACTIVATION_FUNCTION_H_
#define _ACTIVATION_FUNCTION_H_

#include "tensor.h"

typedef enum{
    _no_activation_function,
    _softmax,
    _relu,
    _tanh,
    _sigmoid
}activation_function;

void softmax(tensor* input,tensor* output);
void ReLu(tensor* in,tensor* out);
void tensortanh(tensor* in,tensor* out);
void sigmoid(tensor* in,tensor* out);

inline void _activationfunction(activation_function f,tensor* in,tensor* out);

#endif
