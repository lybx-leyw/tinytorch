#ifndef __NN_H__
#define __NN_H__
#include "tensor.h"

// Linear
typedef struct _linearlayer{
    Tensor* weight;
    Tensor* bias;
    int in_features;
    int out_features;
} LinearLayer;

LinearLayer* linear_create(int in_features, int out_features);
Tensor* linear_forward(LinearLayer* layer, Tensor* input);
Tensor* linear_backward(LinearLayer* layer,Tensor* input,Tensor* upstream_grad);
void linear_sgd_update(LinearLayer* layer, float lr);

Tensor* relu_forward(Tensor* input);
Tensor* relu_backward(Tensor* input,Tensor* upstream_grad);

float cross_entropy_loss(Tensor* input,Tensor* target);

// Conv
typedef struct _convlayer{
    Tensor* weight;
    int kernel_size;
    int stride;
    int padding;
} ConvLayer;

ConvLayer* conv_create(int in_channels, int out_channels, int kernel_size, int stride, int padding);
Tensor* conv_forward(ConvLayer* layer, Tensor* input);
Tensor* conv_backward(ConvLayer* layer, Tensor* input, Tensor* upstream_grad);
void conv_sgd_update(ConvLayer* layer, float lr);
void conv_free(ConvLayer* layer);

#endif