#include "nn.h"

Tensor* relu_forward(Tensor* input)
{
    if (input->grad) tensor_free(input->grad);
    Tensor* input_grad = tensor_relu_index(input);
    input->grad = input_grad;

    Tensor* ret = tensor_relu(input);
    return ret;
}

Tensor* relu_backward(Tensor* input,Tensor* upstream_grad)
{
    Tensor* new_grad = tensor_mul(upstream_grad,input->grad);
    tensor_free(input->grad);
    input->grad = new_grad;

    return new_grad;
}