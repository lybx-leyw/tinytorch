#include "nn.h"
#define Calloc(n,type) ((type*)calloc(n,sizeof(type)))

float cross_entropy_loss(Tensor* input,Tensor* target)
{
    Tensor* pred = tensor_softmax(input);
    int d_size = pred->shape[pred->ndim-1];
    int data_size = target->size;
    float tatal_loss = 0;

    Tensor* input_grad = tensor_clone(pred);
    for (int i=0; i<data_size; i++) {
        int index = (int)(d_size*i+target->data[i]);
        tatal_loss += -log(pred->data[index]);
        input_grad->data[index] -= 1;
    }
    
    if (input->grad) tensor_free(input->grad);
    input->grad = input_grad;

    tensor_free(pred);
    return tatal_loss/target->size;
}