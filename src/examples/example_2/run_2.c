#include "nn.h"
#include "optim.h"
#include "autograd.h"
#define Calloc(n,type) ((type*)calloc(n,sizeof(type)))

Operation* create_conv_model(ConvLayer* conv,LinearLayer* fc)
{
    Operation* model = Calloc(1,Operation);
    add_layer(CONV_LAYER,conv,model);
    add_layer(RELU_ACTIVE_LAYER,NULL,model);
    add_layer(LINEAR_LAYER,fc,model);
    return model;
}

float model_run(Tensor* input,int label,float lr,Operation* model)
{
    Tensor* out = model_forward(model,input);

    int shape = 1;
    Tensor* target = tensor_create((float*)&label,&shape,1);
    float loss = cross_entropy_loss(out,target);
    Tensor* loss_grad = out->grad;

    model_backward(model,loss_grad);
    model_step(model,lr);

    tensor_free(out);
    tensor_free(target);
    return loss;
}



