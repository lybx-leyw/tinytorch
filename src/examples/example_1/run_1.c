#include "nn.h"
#include "optim.h"
#include "autograd.h"
#define Calloc(n,type) ((type*)calloc(n,sizeof(type)))

Operation* create_model(LinearLayer* fc1,LinearLayer* fc2)
{
    Operation* model = Calloc(1,Operation);
    add_layer(LINEAR_LAYER,fc1,model);
    add_layer(RELU_ACTIVE_LAYER,NULL,model);
    add_layer(LINEAR_LAYER,fc2,model);
    return model;
}
float model_run(Tensor* input,int lable,float lr,Operation* model) 
{
    Tensor* out = model_forward(model,input);

    int shape = 1;
    Tensor* target = tensor_create((float*)&lable,&shape,1);
    float loss = cross_entropy_loss(out,target);
    Tensor* loss_grad = out->grad;

    model_backward(model,loss_grad);
    model_step(model,lr);

    tensor_free(out);
    tensor_free(target);
    return loss;
}