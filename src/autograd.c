#include "autograd.h"
#define Calloc(n,type) ((type*)calloc(n,sizeof(type)));

// 统一前向传播函数
Tensor* forward(LAYERTYPE layer_type,void* params,Tensor* input)
{
    switch (layer_type) {
        case LINEAR_LAYER:
            return linear_forward((LinearLayer*)params,input);
        case RELU_ACTIVE_LAYER:
            return relu_forward(input);
        case CONV_LAYER:
            return conv_forward((ConvLayer*)params,input);
    }
}
// 统一后向传播函数
Tensor* backward(LAYERTYPE layer_type,void* params,Tensor* input,Tensor* upstream_grad)
{
    switch (layer_type) {
        case LINEAR_LAYER:
            return linear_backward((LinearLayer*)params,input,upstream_grad);
        case RELU_ACTIVE_LAYER:
            return relu_backward(input,upstream_grad);
        case CONV_LAYER:
            return conv_backward((ConvLayer*)params,input,upstream_grad);
    }
}

void add_layer(LAYERTYPE layer_type, void* params, Operation* model)
{
    if (model->layer_type == 0 && model->params == NULL) {
        model->layer_type = layer_type;
        model->last_op = NULL;
        model->next_op = NULL;
        model->input = NULL;
        switch (layer_type) {
            case LINEAR_LAYER:
                model->params = (LinearLayer*)params;
                break;
            case CONV_LAYER:
                model->params = (ConvLayer*)params;
                break;
        }
        return;
    }
    
    while (model->next_op) {
        model = model->next_op;
    }
    model->next_op = Calloc(1,Operation);
    model->next_op->last_op = model;
    model->next_op->next_op = NULL;
    model = model->next_op;
    
    model->layer_type = layer_type;
    model->input = NULL;
    switch (layer_type) {
        case LINEAR_LAYER:
            model->params = (LinearLayer*)params;
            break;
        case CONV_LAYER:
            model->params = (ConvLayer*)params;
            break;
    }
    return;
}

Tensor* model_forward(Operation* model,Tensor* input)
{
    Tensor* new_input=NULL;
    if (model) {
        if (model->input) tensor_free(model->input);
        model->input = input;
        new_input = forward(model->layer_type,model->params,input);
        
        model = model->next_op;
    } 
    
    while (model) {
        if (model->input) tensor_free(model->input);
        model->input = new_input;
        new_input = forward(model->layer_type,model->params,new_input);
        
        model= model->next_op;
    }
    return new_input;
}

void model_backward(Operation* model,Tensor* upstream_grad)
{
    Tensor* new_upstream_grad;
    while (model->next_op) model=model->next_op;
    if (model) {
        new_upstream_grad = backward(model->layer_type,model->params,model->input,upstream_grad);
        model= model->last_op;
    } 
    
    while (model) {
        new_upstream_grad = backward(model->layer_type,model->params,model->input,new_upstream_grad);
        model= model->last_op;
    }
}