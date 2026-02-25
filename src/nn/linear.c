#include "nn.h"
#define Calloc(n,type) ((type*)calloc(n,sizeof(type)))

// 初始化
LinearLayer* linear_create(int in_features, int out_features)
{
    // weight:[1,in,out],bias:[1,out]
    int weight_shape[] = {in_features,out_features};
    int bias_shape[] = {out_features};

    LinearLayer* ret_layer = Calloc(1,LinearLayer);
    ret_layer->weight = tensor_rand(weight_shape,2,-1,1);
    ret_layer->bias = tensor_rand(bias_shape,1,-1,1);

    ret_layer->in_features = in_features;
    ret_layer->out_features = out_features;
    return ret_layer;
}

// 正向传播顺便计算初始梯度
Tensor* linear_forward(LinearLayer* layer, Tensor* input)
{
    // 清空梯度
    if (layer->weight->grad) tensor_free(layer->weight->grad);
    if (layer->bias->grad) tensor_free(layer->bias->grad);
    if (input->grad) tensor_free(input->grad);
    
    Tensor* weight_grad = tensor_clone(input); //weight相对x的梯度就是x的转置
    tensor_transpose(weight_grad);
    int bias_grad_shape[] = {1,1};

    Tensor* bias_grad = tensor_ones(bias_grad_shape,2);
    layer->weight->grad = weight_grad;
    layer->bias->grad = bias_grad;

    Tensor* input_grad = tensor_clone(layer->weight);
    tensor_transpose(input_grad);
    input->grad = input_grad;

    // 前向传播
    Tensor* ret_1 = tensor_matmul(input,layer->weight);
    Tensor* ret_2 = tensor_add(ret_1,layer->bias);
    tensor_free(ret_1);
    return ret_2;
}

Tensor* linear_backward(LinearLayer* layer,Tensor* input,Tensor* upstream_grad)
{
    tensor_free(layer->bias->grad);
    layer->bias->grad = upstream_grad;

    Tensor* new_grad_1 = tensor_matmul(layer->weight->grad,upstream_grad);
    tensor_free(layer->weight->grad);
    layer->weight->grad = new_grad_1;

    Tensor* new_grad_2 = tensor_matmul(upstream_grad,input->grad);
    tensor_free(input->grad);
    input->grad = new_grad_2;
    return new_grad_2;
}

void linear_sgd_update(LinearLayer* layer, float lr)
{
    if (layer->weight->requires_grad) {
        Tensor* step1 = tensor_scalar_mul(layer->weight->grad,-lr);
        Tensor* new_weight = tensor_add(layer->weight,step1);
        tensor_free(layer->weight);
        tensor_free(step1);
        layer->weight = new_weight;
    }
    if (layer->bias->requires_grad) {
        Tensor* step2 = tensor_scalar_mul(layer->bias->grad,-lr);
        Tensor* new_bias = tensor_add(layer->bias,step2);
        tensor_free(layer->bias);
        tensor_free(step2);
        layer->bias = new_bias;
    }
}