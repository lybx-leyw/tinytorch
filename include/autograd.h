#ifndef __AUTOGRAD_H__
#define __AUTOGRAD_H__
#include "nn.h"

// 统一的计算图
typedef struct _operation 
{
    LAYERTYPE layer_type;
    void* params;
    Tensor* input;
    struct _operation* next_op;
    struct _operation* last_op;
} Operation;

// 统一前向传播函数
Tensor* forward(LAYERTYPE layer_type,void* params,Tensor* input);

// 统一后向传播函数
Tensor* backward(LAYERTYPE layer_type,void* params,Tensor* input,Tensor* upstream_grad);

// 统一的添加计算层的函数
void add_layer(LAYERTYPE layer_type,void* params,Operation* model);

// 传入整个model，利用计算图前向传播
Tensor* model_forward(Operation* model,Tensor* input);

// 自动微分
void model_backward(Operation* model,Tensor* upstream_grad);

#endif